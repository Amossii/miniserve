"""MiniServe CLI 的公共运行时装配：模型、prompt、Engine 与正确性对照。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaConfig,
    LlamaForCausalLM,
)

from miniserve.decode_batch import DecodeBatchRunner, normalize_eos_token_ids
from miniserve.engine import Engine
from miniserve.profiling import EngineProfiler
from miniserve.request import Request
from miniserve.scheduler import Scheduler


@dataclass(frozen=True)
class RuntimeBundle:
    """已加载的共享运行资源；模型只读复用，Engine/Request/KV 必须按实验重建。"""

    model: Any
    tokenizer: Any | None
    prompts: list[list[int]]
    device: torch.device
    model_name: str


def validate_device(device_name: str) -> torch.device:
    """输入 cpu/cuda 名称；返回 torch.device；不改状态并为不可用 CUDA 给出明确错误。"""
    if device_name not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this process; use --device cpu")
    return torch.device(device_name)


def load_runtime(
    model_path: str | None,
    device_name: str,
    *,
    seed: int = 18,
    cpu_threads: int = 1,
) -> RuntimeBundle:
    """输入模型路径、设备和确定性配置；返回模型、tokenizer、固定 prompts；加载一次供实验复用。"""
    if cpu_threads <= 0:
        raise ValueError("cpu_threads must be positive")
    device = validate_device(device_name)
    torch.manual_seed(seed)
    if device.type == "cpu":
        torch.set_num_threads(cpu_threads)

    if model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        dtype = torch.float32
        if device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                dtype=dtype,
                attn_implementation="eager",
            )
            .to(device)
            .eval()
        )
        prompts = [_encode_prompt(tokenizer, text) for text in _DEFAULT_TEXT_PROMPTS]
        return RuntimeBundle(model, tokenizer, prompts, device, model_path)

    config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        bos_token_id=1,
        eos_token_id=None,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).to(device).eval()
    prompts = [[1, 2, 3], [1, 4, 5, 6], [1, 7], [1, 8, 9, 10, 11]]
    return RuntimeBundle(model, None, prompts, device, "random-tiny-llama")


_DEFAULT_TEXT_PROMPTS = (
    "What is KV cache?",
    "Explain continuous batching briefly.",
    "Define TTFT.",
    "Why does decoding use a KV cache?",
)


def _encode_prompt(tokenizer: Any, text: str) -> list[int]:
    """输入 tokenizer 与文本；返回 chat-template 或普通编码 token；不保留 tensor 状态。"""
    if tokenizer.chat_template:
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        return encoded["input_ids"][0].tolist()
    return tokenizer.encode(text)


def resolve_token_budget(model_path: str | None, explicit_budget: int | None) -> int:
    """输入模型类型和可选预算；返回统一默认值；无状态，真实模型默认 256。"""
    return explicit_budget if explicit_budget is not None else (256 if model_path else 6)


def validate_engine_limits(
    prompts: list[list[int]], max_running: int, token_budget: int
) -> None:
    """输入 prompt 与调度限制；无返回；验证当前不支持拆分 prefill 的必要约束。"""
    if max_running <= 0 or token_budget < max_running:
        raise ValueError("token budget must be >= positive max_running")
    if not prompts or any(not prompt for prompt in prompts):
        raise ValueError("prompts must be non-empty")
    longest = max(map(len, prompts))
    if longest > token_budget:
        raise ValueError(
            f"Longest prompt has {longest} tokens; increase token budget "
            "because full prefill cannot be split."
        )


def build_engine(
    bundle: RuntimeBundle,
    *,
    max_running: int,
    token_budget: int,
    profiler: EngineProfiler | None = None,
    annotate_profiler: bool = False,
    kv_backend: str = "dynamic",
    num_kv_blocks: int = 256,
    kv_block_size: int = 16,
) -> Engine:
    """输入共享模型、调度和 KV backend；返回全新 Engine；队列、runner 与 KV 状态隔离。"""
    validate_engine_limits(bundle.prompts, max_running, token_budget)
    eos_token_ids = normalize_eos_token_ids(
        bundle.model.generation_config.eos_token_id
    )
    if kv_backend == "dynamic":
        runner = DecodeBatchRunner(
            model=bundle.model,
            device=bundle.device,
            eos_token_ids=eos_token_ids,
            annotate_profiler=annotate_profiler,
        )
    elif kv_backend == "paged":
        from miniserve.paged_kv import PagedDecodeBatchRunner

        runner = PagedDecodeBatchRunner(
            model=bundle.model,
            device=bundle.device,
            eos_token_ids=eos_token_ids,
            num_blocks=num_kv_blocks,
            block_size=kv_block_size,
            annotate_profiler=annotate_profiler,
        )
    else:
        raise ValueError("kv_backend must be 'dynamic' or 'paged'")
    return Engine(
        scheduler=Scheduler(
            max_num_running=max_running,
            max_num_batched_tokens=token_budget,
        ),
        decode_runner=runner,
        profiler=profiler,
        annotate_profiler=annotate_profiler,
    )


@torch.inference_mode()
def check_reference(bundle: RuntimeBundle, requests: list[Request]) -> None:
    """输入运行时和已完成请求；无返回；用独立 HF greedy generate 逐请求验证输出。"""
    for request in requests:
        inputs = torch.tensor([request.prompt_token_ids], device=bundle.device)
        config = GenerationConfig(
            do_sample=False,
            repetition_penalty=1.0,
            temperature=1.0,
            top_p=1.0,
            top_k=50,
            max_new_tokens=request.max_new_tokens,
            use_cache=True,
            eos_token_id=bundle.model.generation_config.eos_token_id,
            pad_token_id=bundle.model.generation_config.pad_token_id or 0,
        )
        output = bundle.model.generate(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            generation_config=config,
        )[0, request.prompt_length :].tolist()
        if output != request.generated_token_ids:
            raise AssertionError(
                f"{request.request_id}: HF mismatch\n"
                f"engine={request.generated_token_ids}\nHF={output}"
            )
        print(f"HF reference {request.request_id}: PASS")
