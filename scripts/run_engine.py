"""MiniServe 联调入口：python scripts/run_engine.py --check-reference。"""

from __future__ import annotations

import argparse
from time import perf_counter

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaConfig,
    LlamaForCausalLM,
)

from miniserve.benchmark import summarize_serving
from miniserve.decode_batch import DecodeBatchRunner, normalize_eos_token_ids
from miniserve.engine import Engine
from miniserve.request import Request
from miniserve.scheduler import Scheduler


def parse_args() -> argparse.Namespace:
    """输入命令行；返回配置；不改 Engine 状态，显式区分离线演示与本地模型。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", help="本地 HF 模型目录；省略则使用随机初始化的小型 Llama"
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--max-running", type=int, default=3)
    parser.add_argument(
        "--token-budget", type=int, help="每轮输入预算；小模型默认 6，本地模型默认 256"
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--check-reference",
        action="store_true",
        help="计时结束后与独立 HF generate 逐 token 比较",
    )
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.max_running <= 0:
        parser.error("max-running and max-new-tokens must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available; use --device cpu")
    if args.token_budget is None:
        args.token_budget = 256 if args.model else 6
    if args.token_budget < args.max_running:
        parser.error("token-budget must be >= max-running")
    return args


def load_model_and_prompts(args):
    """输入运行配置；返回模型、可选 tokenizer 和四组输入；初始化模型，不执行请求。"""
    torch.manual_seed(18)
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(1)
    if args.model:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        dtype = torch.float32
        if device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = (
            AutoModelForCausalLM.from_pretrained(
                args.model,
                local_files_only=True,
                dtype=dtype,
                attn_implementation="eager",
            )
            .to(device)
            .eval()
        )
        prompts = []
        for text in [
            "What is KV cache?",
            "Explain continuous batching briefly.",
            "Define TTFT.",
            "Why does decoding use a KV cache?",
        ]:
            if tokenizer.chat_template:
                encoded = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                ids = encoded["input_ids"][0].tolist()
            else:
                ids = tokenizer.encode(text)
            prompts.append(ids)
    else:
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
        tokenizer = None
        prompts = [[1, 2, 3], [1, 4, 5, 6], [1, 7], [1, 8, 9, 10, 11]]
    return model, tokenizer, prompts


@torch.inference_mode()
def check_reference(model, requests, device) -> None:
    """输入模型与已完成请求；无返回；独立 greedy 生成对照，失败抛错，不纳入计时。"""
    for request in requests:
        inputs = torch.tensor([request.prompt_token_ids], device=device)
        # 显式使用原始 greedy argmax，避免模型仓库的采样/惩罚默认配置改变对照口径。
        config = GenerationConfig(
            do_sample=False,
            repetition_penalty=1.0,
            temperature=1.0,
            top_p=1.0,
            top_k=50,
            max_new_tokens=request.max_new_tokens,
            use_cache=True,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=model.generation_config.pad_token_id or 0,
        )
        output = model.generate(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            generation_config=config,
        )[0, request.prompt_length :].tolist()
        if output != request.generated_token_ids:
            raise AssertionError(
                f"{request.request_id}: HF mismatch\nengine={request.generated_token_ids}\nHF={output}"
            )
        print(f"HF reference {request.request_id}: PASS")


def main() -> None:
    """输入命令行；无返回；完成预热、动态入队、Engine 循环与报告，串联已有模块。"""
    args = parse_args()
    model, tokenizer, prompts = load_model_and_prompts(args)
    device = torch.device(args.device)
    if max(map(len, prompts)) > args.token_budget:
        raise ValueError(
            f"Longest prompt has {max(map(len, prompts))} tokens; "
            "increase --token-budget (full prefill cannot be split)."
        )
    runner = DecodeBatchRunner(
        model=model,
        device=device,
        eos_token_ids=normalize_eos_token_ids(model.generation_config.eos_token_id),
    )

    def new_engine() -> Engine:
        """输入无；返回空 Engine；复用模型但隔离队列和 KV，避免预热污染测量。"""
        return Engine(
            scheduler=Scheduler(
                max_num_running=args.max_running,
                max_num_batched_tokens=args.token_budget,
            ),
            decode_runner=runner,
        )

    warmup = new_engine()
    warmup.add_request(Request("warmup", max(prompts, key=len), max_new_tokens=2))
    while warmup.has_unfinished_requests():
        warmup.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # 按轮次而非墙钟注入请求：0 轮 A，1 轮 B/C，2 轮 D。
    # 长短输出使 batch 成员发生变化；这是确定性演示，不是请求率压测。
    lengths = [
        args.max_new_tokens,
        min(3, args.max_new_tokens),
        args.max_new_tokens,
        min(2, args.max_new_tokens),
    ]
    requests = [
        Request(name, prompt, max_new_tokens=length)
        for name, prompt, length in zip("ABCD", prompts, lengths, strict=True)
    ]
    arrivals = [(0, requests[0]), (1, requests[1]), (1, requests[2]), (2, requests[3])]
    by_id = {r.request_id: r for r in requests}
    engine = new_engine()
    traces = []
    cursor = iteration = 0
    started = perf_counter()
    while cursor < len(arrivals) or engine.has_unfinished_requests():
        while cursor < len(arrivals) and arrivals[cursor][0] <= iteration:
            engine.add_request(arrivals[cursor][1])
            cursor += 1
        result = engine.step()
        used = sum(by_id[rid].prompt_length for rid in result.newly_prefilled) + len(
            result.decoded_tokens
        )
        assert used <= args.token_budget
        traces.append(
            (
                iteration,
                result,
                used,
                tuple(r.request_id for r in engine.scheduler.waiting),
            )
        )
        iteration += 1
    ended = perf_counter()
    report = summarize_serving(requests, workload_start=started, workload_end=ended)
    assert not engine.decode_states and not engine.scheduler.running

    # 所有打印、文本解码和 HF 对照都在计时结束后执行。
    print(
        f"Model: {args.model or 'random tiny Llama (not meaningful text)'} | device={device}"
    )
    print(f"max_running={args.max_running}, token_budget={args.token_budget}")
    for index, output, used, waiting in traces:
        print(
            f"step={index:02d} prefill={output.newly_prefilled} "
            f"decode={tuple(output.decoded_tokens)} tokens={used}/{args.token_budget} "
            f"waiting={waiting} running={output.running_requests} finished={output.finished_requests}"
        )
    for request in requests:
        result = (
            tokenizer.decode(request.generated_token_ids, skip_special_tokens=True)
            if tokenizer
            else request.generated_token_ids
        )
        print(
            f"{request.request_id}: prompt={request.prompt_length} output={request.num_generated_tokens} result={result}"
        )
    print(
        f"\nDuration: {report.duration_seconds:.4f} s | "
        f"output: {report.output_tokens_per_second:.2f} tokens/s | "
        f"requests: {report.requests_per_second:.2f} requests/s"
    )
    for name in ["queue_wait_ms", "ttft_ms", "tpot_ms", "itl_ms", "e2e_ms"]:
        stats = getattr(report, name)
        if stats is None:
            print(f"{name}: N/A (no samples)")
        else:
            print(
                f"{name}: n={len(stats.samples_ms)} mean={stats.mean_ms:.3f} "
                f"p50={stats.p50_ms:.3f} p99={stats.p99_ms:.3f}"
            )
    print(
        "Demo only: 4 requests, iteration-based arrivals; not a performance benchmark."
    )
    if args.check_reference:
        check_reference(model, requests, device)
    print("Engine integration: PASS")


if __name__ == "__main__":
    main()
