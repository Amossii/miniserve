"""在相同变长 burst workload 下比较 HF static batch 与 MiniServe continuous batching。"""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import GenerationConfig
from transformers.generation.streamers import BaseStreamer

from miniserve.benchmark import ServingMetrics, TimingStats, summarize_serving
from miniserve.decode_batch import DecodeBatchRunner
from miniserve.engine import Engine
from miniserve.request import Request
from miniserve.runtime import load_runtime
from miniserve.scheduler import Scheduler

BASE_PROMPTS = (
    "Explain how KV cache accelerates autoregressive LLM inference.",
    "Explain continuous batching in an LLM serving system.",
    "Describe the difference between prefill and decode.",
    "Explain why inference schedulers track a token budget.",
)


class BatchTokenTimingStreamer(BaseStreamer):
    """记录 HF generate 每轮 batch token 与可用时间；第一次 put 是输入 prompt，故跳过。"""

    def __init__(self) -> None:
        """输入为空；无返回；创建尚未收到 prompt/token 的 batch 记录器。"""
        self._received_prompt = False
        self.token_steps: list[list[int]] = []
        self.timestamps: list[float] = []

    def put(self, value: torch.Tensor) -> None:
        """输入 HF streamer tensor；无返回；忽略 prompt，之后每次记录整批生成 token。"""
        if not self._received_prompt:
            self._received_prompt = True
            return
        self.token_steps.append(
            [int(token_id) for token_id in value.reshape(-1).tolist()]
        )
        self.timestamps.append(time.perf_counter())

    def end(self) -> None:
        """输入为空；无返回；同步 streamer 接口，时间线已由 put 完整记录。"""


def parse_args() -> argparse.Namespace:
    """输入命令行；返回经过基本校验的实验配置；不加载模型。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="本地 Hugging Face 模型路径")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--input-lengths", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument(
        "--output-lengths", type=int, nargs="+", default=[16, 32, 64, 128]
    )
    parser.add_argument("--num-requests", type=int, default=80)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument(
        "--token-budget",
        type=int,
        help="MiniServe 每轮 token budget；默认 max(input_lengths) * max_concurrency",
    )
    parser.add_argument("--warmup-output-length", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/hf_vs_miniserve.json"),
    )
    args = parser.parse_args()
    counts = (
        args.num_requests,
        args.max_concurrency,
        args.warmup_output_length,
        args.repeats,
    )
    if min(counts) <= 0 or min(*args.input_lengths, *args.output_lengths) <= 0:
        parser.error(
            "Lengths, counts, concurrency, warmup and repeats must be positive"
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu")
    if args.token_budget is None:
        args.token_budget = max(args.input_lengths) * args.max_concurrency
    if args.token_budget < max(args.input_lengths):
        parser.error("token-budget must fit at least one complete prompt")
    return args


def build_workload(
    tokenizer: Any,
    *,
    num_requests: int,
    input_lengths: list[int],
    output_lengths: list[int],
    seed: int,
) -> tuple[list[list[int]], list[int]]:
    """输入 tokenizer、候选长度和 seed；返回可复现的变长 prompts 与输出目标。"""
    forbidden = {
        token_id
        for token_id in (tokenizer.eos_token_id, tokenizer.pad_token_id)
        if token_id is not None
    }
    rng = random.Random(seed)
    prompts: list[list[int]] = []
    targets: list[int] = []
    for index in range(num_requests):
        encoded = tokenizer.encode(BASE_PROMPTS[index % len(BASE_PROMPTS)])
        usable = [int(token_id) for token_id in encoded if token_id not in forbidden]
        if not usable:
            raise RuntimeError("Tokenizer produced no usable prompt tokens")
        input_length = rng.choice(input_lengths)
        repeats = (input_length + len(usable) - 1) // len(usable)
        prompts.append((usable * repeats)[:input_length])
        targets.append(rng.choice(output_lengths))
    return prompts, targets


def build_generation_config(model: Any, output_length: int) -> GenerationConfig:
    """输入模型与固定输出长度；返回两端共同语义的 greedy、无 EOS 提前停止配置。"""
    pad_token_id = model.generation_config.pad_token_id
    if pad_token_id is None:
        pad_token_id = 0
    return GenerationConfig(
        do_sample=False,
        use_cache=True,
        max_new_tokens=output_length,
        min_new_tokens=output_length,
        eos_token_id=None,
        pad_token_id=pad_token_id,
    )


def run_hf_static(
    model: Any,
    device: torch.device,
    prompts: list[list[int]],
    *,
    output_lengths: list[int],
    batch_size: int,
) -> tuple[list[Request], ServingMetrics]:
    """输入变长 workload；返回完成请求和指标；每个固定 batch 完成后才接收下一组。"""
    if len(prompts) != len(output_lengths):
        raise ValueError("Every prompt must have one output length")
    generation_lock = threading.Lock()
    requests = [
        Request(
            f"req-{index:04d}",
            list(prompt),
            max_new_tokens=output_lengths[index],
        )
        for index, prompt in enumerate(prompts)
    ]
    workload_start = time.perf_counter()
    for request in requests:
        request.arrival_time = workload_start
    for batch_start in range(0, len(requests), batch_size):
        batch_requests = requests[batch_start : batch_start + batch_size]
        max_output_length = max(r.max_new_tokens for r in batch_requests)
        config = build_generation_config(model, max_output_length)
        with generation_lock:
            # Static batch 成员一起 admission；短请求完成后也不在本批中补入新成员。
            for request in batch_requests:
                request.mark_running()
                request.mark_prefill_completed()
            max_input_length = max(r.prompt_length for r in batch_requests)
            pad_token_id = int(config.pad_token_id)
            padded = [
                [pad_token_id] * (max_input_length - request.prompt_length)
                + request.prompt_token_ids
                for request in batch_requests
            ]
            masks = [
                [0] * (max_input_length - request.prompt_length)
                + [1] * request.prompt_length
                for request in batch_requests
            ]
            input_ids = torch.tensor(padded, dtype=torch.long, device=device)
            attention_mask = torch.tensor(masks, dtype=torch.long, device=device)
            streamer = BatchTokenTimingStreamer()
            with torch.inference_mode():
                model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    repetition_penalty=1.0,
                    generation_config=config,
                    streamer=streamer,
                )
            if len(streamer.token_steps) != max_output_length:
                raise RuntimeError(
                    "HF static batch did not run to its maximum output length"
                )
            if any(len(step) != len(batch_requests) for step in streamer.token_steps):
                raise RuntimeError("HF streamer batch size changed during generation")
            for batch_index, request in enumerate(batch_requests):
                for step in range(request.max_new_tokens):
                    request.append_generated_token(
                        streamer.token_steps[step][batch_index],
                        timestamp=streamer.timestamps[step],
                    )
                # 请求达到自己的目标时即可返回，但 static row 仍计算到整批结束。
                request.mark_finished(
                    timestamp=streamer.timestamps[request.max_new_tokens - 1]
                )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    workload_end = time.perf_counter()
    return requests, summarize_serving(
        requests, workload_start=workload_start, workload_end=workload_end
    )


def run_miniserve(
    model: Any,
    device: torch.device,
    prompts: list[list[int]],
    *,
    output_lengths: list[int],
    max_concurrency: int,
    token_budget: int,
) -> tuple[list[Request], ServingMetrics]:
    """输入同一 workload；返回完成请求和指标；使用 dynamic KV continuous batching。"""
    if len(prompts) != len(output_lengths):
        raise ValueError("Every prompt must have one output length")
    runner = DecodeBatchRunner(model=model, device=device, eos_token_ids=set())
    engine = Engine(
        scheduler=Scheduler(
            max_num_running=max_concurrency,
            max_num_batched_tokens=token_budget,
        ),
        decode_runner=runner,
    )
    requests = [
        Request(
            f"req-{index:04d}",
            list(prompt),
            max_new_tokens=output_lengths[index],
        )
        for index, prompt in enumerate(prompts)
    ]
    workload_start = time.perf_counter()
    for request in requests:
        engine.add_request(request)
        # add_request 会记录调用时间；本实验覆盖为共同起点，表达 burst arrival。
        request.arrival_time = workload_start
    while engine.has_unfinished_requests():
        engine.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    workload_end = time.perf_counter()
    return requests, summarize_serving(
        requests, workload_start=workload_start, workload_end=workload_end
    )


def metrics_dict(metrics: ServingMetrics) -> dict[str, Any]:
    """输入 ServingMetrics；返回含原始样本和常用分位数的 JSON 字典。"""
    result = asdict(metrics)
    for name in ("ttft_ms", "tpot_ms", "e2e_ms"):
        stats = getattr(metrics, name)
        if stats is not None:
            result[name].update(p50=stats.p50_ms, p99=stats.p99_ms)
    return result


def aggregate(runs: list[dict[str, Any]], backend: str) -> dict[str, Any]:
    """输入原始重复记录；返回 workload 吞吐中位数与合并请求延迟分位数。"""
    selected = [run["metrics"] for run in runs if run["backend"] == backend]
    if not selected:
        raise ValueError(f"No runs for backend: {backend}")

    def pooled(metric: str) -> TimingStats:
        """输入指标名；返回跨重复合并原始请求样本后的统计对象。"""
        samples = [
            float(value) for item in selected for value in item[metric]["samples_ms"]
        ]
        return TimingStats(samples)

    throughputs = [float(item["output_tokens_per_second"]) for item in selected]
    request_rates = [float(item["requests_per_second"]) for item in selected]
    result: dict[str, Any] = {
        "repeats": len(selected),
        "output_tokens_per_second_median": statistics.median(throughputs),
        "output_tokens_per_second_samples": throughputs,
        "requests_per_second_median": statistics.median(request_rates),
        "requests_per_second_samples": request_rates,
    }
    for metric in ("ttft_ms", "tpot_ms", "e2e_ms"):
        stats = pooled(metric)
        result[metric] = {
            "count": len(stats.samples_ms),
            "p50": stats.p50_ms,
            "p99": stats.p99_ms,
        }
    return result


def print_summary(summary: dict[str, Any]) -> None:
    """输入聚合结果；无返回；打印适合人工检查的统一指标表。"""
    print(
        "backend output_tok/s request/s TTFT_p50 TTFT_p99 "
        "TPOT_p50 TPOT_p99 E2E_p50 E2E_p99"
    )
    for backend in ("hf_static", "miniserve_continuous"):
        item = summary[backend]
        print(
            f"{backend:20s} "
            f"{item['output_tokens_per_second_median']:12.2f} "
            f"{item['requests_per_second_median']:9.3f} "
            f"{item['ttft_ms']['p50']:9.2f} {item['ttft_ms']['p99']:9.2f} "
            f"{item['tpot_ms']['p50']:9.2f} {item['tpot_ms']['p99']:9.2f} "
            f"{item['e2e_ms']['p50']:8.2f} {item['e2e_ms']['p99']:8.2f}"
        )


def compare_outputs(
    reference: list[list[int]], candidate: list[list[int]]
) -> dict[str, Any]:
    """输入两组逐请求 token；返回请求与 token 匹配率；允许不同 batch shape 导致数值分叉。"""
    if len(reference) != len(candidate):
        raise ValueError("Backends returned different request counts")
    request_matches = sum(a == b for a, b in zip(reference, candidate, strict=True))
    total_tokens = sum(len(tokens) for tokens in reference)
    matched_tokens = sum(
        sum(left == right for left, right in zip(a, b, strict=True))
        for a, b in zip(reference, candidate, strict=True)
    )
    return {
        "exact_request_matches": request_matches,
        "num_requests": len(reference),
        "exact_request_match_rate": request_matches / len(reference),
        "matched_token_positions": matched_tokens,
        "num_token_positions": total_tokens,
        "token_match_rate": matched_tokens / total_tokens,
        "all_equal": request_matches == len(reference),
    }


def main() -> None:
    """输入命令行；无返回；预热、交错运行两后端、校验 token 并保存完整实验记录。"""
    args = parse_args()
    torch.manual_seed(args.seed)
    runtime = load_runtime(args.model, args.device)
    if runtime.tokenizer is None:
        raise RuntimeError("This comparison requires a real tokenizer/model")
    model, device = runtime.model, runtime.device
    prompts, output_lengths = build_workload(
        runtime.tokenizer,
        num_requests=args.num_requests,
        input_lengths=args.input_lengths,
        output_lengths=args.output_lengths,
        seed=args.seed,
    )
    max_positions = getattr(model.config, "max_position_embeddings", None)
    longest_request = max(
        len(prompt) + output_length
        for prompt, output_length in zip(prompts, output_lengths, strict=True)
    )
    if max_positions is not None and longest_request > max_positions:
        raise ValueError("A request exceeds model context capacity")

    # 两条路径分别预热；预热不进入正式时间窗口。
    warmup_prompts = prompts[:1]
    warmup_lengths = [args.warmup_output_length]
    run_hf_static(
        model,
        device,
        warmup_prompts,
        output_lengths=warmup_lengths,
        batch_size=args.max_concurrency,
    )
    run_miniserve(
        model,
        device,
        warmup_prompts,
        output_lengths=warmup_lengths,
        max_concurrency=args.max_concurrency,
        token_budget=args.token_budget,
    )

    runs: list[dict[str, Any]] = []
    outputs_by_backend: dict[str, list[list[int]]] = {}
    for repeat in range(args.repeats):
        # 交替顺序减轻温度、频率和固定先后次序带来的偏差。
        order = (
            ("hf_static", "miniserve_continuous")
            if repeat % 2 == 0
            else ("miniserve_continuous", "hf_static")
        )
        for backend in order:
            if backend == "hf_static":
                requests, metrics = run_hf_static(
                    model,
                    device,
                    prompts,
                    output_lengths=output_lengths,
                    batch_size=args.max_concurrency,
                )
            else:
                requests, metrics = run_miniserve(
                    model,
                    device,
                    prompts,
                    output_lengths=output_lengths,
                    max_concurrency=args.max_concurrency,
                    token_budget=args.token_budget,
                )
            outputs = [request.generated_token_ids for request in requests]
            previous = outputs_by_backend.setdefault(backend, outputs)
            if outputs != previous:
                raise AssertionError(f"{backend} outputs changed across repeats")
            runs.append(
                {"backend": backend, "repeat": repeat, "metrics": metrics_dict(metrics)}
            )

    cross_backend_correctness = compare_outputs(
        outputs_by_backend["hf_static"],
        outputs_by_backend["miniserve_continuous"],
    )
    summary = {
        backend: aggregate(runs, backend)
        for backend in ("hf_static", "miniserve_continuous")
    }
    payload = {
        "schema_version": 1,
        "config": {
            **vars(args),
            "output": str(args.output),
            "arrival": "burst",
            "sampling": "greedy",
            "eos_stopping": False,
            "hf_static_batch_size": args.max_concurrency,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": str(device),
            "cuda_device": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
            "model": args.model,
            "dtype": str(model.dtype),
            "attention": getattr(model.config, "_attn_implementation", None),
        },
        "workload": {
            "num_requests": len(prompts),
            "input_lengths": [len(prompt) for prompt in prompts],
            "output_lengths": output_lengths,
        },
        "runs": runs,
        "summary": summary,
        "correctness": {
            "cross_repeat_equal": True,
            "hf_vs_miniserve": cross_backend_correctness,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print_summary(summary)
    print(
        "HF vs MiniServe token match: "
        f"{cross_backend_correctness['token_match_rate']:.6f}; "
        f"exact requests: {cross_backend_correctness['exact_request_matches']}/"
        f"{cross_backend_correctness['num_requests']}"
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
