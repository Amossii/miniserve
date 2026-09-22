"""Step 20 phase profiler：记录 Engine iteration 的结构化 JSON trace。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from miniserve.profiling import EngineProfiler
from miniserve.request import Request
from miniserve.runtime import (
    build_engine,
    load_runtime,
    resolve_token_budget,
    validate_engine_limits,
)


def main() -> None:
    """输入命令行；无返回；预热后运行固定 workload，打印并保存 per-phase trace。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--max-running", type=int, default=3)
    parser.add_argument("--token-budget", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/results/profile.json")
    )
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu")
    runtime = load_runtime(args.model, args.device)
    prompts, device = runtime.prompts, runtime.device
    budget = resolve_token_budget(args.model, args.token_budget)
    try:
        validate_engine_limits(prompts, args.max_running, budget)
    except ValueError as error:
        parser.error(str(error))

    def new_engine(profiler=None):
        """输入可选 profiler；返回隔离 Engine；复用模型但不复用 KV。"""
        return build_engine(
            runtime,
            max_running=args.max_running,
            token_budget=budget,
            profiler=profiler,
        )

    warmup = new_engine()
    warmup.add_request(Request("warmup", list(max(prompts, key=len)), max_new_tokens=2))
    while warmup.has_unfinished_requests():
        warmup.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    profiler = EngineProfiler()
    engine = new_engine(profiler)
    requests = [
        Request(chr(65 + i), list(prompt), max_new_tokens=args.max_new_tokens)
        for i, prompt in enumerate(prompts)
    ]
    for request in requests:
        engine.add_request(request)
    while engine.has_unfinished_requests():
        engine.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    records = [
        {
            "iteration": record.iteration,
            "total_ms": record.total_seconds * 1000,
            "scheduler_ms": record.scheduler_seconds * 1000,
            "prefill_ms": record.prefill_seconds * 1000,
            "decode_ms": record.decode_seconds * 1000,
            "reclaim_ms": record.reclaim_seconds * 1000,
            "num_running": record.num_running,
            "num_prefill": record.num_prefill,
            "num_decode": record.num_decode,
            "prefill_tokens": record.prefill_tokens,
            "decode_tokens": record.decode_tokens,
            "scheduled_tokens": record.scheduled_tokens,
        }
        for record in profiler.records
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n")
    print(
        "iteration total_ms scheduler_ms prefill_ms decode_ms reclaim_ms prefill_tokens decode_tokens"
    )
    for record in records:
        print(
            f"{record['iteration']:9d} {record['total_ms']:8.3f} "
            f"{record['scheduler_ms']:13.3f} {record['prefill_ms']:10.3f} "
            f"{record['decode_ms']:9.3f} {record['reclaim_ms']:10.3f} "
            f"{record['prefill_tokens']:14d} {record['decode_tokens']:13d}"
        )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
