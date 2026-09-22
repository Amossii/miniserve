"""Step 21：使用 PyTorch Profiler 导出 MiniServe operator/kernel trace。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from miniserve.engine import Engine
from miniserve.request import Request
from miniserve.runtime import (
    build_engine,
    load_runtime,
    resolve_token_budget,
    validate_engine_limits,
)


def main() -> None:
    """输入命令行；无返回；预热后采集固定数量 iteration，导出 trace、表格和元数据。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--max-running", type=int, default=3)
    parser.add_argument("--token-budget", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--row-limit", type=int, default=30)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/profiles/torch"),
    )
    args = parser.parse_args()
    if args.steps <= 0 or args.row_limit <= 0:
        parser.error("steps and row-limit must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu")

    runtime = load_runtime(args.model, args.device)
    model, prompts, device = runtime.model, runtime.prompts, runtime.device
    budget = resolve_token_budget(args.model, args.token_budget)
    try:
        validate_engine_limits(prompts, args.max_running, budget)
    except ValueError as error:
        parser.error(str(error))

    def new_engine(*, annotated: bool) -> Engine:
        """输入 annotation 开关；返回隔离 Engine；profile run 才启用用户区间。"""
        return build_engine(
            runtime,
            max_running=args.max_running,
            token_budget=budget,
            annotate_profiler=annotated,
        )

    warmup = new_engine(annotated=False)
    for index, prompt in enumerate(prompts[: args.max_running]):
        warmup.add_request(Request(f"warmup-{index}", list(prompt), max_new_tokens=2))
    while warmup.has_unfinished_requests():
        warmup.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    engine = new_engine(annotated=True)
    for index, prompt in enumerate(prompts):
        engine.add_request(
            Request(
                f"profile-{index}", list(prompt), max_new_tokens=args.max_new_tokens
            )
        )
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profile:
        executed = 0
        while engine.has_unfinished_requests() and executed < args.steps:
            engine.step()
            profile.step()
            executed += 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    trace_path = args.output_dir / "trace.json"
    table_path = args.output_dir / "operator_table.txt"
    metadata_path = args.output_dir / "metadata.json"
    profile.export_chrome_trace(str(trace_path))
    sort_by = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    table = profile.key_averages(group_by_input_shape=True).table(
        sort_by=sort_by, row_limit=args.row_limit
    )
    table_path.write_text(table + "\n")
    metadata_path.write_text(
        json.dumps(
            {
                "device": args.device,
                "model": args.model or "random-tiny-llama",
                "dtype": str(model.dtype),
                "torch": torch.__version__,
                "activities": [activity.name for activity in activities],
                "executed_steps": executed,
                "unfinished_after_profile": engine.has_unfinished_requests(),
                "max_running": args.max_running,
                "token_budget": budget,
                "record_shapes": True,
                "profile_memory": True,
                "sort_by": sort_by,
            },
            indent=2,
        )
        + "\n"
    )
    print(table)
    print(f"Trace: {trace_path}")
    print(f"Table: {table_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
