"""Step 22：可复现 serving benchmark，保存原始指标、策略和峰值显存。"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict
from pathlib import Path

import torch
import transformers

from miniserve.runtime import build_engine, check_reference, load_runtime
from miniserve.workload import generate_workload, run_workload, validate_workload


def parse_args():
    """输入命令行；返回已校验配置；不加载模型，尽早拒绝无效实验规模。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="本地模型路径；省略使用离线随机小型 Llama")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--num-requests", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--arrival", choices=["burst", "constant", "poisson"], default="poisson"
    )
    parser.add_argument("--request-rate", type=float, default=50)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--max-running", type=int, nargs="+", default=[3])
    parser.add_argument("--token-budgets", type=int, nargs="+")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--check-reference", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/results/serving.json")
    )
    args = parser.parse_args()
    if (
        min(args.num_requests, args.max_new_tokens, args.repeats, *args.max_running)
        <= 0
    ):
        parser.error("Counts and max-running must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu.")
    if args.token_budgets is None:
        args.token_budgets = [64, 128] if args.model else [6, 12]
    if min(args.token_budgets) < max(args.max_running):
        parser.error("Every token budget must cover every max-running value.")
    return args


def metric_dict(report):
    """输入指标快照；返回可序列化字典；保留原始样本并添加分位数，方便复核。"""
    result = asdict(report)
    for name in ["queue_wait_ms", "ttft_ms", "tpot_ms", "itl_ms", "e2e_ms"]:
        stats = getattr(report, name)
        if stats is not None:
            result[name].update(
                count=len(stats.samples_ms),
                mean=stats.mean_ms,
                p50=stats.p50_ms,
                p99=stats.p99_ms,
            )
    return result


def main():
    """输入命令行；无返回；同一模型/计划比较配置，隔离预热并保存每次结果。"""
    args = parse_args()
    runtime = load_runtime(args.model, args.device)
    model, prompts = runtime.model, runtime.prompts
    specs = generate_workload(
        prompts,
        num_requests=args.num_requests,
        max_new_tokens=args.max_new_tokens,
        arrival=args.arrival,
        request_rate=args.request_rate,
        seed=args.seed,
    )
    policies = list(
        dict.fromkeys(
            (capacity, budget)
            for capacity in args.max_running
            for budget in args.token_budgets
        )
    )

    def new_engine(capacity, budget):
        """输入策略；返回空 Engine；复用模型但不复用 Request、队列或 KV。"""
        return build_engine(
            runtime,
            max_running=capacity,
            token_budget=budget,
        )

    for capacity, budget in policies:
        validate_workload(specs, new_engine(capacity, budget))
    payload = {
        "schema_version": 2,
        "config": {**vars(args), "output": str(args.output)},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": args.device,
            "dtype": str(model.dtype),
            "cpu_threads": torch.get_num_threads(),
            "cuda_device": torch.cuda.get_device_name()
            if args.device == "cuda"
            else None,
            "model": args.model or "random-tiny-llama",
            "model_seed": 18,
            "attention": "eager",
        },
        "workload": [asdict(s) for s in specs],
        "runs": [],
    }
    baseline_tokens = reference_requests = None
    print(
        "capacity budget repeat output_tok/s planned_TTFT_p50_ms ITL_p99_ms dispatch_p99_ms"
    )
    for repeat in range(args.repeats):
        # 轮换执行顺序减轻固定次序偏差；保留每次独立结果，不平均 P99。
        order = policies[repeat % len(policies) :] + policies[: repeat % len(policies)]
        for capacity, budget in order:
            warmup_specs = generate_workload(
                prompts,
                num_requests=capacity,
                max_new_tokens=2,
                arrival="burst",
                seed=args.seed,
            )
            run_workload(new_engine(capacity, budget), warmup_specs)
            if args.device == "cuda":
                torch.cuda.synchronize()
                # 输入是当前 CUDA device；输出为空；重置统计窗口但不释放模型。
                # 这样记录的是模型常驻显存之上的本次 workload 峰值。
                torch.cuda.reset_peak_memory_stats()
            result = run_workload(new_engine(capacity, budget), specs)
            if args.device == "cuda":
                # run_workload 返回前 token 已读回 CPU，但这里仍显式同步，保证
                # allocator 统计覆盖测量窗口内提交的全部 CUDA 工作。
                torch.cuda.synchronize()
                peak_allocated_bytes = torch.cuda.max_memory_allocated()
                peak_reserved_bytes = torch.cuda.max_memory_reserved()
            else:
                peak_allocated_bytes = None
                peak_reserved_bytes = None
            actual = [r.generated_token_ids for r in result.requests]
            if baseline_tokens is None:
                baseline_tokens, reference_requests = actual, result.requests
            elif actual != baseline_tokens:
                raise AssertionError(
                    "Outputs differ across policies/repeats; investigate before comparing speed."
                )
            record = {
                "policy": "sequential" if capacity == 1 else "continuous",
                "max_running": capacity,
                "token_budget": budget,
                "repeat": repeat,
                "iterations": result.iterations,
                "peak_allocated_bytes": peak_allocated_bytes,
                "peak_reserved_bytes": peak_reserved_bytes,
                "submitted_metrics": metric_dict(result.submitted_metrics),
                "scheduled_metrics": metric_dict(result.scheduled_metrics),
                "dispatch_lag_ms": result.dispatch_lag_ms.samples_ms,
                "requests": [
                    {
                        "request_id": r.request_id,
                        "submitted_offset": r.arrival_time - result.started,
                        "admitted_offset": r.start_time - result.started,
                        "token_offsets": [
                            t - result.started for t in r.token_timestamps
                        ],
                        "finish_offset": r.finish_time - result.started,
                        "output_token_ids": r.generated_token_ids,
                    }
                    for r in result.requests
                ],
            }
            payload["runs"].append(record)
            metrics = result.scheduled_metrics
            itl = f"{metrics.itl_ms.p99_ms:.3f}" if metrics.itl_ms else "N/A"
            print(
                f"{capacity:8d} {budget:6d} {repeat:6d} {metrics.output_tokens_per_second:12.2f} "
                f"{metrics.ttft_ms.p50_ms:19.3f} {itl:>10} {result.dispatch_lag_ms.p99_ms:15.3f}"
            )
    if args.check_reference:
        check_reference(runtime, reference_requests)
    payload["correctness"] = {
        "cross_run_equal": True if len(payload["runs"]) > 1 else None,
        "hf_reference": "passed" if args.check_reference else "not_run",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"Saved: {args.output}")
    print(
        "Finite synchronous-engine benchmark; scheduled TTFT includes dispatch lag. No network overhead."
    )


if __name__ == "__main__":
    main()
