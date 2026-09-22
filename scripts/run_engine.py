"""MiniServe 联调入口：python scripts/run_engine.py --check-reference。"""

from __future__ import annotations

import argparse
from time import perf_counter

import torch

from miniserve.benchmark import summarize_serving
from miniserve.engine import Engine
from miniserve.request import Request
from miniserve.runtime import (
    build_engine,
    check_reference,
    load_runtime,
    resolve_token_budget,
    validate_device,
    validate_engine_limits,
)


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
    try:
        validate_device(args.device)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    args.token_budget = resolve_token_budget(args.model, args.token_budget)
    return args

def main() -> None:
    """输入命令行；无返回；完成预热、动态入队、Engine 循环与报告，串联已有模块。"""
    args = parse_args()
    runtime = load_runtime(args.model, args.device)
    tokenizer, prompts = runtime.tokenizer, runtime.prompts
    device = runtime.device
    validate_engine_limits(prompts, args.max_running, args.token_budget)

    def new_engine() -> Engine:
        """输入无；返回空 Engine；复用模型但隔离队列和 KV，避免预热污染测量。"""
        return build_engine(
            runtime,
            max_running=args.max_running,
            token_budget=args.token_budget,
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
        check_reference(runtime, requests)
    print("Engine integration: PASS")


if __name__ == "__main__":
    main()
