"""有限、可复现的到达计划与同步 Engine benchmark 驱动。"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from miniserve.benchmark import ServingMetrics, TimingStats, summarize_serving
from miniserve.engine import Engine
from miniserve.request import Request


@dataclass(frozen=True)
class RequestSpec:
    """不可变请求规格；offset 是相对 workload 起点的计划到达秒数，不是轮号。"""

    request_id: str
    arrival_offset: float
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int


@dataclass(frozen=True)
class WorkloadResult:
    """单次排空实验；区分实际提交后的 Engine 指标和包含提交延迟的计划到达指标。"""

    requests: tuple[Request, ...]
    started: float
    ended: float
    iterations: int
    submitted_metrics: ServingMetrics
    scheduled_metrics: ServingMetrics
    dispatch_lag_ms: TimingStats


def generate_workload(
    prompts: Sequence[Sequence[int]],
    *,
    num_requests: int,
    max_new_tokens: int,
    arrival: str = "poisson",
    request_rate: float = 50.0,
    seed: int = 19,
) -> tuple[RequestSpec, ...]:
    """输入 prompt 池、规模与到达配置；返回不可变计划；用局部 RNG 避免污染全局状态。

    首请求固定在 0；后续 poisson 间隔服从指数分布。输出上限在 [1, max] 均匀采样。
    到达与内容使用独立随机流，因此改变到达模式不会改变同种子的请求内容。
    """
    if (
        num_requests <= 0
        or max_new_tokens <= 0
        or not prompts
        or any(not p for p in prompts)
    ):
        raise ValueError("Workload needs positive counts and nonempty prompts.")
    if arrival not in {"burst", "constant", "poisson"}:
        raise ValueError("arrival must be burst, constant or poisson.")
    if not math.isfinite(request_rate) or request_rate <= 0:
        raise ValueError("request_rate must be finite and positive.")
    timing_rng, content_rng = random.Random(seed), random.Random(seed + 1)
    specs = []
    offset = 0.0
    for index in range(num_requests):
        if index:
            if arrival == "constant":
                offset = index / request_rate
            elif arrival == "poisson":
                offset += timing_rng.expovariate(request_rate)
        specs.append(
            RequestSpec(
                request_id=f"req-{index:04d}",
                arrival_offset=offset,
                prompt_token_ids=tuple(content_rng.choice(prompts)),
                max_new_tokens=content_rng.randint(1, max_new_tokens),
            )
        )
    return tuple(specs)


def validate_workload(specs: Sequence[RequestSpec], engine: Engine) -> None:
    """输入计划与目标 Engine；无返回、不修改状态；执行前拒绝非法计划或非空 Engine。"""
    if not specs:
        raise ValueError("Workload must contain at least one request.")
    scheduler = engine.scheduler
    if (
        scheduler.waiting
        or scheduler.running
        or scheduler.finished_requests
        or engine.decode_states
    ):
        raise ValueError("Each workload run requires a fresh Engine.")
    seen = set()
    previous = 0.0
    for spec in specs:
        if spec.request_id in seen:
            raise ValueError("Duplicate workload request_id.")
        seen.add(spec.request_id)
        if not math.isfinite(spec.arrival_offset) or spec.arrival_offset < previous:
            raise ValueError("Arrival offsets must be finite, nonnegative and sorted.")
        previous = spec.arrival_offset
        if not spec.prompt_token_ids or spec.max_new_tokens <= 0:
            raise ValueError("Prompt and output limit must be positive.")
        budget = scheduler.max_num_batched_tokens
        if budget is not None and len(spec.prompt_token_ids) > budget:
            raise ValueError(
                "Prompt exceeds token budget; increase budget before running."
            )


def run_workload(
    engine: Engine,
    specs: Sequence[RequestSpec],
    *,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> WorkloadResult:
    """输入空 Engine 与计划；返回两种时间边界的指标；推进队列直到全部请求完成。

    clock 必须与 Engine/Request 同源；注入参数用于假时钟测试。忙时在 step 边界接收
    所有已到期请求，闲时 sleep，不忙等。保留原计划时间，不根据实际提交推迟后续到达。
    模型加载、tokenization、warmup 和打印由调用方放在本函数外。
    """
    validate_workload(specs, engine)
    requests = tuple(
        Request(s.request_id, list(s.prompt_token_ids), max_new_tokens=s.max_new_tokens)
        for s in specs
    )
    cursor = iterations = 0
    started = clock()
    while cursor < len(specs) or engine.has_unfinished_requests():
        while cursor < len(specs) and started + specs[cursor].arrival_offset <= clock():
            engine.add_request(requests[cursor])
            cursor += 1
        if engine.has_unfinished_requests():
            engine.step()
            iterations += 1
        elif cursor < len(specs):
            sleep(min(0.05, max(0.0, started + specs[cursor].arrival_offset - clock())))
    ended = clock()
    if engine.decode_states or engine.scheduler.running:
        raise RuntimeError("Engine did not release completed request state.")
    # 复制逻辑请求来改变统计边界，绝不篡改 Engine 实际记录的 arrival_time。
    scheduled = [
        replace(r, arrival_time=started + s.arrival_offset)
        for r, s in zip(requests, specs, strict=True)
    ]
    lags = [
        (r.arrival_time - s.arrival_time) * 1000
        for r, s in zip(requests, scheduled, strict=True)
    ]
    if any(lag < 0 for lag in lags):
        raise RuntimeError(
            "Clock mismatch: request submitted before its scheduled arrival."
        )
    return WorkloadResult(
        requests=requests,
        started=started,
        ended=ended,
        iterations=iterations,
        submitted_metrics=summarize_serving(
            requests, workload_start=started, workload_end=ended
        ),
        scheduled_metrics=summarize_serving(
            scheduled, workload_start=started, workload_end=ended
        ),
        dispatch_lag_ms=TimingStats(lags),
    )
