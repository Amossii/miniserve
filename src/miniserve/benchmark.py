from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import pairwise
from math import isfinite
from typing import TypeVar

import torch

from miniserve.request import Request

T = TypeVar("T")


@dataclass
class TimingStats:
    samples_ms: list[float]

    @property
    def mean_ms(self) -> float:
        """输入自身；返回非空样本均值；只读，统一毫秒单位。"""
        return statistics.mean(self.samples_ms)

    @property
    def min_ms(self) -> float:
        """输入自身；返回非空样本最小值；只读，用于延迟报告。"""
        return min(self.samples_ms)

    @property
    def max_ms(self) -> float:
        """输入自身；返回非空样本最大值；只读，显示最慢观测。"""
        return max(self.samples_ms)

    def percentile(self, p: float) -> float:
        """输入百分位；返回线性插值分位数；排序副本，保留原始非空样本顺序。"""
        if not 0 <= p <= 100:
            raise ValueError("p must be between 0 and 100")

        values = sorted(self.samples_ms)

        if len(values) == 1:
            return values[0]

        position = (len(values) - 1) * p / 100
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)

        weight = position - lower

        return values[lower] * (1 - weight) + values[upper] * weight

    @property
    def p50_ms(self) -> float:
        """输入自身；返回中位数；只读，描述典型延迟。"""
        return self.percentile(50)

    @property
    def p95_ms(self) -> float:
        """输入自身；返回 P95；只读，描述尾部延迟。"""
        return self.percentile(95)

    @property
    def p99_ms(self) -> float:
        """输入自身；返回 P99；只读，小样本分位数不能代表稳定性能。"""
        return self.percentile(99)


def benchmark_wall_clock(
    fn: Callable[[], T],
    *,
    warmup: int = 3,
    repeats: int = 10,
    synchronize_cuda: bool = True,
) -> tuple[T, TimingStats]:
    """输入函数及预热/重复配置；返回末次结果与墙钟样本；执行函数并按需同步 GPU。"""
    if warmup < 0:
        raise ValueError("warmup must be >= 0")

    if repeats <= 0:
        raise ValueError("repeats must be > 0")

    result: T | None = None

    # ------------------------------
    # Warmup
    # ------------------------------

    for _ in range(warmup):
        result = fn()

    if synchronize_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()

    # ------------------------------
    # Measurements
    # ------------------------------

    samples_ms: list[float] = []

    for _ in range(repeats):
        if synchronize_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()

        result = fn()

        if synchronize_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

        end = time.perf_counter()

        samples_ms.append((end - start) * 1000)

    assert result is not None

    return result, TimingStats(samples_ms)


def benchmark_cuda_event(
    fn: Callable[[], T],
    *,
    warmup: int = 3,
    repeats: int = 10,
) -> tuple[T, TimingStats]:
    """输入函数及重复配置；返回末次结果与 CUDA 时间；执行并同步事件以测 GPU 区间。"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA event timing requires CUDA.")

    if warmup < 0:
        raise ValueError("warmup must be >= 0")

    if repeats <= 0:
        raise ValueError("repeats must be > 0")

    result: T | None = None

    # Warmup
    for _ in range(warmup):
        result = fn()

    torch.cuda.synchronize()

    samples_ms: list[float] = []

    for _ in range(repeats):
        start_event = torch.cuda.Event(enable_timing=True)

        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        result = fn()

        end_event.record()

        # elapsed_time() requires the events
        # to have completed.
        torch.cuda.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event)

        samples_ms.append(elapsed_ms)

    assert result is not None

    return result, TimingStats(samples_ms)


def tokens_per_second(
    num_tokens: int,
    elapsed_ms: float,
) -> float:
    """输入 token 数和毫秒耗时；返回 tokens/s；无状态，拒绝无效分母。"""
    if num_tokens < 0:
        raise ValueError("num_tokens must be >= 0")

    if elapsed_ms <= 0:
        raise ValueError("elapsed_ms must be > 0")

    return num_tokens / (elapsed_ms / 1000)


@dataclass(frozen=True)
class ServingMetrics:
    """完整排空 workload 的快照；分布单位为 ms，空分布为 None，EOS 计入输出。"""

    duration_seconds: float
    num_requests: int
    num_output_tokens: int
    output_tokens_per_second: float
    requests_per_second: float
    queue_wait_ms: TimingStats | None
    ttft_ms: TimingStats | None
    tpot_ms: TimingStats | None
    itl_ms: TimingStats | None
    e2e_ms: TimingStats | None


def _latency_stats(samples_seconds: Iterable[float | None]) -> TimingStats | None:
    """输入秒级样本；返回毫秒分布或 None；不修改请求，忽略无定义指标而非填零。"""
    samples_ms = [value * 1000 for value in samples_seconds if value is not None]
    return TimingStats(samples_ms) if samples_ms else None


def summarize_serving(
    requests: Iterable[Request],
    *,
    workload_start: float,
    workload_end: float,
) -> ServingMetrics:
    """输入全部完成请求与显式窗口；返回汇总快照；只读，拒绝不完整或越界时间线。

    窗口采用同一个 perf_counter 时钟，覆盖到达间隔、排队和执行，排除 warmup。
    TTFT/TPOT/E2E 每请求一个样本；ITL 汇集所有 token 间隔，长输出权重更大。
    这是 Engine 内部 token 可用时间，不包含网络，也不支持截断窗口或失败请求统计。
    """
    if (
        not isfinite(workload_start)
        or not isfinite(workload_end)
        or workload_end <= workload_start
    ):
        raise ValueError("Workload window must be finite and have positive duration.")
    completed = list(requests)
    seen_ids: set[str] = set()
    for request in completed:
        if request.request_id in seen_ids:
            raise ValueError(f"Duplicate request_id: {request.request_id}")
        seen_ids.add(request.request_id)
        if (
            not request.is_finished
            or request.start_time is None
            or request.finish_time is None
            or not request.token_timestamps
            or len(request.token_timestamps) != request.num_generated_tokens
            or request.first_token_time != request.token_timestamps[0]
        ):
            raise ValueError(f"Incomplete request metrics: {request.request_id}")
        timeline = [
            workload_start,
            request.arrival_time,
            request.start_time,
            *request.token_timestamps,
            request.finish_time,
            workload_end,
        ]
        if not all(isfinite(value) for value in timeline) or any(
            later < earlier for earlier, later in pairwise(timeline)
        ):
            raise ValueError(f"Invalid request timeline: {request.request_id}")

    duration = workload_end - workload_start
    output_tokens = sum(r.num_generated_tokens for r in completed)
    return ServingMetrics(
        duration_seconds=duration,
        num_requests=len(completed),
        num_output_tokens=output_tokens,
        output_tokens_per_second=output_tokens / duration,
        requests_per_second=len(completed) / duration,
        queue_wait_ms=_latency_stats(r.queue_wait_seconds for r in completed),
        ttft_ms=_latency_stats(r.ttft_seconds for r in completed),
        tpot_ms=_latency_stats(r.tpot_seconds for r in completed),
        itl_ms=_latency_stats(value for r in completed for value in r.itl_seconds),
        e2e_ms=_latency_stats(r.e2e_latency_seconds for r in completed),
    )
