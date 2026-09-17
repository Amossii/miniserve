from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

import torch


T = TypeVar("T")


@dataclass
class TimingStats:
    samples_ms: list[float]

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.samples_ms)

    @property
    def min_ms(self) -> float:
        return min(self.samples_ms)

    @property
    def max_ms(self) -> float:
        return max(self.samples_ms)

    def percentile(self, p: float) -> float:
        if not 0 <= p <= 100:
            raise ValueError("p must be between 0 and 100")

        values = sorted(self.samples_ms)

        if len(values) == 1:
            return values[0]

        position = (len(values) - 1) * p / 100
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)

        weight = position - lower

        return (
            values[lower] * (1 - weight)
            + values[upper] * weight
        )

    @property
    def p50_ms(self) -> float:
        return self.percentile(50)

    @property
    def p95_ms(self) -> float:
        return self.percentile(95)

    @property
    def p99_ms(self) -> float:
        return self.percentile(99)


def benchmark_wall_clock(
    fn: Callable[[], T],
    *,
    warmup: int = 3,
    repeats: int = 10,
    synchronize_cuda: bool = True,
) -> tuple[T, TimingStats]:

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

        samples_ms.append(
            (end - start) * 1000
        )

    assert result is not None

    return result, TimingStats(samples_ms)
def benchmark_cuda_event(
    fn: Callable[[], T],
    *,
    warmup: int = 3,
    repeats: int = 10,
) -> tuple[T, TimingStats]:

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA event timing requires CUDA."
        )

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

        start_event = torch.cuda.Event(
            enable_timing=True
        )

        end_event = torch.cuda.Event(
            enable_timing=True
        )

        start_event.record()

        result = fn()

        end_event.record()

        # elapsed_time() requires the events
        # to have completed.
        torch.cuda.synchronize()

        elapsed_ms = start_event.elapsed_time(
            end_event
        )

        samples_ms.append(elapsed_ms)

    assert result is not None

    return result, TimingStats(samples_ms)
def tokens_per_second(
    num_tokens: int,
    elapsed_ms: float,
) -> float:

    if num_tokens < 0:
        raise ValueError(
            "num_tokens must be >= 0"
        )

    if elapsed_ms <= 0:
        raise ValueError(
            "elapsed_ms must be > 0"
        )

    return num_tokens / (elapsed_ms / 1000)