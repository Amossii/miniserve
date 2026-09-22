from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter


@dataclass(frozen=True)
class StepProfile:
    """一轮 Engine 的阶段耗时与工作量快照；时间单位为秒。"""

    iteration: int
    total_seconds: float
    scheduler_seconds: float
    prefill_seconds: float
    decode_seconds: float
    reclaim_seconds: float
    num_running: int
    num_prefill: int
    num_decode: int
    prefill_tokens: int
    decode_tokens: int
    scheduled_tokens: int


class EngineProfiler:
    """轻量 phase profiler；默认不挂到 Engine，避免普通路径增加计时开销。"""

    def __init__(self) -> None:
        """输入无；无返回；初始化空 trace，记录顺序与累计统计所需原始样本。"""
        self.records: list[StepProfile] = []

    def record(self, profile: StepProfile) -> None:
        """输入一轮 profile；无返回；追加不可变快照，不修改 Engine 或请求状态。"""
        self.records.append(profile)

    def reset(self) -> None:
        """输入无；无返回；清空 trace，便于区分 warmup 与正式 workload。"""
        self.records.clear()


class _PhaseTimer:
    """内部计时器；只在 profiler 启用时创建，用同一 monotonic clock 计算区间。"""

    def __init__(self) -> None:
        """输入无；返回计时器；记录起点。"""
        self.started = perf_counter()

    def elapsed(self) -> float:
        """输入无；返回起点至当前秒数；不改变起点，允许读取多个阶段结果。"""
        return perf_counter() - self.started
