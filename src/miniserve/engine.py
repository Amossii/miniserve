from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from torch.profiler import record_function

from miniserve.decode_batch import DecodeBatchRunner
from miniserve.profiling import EngineProfiler, StepProfile, _PhaseTimer
from miniserve.request import Request
from miniserve.scheduler import Scheduler


@dataclass(frozen=True)
class EngineStepOutput:
    """一轮结果；decoded_tokens 只含 decode 输出，prefill 首 token 在 Request 中。"""

    newly_prefilled: tuple[str, ...]
    decoded_tokens: dict[str, int]
    finished_requests: tuple[str, ...]
    running_requests: tuple[str, ...]


class Engine:
    """每轮先逐请求 prefill，再批量 decode；同一请求一轮只执行其中一种。"""

    def __init__(
        self,
        *,
        scheduler: Scheduler,
        decode_runner: DecodeBatchRunner,
        profiler: EngineProfiler | None = None,
        annotate_profiler: bool = False,
    ) -> None:
        """输入调度器、执行器和可选 profiler；无返回；建立 KV 索引与 iteration 计数。"""
        self.scheduler = scheduler
        self.decode_runner = decode_runner
        self.decode_states: dict[str, Any] = {}
        self.profiler = profiler
        self.annotate_profiler = annotate_profiler
        self.iteration = 0
        self.last_profile: StepProfile | None = None

    def add_request(self, request: Request) -> None:
        """输入请求；无返回；成功入队后采用调用入口时间，排除预创建对象的等待。"""
        arrival_time = perf_counter()
        self.scheduler.add_request(request)
        # 校验失败时不覆盖已有请求的时间，尤其是重复提交同一对象。
        request.arrival_time = arrival_time

    def has_unfinished_requests(self) -> bool:
        """输入自身；返回是否仍有工作；只读，供调用方驱动循环。"""
        return self.scheduler.has_unfinished_requests()

    def step(self) -> EngineStepOutput:
        """输入队列与 KV 状态；返回本轮结果；可选记录 scheduler/forward/reclaim 阶段。"""
        total_timer = _PhaseTimer() if self.profiler is not None else None
        scheduler_timer = _PhaseTimer() if self.profiler is not None else None
        scope = record_function if self.annotate_profiler else lambda _: nullcontext()
        with scope("miniserve::scheduler"):
            plan = self.scheduler.schedule()
        scheduler_seconds = scheduler_timer.elapsed() if scheduler_timer else 0.0
        for request in plan.finished_requests:
            state = self.decode_states.pop(request.request_id, None)
            if state is not None:
                self.decode_runner.release_state(state)

        # 在执行任何 prefill 前检查已有 decode 状态，缺失时不能静默跳过。
        active_states = []
        for request in plan.decode_requests:
            if request.request_id not in self.decode_states:
                raise RuntimeError(f"Missing decode state: {request.request_id}")
            active_states.append(self.decode_states[request.request_id])

        prefill_timer = _PhaseTimer() if self.profiler is not None else None
        with scope("miniserve::prefill"):
            for request in plan.prefill_requests:
                state = self.decode_runner.prefill_request(request)
                self.decode_states[request.request_id] = state
        prefill_seconds = prefill_timer.elapsed() if prefill_timer else 0.0

        decoded_tokens = {}
        decode_timer = _PhaseTimer() if self.profiler is not None else None
        with scope("miniserve::decode"):
            if active_states:
                output = self.decode_runner.decode_step(active_states)
                decoded_tokens = dict(
                    zip(output.request_ids, output.token_ids, strict=True)
                )
        decode_seconds = decode_timer.elapsed() if decode_timer else 0.0

        # 轮末回收不做 admission；新空位下一轮再接纳，保持执行计划边界清晰。
        reclaim_timer = _PhaseTimer() if self.profiler is not None else None
        with scope("miniserve::reclaim"):
            finished = plan.finished_requests + self.scheduler.reclaim_finished()
            for request in finished:
                state = self.decode_states.pop(request.request_id, None)
                if state is not None:
                    self.decode_runner.release_state(state)
        reclaim_seconds = reclaim_timer.elapsed() if reclaim_timer else 0.0

        if total_timer is not None:
            self.last_profile = StepProfile(
                iteration=self.iteration,
                total_seconds=total_timer.elapsed(),
                scheduler_seconds=scheduler_seconds,
                prefill_seconds=prefill_seconds,
                decode_seconds=decode_seconds,
                reclaim_seconds=reclaim_seconds,
                num_running=len(plan.running_requests),
                num_prefill=len(plan.prefill_requests),
                num_decode=len(plan.decode_requests),
                prefill_tokens=sum(r.prompt_length for r in plan.prefill_requests),
                decode_tokens=len(plan.decode_requests),
                scheduled_tokens=plan.num_scheduled_tokens,
            )
            self.profiler.record(self.last_profile)
        self.iteration += 1

        return EngineStepOutput(
            newly_prefilled=tuple(r.request_id for r in plan.prefill_requests),
            decoded_tokens=decoded_tokens,
            finished_requests=tuple(r.request_id for r in finished),
            running_requests=tuple(r.request_id for r in self.scheduler.running),
        )
