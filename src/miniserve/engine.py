from __future__ import annotations

from dataclasses import dataclass

from miniserve.decode_batch import DecodeBatchRunner, DecodeState
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
        self, *, scheduler: Scheduler, decode_runner: DecodeBatchRunner
    ) -> None:
        """输入调度器与执行器；无返回；建立独立 KV 索引，分离控制与模型执行。"""
        self.scheduler = scheduler
        self.decode_runner = decode_runner
        self.decode_states: dict[str, DecodeState] = {}

    def add_request(self, request: Request) -> None:
        """输入请求；无返回；委托调度器入队，模型状态到 prefill 时才创建。"""
        self.scheduler.add_request(request)

    def has_unfinished_requests(self) -> bool:
        """输入自身；返回是否仍有工作；只读，供调用方驱动循环。"""
        return self.scheduler.has_unfinished_requests()

    def step(self) -> EngineStepOutput:
        """输入队列与 KV 状态；返回本轮结果；推进两组请求并回收，空轮返回空结果。"""
        plan = self.scheduler.schedule()
        for request in plan.finished_requests:
            self.decode_states.pop(request.request_id, None)

        # 在执行任何 prefill 前检查已有 decode 状态，缺失时不能静默跳过。
        active_states = []
        for request in plan.decode_requests:
            if request.request_id not in self.decode_states:
                raise RuntimeError(f"Missing decode state: {request.request_id}")
            active_states.append(self.decode_states[request.request_id])

        for request in plan.prefill_requests:
            state = self.decode_runner.prefill_request(request)
            self.decode_states[request.request_id] = state

        decoded_tokens = {}
        if active_states:
            output = self.decode_runner.decode_step(active_states)
            decoded_tokens = dict(
                zip(output.request_ids, output.token_ids, strict=True)
            )

        # 轮末回收不做 admission；新空位下一轮再接纳，保持执行计划边界清晰。
        finished = plan.finished_requests + self.scheduler.reclaim_finished()
        for request in finished:
            self.decode_states.pop(request.request_id, None)

        return EngineStepOutput(
            newly_prefilled=tuple(r.request_id for r in plan.prefill_requests),
            decoded_tokens=decoded_tokens,
            finished_requests=tuple(r.request_id for r in finished),
            running_requests=tuple(r.request_id for r in self.scheduler.running),
        )
