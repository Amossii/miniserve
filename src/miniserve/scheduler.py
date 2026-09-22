from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from miniserve.request import Request


@dataclass(frozen=True)
class ScheduleOutput:
    """调度时固定两组成员；Request 可变，但本轮不再按变化后的 phase 重新分组。"""

    newly_admitted: tuple[Request, ...]
    running_requests: tuple[Request, ...]
    finished_requests: tuple[Request, ...]
    prefill_requests: tuple[Request, ...]
    decode_requests: tuple[Request, ...]
    num_scheduled_tokens: int

    @property
    def scheduled_requests(self) -> tuple[Request, ...]:
        """输入自身；返回本轮执行集合；只读，兼容早期命名。"""
        return self.running_requests

    @property
    def newly_finished(self) -> tuple[Request, ...]:
        """输入自身；返回轮首回收集合；只读，兼容早期命名。"""
        return self.finished_requests


class Scheduler:
    """FIFO + 请求数容量 + 可选输入 token 预算；完整 prefill，不分块、不抢占。"""

    def __init__(
        self,
        *,
        max_num_running: int | None = None,
        max_num_seqs: int | None = None,
        max_num_batched_tokens: int | None = None,
    ) -> None:
        """输入容量与可选预算；无返回；初始化队列，None 保持原有无预算行为。"""
        if (max_num_running is None) == (max_num_seqs is None):
            raise ValueError("Specify exactly one capacity argument.")
        capacity = max_num_running if max_num_running is not None else max_num_seqs
        if capacity is None or capacity <= 0:
            raise ValueError("Capacity must be positive.")
        if max_num_batched_tokens is not None:
            if type(max_num_batched_tokens) is not int or max_num_batched_tokens <= 0:
                raise ValueError("max_num_batched_tokens must be a positive integer.")
            # 保障所有已有请求都能执行一次 decode，暂不引入 decode 轮转。
            if max_num_batched_tokens < capacity:
                raise ValueError("Token budget must be >= max_num_running.")
        self.max_num_running = capacity
        self.max_num_batched_tokens = max_num_batched_tokens
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self._finished: list[Request] = []
        self._request_ids: set[str] = set()

    @property
    def waiting_requests(self) -> tuple[Request, ...]:
        """输入自身；返回等待快照；不修改状态，避免经此接口修改容器。"""
        return tuple(self.waiting)

    @property
    def running_requests(self) -> tuple[Request, ...]:
        """输入自身；返回运行快照；只读，供观测和旧调用方使用。"""
        return tuple(self.running)

    @property
    def finished_requests(self) -> tuple[Request, ...]:
        """输入自身；返回累计回收结果；只读，保留逻辑结果而非 KV。"""
        return tuple(self._finished)

    @property
    def num_free_slots(self) -> int:
        """输入自身；返回空闲请求槽数；只读，这不是 KV 显存预算。"""
        return self.max_num_running - len(self.running)

    def add_request(self, request: Request) -> None:
        """输入新请求；无返回；校验后入队，拒绝重复 ID 以保护 KV 索引。"""
        if not request.is_waiting:
            raise ValueError("Only waiting request can be added.")
        if request.request_id in self._request_ids:
            raise ValueError(f"Duplicate request_id: {request.request_id}")
        if request.prompt_length <= 0 or request.max_new_tokens <= 0:
            raise ValueError("Prompt and max_new_tokens must be nonempty/positive.")
        if (
            self.max_num_batched_tokens is not None
            and request.prompt_length > self.max_num_batched_tokens
        ):
            # 必须在入队与登记 ID 之前拒绝，避免永远无法调度的队首堵住系统。
            raise ValueError(
                "Prompt exceeds token budget; chunked prefill is required."
            )
        self.waiting.append(request)
        self._request_ids.add(request.request_id)

    def has_unfinished_requests(self) -> bool:
        """输入自身；返回是否有待执行请求；只读，不把待回收结束项当作工作。"""
        return bool(self.waiting) or any(not r.is_finished for r in self.running)

    def reclaim_finished(self) -> tuple[Request, ...]:
        """输入自身；返回本次完成项；移出 running 并保存结果，不触发新 admission。"""
        finished = tuple(r for r in self.running if r.is_finished)
        self.running = [r for r in self.running if not r.is_finished]
        self._finished.extend(finished)
        return finished

    def schedule(self) -> ScheduleOutput:
        """输入队列状态；返回分组与计费快照；先预留已有工作，再按 FIFO 接纳。"""
        finished = self.reclaim_finished()

        # 正常 Engine 循环中，已有 running 请求都已完成 prefill。
        # 若调用方尚未执行上次计划，则仍需为其完整 prompt 预留预算。
        used_tokens = sum(
            1 if r.needs_decode else r.prompt_length for r in self.running
        )
        budget = self.max_num_batched_tokens
        if budget is not None and used_tokens > budget:
            raise RuntimeError("Existing running work exceeds token budget.")

        admitted = []
        while self.waiting and self.num_free_slots > 0:
            request = self.waiting[0]
            if budget is not None and used_tokens + request.prompt_length > budget:
                # 严格 FIFO：不跳过队首，也不提前占用 running 槽位。
                break
            request.mark_running()
            self.waiting.popleft()
            self.running.append(request)
            admitted.append(request)
            used_tokens += request.prompt_length

        # 必须在任何 forward 之前分组：新请求本轮只走 prefill。
        return ScheduleOutput(
            newly_admitted=tuple(admitted),
            running_requests=tuple(self.running),
            finished_requests=finished,
            prefill_requests=tuple(r for r in self.running if r.needs_prefill),
            decode_requests=tuple(r for r in self.running if r.needs_decode),
            # 保存整数快照，避免 prefill 改变 phase 后重新计算得到错误结果。
            num_scheduled_tokens=used_tokens,
        )
