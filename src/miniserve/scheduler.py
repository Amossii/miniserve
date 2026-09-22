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

    @property
    def scheduled_requests(self) -> tuple[Request, ...]:
        """输入自身；返回本轮执行集合；只读，兼容早期命名。"""
        return self.running_requests

    @property
    def newly_finished(self) -> tuple[Request, ...]:
        """输入自身；返回轮首回收集合；只读，兼容早期命名。"""
        return self.finished_requests


class Scheduler:
    """FIFO + 请求数容量；不含 token budget、chunked prefill 或抢占。"""

    def __init__(
        self, *, max_num_running: int | None = None, max_num_seqs: int | None = None
    ) -> None:
        """输入容量（二选一）；无返回；初始化队列，兼容早期 max_num_seqs 参数。"""
        if (max_num_running is None) == (max_num_seqs is None):
            raise ValueError("Specify exactly one capacity argument.")
        capacity = max_num_running if max_num_running is not None else max_num_seqs
        if capacity is None or capacity <= 0:
            raise ValueError("Capacity must be positive.")
        self.max_num_running = capacity
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
        """输入队列状态；返回固定执行分组；先回收再接纳，只调度、不执行模型。"""
        finished = self.reclaim_finished()
        admitted = []
        while self.waiting and self.num_free_slots > 0:
            request = self.waiting.popleft()
            request.mark_running()
            self.running.append(request)
            admitted.append(request)

        # 必须在任何 forward 之前分组：新请求本轮只走 prefill。
        return ScheduleOutput(
            newly_admitted=tuple(admitted),
            running_requests=tuple(self.running),
            finished_requests=finished,
            prefill_requests=tuple(r for r in self.running if r.needs_prefill),
            decode_requests=tuple(r for r in self.running if r.needs_decode),
        )
