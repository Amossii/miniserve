from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from miniserve.request import Request


@dataclass(frozen=True)
class PrefillChunk:
    """一轮 prefill 工作快照；start/end 是 prompt token 的左闭右开区间。"""

    request: Request
    start: int
    end: int

    @property
    def num_tokens(self) -> int:
        """输入自身；返回本 chunk token 数；只读，供预算验证和 profiler 使用。"""
        return self.end - self.start


@dataclass(frozen=True)
class ScheduleOutput:
    """调度时固定两组成员；Request 可变，但本轮不再按变化后的 phase 重新分组。"""

    newly_admitted: tuple[Request, ...]
    running_requests: tuple[Request, ...]
    finished_requests: tuple[Request, ...]
    prefill_requests: tuple[Request, ...]
    prefill_chunks: tuple[PrefillChunk, ...]
    recompute_requests: tuple[Request, ...]
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
        enable_chunked_prefill: bool = False,
        max_preemptions: int = 3,
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
        self.enable_chunked_prefill = enable_chunked_prefill
        if type(max_preemptions) is not int or max_preemptions <= 0:
            raise ValueError("max_preemptions must be a positive integer")
        self.max_preemptions = max_preemptions
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
            not self.enable_chunked_prefill
            and self.max_num_batched_tokens is not None
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

    def preempt_request(self, request: Request) -> None:
        """输入 active decode victim；无返回；移出 running 并追加 waiting 队尾，保留 FIFO 进展。"""
        if request not in self.running:
            raise ValueError("Preemption victim must be running")
        if request.num_preemptions >= self.max_preemptions:
            raise RuntimeError("Request exceeded maximum preemptions")
        recompute_tokens = len(request.recompute_token_ids)
        if (
            self.max_num_batched_tokens is not None
            and recompute_tokens > self.max_num_batched_tokens
        ):
            raise RuntimeError("Recompute context exceeds token budget")
        request.preempt_for_recompute()
        self.running.remove(request)
        self.waiting.append(request)

    def schedule(self) -> ScheduleOutput:
        """输入队列状态；返回分组与计费快照；先预留已有工作，再按 FIFO 接纳。"""
        finished = self.reclaim_finished()

        if self.enable_chunked_prefill:
            return self._schedule_with_chunked_prefill(finished)

        # 正常 Engine 循环中，已有 running 请求都已完成 prefill。
        # 若调用方尚未执行上次计划，则仍需为其完整 prompt 预留预算。
        used_tokens = sum(
            1
            if r.needs_decode
            else len(r.recompute_token_ids)
            if r.needs_recompute
            else r.prompt_length
            for r in self.running
        )
        budget = self.max_num_batched_tokens
        if budget is not None and used_tokens > budget:
            raise RuntimeError("Existing running work exceeds token budget.")

        admitted = []
        while self.waiting and self.num_free_slots > 0:
            request = self.waiting[0]
            request_tokens = (
                len(request.recompute_token_ids)
                if request.needs_recompute
                else request.prompt_length
            )
            if budget is not None and used_tokens + request_tokens > budget:
                # 严格 FIFO：不跳过队首，也不提前占用 running 槽位。
                break
            request.mark_running()
            self.waiting.popleft()
            self.running.append(request)
            admitted.append(request)
            used_tokens += request_tokens

        # 必须在任何 forward 之前分组：新请求本轮只走 prefill。
        return ScheduleOutput(
            newly_admitted=tuple(admitted),
            running_requests=tuple(self.running),
            finished_requests=finished,
            prefill_requests=tuple(r for r in self.running if r.needs_prefill),
            prefill_chunks=tuple(
                PrefillChunk(r, r.num_prefilled_tokens, r.prompt_length)
                for r in self.running
                if r.needs_prefill
            ),
            recompute_requests=tuple(r for r in self.running if r.needs_recompute),
            decode_requests=tuple(r for r in self.running if r.needs_decode),
            # 保存整数快照，避免 prefill 改变 phase 后重新计算得到错误结果。
            num_scheduled_tokens=used_tokens,
        )

    def _schedule_with_chunked_prefill(
        self, finished: tuple[Request, ...]
    ) -> ScheduleOutput:
        """输入轮首回收项；返回 decode-priority chunk 计划；推进由执行器完成而非调度器预写。"""
        budget = self.max_num_batched_tokens
        if budget is None:
            raise RuntimeError("Chunked prefill requires a finite token budget")
        if any(r.needs_recompute for r in (*self.running, *self.waiting)):
            # 当前 runtime 只支持 dynamic + chunked；recompute 属于 paged backend。
            raise RuntimeError("Chunked prefill cannot schedule recompute requests")
        decode_requests = tuple(r for r in self.running if r.needs_decode)
        used_tokens = len(decode_requests)
        if used_tokens > budget:
            raise RuntimeError("Existing decode work exceeds token budget")
        chunks: list[PrefillChunk] = []

        def schedule_chunk(request: Request) -> bool:
            """输入 partial request；返回是否安排了非空 chunk；只更新本轮局部计划。"""
            nonlocal used_tokens
            available = budget - used_tokens
            if available <= 0:
                return False
            length = min(request.remaining_prompt_tokens, available)
            chunks.append(
                PrefillChunk(
                    request,
                    request.num_prefilled_tokens,
                    request.num_prefilled_tokens + length,
                )
            )
            used_tokens += length
            return True

        # 已接纳的 partial prefill 保持 FIFO running 顺序，且排在新 admission 前。
        for request in self.running:
            if request.needs_prefill and not schedule_chunk(request):
                break

        admitted: list[Request] = []
        while self.waiting and self.num_free_slots > 0 and used_tokens < budget:
            request = self.waiting.popleft()
            request.mark_running()
            self.running.append(request)
            admitted.append(request)
            schedule_chunk(request)

        return ScheduleOutput(
            newly_admitted=tuple(admitted),
            running_requests=tuple(self.running),
            finished_requests=finished,
            prefill_requests=tuple(chunk.request for chunk in chunks),
            prefill_chunks=tuple(chunks),
            recompute_requests=(),
            decode_requests=decode_requests,
            num_scheduled_tokens=used_tokens,
        )
