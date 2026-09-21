from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from miniserve.request import Request


@dataclass(frozen=True)
class ScheduleOutput:
    """
    表示一次 Scheduler.schedule() 的完整调度结果。

    scheduled_requests:
        本轮应该被执行的所有 Request。

        Scheduler V1 中：
            scheduled_requests == 当前全部 running requests

    prefill_requests:
        本轮需要执行 PREFILL 的 Request。

    decode_requests:
        本轮需要执行 DECODE 的 Request。

    newly_admitted:
        本轮刚刚从 WAITING 转换成 RUNNING 的 Request。

    newly_finished:
        本轮开始时从 running 中回收的 FINISHED Request。
    """

    scheduled_requests: tuple[Request, ...]
    prefill_requests: tuple[Request, ...]
    decode_requests: tuple[Request, ...]
    newly_admitted: tuple[Request, ...]
    newly_finished: tuple[Request, ...]


class Scheduler:
    """
    MiniServe Scheduler V1。

    Scheduler 当前只负责 control-plane scheduling：

    - waiting queue
    - running set
    - finished reclamation
    - FIFO admission
    - max_num_seqs capacity
    - PREFILL / DECODE 分类

    Scheduler 明确不负责：

    - model.forward()
    - Tensor batching
    - attention mask
    - KV Cache
    - token sampling
    - token budget
    - preemption
    """

    # --------------------------------------------------
    # 初始化 Scheduler。
    #
    # max_num_seqs:
    #     同一时刻最多允许多少个 active Request。
    #
    # 必须 > 0，否则 Scheduler 永远无法接纳 Request。
    # --------------------------------------------------
    def __init__(
        self,
        *,
        max_num_seqs: int,
    ) -> None:

        if max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be > 0.")

        self.max_num_seqs = max_num_seqs

        # WAITING Request 使用 FIFO queue。
        self._waiting: deque[Request] = deque()

        # 当前已经被接纳、尚未被 Scheduler 回收的 Request。
        #
        # 使用 list 保持稳定顺序。
        self._running: list[Request] = []

        # 已经完成并从 running 中回收的 Request。
        self._finished: list[Request] = []

        # 防止同一个 Request 被重复加入 Scheduler。
        self._request_ids: set[str] = set()

    # --------------------------------------------------
    # 返回当前 waiting requests 的只读快照。
    #
    # 返回 tuple，避免外部直接修改 Scheduler 内部容器。
    # --------------------------------------------------
    @property
    def waiting_requests(
        self,
    ) -> tuple[Request, ...]:

        return tuple(self._waiting)

    # --------------------------------------------------
    # 返回当前 running requests 的只读快照。
    # --------------------------------------------------
    @property
    def running_requests(
        self,
    ) -> tuple[Request, ...]:

        return tuple(self._running)

    # --------------------------------------------------
    # 返回已经被 Scheduler 回收的 finished requests。
    # --------------------------------------------------
    @property
    def finished_requests(
        self,
    ) -> tuple[Request, ...]:

        return tuple(self._finished)

    # --------------------------------------------------
    # 当前还有多少 active sequence slot。
    #
    # Scheduler V1 只考虑 sequence 数量，
    # 暂时不考虑 token budget / KV memory。
    # --------------------------------------------------
    @property
    def num_free_slots(self) -> int:

        return self.max_num_seqs - len(self._running)

    # --------------------------------------------------
    # 将新 Request 放入 waiting queue。
    #
    # 新加入的 Request 必须处于 WAITING 状态。
    #
    # 注意：
    # add_request() 不会立即 mark_running()。
    #
    # 真正 admission 发生在 schedule()，
    # 因为只有 Scheduler 才知道当前还有没有容量。
    # --------------------------------------------------
    def add_request(
        self,
        request: Request,
    ) -> None:

        if not request.is_waiting:
            raise ValueError("A newly added request must be WAITING.")

        if request.request_id in self._request_ids:
            raise ValueError(f"Duplicate request_id: {request.request_id}")

        self._waiting.append(request)

        self._request_ids.add(request.request_id)

    # --------------------------------------------------
    # 判断系统中是否还有尚未完成的工作。
    #
    # waiting 非空：
    #     还有请求等待 admission。
    #
    # running 非空：
    #     还有 active 请求尚未被回收。
    # --------------------------------------------------
    def has_unfinished_requests(
        self,
    ) -> bool:

        return bool(self._waiting or self._running)

    # --------------------------------------------------
    # 将已经 FINISHED 的 Request 从 running 中移除。
    #
    # 返回：
    #     本轮刚刚回收的 Request。
    #
    # 这里必须在 admission 之前执行，
    # 否则 finished Request 会错误占据 active slot。
    # --------------------------------------------------
    def _reclaim_finished(
        self,
    ) -> list[Request]:

        still_running: list[Request] = []
        newly_finished: list[Request] = []

        for request in self._running:
            if request.is_finished:
                newly_finished.append(request)
            else:
                still_running.append(request)

        self._running = still_running

        self._finished.extend(newly_finished)

        return newly_finished

    # --------------------------------------------------
    # 按 FIFO 从 waiting queue 接纳新 Request。
    #
    # 每 admission 一个 Request：
    #
    # WAITING
    #   ->
    # RUNNING
    #
    # 直到：
    #
    # - waiting 为空
    # 或
    # - running 达到 max_num_seqs
    # --------------------------------------------------
    def _admit_waiting(
        self,
    ) -> list[Request]:

        newly_admitted: list[Request] = []

        while self._waiting and len(self._running) < self.max_num_seqs:
            request = self._waiting.popleft()

            if not request.is_waiting:
                raise RuntimeError("Waiting queue contains a non-WAITING request.")

            request.mark_running()

            self._running.append(request)

            newly_admitted.append(request)

        return newly_admitted

    # --------------------------------------------------
    # 验证 Scheduler 的核心 invariants。
    #
    # 这些检查主要用于尽早发现 Scheduler bug。
    #
    # Phase A 的规模很小，因此检查成本可以忽略。
    # --------------------------------------------------
    def _validate_invariants(
        self,
    ) -> None:

        if len(self._running) > self.max_num_seqs:
            raise RuntimeError("Scheduler capacity invariant violated.")

        for request in self._waiting:
            if not request.is_waiting:
                raise RuntimeError("Waiting queue contains a non-WAITING request.")

        for request in self._running:
            if not request.is_running:
                raise RuntimeError("Running set contains a non-RUNNING request.")

        for request in self._finished:
            if not request.is_finished:
                raise RuntimeError("Finished set contains a non-FINISHED request.")

        # 一个 Request 不应该同时存在于多个 lifecycle collection。
        waiting_ids = {request.request_id for request in self._waiting}

        running_ids = {request.request_id for request in self._running}

        finished_ids = {request.request_id for request in self._finished}

        if waiting_ids & running_ids:
            raise RuntimeError("A request appears in both waiting and running.")

        if waiting_ids & finished_ids:
            raise RuntimeError("A request appears in both waiting and finished.")

        if running_ids & finished_ids:
            raise RuntimeError("A request appears in both running and finished.")

    # --------------------------------------------------
    # 执行一次 scheduling iteration。
    #
    # 固定顺序：
    #
    # 1. reclaim finished
    # 2. admit waiting
    # 3. 获取当前 running
    # 4. 按 PREFILL / DECODE 分类
    # 5. 返回 ScheduleOutput
    #
    # Scheduler V1 中：
    #
    # scheduled_requests == running_requests
    #
    # Step 17 加入 token budget 后，
    # 两者才可能不同。
    # --------------------------------------------------
    def schedule(
        self,
    ) -> ScheduleOutput:

        newly_finished = self._reclaim_finished()

        newly_admitted = self._admit_waiting()

        self._validate_invariants()

        scheduled_requests = list(self._running)

        prefill_requests = [
            request for request in scheduled_requests if request.needs_prefill
        ]

        decode_requests = [
            request for request in scheduled_requests if request.needs_decode
        ]

        return ScheduleOutput(
            scheduled_requests=tuple(scheduled_requests),
            prefill_requests=tuple(prefill_requests),
            decode_requests=tuple(decode_requests),
            newly_admitted=tuple(newly_admitted),
            newly_finished=tuple(newly_finished),
        )
