from __future__ import annotations

import time
from collections.abc import Hashable
from dataclasses import dataclass, field
from enum import Enum, auto


class RequestStatus(Enum):
    """
    描述 Request 在整个 serving system 中的生命周期状态。

    WAITING:
        请求已经进入 Engine，但尚未被 Scheduler 接纳执行。

    RUNNING:
        请求已经被接纳，正在进行 prefill 或 decode。

    FINISHED:
        请求已经完成，不应该再次被执行。
    """

    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class ExecutionPhase(Enum):
    """
    描述一个活跃 Request 下一次需要执行的模型计算类型。

    PREFILL:
        prompt 尚未完成第一次 forward，
        下一次需要处理完整 prompt。

    DECODE:
        prompt 已经 prefill 完成，
        后续每轮只生成一个新 token。
    """

    PREFILL = auto()
    DECODE = auto()


@dataclass
class Request:
    """
    表示 MiniServe 中的一个逻辑推理请求。

    Request 保存：
    - 请求身份
    - token 状态
    - 生命周期状态
    - execution phase
    - timing 信息
    - cache 的逻辑句柄

    Request 不直接负责：
    - GPU tensor
    - model.forward()
    - Scheduler policy
    - Hugging Face DynamicCache
    """

    request_id: str

    # 用户 prompt tokenizer 后得到的 token IDs。
    prompt_token_ids: list[int]

    # 最多允许新生成多少个 token。
    max_new_tokens: int = 32

    # 当前已经生成的 output tokens。
    generated_token_ids: list[int] = field(default_factory=list)

    # 新请求进入 Engine 后默认处于 WAITING。
    status: RequestStatus = RequestStatus.WAITING

    # 新请求第一次运行一定需要先 prefill。
    phase: ExecutionPhase = ExecutionPhase.PREFILL

    # Request 被创建/进入 Engine 的时间。
    arrival_time: float = field(default_factory=time.perf_counter)

    # 第一个 output token 产生的时间。
    first_token_time: float | None = None

    # 请求完成时间。
    finish_time: float | None = None

    # KV cache 的逻辑句柄。
    #
    # 当前 Phase A 暂时只保留接口，
    # 不直接让 Request 持有 DynamicCache。
    cache_handle: Hashable | None = None

    @property
    def prompt_length(self) -> int:
        """
        返回 prompt token 数量。
        """

        return len(self.prompt_token_ids)

    @property
    def num_generated_tokens(self) -> int:
        """
        返回已经生成的 token 数量。
        """

        return len(self.generated_token_ids)

    @property
    def sequence_length(self) -> int:
        """
        返回当前完整逻辑 sequence 长度：

        prompt tokens
        +
        generated tokens
        """

        return self.prompt_length + self.num_generated_tokens

    @property
    def reached_max_new_tokens(self) -> bool:
        """
        判断是否已经达到最大生成长度。
        """

        return self.num_generated_tokens >= self.max_new_tokens

    @property
    def all_token_ids(self) -> list[int]:
        """
        返回完整逻辑 token sequence。

        返回新 list，避免外部意外修改 Request 内部状态。
        """

        return self.prompt_token_ids + self.generated_token_ids

    @property
    def is_waiting(self) -> bool:
        """
        判断请求是否仍在等待 Scheduler admission。
        """

        return self.status is RequestStatus.WAITING

    @property
    def is_running(self) -> bool:
        """
        判断请求是否正在由 Engine 执行。
        """

        return self.status is RequestStatus.RUNNING

    @property
    def is_finished(self) -> bool:
        """
        判断请求是否已经结束。
        """

        return self.status is RequestStatus.FINISHED

    @property
    def needs_prefill(self) -> bool:
        """
        判断请求下一次是否需要执行 prefill。

        只有 RUNNING request 才应该真正进入模型，
        但这个属性只描述 phase 本身。
        """

        return self.phase is ExecutionPhase.PREFILL

    @property
    def needs_decode(self) -> bool:
        """
        判断请求下一次是否需要执行 decode。
        """

        return self.phase is ExecutionPhase.DECODE

    def mark_running(self) -> None:
        """
        将 WAITING request 转换为 RUNNING。

        合法状态转换：

        WAITING -> RUNNING

        其他状态调用该函数都说明 Engine/Scheduler
        出现逻辑错误，因此直接抛异常。
        """

        if self.status is not RequestStatus.WAITING:
            raise RuntimeError("Only a WAITING request can become RUNNING.")

        self.status = RequestStatus.RUNNING

    def mark_prefill_completed(self) -> None:
        """
        标记当前 Request 的 prefill 已经完成。

        合法转换：

        RUNNING + PREFILL
            ->
        RUNNING + DECODE

        注意 lifecycle status 并没有变化，
        Request 仍然处于 RUNNING。
        """

        if self.status is not RequestStatus.RUNNING:
            raise RuntimeError("Prefill can only complete for a RUNNING request.")

        if self.phase is not ExecutionPhase.PREFILL:
            raise RuntimeError("Request is not in PREFILL phase.")

        self.phase = ExecutionPhase.DECODE

    def append_generated_token(
        self,
        token_id: int,
        *,
        timestamp: float | None = None,
    ) -> None:
        """
        向 RUNNING request 追加一个模型生成 token。

        同时：

        1. 检查 request 是否真的处于 RUNNING。
        2. 检查是否已经达到 max_new_tokens。
        3. 第一个 token 时记录 first_token_time。

        这里暂时不自动判断 EOS。
        EOS 判断属于 ModelRunner / Engine 获得 token 后的逻辑，
        后续会统一处理。
        """

        if self.status is not RequestStatus.RUNNING:
            raise RuntimeError(
                "Generated tokens can only be appended to a RUNNING request."
            )

        if self.reached_max_new_tokens:
            raise RuntimeError(
                "Cannot append token: request has already reached max_new_tokens."
            )

        self.generated_token_ids.append(token_id)

        if self.first_token_time is None:
            self.first_token_time = (
                time.perf_counter() if timestamp is None else timestamp
            )

    def mark_finished(
        self,
        *,
        timestamp: float | None = None,
    ) -> None:
        """
        将一个 RUNNING request 标记为 FINISHED。

        合法转换：

        RUNNING -> FINISHED

        FINISHED request 不应该再次进入 Scheduler 的 active set。
        """

        if self.status is not RequestStatus.RUNNING:
            raise RuntimeError("Only a RUNNING request can become FINISHED.")

        self.status = RequestStatus.FINISHED

        self.finish_time = time.perf_counter() if timestamp is None else timestamp

    @property
    def ttft_seconds(self) -> float | None:
        """
        计算 Time To First Token：

        first_token_time - arrival_time

        尚未生成首 token 时返回 None。
        """

        if self.first_token_time is None:
            return None

        return self.first_token_time - self.arrival_time

    @property
    def e2e_latency_seconds(
        self,
    ) -> float | None:
        """
        计算端到端请求延迟：

        finish_time - arrival_time

        请求尚未结束时返回 None。
        """

        if self.finish_time is None:
            return None

        return self.finish_time - self.arrival_time
