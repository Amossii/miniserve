from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import pairwise
from math import isfinite
from time import perf_counter


class RequestStatus(Enum):
    """生命周期；RUNNING 表示已接纳，不表示每时每刻都在 GPU 上执行。"""

    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class ExecutionPhase(Enum):
    """计算阶段，与生命周期分开描述。"""

    PREFILL = auto()
    DECODE = auto()


@dataclass
class Request:
    """逻辑请求与计时信息；GPU KV 由 DecodeState 持有。"""

    request_id: str
    prompt_token_ids: list[int]
    max_new_tokens: int = 32
    status: RequestStatus = RequestStatus.WAITING
    generated_token_ids: list[int] = field(default_factory=list)
    arrival_time: float = field(default_factory=perf_counter)
    start_time: float | None = None
    finish_time: float | None = None
    prefill_completed: bool = False
    num_prefilled_tokens: int = 0
    first_token_time: float | None = None
    token_timestamps: list[float] = field(default_factory=list)

    @property
    def phase(self) -> ExecutionPhase:
        """输入自身；返回阶段；只读，从布尔值派生以免两份状态冲突。"""
        return (
            ExecutionPhase.DECODE if self.prefill_completed else ExecutionPhase.PREFILL
        )

    @property
    def prompt_length(self) -> int:
        """输入自身；返回 prompt 长度；只读，供构造模型输入。"""
        return len(self.prompt_token_ids)

    @property
    def num_generated_tokens(self) -> int:
        """输入自身；返回输出长度；只读，统一长度统计。"""
        return len(self.generated_token_ids)

    @property
    def sequence_length(self) -> int:
        """输入自身；返回逻辑总长度；只读，包含尚未写入 KV 的最新 token。"""
        return self.prompt_length + self.num_generated_tokens

    @property
    def is_waiting(self) -> bool:
        """输入自身；返回是否等待；只读，供 admission 校验。"""
        return self.status is RequestStatus.WAITING

    @property
    def is_running(self) -> bool:
        """输入自身；返回是否已接纳；只读，供执行器校验。"""
        return self.status is RequestStatus.RUNNING

    @property
    def is_finished(self) -> bool:
        """输入自身；返回是否完成；只读，供资源回收。"""
        return self.status is RequestStatus.FINISHED

    @property
    def needs_decode(self) -> bool:
        """输入自身；返回是否还需 decode；只读，排除结束请求。"""
        return self.prefill_completed and not self.is_finished

    @property
    def needs_prefill(self) -> bool:
        """输入自身；返回是否还需 prefill；只读，执行前另检查 RUNNING。"""
        return not self.prefill_completed and not self.is_finished

    @property
    def remaining_prompt_tokens(self) -> int:
        """输入自身；返回尚未写入 prefill KV 的 prompt token 数；只读，支持 chunk 调度。"""
        return self.prompt_length - self.num_prefilled_tokens

    @property
    def reached_max_new_tokens(self) -> bool:
        """输入自身；返回是否达到生成上限；只读，统一停止条件。"""
        return self.num_generated_tokens >= self.max_new_tokens

    def mark_running(self, *, timestamp: float | None = None) -> None:
        """输入可选接纳时间；无返回；更新状态与时间，允许测试注入同一时钟。"""
        if not self.is_waiting:
            raise RuntimeError("Only waiting request can become running.")
        self.status = RequestStatus.RUNNING
        self.start_time = perf_counter() if timestamp is None else timestamp

    def mark_prefill_completed(self) -> None:
        """输入自身；无返回；更新阶段，拒绝未接纳或重复 prefill。"""
        if not self.is_running or self.prefill_completed:
            raise RuntimeError("Request must be RUNNING and need PREFILL.")
        self.num_prefilled_tokens = self.prompt_length
        self.prefill_completed = True

    def advance_prefill(self, num_tokens: int) -> None:
        """输入本轮完成的 prompt token 数；无返回；推进 cursor，到达末尾时切换 DECODE。"""
        if not self.is_running or not self.needs_prefill:
            raise RuntimeError("Request must be RUNNING and need PREFILL.")
        if type(num_tokens) is not int or num_tokens <= 0:
            raise ValueError("num_tokens must be a positive integer")
        if num_tokens > self.remaining_prompt_tokens:
            raise ValueError("Prefill progress cannot exceed prompt length")
        self.num_prefilled_tokens += num_tokens
        if self.num_prefilled_tokens == self.prompt_length:
            self.prefill_completed = True

    def append_generated_token(
        self, token_id: int, *, timestamp: float | None = None
    ) -> None:
        """输入 token 与可选时间；无返回；同步追加 token/时间，拒绝倒退时间。"""
        if not self.is_running or self.reached_max_new_tokens:
            raise RuntimeError("Request cannot accept another generated token.")
        timestamp = perf_counter() if timestamp is None else timestamp
        if not isfinite(timestamp) or (
            self.token_timestamps and timestamp < self.token_timestamps[-1]
        ):
            raise ValueError("Token timestamps must be finite and nondecreasing.")
        self.generated_token_ids.append(token_id)
        self.token_timestamps.append(timestamp)
        if self.first_token_time is None:
            self.first_token_time = timestamp

    def mark_finished(self, *, timestamp: float | None = None) -> None:
        """输入可选时间；无返回；记录完成，重复调用幂等，禁止 WAITING 直接完成。"""
        if self.is_finished:
            return
        if not self.is_running:
            raise RuntimeError("Only running request can finish.")
        self.status = RequestStatus.FINISHED
        self.finish_time = perf_counter() if timestamp is None else timestamp

    @property
    def ttft_seconds(self) -> float | None:
        """输入自身；返回首 token 延迟或 None；只读，包含排队时间。"""
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.arrival_time

    @property
    def e2e_latency_seconds(self) -> float | None:
        """输入自身；返回完成延迟或 None；只读，用于请求级统计。"""
        if self.finish_time is None:
            return None
        return self.finish_time - self.arrival_time

    @property
    def queue_wait_seconds(self) -> float | None:
        """输入自身；返回接纳前等待秒数或 None；只读，区别排队与 prefill 时间。"""
        if self.start_time is None:
            return None
        return self.start_time - self.arrival_time

    @property
    def itl_seconds(self) -> list[float]:
        """输入自身；返回相邻输出间隔副本；只读，少于两个 token 时无间隔样本。"""
        return [b - a for a, b in pairwise(self.token_timestamps)]

    @property
    def tpot_seconds(self) -> float | None:
        """输入自身；返回首末 token 间平均间隔；只读，单 token 请求返回 None。"""
        if len(self.token_timestamps) < 2:
            return None
        return (self.token_timestamps[-1] - self.token_timestamps[0]) / (
            len(self.token_timestamps) - 1
        )
