from __future__ import annotations

import time
from collections.abc import Hashable
from dataclasses import dataclass, field


@dataclass
class Request:
    """
    表示 MiniServe 中的一个逻辑推理请求。

    Request 只负责保存请求自身的逻辑状态。

    它暂时不负责：
    - GPU tensor
    - Hugging Face DynamicCache
    - batching
    - scheduling policy
    - model.forward()

    这些职责后续分别交给 Batch Builder、
    Scheduler 和 ModelRunner。
    """

    request_id: str

    # 用户 prompt tokenizer 后得到的 token IDs。
    #
    # 使用 list[int] 而不是 CUDA Tensor，
    # 让 Request 保持为轻量的 control-plane object。
    prompt_token_ids: list[int]

    # 当前已经生成的 token。
    #
    # 初始为空，随着 decode 逐 token append。
    generated_token_ids: list[int] = field(default_factory=list)

    # 最多允许模型新生成多少个 token。
    max_new_tokens: int = 32

    # 请求进入 Engine 的 wall-clock 时间。
    #
    # default_factory 保证每个 Request 创建时
    # 都会单独调用一次 time.perf_counter()。
    arrival_time: float = field(default_factory=time.perf_counter)

    # 第一个 output token 真正产生的时间。
    #
    # 在请求还没有生成 token 时为 None。
    first_token_time: float | None = None

    # 请求最终完成的时间。
    finish_time: float | None = None

    # Request 暂时只持有 KV Cache 的逻辑句柄。
    #
    # 以后可以变成：
    #
    # cache_handle = block_table_id
    #
    # 或其他 CacheManager 能识别的 ID。
    #
    # 这里故意不直接依赖 DynamicCache。
    cache_handle: Hashable | None = None

    # --------------------------------------------------
    # 返回 prompt 本身的 token 数量。
    # --------------------------------------------------
    @property
    def prompt_length(self) -> int:
        return len(self.prompt_token_ids)

    # --------------------------------------------------
    # 返回当前已经生成的 token 数量。
    # --------------------------------------------------
    @property
    def num_generated_tokens(self) -> int:
        return len(self.generated_token_ids)

    # --------------------------------------------------
    # 返回当前完整逻辑 sequence 的长度。
    #
    # sequence:
    #
    # prompt tokens
    # +
    # generated tokens
    # --------------------------------------------------
    @property
    def sequence_length(self) -> int:
        return self.prompt_length + self.num_generated_tokens

    # --------------------------------------------------
    # 判断是否已经达到 max_new_tokens。
    #
    # 注意：
    # 这只是“长度终止条件”。
    #
    # EOS 等其他结束原因将在后续状态机中处理。
    # --------------------------------------------------
    @property
    def reached_max_new_tokens(self) -> bool:
        return self.num_generated_tokens >= self.max_new_tokens

    # --------------------------------------------------
    # 返回当前完整 sequence。
    #
    # 返回新 list，避免外部修改返回值时
    # 意外修改 Request 内部状态。
    # --------------------------------------------------
    @property
    def all_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.generated_token_ids

    # --------------------------------------------------
    # 向该 Request 追加一个新生成 token。
    #
    # 这就是之后每次 decode iteration
    # 更新 Request 的最基本操作。
    # --------------------------------------------------
    def append_generated_token(
        self,
        token_id: int,
        *,
        timestamp: float | None = None,
    ) -> None:

        if self.reached_max_new_tokens:
            raise RuntimeError(
                "Cannot append token: request has already reached max_new_tokens."
            )

        self.generated_token_ids.append(token_id)

        # 第一次 append token 时，记录 TTFT 的结束时间。
        if self.first_token_time is None:
            self.first_token_time = (
                time.perf_counter() if timestamp is None else timestamp
            )

    # --------------------------------------------------
    # 将请求标记为结束，并记录结束时间。
    #
    # Step 8 会把这个行为进一步整合进正式
    # Request state machine。
    # --------------------------------------------------
    def mark_finished(
        self,
        *,
        timestamp: float | None = None,
    ) -> None:

        if self.finish_time is not None:
            raise RuntimeError("Request is already finished.")

        self.finish_time = time.perf_counter() if timestamp is None else timestamp

    # --------------------------------------------------
    # 计算 TTFT。
    #
    # TTFT:
    #
    # first token time
    # -
    # request arrival time
    #
    # 如果还没产生首 token，则返回 None。
    # --------------------------------------------------
    @property
    def ttft_seconds(self) -> float | None:

        if self.first_token_time is None:
            return None

        return self.first_token_time - self.arrival_time

    # --------------------------------------------------
    # 计算请求端到端 latency。
    #
    # E2E:
    #
    # finish time
    # -
    # arrival time
    #
    # 请求没结束时返回 None。
    # --------------------------------------------------
    @property
    def e2e_latency_seconds(
        self,
    ) -> float | None:

        if self.finish_time is None:
            return None

        return self.finish_time - self.arrival_time
