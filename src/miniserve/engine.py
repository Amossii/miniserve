from __future__ import annotations

from typing import Any

import torch

from miniserve.request import (
    ExecutionPhase,
    Request,
)


# ---------------------------------------------------------
# 将模型配置中的 EOS token ID 统一转换成 set[int]。
#
# 不同模型可能：
# - 没有 EOS
# - 有一个 EOS
# - 有多个 EOS
# ---------------------------------------------------------
def _normalize_eos_token_ids(
    eos_token_id: int | list[int] | None,
) -> set[int]:

    if eos_token_id is None:
        return set()

    if isinstance(eos_token_id, int):
        return {eos_token_id}

    return set(eos_token_id)


class Engine:
    """
    MiniServe 最小单请求 inference engine。

    当前职责：

    1. 接收一个 Request。
    2. 推进 Request lifecycle。
    3. 执行 prefill。
    4. 保存 KV Cache。
    5. 执行逐 token decode。
    6. 处理 EOS / max_new_tokens。
    7. 将生成 token 写回 Request。

    当前明确不负责：

    - 多请求调度
    - batching
    - continuous batching
    - token budget
    - block-based KV cache
    - HTTP serving

    这些会在后续步骤逐渐加入。
    """

    # -----------------------------------------------------
    # 初始化 Engine。
    #
    # model:
    #     Hugging Face causal language model。
    #
    # device:
    #     模型所在设备，例如 torch.device("cuda")。
    #
    # eos_token_ids:
    #     可显式指定 EOS。
    #     如果不指定，则尝试从 model.generation_config 获取。
    # -----------------------------------------------------
    def __init__(
        self,
        model,
        device: torch.device,
        eos_token_ids: set[int] | None = None,
    ) -> None:

        self.model = model
        self.device = device

        # 当前版本只允许一个 active Request。
        self._request: Request | None = None

        # 真正的 KV Cache 属于 Engine 的执行状态，
        # 而不是 Request 的逻辑状态。
        #
        # 第一次 prefill 前为 None。
        self._past_key_values: Any | None = None

        if eos_token_ids is None:
            generation_config = getattr(
                model,
                "generation_config",
                None,
            )

            eos_token_id = (
                None if generation_config is None else generation_config.eos_token_id
            )

            self._eos_token_ids = _normalize_eos_token_ids(eos_token_id)
        else:
            self._eos_token_ids = set(eos_token_ids)

    # -----------------------------------------------------
    # 返回当前 Request。
    #
    # 主要用于：
    # - demo
    # - tests
    # - 后续 Engine 状态观察
    # -----------------------------------------------------
    @property
    def request(self) -> Request | None:
        return self._request

    # -----------------------------------------------------
    # 返回当前 KV Cache 已经真正缓存多少个 token。
    #
    # prefill 前：
    #     0
    #
    # prefill 后：
    #     prompt_length
    #
    # 每次 decode forward 后：
    #     +1
    # -----------------------------------------------------
    @property
    def cache_length(self) -> int:

        if self._past_key_values is None:
            return 0

        get_seq_length = getattr(
            self._past_key_values,
            "get_seq_length",
            None,
        )

        if get_seq_length is None:
            raise RuntimeError("Current cache object does not expose get_seq_length().")

        return int(get_seq_length())

    # -----------------------------------------------------
    # 将一个 Request 加入 Engine。
    #
    # 当前限制：
    #
    # 同一时间只能存在一个 unfinished Request。
    #
    # 已完成旧 Request 后，可以加入新的 Request。
    # 加入新 Request 时必须清空旧 KV Cache。
    # -----------------------------------------------------
    def add_request(
        self,
        request: Request,
    ) -> None:

        if not request.is_waiting:
            raise ValueError("A newly added request must be WAITING.")

        if self._request is not None and not self._request.is_finished:
            raise RuntimeError(
                "Single-request Engine already has an unfinished request."
            )

        # 一个新的 Request 必须拥有自己的全新 KV state。
        self._past_key_values = None

        self._request = request

    # -----------------------------------------------------
    # Engine 是否还有尚未完成的工作。
    #
    # 以后主循环就是：
    #
    # while engine.has_unfinished_requests():
    #     engine.step()
    # -----------------------------------------------------
    def has_unfinished_requests(self) -> bool:

        return self._request is not None and not self._request.is_finished

    # -----------------------------------------------------
    # 根据 Request phase 构造本轮真正送进模型的 input_ids。
    #
    # PREFILL:
    #
    #     完整 prompt
    #
    # DECODE:
    #
    #     只传入最近刚生成、但尚未进入 KV Cache 的 token
    #
    # 这是 KV Cache 推理最关键的区别之一。
    # -----------------------------------------------------
    def _build_input_ids(
        self,
        request: Request,
    ) -> torch.Tensor:

        if request.phase is ExecutionPhase.PREFILL:
            token_ids = request.prompt_token_ids

        elif request.phase is ExecutionPhase.DECODE:
            if not request.generated_token_ids:
                raise RuntimeError(
                    "DECODE request has no generated token to feed into the model."
                )

            # 只取最后生成的一个 token。
            token_ids = [request.generated_token_ids[-1]]

        else:
            raise RuntimeError(f"Unknown execution phase: {request.phase}")

        return torch.tensor(
            [token_ids],
            dtype=torch.long,
            device=self.device,
        )

    # -----------------------------------------------------
    # 构造当前 attention_mask。
    #
    # 当前只支持单请求、无 padding，因此全部都是 1。
    #
    # 注意 decode 时：
    #
    # input_ids length = 1
    #
    # 但 attention_mask length =
    #
    # historical cached tokens
    # +
    # current token
    #
    # 也就是完整逻辑 sequence length。
    # -----------------------------------------------------
    def _build_attention_mask(
        self,
        request: Request,
    ) -> torch.Tensor:

        return torch.ones(
            (1, request.sequence_length),
            dtype=torch.long,
            device=self.device,
        )

    # -----------------------------------------------------
    # 判断当前生成 token 是否意味着 Request 应该结束。
    #
    # 两种基本条件：
    #
    # 1. 生成 EOS。
    # 2. 达到 max_new_tokens。
    # -----------------------------------------------------
    def _should_finish(
        self,
        request: Request,
        token_id: int,
    ) -> bool:

        hit_eos = token_id in self._eos_token_ids

        hit_length_limit = request.reached_max_new_tokens

        return hit_eos or hit_length_limit

    # -----------------------------------------------------
    # 推进 Engine 一个 inference iteration。
    #
    # PREFILL step:
    #
    # full prompt
    #   -> model
    #   -> KV cache
    #   -> first output token
    #
    # DECODE step:
    #
    # latest generated token
    #   + historical KV
    #   -> model
    #   -> updated KV
    #   -> next output token
    #
    # 返回：
    #     当前 step 新生成的 token ID。
    # -----------------------------------------------------
    @torch.inference_mode()
    def step(self) -> int:

        if self._request is None:
            raise RuntimeError("Engine has no request.")

        request = self._request

        if request.is_finished:
            raise RuntimeError("Cannot step a finished request.")

        # ---------------------------------------------
        # 第一次真正执行 request 时，
        # 完成 WAITING -> RUNNING admission。
        #
        # Step 13 有 Scheduler 后，
        # 这个行为会由 Scheduler 负责。
        # ---------------------------------------------

        if request.is_waiting:
            request.mark_running()

        # 记录这次 forward 属于哪个 phase。
        #
        # 后面 append token 以后还需要用它判断：
        # 是否应该执行 PREFILL -> DECODE。
        current_phase = request.phase

        # ---------------------------------------------
        # 构造本轮 model input
        # ---------------------------------------------

        input_ids = self._build_input_ids(request)

        attention_mask = self._build_attention_mask(request)

        # ---------------------------------------------
        # Model forward
        #
        # PREFILL:
        #     _past_key_values == None
        #
        # DECODE:
        #     使用上一轮 cache
        # ---------------------------------------------

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=self._past_key_values,
            use_cache=True,
        )

        # 保存更新后的 KV Cache。
        self._past_key_values = outputs.past_key_values

        # ---------------------------------------------
        # 从最后位置 logits 选择 next token。
        #
        # 当前固定使用 greedy decoding。
        # ---------------------------------------------

        next_token_logits = outputs.logits[:, -1, :]

        next_token = torch.argmax(
            next_token_logits,
            dim=-1,
        )

        token_id = int(next_token.item())

        # ---------------------------------------------
        # 将模型生成结果写回 Request。
        # ---------------------------------------------

        request.append_generated_token(token_id)

        # ---------------------------------------------
        # Prefill 只执行一次。
        #
        # prefill forward 已经：
        #
        # 1. 建立 prompt KV cache
        # 2. 产生第一个 output token
        #
        # 因此下一轮开始进入 DECODE。
        # ---------------------------------------------

        if current_phase is ExecutionPhase.PREFILL:
            request.mark_prefill_completed()

        # ---------------------------------------------
        # EOS / max_new_tokens
        # ---------------------------------------------

        if self._should_finish(
            request=request,
            token_id=token_id,
        ):
            request.mark_finished()

        return token_id
