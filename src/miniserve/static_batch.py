from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from transformers import DynamicCache

from miniserve.request import Request


@dataclass
class StaticBatchTrace:
    """
    保存 static batching 的关键运行轨迹。

    forward_input_shapes:
        每次 model.forward() 真正收到的 input_ids shape。

        例如：

        [
            (3, 20),   # prefill
            (3, 1),    # decode
            (3, 1),
        ]

    cache_lengths:
        每轮 forward 后 batched KV Cache 的物理 sequence length。

    active_counts:
        每轮 forward 前仍然 active 的 Request 数量。

        即使 active_count 下降，batch dimension 仍然保持不变，
        这正是 static batching 的特征。
    """

    forward_input_shapes: list[tuple[int, int]]
    cache_lengths: list[int]
    active_counts: list[int]


class StaticBatchRunner:
    """
    MiniServe Static Batching baseline。

    一批 Request 在开始时固定组成 batch。

    支持：

    - 不同 prompt length
    - left padding
    - batched prefill
    - batched KV cache
    - batched decode
    - EOS
    - 不同 max_new_tokens
    - Request 提前结束

    不支持：

    - 新 Request 动态加入
    - finished slot replacement
    - continuous batching

    如果某个 Request 提前结束，它仍然占据 batch 中原来的 row，
    后续使用 masked padding token 维持矩形 batch。
    """

    # --------------------------------------------------
    # 初始化 StaticBatchRunner。
    #
    # model:
    #     Hugging Face causal LM。
    #
    # device:
    #     模型所在设备。
    #
    # pad_token_id:
    #     用于 left padding，以及 finished Request 的 dummy input。
    #
    # eos_token_ids:
    #     模型终止 token 集合。
    # --------------------------------------------------
    def __init__(
        self,
        model,
        device: torch.device,
        pad_token_id: int,
        eos_token_ids: set[int],
    ) -> None:

        self.model = model
        self.device = device

        self.pad_token_id = pad_token_id
        self.eos_token_ids = set(eos_token_ids)

    # --------------------------------------------------
    # 将不同长度 prompt 构造成 left-padded batch。
    #
    # 例如：
    #
    # A = [1, 2, 3, 4]
    # B = [5, 6]
    #
    # 得到：
    #
    # input_ids:
    #
    # [1, 2, 3, 4]
    # [P, P, 5, 6]
    #
    # attention_mask:
    #
    # [1, 1, 1, 1]
    # [0, 0, 1, 1]
    # --------------------------------------------------
    def _build_prefill_batch(
        self,
        requests: list[Request],
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if not requests:
            raise ValueError("Static batch cannot be empty.")

        max_prompt_length = max(request.prompt_length for request in requests)

        if max_prompt_length <= 0:
            raise ValueError("Empty prompts are not supported.")

        batch_size = len(requests)

        input_ids = torch.full(
            (
                batch_size,
                max_prompt_length,
            ),
            fill_value=self.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )

        attention_mask = torch.zeros(
            (
                batch_size,
                max_prompt_length,
            ),
            dtype=torch.long,
            device=self.device,
        )

        for batch_index, request in enumerate(requests):
            prompt = torch.tensor(
                request.prompt_token_ids,
                dtype=torch.long,
                device=self.device,
            )

            prompt_length = prompt.shape[0]

            # 左 padding，因此真实 prompt 放在 tensor 最右侧。
            input_ids[
                batch_index,
                -prompt_length:,
            ] = prompt

            attention_mask[
                batch_index,
                -prompt_length:,
            ] = 1

        return input_ids, attention_mask

    # --------------------------------------------------
    # 根据 attention_mask 构造 position_ids。
    #
    # 左 padding：
    #
    # mask:
    # [0, 0, 1, 1, 1]
    #
    # position:
    # [0, 0, 0, 1, 2]
    #
    # padding positions 最终统一设成 0。
    #
    # 这样每个 Request 的第一个真实 token 都从逻辑 position 0 开始。
    # --------------------------------------------------
    def _build_position_ids(
        self,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:

        position_ids = attention_mask.cumsum(dim=-1) - 1

        position_ids = position_ids.masked_fill(
            attention_mask == 0,
            0,
        )

        return position_ids

    # --------------------------------------------------
    # 判断一个新生成 token 是否使 Request 结束。
    #
    # 结束条件：
    #
    # 1. EOS
    # 2. max_new_tokens
    # --------------------------------------------------
    def _should_finish(
        self,
        request: Request,
        token_id: int,
    ) -> bool:

        if token_id in self.eos_token_ids:
            return True

        return request.reached_max_new_tokens

    # --------------------------------------------------
    # 处理一次 model.forward() 得到的 batched logits。
    #
    # 对仍 active 的 Request：
    #
    # 1. argmax 得到 token
    # 2. 写入 Request
    # 3. 必要时完成 PREFILL -> DECODE
    # 4. 检查 EOS / max_new_tokens
    #
    # 已经 FINISHED 的 row 不再修改。
    #
    # 返回下一次 decode 要输入的：
    #
    # [batch_size, 1]
    #
    # active row:
    #     最近生成的 token
    #
    # finished row:
    #     pad token
    # --------------------------------------------------
    def _consume_logits(
        self,
        requests: list[Request],
        logits: torch.Tensor,
        *,
        completing_prefill: bool,
    ) -> torch.Tensor:

        next_token_ids = torch.argmax(
            logits[:, -1, :],
            dim=-1,
        )

        next_inputs = torch.full(
            (
                len(requests),
                1,
            ),
            fill_value=self.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )

        # 同一个 batch forward 完成的 token
        # 使用同一个逻辑时间点。
        timestamp = time.perf_counter()

        for batch_index, request in enumerate(requests):
            if request.is_finished:
                continue

            token_id = int(next_token_ids[batch_index].item())

            request.append_generated_token(
                token_id,
                timestamp=timestamp,
            )

            # Prefill 完成以后，
            # Request 下一次进入 DECODE。
            if completing_prefill:
                request.mark_prefill_completed()

            if self._should_finish(
                request=request,
                token_id=token_id,
            ):
                request.mark_finished(timestamp=timestamp)
                continue

            # 这个 token 在下一轮 decode 中
            # 真正进入 Transformer，并写入 KV Cache。
            next_inputs[
                batch_index,
                0,
            ] = token_id

        return next_inputs

    # --------------------------------------------------
    # Static batch 主执行函数。
    #
    # 整体流程：
    #
    # 1. 所有 Request 同时 admission
    # 2. left-pad prompts
    # 3. batched prefill
    # 4. 生成每个 Request 的 first token
    # 5. 固定 batch 逐轮 decode
    # 6. 所有 Request 完成后退出
    # --------------------------------------------------
    @torch.inference_mode()
    def run_requests(
        self,
        requests: list[Request],
    ) -> StaticBatchTrace:

        if not requests:
            return StaticBatchTrace(
                forward_input_shapes=[],
                cache_lengths=[],
                active_counts=[],
            )

        # Static batch 开始时，
        # 所有 Request 必须都是全新的 WAITING request。
        for request in requests:
            if not request.is_waiting:
                raise ValueError(
                    "Every request must be WAITING before entering a static batch."
                )

            request.mark_running()

        input_ids, attention_mask = self._build_prefill_batch(requests)

        position_ids = self._build_position_ids(attention_mask)

        max_prompt_length = input_ids.shape[1]

        # DynamicCache 的物理 cache position
        # 与 padding 无关。
        #
        # batch 中所有 row 共用：
        #
        # 0 ... max_prompt_length - 1
        cache_position = torch.arange(
            max_prompt_length,
            dtype=torch.long,
            device=self.device,
        )

        past_key_values = DynamicCache(config=self.model.config)

        trace = StaticBatchTrace(
            forward_input_shapes=[],
            cache_lengths=[],
            active_counts=[],
        )

        # ==================================================
        # PREFILL
        # ==================================================

        trace.active_counts.append(len(requests))

        trace.forward_input_shapes.append(tuple(input_ids.shape))

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=past_key_values,
            use_cache=True,
        )

        past_key_values = outputs.past_key_values

        trace.cache_lengths.append(int(past_key_values.get_seq_length()))

        current_input_ids = self._consume_logits(
            requests=requests,
            logits=outputs.logits,
            completing_prefill=True,
        )

        # ==================================================
        # DECODE
        # ==================================================

        while any(not request.is_finished for request in requests):
            active_mask = torch.tensor(
                [0 if request.is_finished else 1 for request in requests],
                dtype=attention_mask.dtype,
                device=self.device,
            ).unsqueeze(1)

            # 当前 decode token 也属于 attention context。
            #
            # finished row 追加 0，
            # active row 追加 1。
            attention_mask = torch.cat(
                [
                    attention_mask,
                    active_mask,
                ],
                dim=1,
            )

            # 每个 active Request 的逻辑 token position
            # 可能不同，因此 position_ids 是 per-row 的。
            #
            # 例如：
            #
            # Request A 当前有效长度 10 -> position 9
            # Request B 当前有效长度 20 -> position 19
            decode_position_ids = (
                attention_mask.sum(
                    dim=1,
                    keepdim=True,
                )
                - 1
            )

            decode_position_ids = decode_position_ids.clamp_min(0)

            # DynamicCache 的物理 seq length
            # 对整个 batch 是统一的。
            physical_cache_length = int(past_key_values.get_seq_length())

            cache_position = torch.tensor(
                [physical_cache_length],
                dtype=torch.long,
                device=self.device,
            )

            active_count = sum(not request.is_finished for request in requests)

            trace.active_counts.append(active_count)

            trace.forward_input_shapes.append(tuple(current_input_ids.shape))

            outputs = self.model(
                input_ids=current_input_ids,
                attention_mask=attention_mask,
                position_ids=decode_position_ids,
                cache_position=cache_position,
                past_key_values=past_key_values,
                use_cache=True,
            )

            past_key_values = outputs.past_key_values

            trace.cache_lengths.append(int(past_key_values.get_seq_length()))

            current_input_ids = self._consume_logits(
                requests=requests,
                logits=outputs.logits,
                completing_prefill=False,
            )

        return trace
