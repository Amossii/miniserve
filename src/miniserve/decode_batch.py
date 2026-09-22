from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from miniserve.request import Request


@dataclass
class DecodeState:
    """
    一个 Request 在 Decode 阶段需要保存的执行状态。

    request:
        Request 负责逻辑状态，例如：
        - prompt tokens
        - generated tokens
        - lifecycle
        - max_new_tokens

    cache:
        该 Request 当前独立拥有的 KV Cache。

        注意一个重要 invariant：

            request.sequence_length
            ==
            cache.get_seq_length() + 1

        因为 Request 最后一个 generated token
        还没有真正作为输入经过 Transformer。
    """

    request: Request
    cache: DynamicCache


@dataclass(frozen=True)
class DecodeStepOutput:
    """
    保存一次 batched decode iteration 的观测信息。

    request_ids:
        本轮参与 batched forward 的 Request ID。

    token_ids:
        本轮模型为每个 Request 新预测出的 token。

    logical_cache_lengths_before:
        forward 前每个 Request 自己的有效 KV 长度。

    physical_cache_length:
        padding / packing 后统一的 KV sequence length。

    input_shape:
        本轮真正送入模型的 input_ids shape。

        Decode 时应该始终类似：

            [batch_size, 1]
    """

    request_ids: tuple[str, ...]
    token_ids: tuple[int, ...]
    logical_cache_lengths_before: tuple[int, ...]
    physical_cache_length: int
    input_shape: tuple[int, int]


# ---------------------------------------------------------
# 将模型中的 EOS 配置统一转换为 set[int]。
# ---------------------------------------------------------
def normalize_eos_token_ids(
    eos_token_id: int | list[int] | None,
) -> set[int]:

    if eos_token_id is None:
        return set()

    if isinstance(eos_token_id, int):
        return {eos_token_id}

    return set(eos_token_id)


# ---------------------------------------------------------
# 对一个 KV tensor 在 sequence dimension 做左 padding。
#
# 输入：
#
# [1, H, seq_len, D]
#
# 输出：
#
# [1, H, target_length, D]
#
# 真实 KV 始终靠右。
# ---------------------------------------------------------
def left_pad_kv_tensor(
    tensor: torch.Tensor,
    *,
    target_length: int,
) -> torch.Tensor:

    current_length = tensor.shape[-2]

    if current_length > target_length:
        raise ValueError("target_length cannot be smaller than current KV length.")

    pad_length = target_length - current_length

    if pad_length == 0:
        return tensor

    return F.pad(
        tensor,
        (
            0,
            0,
            pad_length,
            0,
        ),
        value=0.0,
    )


# ---------------------------------------------------------
# 获取单个 DynamicCache 当前逻辑 sequence length。
# ---------------------------------------------------------
def cache_length(
    cache: DynamicCache,
) -> int:

    return int(cache.get_seq_length())


# ---------------------------------------------------------
# 将多个独立 Request KV Cache 打包成一个 batched cache。
#
# 例如：
#
# A: [1, H, 20, D]
# B: [1, H, 50, D]
# C: [1, H, 18, D]
#
# padding 后：
#
# A: [1, H, 50, D]
# B: [1, H, 50, D]
# C: [1, H, 50, D]
#
# batch cat：
#
# [3, H, 50, D]
#
# 返回：
#
# batched_cache
# logical_lengths
# physical_length
# ---------------------------------------------------------
def pack_dynamic_caches(
    states: list[DecodeState],
    *,
    model_config=None,
) -> tuple[
    DynamicCache,
    list[int],
    int,
]:

    if not states:
        raise ValueError("Cannot pack an empty DecodeState list.")

    logical_lengths = [cache_length(state.cache) for state in states]

    physical_length = max(logical_lengths)

    num_layers = len(states[0].cache.layers)

    # 确保所有 Request 使用相同模型结构。
    for state in states:
        if len(state.cache.layers) != num_layers:
            raise RuntimeError("All caches must have the same number of layers.")

    batched_layer_data: list[tuple[torch.Tensor, torch.Tensor]] = []

    for layer_index in range(num_layers):
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []

        for state in states:
            layer = state.cache.layers[layer_index]

            key = left_pad_kv_tensor(
                layer.keys,
                target_length=physical_length,
            )

            value = left_pad_kv_tensor(
                layer.values,
                target_length=physical_length,
            )

            keys.append(key)
            values.append(value)

        batched_key = torch.cat(
            keys,
            dim=0,
        )

        batched_value = torch.cat(
            values,
            dim=0,
        )

        batched_layer_data.append(
            (
                batched_key,
                batched_value,
            )
        )

    # 当前 Transformers v5 DynamicCache
    # 使用 ddp_cache_data 接收已经准备好的
    # per-layer K/V tensors。
    batched_cache = DynamicCache(
        ddp_cache_data=batched_layer_data,
        config=model_config,
    )

    return (
        batched_cache,
        logical_lengths,
        physical_length,
    )


# ---------------------------------------------------------
# 将一次 batched forward 更新后的 KV Cache
# 再拆成独立的 per-request caches。
#
# forward 前：
#
# physical length = P
# Request i logical length = L_i
#
# padding 后有效历史位于：
#
# [P - L_i, ..., P - 1]
#
# 当前 token forward 后写入 physical position P，
# 因此新的有效范围是：
#
# [P - L_i, ..., P]
#
# 长度正好：
#
# L_i + 1
# ---------------------------------------------------------
def unpack_dynamic_cache(
    batched_cache: DynamicCache,
    *,
    logical_lengths_before: list[int],
    physical_length_before: int,
    model_config=None,
) -> list[DynamicCache]:

    batch_size = len(logical_lengths_before)

    if batch_size == 0:
        return []

    result: list[DynamicCache] = []

    for batch_index, logical_length in enumerate(logical_lengths_before):
        start = physical_length_before - logical_length

        per_layer_data: list[tuple[torch.Tensor, torch.Tensor]] = []

        for layer in batched_cache.layers:
            # clone() 是故意的。
            #
            # Phase A 希望每个 Request 真正重新拥有
            # 独立 KV storage，而不是持有 batched
            # tensor 的 view。
            #
            # 这会产生 copy overhead，
            # 后面 benchmark 会如实暴露它。
            key = layer.keys[
                batch_index : batch_index + 1,
                :,
                start:,
                :,
            ].clone()

            value = layer.values[
                batch_index : batch_index + 1,
                :,
                start:,
                :,
            ].clone()

            expected_length = logical_length + 1

            if key.shape[-2] != expected_length:
                raise RuntimeError("Unexpected unpacked KV length.")

            per_layer_data.append(
                (
                    key,
                    value,
                )
            )

        individual_cache = DynamicCache(
            ddp_cache_data=per_layer_data,
            config=model_config,
        )

        result.append(individual_cache)

    return result


# ---------------------------------------------------------
# 构造 heterogeneous decode 的 attention mask。
#
# 例如：
#
# logical:
# [20, 50, 18]
#
# physical:
# 50
#
# 输出 shape：
#
# [3, 51]
#
# 最后一列是本轮 current token。
# ---------------------------------------------------------
def build_decode_attention_mask(
    logical_lengths: list[int],
    *,
    physical_length: int,
    device: torch.device,
) -> torch.Tensor:

    batch_size = len(logical_lengths)

    attention_mask = torch.zeros(
        (
            batch_size,
            physical_length + 1,
        ),
        dtype=torch.long,
        device=device,
    )

    for batch_index, logical_length in enumerate(logical_lengths):
        if logical_length > physical_length:
            raise ValueError(
                "logical cache length cannot exceed physical cache length."
            )

        start = physical_length - logical_length

        # 这里一直写到最后一列。
        #
        # 所以同时包含：
        #
        # historical KV
        # +
        # current token
        attention_mask[
            batch_index,
            start:,
        ] = 1

    return attention_mask


class DecodeBatchRunner:
    """
    MiniServe Phase A heterogeneous Decode Batch Runner。

    它实现：

    per-request KV
        ↓
    padding + pack
        ↓
    one batched model.forward()
        ↓
    unpack KV
        ↓
    per-request KV

    每调用一次 decode_step()：

    每个 active Request 正好推进一个 output token。
    """

    # -----------------------------------------------------
    # 初始化 DecodeBatchRunner。
    # -----------------------------------------------------
    def __init__(
        self,
        *,
        model,
        device: torch.device,
        eos_token_ids: set[int],
    ) -> None:

        self.model = model
        self.device = device
        self.eos_token_ids = set(eos_token_ids)

    # -----------------------------------------------------
    # 对一个 RUNNING + PREFILL Request
    # 执行独立 prefill。
    #
    # prefill 之后：
    #
    # cache:
    #     保存完整 prompt KV。
    #
    # Request:
    #     已经拥有 first generated token。
    #
    # 因此：
    #
    # request.sequence_length
    # =
    # cache_length + 1
    # -----------------------------------------------------
    @torch.inference_mode()
    def prefill_request(
        self,
        request: Request,
    ) -> DecodeState:

        if not request.is_running:
            raise RuntimeError("Request must be RUNNING before prefill.")

        if not request.needs_prefill:
            raise RuntimeError("Request is not in PREFILL phase.")

        if request.prompt_length <= 0:
            raise ValueError("Empty prompts are not supported.")

        input_ids = torch.tensor(
            [request.prompt_token_ids],
            dtype=torch.long,
            device=self.device,
        )

        attention_mask = torch.ones_like(
            input_ids,
            dtype=torch.long,
        )

        position_ids = torch.arange(
            request.prompt_length,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)

        cache_position = torch.arange(
            request.prompt_length,
            dtype=torch.long,
            device=self.device,
        )

        cache = DynamicCache(config=self.model.config)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=cache,
            use_cache=True,
        )

        cache = outputs.past_key_values

        next_token_id = int(
            torch.argmax(
                outputs.logits[:, -1, :],
                dim=-1,
            ).item()
        )

        timestamp = time.perf_counter()

        request.append_generated_token(
            next_token_id,
            timestamp=timestamp,
        )

        request.mark_prefill_completed()

        if next_token_id in self.eos_token_ids or request.reached_max_new_tokens:
            request.mark_finished(timestamp=timestamp)

        state = DecodeState(
            request=request,
            cache=cache,
        )

        # 如果没有直接结束，
        # 应满足最重要的 decode invariant。
        if not request.is_finished:
            if request.sequence_length != cache_length(cache) + 1:
                raise RuntimeError(
                    "Prefill invariant violated: "
                    "request.sequence_length must equal "
                    "cache_length + 1."
                )

        return state

    # -----------------------------------------------------
    # 对多个 heterogeneous Request 执行一次 batched decode。
    #
    # 每个 Request：
    #
    # input:
    #     最新生成但尚未进入 KV Cache 的一个 token。
    #
    # cache:
    #     之前所有真正经过 Transformer 的 token。
    #
    # 输出：
    #     每个 Request 再生成一个 token。
    # -----------------------------------------------------
    @torch.inference_mode()
    def decode_step(
        self,
        states: list[DecodeState],
    ) -> DecodeStepOutput:

        if not states:
            raise ValueError("decode_step requires at least one state.")

        # -------------------------------------------------
        # 验证每个 Request 当前都可以进行 Decode。
        # -------------------------------------------------

        for state in states:
            request = state.request

            if request.is_finished:
                raise RuntimeError(
                    "Finished requests must not enter DecodeBatchRunner."
                )

            if not request.is_running:
                raise RuntimeError("Decode request must be RUNNING.")

            if not request.needs_decode:
                raise RuntimeError("Decode request must be in DECODE phase.")

            if not request.generated_token_ids:
                raise RuntimeError(
                    "Decode request must have a pending generated token."
                )

            logical_length = cache_length(state.cache)

            if request.sequence_length != logical_length + 1:
                raise RuntimeError(
                    f"Decode invariant violated for {request.request_id}."
                )

        # -------------------------------------------------
        # 1. Pack KV Cache
        # -------------------------------------------------

        (
            batched_cache,
            logical_lengths,
            physical_length,
        ) = pack_dynamic_caches(
            states,
            model_config=self.model.config,
        )

        # -------------------------------------------------
        # 2. 构造 [B, 1] decode input
        #
        # 每个 Request 只提交自己最后一个
        # 尚未进入 KV Cache 的 generated token。
        # -------------------------------------------------

        input_ids = torch.tensor(
            [[state.request.generated_token_ids[-1]] for state in states],
            dtype=torch.long,
            device=self.device,
        )

        # -------------------------------------------------
        # 3. Attention Mask
        # -------------------------------------------------

        attention_mask = build_decode_attention_mask(
            logical_lengths,
            physical_length=(physical_length),
            device=self.device,
        )

        # -------------------------------------------------
        # 4. Logical Position IDs
        #
        # A cache length 20 -> current token position 20
        # B cache length 50 -> current token position 50
        #
        # 不能统一使用 physical_length。
        # -------------------------------------------------

        position_ids = torch.tensor(
            logical_lengths,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(1)

        # -------------------------------------------------
        # 5. Physical Cache Position
        #
        # batched KV 的 physical sequence length 是统一的。
        #
        # 当前 token 会被 append 到 physical_length。
        # -------------------------------------------------

        cache_position = torch.tensor(
            [physical_length],
            dtype=torch.long,
            device=self.device,
        )

        # -------------------------------------------------
        # 6. 真正的一次 Batched Decode Forward
        # -------------------------------------------------

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=batched_cache,
            use_cache=True,
        )

        next_token_ids = torch.argmax(
            outputs.logits[:, -1, :],
            dim=-1,
        )

        # -------------------------------------------------
        # 7. 将更新后的 Batch Cache 拆回 Request Cache
        #
        # 每个 Request 的 logical cache length 都 +1。
        # -------------------------------------------------

        individual_caches = unpack_dynamic_cache(
            outputs.past_key_values,
            logical_lengths_before=(logical_lengths),
            physical_length_before=(physical_length),
            model_config=self.model.config,
        )

        timestamp = time.perf_counter()

        produced_token_ids: list[int] = []

        # -------------------------------------------------
        # 8. 更新每个 Request 的逻辑状态
        # -------------------------------------------------

        for batch_index, state in enumerate(states):
            request = state.request

            # 新 cache 已经包含：
            #
            # previous cached history
            # +
            # 本轮 input token
            state.cache = individual_caches[batch_index]

            token_id = int(next_token_ids[batch_index].item())

            produced_token_ids.append(token_id)

            # token_id 是这一轮新预测出来的。
            #
            # 它还没有进入新的 cache，
            # 下一轮才作为 input。
            request.append_generated_token(
                token_id,
                timestamp=timestamp,
            )

            if token_id in self.eos_token_ids or request.reached_max_new_tokens:
                request.mark_finished(timestamp=timestamp)

            # 未结束 Request 应继续满足：
            #
            # request sequence
            # =
            # cache sequence + 1
            if not request.is_finished:
                if request.sequence_length != cache_length(state.cache) + 1:
                    raise RuntimeError(
                        f"Post-decode invariant violated for {request.request_id}."
                    )

        return DecodeStepOutput(
            request_ids=tuple(state.request.request_id for state in states),
            token_ids=tuple(produced_token_ids),
            logical_cache_lengths_before=tuple(logical_lengths),
            physical_cache_length=(physical_length),
            input_shape=tuple(input_ids.shape),
        )
