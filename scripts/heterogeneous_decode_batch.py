from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache,
)

MODEL_ID = "/home/henry/project/models/Qwen2.5-0.5B-Instruct"


@dataclass
class PrefilledState:
    """
    表示一个已经完成 PREFILL 的 Request。

    request_id:
        请求 ID。

    prompt_token_ids:
        原始 prompt tokens。

    first_token_id:
        PREFILL logits 产生的第一个 output token。

        注意：
        first_token_id 还没有作为 input 再经过 Transformer，
        因此它当前还不在 KV Cache 中。

    cache:
        当前独立 Request 的 KV Cache。

        此时 cache 中只包含 prompt 对应的 K/V。

    logical_cache_length:
        当前 Request 真正有效的 KV token 数。

        PREFILL 刚完成时：
            logical_cache_length == prompt_length
    """

    request_id: str
    prompt_token_ids: list[int]
    first_token_id: int
    cache: DynamicCache
    logical_cache_length: int


# ---------------------------------------------------------
# 将模型 generation_config 中的 EOS token 配置
# 统一转换成 set[int]。
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
# 把一段用户文本转换成模型 prompt token IDs。
#
# 使用与之前实验一致的 chat template，
# 保证 reference generation 和我们的实现输入完全一致。
# ---------------------------------------------------------
def tokenize_prompt(
    tokenizer,
    text: str,
) -> list[int]:

    messages = [
        {
            "role": "user",
            "content": text,
        }
    ]

    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    return encoded["input_ids"][0].tolist()


# ---------------------------------------------------------
# 对单个 Request 执行 PREFILL。
#
# 输入：
#     完整 prompt。
#
# 输出：
#     1. prompt KV Cache
#     2. 第一个生成 token
#
# 注意：
#     第一个生成 token 只是 logits -> argmax 的结果，
#     它还没有被写入 cache。
# ---------------------------------------------------------
@torch.inference_mode()
def prefill_one_request(
    model,
    *,
    request_id: str,
    prompt_token_ids: list[int],
    device: torch.device,
) -> PrefilledState:

    input_ids = torch.tensor(
        [prompt_token_ids],
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.ones_like(
        input_ids,
        dtype=torch.long,
    )

    cache = DynamicCache(config=model.config)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=cache,
        use_cache=True,
    )

    next_token_logits = outputs.logits[:, -1, :]

    first_token_id = int(
        torch.argmax(
            next_token_logits,
            dim=-1,
        ).item()
    )

    cache = outputs.past_key_values

    cache_length = int(cache.get_seq_length())

    if cache_length != len(prompt_token_ids):
        raise RuntimeError(
            "Unexpected cache length after prefill: "
            f"{cache_length} != {len(prompt_token_ids)}"
        )

    return PrefilledState(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        first_token_id=first_token_id,
        cache=cache,
        logical_cache_length=cache_length,
    )


# ---------------------------------------------------------
# 将一个 KV tensor 在 sequence dimension 左侧 padding。
#
# 输入 tensor shape：
#
# [1, num_kv_heads, seq_len, head_dim]
#
# 输出：
#
# [1, num_kv_heads, target_length, head_dim]
#
# 为什么左 padding？
#
# 因为后面 attention mask 也采用左侧 0 padding，
# 这样真实历史 KV 始终靠右排列。
# ---------------------------------------------------------
def left_pad_kv_tensor(
    tensor: torch.Tensor,
    *,
    target_length: int,
) -> torch.Tensor:

    current_length = tensor.shape[-2]

    if current_length > target_length:
        raise ValueError("target_length is smaller than the current KV length.")

    padding_length = target_length - current_length

    if padding_length == 0:
        return tensor

    # F.pad 参数从最后一个维度开始描述。
    #
    # (0, 0)
    #     head_dim 不 padding。
    #
    # (padding_length, 0)
    #     sequence dimension 左侧 padding。
    return F.pad(
        tensor,
        (
            0,
            0,
            padding_length,
            0,
        ),
        value=0.0,
    )


# ---------------------------------------------------------
# 将多个不同 sequence length 的独立 DynamicCache
# 打包成一个 batched DynamicCache。
#
# 例如：
#
# A cache: [1, H, 100, D]
# B cache: [1, H, 300, D]
# C cache: [1, H,  40, D]
#
# 先全部 left-pad 到 300：
#
# [1, H, 300, D]
#
# 然后沿 batch dimension 拼接：
#
# [3, H, 300, D]
#
# 当前 Transformers 的 DynamicCache 由多个 layer cache 组成；
# 每个 layer 分别保存 keys / values。
# ---------------------------------------------------------
def pack_dynamic_caches(
    model,
    states: list[PrefilledState],
) -> DynamicCache:

    if not states:
        raise ValueError("Cannot pack an empty cache list.")

    max_cache_length = max(state.logical_cache_length for state in states)

    num_layers = len(states[0].cache.layers)

    batched_layer_data: list[tuple[torch.Tensor, torch.Tensor]] = []

    for layer_index in range(num_layers):
        padded_keys: list[torch.Tensor] = []
        padded_values: list[torch.Tensor] = []

        for state in states:
            layer = state.cache.layers[layer_index]

            key = layer.keys
            value = layer.values

            padded_key = left_pad_kv_tensor(
                key,
                target_length=max_cache_length,
            )

            padded_value = left_pad_kv_tensor(
                value,
                target_length=max_cache_length,
            )

            padded_keys.append(padded_key)

            padded_values.append(padded_value)

        # batch dimension:
        #
        # [1, H, S, D]
        # [1, H, S, D]
        # [1, H, S, D]
        #
        # ->
        #
        # [B, H, S, D]
        batched_key = torch.cat(
            padded_keys,
            dim=0,
        )

        batched_value = torch.cat(
            padded_values,
            dim=0,
        )

        batched_layer_data.append(
            (
                batched_key,
                batched_value,
            )
        )

    # 当前 DynamicCache 构造器支持接收
    # 已经准备好的 per-layer K/V 数据。
    batched_cache = DynamicCache(
        batched_layer_data,
        config=model.config,
    )

    return batched_cache


# ---------------------------------------------------------
# 为 heterogeneous decode 构造 attention mask。
#
# 假设：
#
# physical cache length = 300
#
# logical lengths:
#
# A = 100
# B = 300
# C = 40
#
# 当前还要输入 1 个新 token，
# 所以最终 mask length = 301。
#
# A:
# 0 x 200 | 1 x 100 | 1
#
# B:
# 1 x 300 | 1
#
# C:
# 0 x 260 | 1 x 40 | 1
# ---------------------------------------------------------
def build_decode_attention_mask(
    logical_cache_lengths: list[int],
    *,
    physical_cache_length: int,
    device: torch.device,
) -> torch.Tensor:

    batch_size = len(logical_cache_lengths)

    attention_mask = torch.zeros(
        (
            batch_size,
            physical_cache_length + 1,
        ),
        dtype=torch.long,
        device=device,
    )

    for batch_index, logical_length in enumerate(logical_cache_lengths):
        if logical_length > physical_cache_length:
            raise ValueError("Logical KV length cannot exceed physical KV length.")

        # 历史有效 KV 占右侧 logical_length 个位置。
        history_start = physical_cache_length - logical_length

        attention_mask[batch_index, history_start:] = 1

        # 注意这个 slice 同时覆盖：
        #
        # historical valid KV
        # +
        # 最后一列 current token
        #
        # 因此最终有效数量：
        #
        # logical_length + 1

    return attention_mask


# ---------------------------------------------------------
# 对多个不同 history length 的 Request
# 执行“一次真正的 batched decode”。
#
# 输入：
#
# current token:
#     [B, 1]
#
# batched KV:
#     [B, H, max_history, D]
#
# attention mask:
#     [B, max_history + 1]
#
# position IDs:
#     每个 Request 使用自己的 logical position。
#
# 返回：
#
# 每个 Request 的下一个 token。
# ---------------------------------------------------------
@torch.inference_mode()
def heterogeneous_decode_once(
    model,
    states: list[PrefilledState],
    *,
    device: torch.device,
) -> list[int]:

    logical_cache_lengths = [state.logical_cache_length for state in states]

    batched_cache = pack_dynamic_caches(
        model=model,
        states=states,
    )

    physical_cache_length = int(batched_cache.get_seq_length())

    # 每个 Request 的第一个 generated token
    # 是本轮真正输入 Transformer 的 token。
    input_ids = torch.tensor(
        [[state.first_token_id] for state in states],
        dtype=torch.long,
        device=device,
    )

    attention_mask = build_decode_attention_mask(
        logical_cache_lengths,
        physical_cache_length=(physical_cache_length),
        device=device,
    )

    # 非常关键：
    #
    # position 取决于 Request 的逻辑长度，
    # 而不是 padding 后的 physical cache length。
    #
    # A cache logical length=100 -> position=100
    # B cache logical length=300 -> position=300
    # C cache logical length=40  -> position=40
    position_ids = torch.tensor(
        logical_cache_lengths,
        dtype=torch.long,
        device=device,
    ).unsqueeze(1)

    print(
        "input_ids shape:",
        tuple(input_ids.shape),
    )

    print(
        "batched physical cache length:",
        physical_cache_length,
    )

    print(
        "logical cache lengths:",
        logical_cache_lengths,
    )

    print(
        "attention_mask shape:",
        tuple(attention_mask.shape),
    )

    print(
        "position_ids:",
        position_ids.squeeze(1).tolist(),
    )

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=batched_cache,
        use_cache=True,
    )

    next_token_ids = torch.argmax(
        outputs.logits[:, -1, :],
        dim=-1,
    )

    return [int(token_id) for token_id in next_token_ids.tolist()]


# ---------------------------------------------------------
# 使用 Hugging Face generate() 生成两个 token，
# 作为单 Request correctness reference。
#
# 返回：
#
# [first_token, second_token]
#
# 我们自己的路径应该满足：
#
# prefill -> first_token
# heterogeneous decode -> second_token
# ---------------------------------------------------------
@torch.inference_mode()
def reference_two_tokens(
    model,
    *,
    prompt_token_ids: list[int],
    device: torch.device,
) -> list[int]:

    input_ids = torch.tensor(
        [prompt_token_ids],
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.ones_like(
        input_ids,
        dtype=torch.long,
    )

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=2,
        do_sample=False,
        repetition_penalty=1.0,
        use_cache=True,
    )

    generated_ids = output_ids[
        0,
        len(prompt_token_ids) :,
    ]

    return generated_ids.tolist()


# ---------------------------------------------------------
# 主实验。
#
# 使用长度明显不同的三个 prompt：
#
# 1. 分别 PREFILL
# 2. 获得独立 KV Cache
# 3. 将不同长度 KV pad + pack
# 4. 一次 batched decode
# 5. 和 HF 单请求 generate() 做 token-level comparison
# ---------------------------------------------------------
def main() -> None:

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    device = torch.device("cuda")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
    ).to(device)

    model.eval()

    prompts = [
        (
            "A",
            "What is KV cache?",
        ),
        (
            "B",
            (
                "Explain in detail why continuous batching "
                "can improve GPU utilization in an LLM "
                "serving system, including the relationship "
                "between request completion, batch slots, "
                "and decode iterations."
            ),
        ),
        (
            "C",
            "Define TTFT briefly.",
        ),
    ]

    states: list[PrefilledState] = []

    print("=== Individual Prefill ===")

    for request_id, text in prompts:
        prompt_token_ids = tokenize_prompt(
            tokenizer,
            text,
        )

        state = prefill_one_request(
            model,
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            device=device,
        )

        states.append(state)

        print(
            f"{request_id}: "
            f"prompt={len(prompt_token_ids)}, "
            f"cache={state.logical_cache_length}, "
            f"first_token={state.first_token_id}"
        )

    print()
    print("=== One Heterogeneous Batched Decode ===")

    second_token_ids = heterogeneous_decode_once(
        model=model,
        states=states,
        device=device,
    )

    print()
    print("=== Correctness ===")

    for state, second_token_id in zip(
        states,
        second_token_ids,
        strict=True,
    ):
        reference_ids = reference_two_tokens(
            model,
            prompt_token_ids=(state.prompt_token_ids),
            device=device,
        )

        if len(reference_ids) < 2:
            raise RuntimeError(
                f"{state.request_id} terminated before producing two tokens."
            )

        expected_first = reference_ids[0]
        expected_second = reference_ids[1]

        first_ok = state.first_token_id == expected_first

        second_ok = second_token_id == expected_second

        print(f"{state.request_id}: first={first_ok}, second={second_ok}")

        print(f"  our tokens: [{state.first_token_id}, {second_token_id}]")

        print(f"  reference:  {reference_ids}")

        if not first_ok or not second_ok:
            raise AssertionError(f"Token mismatch for request {state.request_id}.")

    print()
    print("Heterogeneous decode batching: PASS")


if __name__ == "__main__":
    main()
