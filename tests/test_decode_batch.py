import torch
from transformers import DynamicCache

from miniserve.decode_batch import (
    DecodeState,
    build_decode_attention_mask,
    cache_length,
    left_pad_kv_tensor,
    pack_dynamic_caches,
    unpack_dynamic_cache,
)
from miniserve.request import Request


# ---------------------------------------------------------
# 创建一个只有 1 layer 的 synthetic DynamicCache。
#
# Tensor shape：
#
# [1, 1, seq_len, 1]
#
# value_start 用来让不同 cache 的内容容易区分。
# ---------------------------------------------------------
def make_fake_cache(
    seq_len: int,
    *,
    value_start: int,
) -> DynamicCache:

    values = torch.arange(
        value_start,
        value_start + seq_len,
        dtype=torch.float32,
    ).reshape(
        1,
        1,
        seq_len,
        1,
    )

    keys = values.clone()

    return DynamicCache(
        ddp_cache_data=[
            (
                keys,
                values,
            )
        ]
    )


# ---------------------------------------------------------
# 验证左 padding。
# ---------------------------------------------------------
def test_left_pad_kv_tensor():

    tensor = torch.tensor([[[[1.0], [2.0]]]])

    padded = left_pad_kv_tensor(
        tensor,
        target_length=4,
    )

    assert padded.shape == (
        1,
        1,
        4,
        1,
    )

    assert padded.flatten().tolist() == [
        0.0,
        0.0,
        1.0,
        2.0,
    ]


# ---------------------------------------------------------
# 验证不同 KV length 可以 pack 成统一 physical length。
# ---------------------------------------------------------
def test_pack_dynamic_caches():

    request_a = Request(
        request_id="A",
        prompt_token_ids=[1],
    )

    request_b = Request(
        request_id="B",
        prompt_token_ids=[1],
    )

    state_a = DecodeState(
        request=request_a,
        cache=make_fake_cache(
            2,
            value_start=10,
        ),
    )

    state_b = DecodeState(
        request=request_b,
        cache=make_fake_cache(
            4,
            value_start=20,
        ),
    )

    (
        batched_cache,
        logical_lengths,
        physical_length,
    ) = pack_dynamic_caches(
        [
            state_a,
            state_b,
        ]
    )

    assert logical_lengths == [
        2,
        4,
    ]

    assert physical_length == 4

    assert (batched_cache.layers[0].keys.shape) == (
        2,
        1,
        4,
        1,
    )

    # A 左侧应该补两个 0。
    assert (batched_cache.layers[0].keys[0].flatten().tolist()) == [
        0.0,
        0.0,
        10.0,
        11.0,
    ]


# ---------------------------------------------------------
# 验证 attention mask。
#
# logical:
#
# A = 2
# B = 4
#
# physical:
#
# 4
#
# 加上 current token 后 mask length = 5。
# ---------------------------------------------------------
def test_decode_attention_mask():

    mask = build_decode_attention_mask(
        [
            2,
            4,
        ],
        physical_length=4,
        device=torch.device("cpu"),
    )

    assert mask.tolist() == [
        [0, 0, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ]


# ---------------------------------------------------------
# 验证 batched cache forward 后能够正确 unpack。
#
# 这里直接构造一个“模拟 forward 后”的 batched cache。
#
# forward 前：
#
# A logical=2
# B logical=4
# physical=4
#
# forward 后 physical=5。
#
# A 有效数据应该是：
#
# old A 2 tokens + current token
#
# 长度 = 3。
#
# B 长度 = 5。
# ---------------------------------------------------------
def test_unpack_dynamic_cache():

    batched_key = torch.tensor(
        [
            # A:
            # 两个 padding + 两个历史 KV + 当前 token
            [
                [
                    [0.0],
                    [0.0],
                    [10.0],
                    [11.0],
                    [12.0],
                ]
            ],
            # B:
            [
                [
                    [20.0],
                    [21.0],
                    [22.0],
                    [23.0],
                    [24.0],
                ]
            ],
        ]
    )

    batched_value = batched_key.clone()

    batched_cache = DynamicCache(
        ddp_cache_data=[
            (
                batched_key,
                batched_value,
            )
        ]
    )

    caches = unpack_dynamic_cache(
        batched_cache,
        logical_lengths_before=[
            2,
            4,
        ],
        physical_length_before=4,
    )

    assert len(caches) == 2

    assert cache_length(caches[0]) == 3

    assert cache_length(caches[1]) == 5

    assert (caches[0].layers[0].keys.flatten().tolist()) == [
        10.0,
        11.0,
        12.0,
    ]

    assert (caches[1].layers[0].keys.flatten().tolist()) == [
        20.0,
        21.0,
        22.0,
        23.0,
        24.0,
    ]
