from types import SimpleNamespace

import pytest
import torch

from miniserve.engine import Engine
from miniserve.request import (
    ExecutionPhase,
    Request,
)


class FakeCache:
    """
    测试使用的最小 KV Cache。

    只保存 cached sequence length，
    不保存真正 K/V tensor。
    """

    # 创建一个指定逻辑长度的 fake cache。
    def __init__(
        self,
        seq_length: int,
    ) -> None:

        self._seq_length = seq_length

    # 模拟 Transformers Cache.get_seq_length()。
    def get_seq_length(self) -> int:

        return self._seq_length


class FakeModel:
    """
    一个 deterministic fake causal LM。

    每调用一次 model(...)，
    按顺序生成 next_tokens 中的 token。

    同时模拟 KV Cache sequence length 增长。

    这样可以测试：
    - prefill input length
    - decode input length
    - cache growth
    - EOS
    - request state transitions

    而完全不需要真实 GPU/model。
    """

    # 初始化 FakeModel。
    def __init__(
        self,
        next_tokens: list[int],
        eos_token_id: int = 99,
        vocab_size: int = 128,
    ) -> None:

        self.next_tokens = next_tokens

        self.vocab_size = vocab_size

        self.call_index = 0

        self.forward_input_lengths: list[int] = []

        self.generation_config = SimpleNamespace(eos_token_id=eos_token_id)

    # 模拟 Hugging Face model forward。
    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values,
        use_cache: bool,
    ):

        assert use_cache

        input_length = input_ids.shape[1]

        self.forward_input_lengths.append(input_length)

        # 第一次 prefill 没有历史 cache。
        if past_key_values is None:
            past_length = 0
        else:
            past_length = past_key_values.get_seq_length()

        new_cache = FakeCache(past_length + input_length)

        # 当前调用应该生成哪个 token。
        token_id = self.next_tokens[self.call_index]

        self.call_index += 1

        # 构造假的 logits。
        #
        # 只需要保证 argmax 最终选中 token_id。
        logits = torch.zeros(
            (
                1,
                input_length,
                self.vocab_size,
            ),
            dtype=torch.float32,
        )

        logits[
            0,
            -1,
            token_id,
        ] = 100.0

        return SimpleNamespace(
            logits=logits,
            past_key_values=new_cache,
        )


# ---------------------------------------------------------
# 验证一个 Request 从：
#
# WAITING + PREFILL
#
# 一步之后：
#
# RUNNING + DECODE
#
# 并且建立 prompt KV Cache。
# ---------------------------------------------------------
def test_prefill_step():

    model = FakeModel(next_tokens=[10])

    engine = Engine(
        model=model,
        device=torch.device("cpu"),
    )

    request = Request(
        request_id="req-1",
        prompt_token_ids=[1, 2, 3],
        max_new_tokens=10,
    )

    engine.add_request(request)

    token_id = engine.step()

    assert token_id == 10

    assert request.is_running

    assert request.phase is ExecutionPhase.DECODE

    assert request.generated_token_ids == [10]

    # Prefill 应处理整个 prompt。
    assert model.forward_input_lengths == [3]

    # Cache 此时只包含真正 forward 过的 prompt。
    assert engine.cache_length == 3

    # Request 逻辑 sequence 已经包含刚预测出来的 token 10。
    assert request.sequence_length == 4


# ---------------------------------------------------------
# 验证：
#
# PREFILL:
#     input length = prompt length
#
# DECODE:
#     input length = 1
# ---------------------------------------------------------
def test_decode_uses_one_token_input():

    model = FakeModel(next_tokens=[10, 11, 12])

    engine = Engine(
        model=model,
        device=torch.device("cpu"),
    )

    request = Request(
        request_id="req-1",
        prompt_token_ids=[1, 2, 3],
        max_new_tokens=10,
    )

    engine.add_request(request)

    engine.step()
    engine.step()
    engine.step()

    assert model.forward_input_lengths == [
        3,
        1,
        1,
    ]

    assert request.generated_token_ids == [
        10,
        11,
        12,
    ]

    # 三次 forward 实际写入 cache：
    #
    # prompt: 3
    # token10: +1
    # token11: +1
    #
    # token12 只是刚刚被预测，
    # 尚未作为下一轮输入进入 cache。
    assert engine.cache_length == 5

    assert request.sequence_length == 6

    # 核心 invariant：
    assert request.sequence_length == engine.cache_length + 1


# ---------------------------------------------------------
# 验证生成 EOS 后 Request 自动结束。
# ---------------------------------------------------------
def test_request_finishes_on_eos():

    model = FakeModel(
        next_tokens=[10, 11, 99],
        eos_token_id=99,
    )

    engine = Engine(
        model=model,
        device=torch.device("cpu"),
    )

    request = Request(
        request_id="req-1",
        prompt_token_ids=[1, 2],
        max_new_tokens=10,
    )

    engine.add_request(request)

    while engine.has_unfinished_requests():
        engine.step()

    assert request.generated_token_ids == [
        10,
        11,
        99,
    ]

    assert request.is_finished

    assert not engine.has_unfinished_requests()


# ---------------------------------------------------------
# 验证达到 max_new_tokens 后自动结束。
# ---------------------------------------------------------
def test_request_finishes_on_length_limit():

    model = FakeModel(next_tokens=[10, 11, 12])

    engine = Engine(
        model=model,
        device=torch.device("cpu"),
    )

    request = Request(
        request_id="req-1",
        prompt_token_ids=[1, 2],
        max_new_tokens=2,
    )

    engine.add_request(request)

    while engine.has_unfinished_requests():
        engine.step()

    assert request.generated_token_ids == [
        10,
        11,
    ]

    assert request.is_finished


# ---------------------------------------------------------
# 当前 Engine 只支持一个 unfinished Request。
# ---------------------------------------------------------
def test_cannot_add_second_active_request():

    model = FakeModel(next_tokens=[10])

    engine = Engine(
        model=model,
        device=torch.device("cpu"),
    )

    request_a = Request(
        request_id="A",
        prompt_token_ids=[1],
    )

    request_b = Request(
        request_id="B",
        prompt_token_ids=[2],
    )

    engine.add_request(request_a)

    with pytest.raises(RuntimeError):
        engine.add_request(request_b)


# ---------------------------------------------------------
# 没有 Request 时不能调用 step()。
# ---------------------------------------------------------
def test_cannot_step_empty_engine():

    model = FakeModel(next_tokens=[10])

    engine = Engine(
        model=model,
        device=torch.device("cpu"),
    )

    with pytest.raises(RuntimeError):
        engine.step()
