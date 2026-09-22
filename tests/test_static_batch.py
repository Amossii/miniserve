from types import SimpleNamespace

import torch

from miniserve.request import Request
from miniserve.static_batch import StaticBatchRunner


class FakeCache:
    """
    测试 StaticBatchRunner 使用的最小 cache。

    只保存统一的物理 sequence length。
    """

    def __init__(
        self,
        seq_length: int = 0,
    ) -> None:

        self._seq_length = seq_length

    def get_seq_length(self) -> int:
        return self._seq_length


class FakeModel:
    """
    deterministic batched fake model。

    每一次 forward 都为 batch 中每个 row
    返回预先设定的 next token。

    next_tokens_per_step 示例：

    [
        [10, 20, 30],
        [11, 21, 31],
        [12, 22, 32],
    ]

    表示：

    prefill:
        A -> 10
        B -> 20
        C -> 30

    decode 1:
        A -> 11
        B -> 21
        C -> 31

    decode 2:
        A -> 12
        B -> 22
        C -> 32
    """

    def __init__(
        self,
        next_tokens_per_step: list[list[int]],
        vocab_size: int = 128,
    ) -> None:

        self.next_tokens_per_step = next_tokens_per_step

        self.vocab_size = vocab_size

        self.call_index = 0

        self.config = SimpleNamespace()

        self.input_shapes: list[tuple[int, int]] = []

        self.attention_masks: list[torch.Tensor] = []

    # 模拟 Hugging Face batched forward。
    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values,
        use_cache: bool,
    ):

        assert use_cache

        batch_size = input_ids.shape[0]
        input_length = input_ids.shape[1]

        self.input_shapes.append(tuple(input_ids.shape))

        self.attention_masks.append(attention_mask.clone())

        if hasattr(
            past_key_values,
            "get_seq_length",
        ):
            past_length = int(past_key_values.get_seq_length())
        else:
            past_length = 0

        # Fake DynamicCache 的实际内容我们不需要。
        #
        # 这里只构造新的统一物理 cache length。
        new_cache = FakeCache(past_length + input_length)

        logits = torch.zeros(
            (
                batch_size,
                input_length,
                self.vocab_size,
            ),
            dtype=torch.float32,
        )

        tokens = self.next_tokens_per_step[self.call_index]

        self.call_index += 1

        for batch_index, token_id in enumerate(tokens):
            logits[
                batch_index,
                -1,
                token_id,
            ] = 100.0

        return SimpleNamespace(
            logits=logits,
            past_key_values=new_cache,
        )


# 为测试创建 Runner。
#
# 测试中不用真实 Transformers DynamicCache，
# 因此下面会通过 subclass 覆盖 cache 创建逻辑不方便；
# 我们改用一个带 monkeypatch 的简化方式测试核心行为。
def make_requests() -> list[Request]:

    return [
        Request(
            request_id="A",
            prompt_token_ids=[1, 2],
            max_new_tokens=3,
        ),
        Request(
            request_id="B",
            prompt_token_ids=[3, 4, 5, 6],
            max_new_tokens=3,
        ),
        Request(
            request_id="C",
            prompt_token_ids=[7],
            max_new_tokens=3,
        ),
    ]


# 验证 left padding 的 input / mask 是否正确。
def test_build_prefill_batch():

    model = SimpleNamespace()

    runner = StaticBatchRunner(
        model=model,
        device=torch.device("cpu"),
        pad_token_id=0,
        eos_token_ids={99},
    )

    requests = make_requests()

    input_ids, attention_mask = runner._build_prefill_batch(requests)

    assert input_ids.tolist() == [
        [0, 0, 1, 2],
        [3, 4, 5, 6],
        [0, 0, 0, 7],
    ]

    assert attention_mask.tolist() == [
        [0, 0, 1, 1],
        [1, 1, 1, 1],
        [0, 0, 0, 1],
    ]


# 验证 position_ids 正确忽略 left padding。
def test_position_ids():

    runner = StaticBatchRunner(
        model=SimpleNamespace(),
        device=torch.device("cpu"),
        pad_token_id=0,
        eos_token_ids={99},
    )

    mask = torch.tensor(
        [
            [0, 0, 1, 1],
            [1, 1, 1, 1],
            [0, 1, 1, 1],
        ],
        dtype=torch.long,
    )

    position_ids = runner._build_position_ids(mask)

    assert position_ids.tolist() == [
        [0, 0, 0, 1],
        [0, 1, 2, 3],
        [0, 0, 1, 2],
    ]


def test_static_tokens_are_read_before_timestamp(monkeypatch):
    """输入补丁工具；无返回；验证静态批处理读回顺序，已结束行不会追加计时样本。"""
    runner = StaticBatchRunner(
        model=SimpleNamespace(),
        device=torch.device("cpu"),
        pad_token_id=0,
        eos_token_ids={99},
    )
    active = Request("A", [1], max_new_tokens=1, arrival_time=0)
    finished = Request("B", [1], max_new_tokens=1, arrival_time=0)
    active.mark_running(timestamp=0)
    finished.mark_running(timestamp=0)
    finished.append_generated_token(99, timestamp=0.1)
    finished.mark_finished(timestamp=0.1)
    events = []

    def readback():
        """输入无；返回采样结果；记录 CPU 读回事件，验证计时顺序。"""
        events.append("readback")
        return [5, 6]

    def clock():
        """输入无；返回固定时间；记录计时事件，避免依赖测试运行速度。"""
        events.append("timestamp")
        return 0.2

    monkeypatch.setattr(
        torch, "argmax", lambda *args, **kwargs: SimpleNamespace(tolist=readback)
    )
    monkeypatch.setattr(
        "miniserve.static_batch.time", SimpleNamespace(perf_counter=clock)
    )
    runner._consume_logits(
        [active, finished], torch.zeros(2, 1, 8), completing_prefill=True
    )
    assert events == ["readback", "timestamp"]
    assert active.token_timestamps == [0.2]
    assert active.finish_time == 0.2
    assert finished.token_timestamps == [0.1]
