import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from miniserve.decode_batch import DecodeBatchRunner, cache_length
from miniserve.engine import Engine
from miniserve.request import Request
from miniserve.scheduler import Scheduler


@pytest.mark.parametrize("budget", [0, -1, 2, 3.5, True])
def test_invalid_budget(budget):
    """输入非法预算；无返回、不建队列；验证正整数及覆盖全部 decode 的约束。"""
    with pytest.raises(ValueError):
        Scheduler(max_num_running=3, max_num_batched_tokens=budget)


def test_oversized_prompt_is_rejected_before_enqueue():
    """输入无；无返回；尝试超长请求，验证失败不污染队列、状态或 ID 登记。"""
    scheduler = Scheduler(max_num_running=2, max_num_batched_tokens=6)
    request = Request("A", [1] * 7)
    with pytest.raises(ValueError, match="Prompt exceeds token budget"):
        scheduler.add_request(request)
    assert request.is_waiting
    assert scheduler.waiting_requests == scheduler.running_requests == ()
    assert not scheduler.has_unfinished_requests()
    replacement = Request("A", [1] * 6)
    scheduler.add_request(replacement)
    plan = scheduler.schedule()
    assert plan.prefill_requests == (replacement,)
    assert plan.num_scheduled_tokens == 6


def test_decode_reservation_and_budget_reset():
    """输入无；无返回；推进三轮阶段，验证 decode 预留、精确边界和每轮重新计费。"""
    scheduler = Scheduler(max_num_running=3, max_num_batched_tokens=6)
    a, b, c = Request("A", [1]), Request("B", [1] * 5), Request("C", [1] * 2)
    scheduler.add_request(a)
    scheduler.schedule()
    a.mark_prefill_completed()
    scheduler.add_request(b)
    scheduler.add_request(c)

    first = scheduler.schedule()
    assert first.decode_requests == (a,)
    assert first.prefill_requests == first.newly_admitted == (b,)
    assert first.num_scheduled_tokens == 6
    assert scheduler.num_free_slots == 1
    assert scheduler.waiting_requests == (c,)
    assert c.is_waiting

    # 未执行计划时重复调度，B 仍按完整 prompt 计费，不能被误计为一个 token。
    repeated = scheduler.schedule()
    assert repeated.newly_admitted == ()
    assert repeated.num_scheduled_tokens == 6
    b.mark_prefill_completed()
    assert first.num_scheduled_tokens == 6

    second = scheduler.schedule()
    assert second.decode_requests == (a, b)
    assert second.prefill_requests == (c,)
    assert second.num_scheduled_tokens == 4
    c.mark_prefill_completed()
    third = scheduler.schedule()
    assert third.decode_requests == (a, b, c)
    assert third.num_scheduled_tokens == 3


def test_fifo_head_blocks_shorter_request_until_decode_finishes():
    """输入无；无返回；模拟 decode 结束，验证不跳队及轮首回收后重用完整预算。"""
    scheduler = Scheduler(max_num_running=3, max_num_batched_tokens=6)
    a, head, tail = Request("A", [1]), Request("head", [1] * 6), Request("tail", [1])
    scheduler.add_request(a)
    scheduler.schedule()
    a.mark_prefill_completed()
    scheduler.add_request(head)
    scheduler.add_request(tail)
    blocked = scheduler.schedule()
    assert blocked.decode_requests == (a,)
    assert blocked.prefill_requests == blocked.newly_admitted == ()
    assert blocked.num_scheduled_tokens == 1
    assert scheduler.waiting_requests == (head, tail)

    a.mark_finished()
    plan = scheduler.schedule()
    assert plan.finished_requests == (a,)
    assert plan.prefill_requests == (head,)
    assert plan.num_scheduled_tokens == 6
    assert scheduler.waiting_requests == (tail,)
    head.mark_prefill_completed()
    following = scheduler.schedule()
    assert following.decode_requests == (head,)
    assert following.prefill_requests == (tail,)
    assert following.num_scheduled_tokens == 2


def test_capacity_still_applies_with_spare_tokens():
    """输入无；无返回；构造预算充足但槽位已满的情况，验证两种约束独立生效。"""
    scheduler = Scheduler(max_num_running=1, max_num_batched_tokens=6)
    a, b = Request("A", [1]), Request("B", [1])
    scheduler.add_request(a)
    scheduler.add_request(b)
    plan = scheduler.schedule()
    assert plan.newly_admitted == (a,)
    assert plan.num_scheduled_tokens == 1
    assert scheduler.waiting_requests == (b,)


def test_empty_and_unlimited_budget():
    """输入无；无返回；验证空计划零成本，以及省略预算时保留原有接纳行为。"""
    bounded = Scheduler(max_num_running=2, max_num_batched_tokens=2)
    assert bounded.schedule().num_scheduled_tokens == 0
    unlimited = Scheduler(max_num_seqs=2)
    a, b = Request("A", [1] * 7), Request("B", [1] * 8)
    unlimited.add_request(a)
    unlimited.add_request(b)
    plan = unlimited.schedule()
    assert plan.prefill_requests == (a, b)
    assert plan.num_scheduled_tokens == 15


def test_budgeted_engine_matches_independent_generation():
    """输入无；无返回；运行 CPU 小模型，核对实际输入量、KV、动态到达与参考输出。"""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            config = LlamaConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                max_position_embeddings=128,
                bos_token_id=1,
                eos_token_id=None,
                pad_token_id=0,
            )
            config._attn_implementation = "eager"
            model = LlamaForCausalLM(config).eval()
        runner = DecodeBatchRunner(
            model=model, device=torch.device("cpu"), eos_token_ids=set()
        )
        scheduler = Scheduler(max_num_running=3, max_num_batched_tokens=6)
        engine = Engine(scheduler=scheduler, decode_runner=runner)
        requests = (
            Request("A", [1, 2, 3], max_new_tokens=4),
            Request("B", [1, 4, 5, 6], max_new_tokens=3),
            Request("C", [1, 7], max_new_tokens=2),
        )
        calls = []

        def record_input_tokens(module, args, kwargs):
            """输入 forward hook 参数；无返回；记录实际输入元素数以独立检查预算。"""
            calls.append(kwargs["input_ids"].numel())

        handle = model.register_forward_pre_hook(record_input_tokens, with_kwargs=True)
        try:
            engine.add_request(requests[0])
            traces = []
            for iteration in range(4):
                if iteration == 1:
                    for request in requests[1:]:
                        engine.add_request(request)
                calls.clear()
                output = engine.step()
                assert sum(calls) <= 6
                traces.append(
                    (output.newly_prefilled, tuple(output.decoded_tokens), sum(calls))
                )
                if iteration == 1:
                    assert scheduler.waiting_requests == (requests[2],)
                    assert set(engine.decode_states) == {"A", "B"}
                    assert requests[2].generated_token_ids == []
                for state in engine.decode_states.values():
                    assert (
                        state.request.sequence_length == cache_length(state.cache) + 1
                    )
            assert traces == [
                (("A",), (), 3),
                (("B",), ("A",), 5),
                (("C",), ("A", "B"), 4),
                ((), ("A", "B", "C"), 3),
            ]
            assert not engine.has_unfinished_requests()
            assert engine.decode_states == {}
            assert scheduler.running_requests == ()
        finally:
            handle.remove()

        for request in requests:
            inputs = torch.tensor([request.prompt_token_ids])
            reference = model.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                max_new_tokens=request.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
            assert (
                request.generated_token_ids == reference[0, inputs.shape[1] :].tolist()
            )
    finally:
        torch.set_num_threads(previous_threads)
