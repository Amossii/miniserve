from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from miniserve.decode_batch import DecodeBatchRunner
from miniserve.engine import Engine
from miniserve.paged_kv import PagedDecodeBatchRunner
from miniserve.request import ExecutionPhase, Request
from miniserve.scheduler import Scheduler


def _system():
    """输入为空；返回小型 paged Engine；容量刻意有限以稳定制造 KV pressure。"""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(29)
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
    runner = PagedDecodeBatchRunner(
        model=model,
        device=torch.device("cpu"),
        eos_token_ids=set(),
        num_blocks=8,
        block_size=2,
    )
    engine = Engine(
        scheduler=Scheduler(
            max_num_running=2,
            max_num_batched_tokens=16,
            max_preemptions=2,
        ),
        decode_runner=runner,
    )
    return model, runner, engine


def test_pressure_preempts_lifo_recomputes_and_preserves_correctness():
    """输入有限 blocks 和三请求；输出 LIFO victim；释放、队尾等待、重建后结果仍等于 HF。"""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model, runner, engine = _system()
        a = Request("A", [1, 2, 3], max_new_tokens=5)
        b = Request("B", [1, 4, 5], max_new_tokens=4)
        c = Request("C", [1, 6, 7], max_new_tokens=1)
        engine.add_request(a)
        engine.add_request(b)
        engine.step()  # A/B prefill，各持有两个 blocks。
        first_token = list(b.generated_token_ids)
        first_timestamp = b.first_token_time
        engine.add_request(c)

        victims = engine.preempt_until_free(6)
        assert victims == ("B",)
        assert b.phase is ExecutionPhase.RECOMPUTE
        assert b.num_preemptions == 1
        assert b.generated_token_ids == first_token
        assert b.first_token_time == first_timestamp
        assert engine.scheduler.waiting_requests == (c, b)
        assert runner.allocator.num_free_blocks == 6

        saw_recompute = False
        while engine.has_unfinished_requests():
            engine.step()
            if b.is_running and not b.recompute_required and not b.is_finished:
                saw_recompute = True
        assert saw_recompute
        assert b.num_recomputed_tokens == 3
        assert runner.allocator.num_free_blocks == runner.allocator.num_blocks

        for request in (a, b, c):
            inputs = torch.tensor([request.prompt_token_ids])
            reference = model.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                max_new_tokens=request.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
            assert request.generated_token_ids == reference[
                0, inputs.shape[1] :
            ].tolist()
    finally:
        torch.set_num_threads(previous_threads)


def test_preemption_limit_rejects_repeated_victim():
    """输入达到抢占次数上限的请求；输出明确异常，避免无限抢占循环。"""
    _, _, engine = _system()
    request = Request("A", [1, 2], max_new_tokens=3)
    engine.add_request(request)
    engine.step()
    engine.scheduler.max_preemptions = 1
    assert engine.preempt_one() == "A"
    # 重新 admission/recompute 后达到限制，不能再次成为 victim。
    engine.step()
    with pytest.raises(RuntimeError, match="No eligible"):
        engine.preempt_one()


def test_dynamic_backend_rejects_kv_preemption():
    """输入 dynamic KV Engine；输出明确异常；当前 preemption 只属于 paged backend。"""
    model, _, _ = _system()
    engine = Engine(
        scheduler=Scheduler(max_num_running=1, max_num_batched_tokens=8),
        decode_runner=DecodeBatchRunner(
            model=model,
            device=torch.device("cpu"),
            eos_token_ids=set(),
        ),
    )
    engine.add_request(Request("A", [1, 2], max_new_tokens=2))
    engine.step()
    with pytest.raises(RuntimeError, match="paged runner"):
        engine.preempt_one()
