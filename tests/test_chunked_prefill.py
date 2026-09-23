from __future__ import annotations

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from miniserve.decode_batch import DecodeBatchRunner
from miniserve.engine import Engine
from miniserve.request import Request
from miniserve.scheduler import Scheduler


def test_scheduler_chunks_oversized_prompt_and_reserves_decode_first():
    """输入短请求和长 prompt；输出逐轮 chunk；验证 decode 优先及每轮预算上限。"""
    scheduler = Scheduler(
        max_num_running=2,
        max_num_batched_tokens=4,
        enable_chunked_prefill=True,
    )
    short = Request("short", [1], max_new_tokens=3)
    long = Request("long", [1] * 9, max_new_tokens=2)
    scheduler.add_request(short)
    scheduler.add_request(long)

    first = scheduler.schedule()
    assert [(c.request.request_id, c.start, c.end) for c in first.prefill_chunks] == [
        ("short", 0, 1),
        ("long", 0, 3),
    ]
    short.mark_prefill_completed()
    long.advance_prefill(3)

    second = scheduler.schedule()
    assert second.decode_requests == (short,)
    assert [(c.start, c.end) for c in second.prefill_chunks] == [(3, 6)]
    assert second.num_scheduled_tokens == 4
    long.advance_prefill(3)

    third = scheduler.schedule()
    assert [(c.start, c.end) for c in third.prefill_chunks] == [(6, 9)]
    assert third.num_scheduled_tokens == 4


def test_chunked_engine_matches_full_hf_generation_for_oversized_prompt():
    """输入超过单轮预算的 prompt；输出与完整 HF greedy 一致；每次 forward 输入不超预算。"""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(28)
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
        engine = Engine(
            scheduler=Scheduler(
                max_num_running=2,
                max_num_batched_tokens=4,
                enable_chunked_prefill=True,
            ),
            decode_runner=runner,
        )
        request = Request("long", [1, 2, 3, 4, 5, 6, 7, 8, 9], max_new_tokens=4)
        calls: list[int] = []

        def record_input_tokens(module, args, kwargs):
            """输入 forward hook；无返回；记录每次真实 input token 数验证 chunk budget。"""
            del module, args
            calls.append(kwargs["input_ids"].numel())

        handle = model.register_forward_pre_hook(record_input_tokens, with_kwargs=True)
        try:
            engine.add_request(request)
            chunks = []
            while engine.has_unfinished_requests():
                output = engine.step()
                chunks.extend(output.prefill_chunks)
        finally:
            handle.remove()

        assert chunks[:3] == [
            ("long", 0, 4),
            ("long", 4, 8),
            ("long", 8, 9),
        ]
        assert max(calls) <= 4
        assert request.num_prefilled_tokens == request.prompt_length
        inputs = torch.tensor([request.prompt_token_ids])
        reference = model.generate(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            max_new_tokens=request.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        assert request.generated_token_ids == reference[0, inputs.shape[1] :].tolist()
    finally:
        torch.set_num_threads(previous_threads)


def test_partial_prefill_produces_no_token_until_final_chunk():
    """输入两段 prompt；中间状态只扩展 KV，最后一段才记录首 token 和 DECODE phase。"""
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=32,
            bos_token_id=1,
            eos_token_id=None,
            pad_token_id=0,
            _attn_implementation="eager",
        )
    ).eval()
    runner = DecodeBatchRunner(
        model=model, device=torch.device("cpu"), eos_token_ids=set()
    )
    request = Request("A", [1, 2, 3, 4, 5], max_new_tokens=2)
    request.mark_running()

    state = runner.prefill_chunk(request, None, start=0, end=3)
    assert request.num_prefilled_tokens == 3
    assert request.generated_token_ids == []
    assert request.needs_prefill

    runner.prefill_chunk(request, state, start=3, end=5)
    assert request.num_prefilled_tokens == 5
    assert request.num_generated_tokens == 1
    assert request.needs_decode
