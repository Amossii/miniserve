from __future__ import annotations

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from miniserve.decode_batch import DecodeBatchRunner
from miniserve.engine import Engine
from miniserve.paged_kv import PagedDecodeBatchRunner
from miniserve.request import Request
from miniserve.scheduler import Scheduler


def _tiny_model():
    """输入为空；返回确定性 CPU Llama；模型足够小，用于 dynamic/paged token 对照。"""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(27)
        config = LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=128,
            bos_token_id=1,
            eos_token_id=None,
            pad_token_id=0,
        )
        config._attn_implementation = "eager"
        return LlamaForCausalLM(config).eval()


def _run_engine(runner, specs):
    """输入 runner 与不可变请求规格；返回完成请求；驱动统一 Engine 路径直到完全排空。"""
    engine = Engine(
        scheduler=Scheduler(max_num_running=2, max_num_batched_tokens=8),
        decode_runner=runner,
    )
    requests = [Request(name, list(prompt), max_new_tokens=length) for name, prompt, length in specs]
    for request in requests:
        engine.add_request(request)
    while engine.has_unfinished_requests():
        engine.step()
    assert engine.decode_states == {}
    return requests, engine


def test_paged_engine_matches_dynamic_tokens_and_reclaims_all_blocks():
    """输入相同模型/workload；输出逐 token 一致；Engine 完成后 paged blocks 全部回收。"""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = _tiny_model()
        specs = [("A", [1, 2, 3], 5), ("B", [1, 4, 5, 6], 3), ("C", [1, 7], 4)]
        dynamic_requests, _ = _run_engine(
            DecodeBatchRunner(model=model, device=torch.device("cpu"), eos_token_ids=set()),
            specs,
        )
        paged_runner = PagedDecodeBatchRunner(
            model=model,
            device=torch.device("cpu"),
            eos_token_ids=set(),
            num_blocks=16,
            block_size=2,
        )
        paged_requests, _ = _run_engine(paged_runner, specs)

        assert [r.generated_token_ids for r in paged_requests] == [
            r.generated_token_ids for r in dynamic_requests
        ]
        assert paged_runner.allocator.num_free_blocks == 16
        assert paged_runner.allocator.num_allocated_blocks == 0
    finally:
        torch.set_num_threads(previous_threads)


def test_paged_state_uses_block_storage_and_appends_one_cached_token():
    """输入一次 prefill/decode；输出 table 长度增长一；长期 state 不持有 DynamicCache。"""
    model = _tiny_model()
    runner = PagedDecodeBatchRunner(
        model=model,
        device=torch.device("cpu"),
        eos_token_ids=set(),
        num_blocks=8,
        block_size=2,
    )
    request = Request("A", [1, 2, 3], max_new_tokens=3)
    request.mark_running()

    state = runner.prefill_request(request)
    assert state.block_table.num_tokens == 3
    assert not hasattr(state, "cache")
    gathered = runner.storage.gather_cache(state.block_table, model_config=model.config)
    assert int(gathered.get_seq_length()) == 3

    runner.decode_step([state])
    assert state.block_table.num_tokens == 4
    assert request.sequence_length == 5
    runner.release_state(state)
    assert runner.allocator.num_free_blocks == 8
