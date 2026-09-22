import torch
from transformers import LlamaConfig, LlamaForCausalLM

from miniserve.decode_batch import DecodeBatchRunner
from miniserve.engine import Engine
from miniserve.request import Request
from miniserve.scheduler import Scheduler


def test_torch_profiler_contains_miniserve_phase_annotations():
    """输入无；无返回；执行真实小模型，验证 Engine 与 KV 子阶段出现在 operator trace。"""
    config = LlamaConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        eos_token_id=None,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).eval()
    runner = DecodeBatchRunner(
        model=model,
        device=torch.device("cpu"),
        eos_token_ids=set(),
        annotate_profiler=True,
    )
    engine = Engine(
        scheduler=Scheduler(max_num_running=2),
        decode_runner=runner,
        annotate_profiler=True,
    )
    engine.add_request(Request("A", [1, 2], max_new_tokens=2))
    engine.step()
    engine.add_request(Request("B", [1, 3, 4], max_new_tokens=2))
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profile:
        engine.step()
    names = {event.key for event in profile.key_averages()}
    assert {
        "miniserve::scheduler",
        "miniserve::prefill",
        "miniserve::prefill_model_forward",
        "miniserve::decode",
        "miniserve::kv_pack",
        "miniserve::decode_model_forward",
        "miniserve::kv_unpack",
        "miniserve::reclaim",
    } <= names
