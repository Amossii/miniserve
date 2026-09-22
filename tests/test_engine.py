import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from miniserve.decode_batch import DecodeBatchRunner, cache_length
from miniserve.engine import Engine
from miniserve.request import Request
from miniserve.scheduler import Scheduler


@pytest.fixture
def engine():
    """输入无；返回 CPU Engine；隔离随机数和线程设置，用真实小模型验证 KV。"""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(16)
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
        yield Engine(scheduler=Scheduler(max_num_running=2), decode_runner=runner)
    finally:
        torch.set_num_threads(previous_threads)


def test_mixed_iteration_and_reference(engine):
    """输入 Engine；无返回；动态加入 B，检查 forward 分组、KV 与独立生成一致性。"""
    a = Request("A", [1, 4, 5], max_new_tokens=4)
    b = Request("B", [1, 6, 7, 8, 9], max_new_tokens=3)
    engine.add_request(a)
    first = engine.step()
    assert first.newly_prefilled == ("A",)
    assert first.decoded_tokens == {}
    assert a.num_generated_tokens == 1
    assert cache_length(engine.decode_states["A"].cache) == 3

    calls = []

    def record_forward(module, args, kwargs):
        """输入 hook 参数；无返回；只记录输入 shape，验证实际模型调用边界。"""
        calls.append(tuple(kwargs["input_ids"].shape))

    handle = engine.decode_runner.model.register_forward_pre_hook(
        record_forward, with_kwargs=True
    )
    try:
        engine.add_request(b)
        second = engine.step()
        assert second.newly_prefilled == ("B",)
        assert tuple(second.decoded_tokens) == ("A",)
        assert calls == [(1, 5), (1, 1)]
        assert (a.num_generated_tokens, b.num_generated_tokens) == (2, 1)
        for state in engine.decode_states.values():
            assert state.request.sequence_length == cache_length(state.cache) + 1

        calls.clear()
        third = engine.step()
        assert third.newly_prefilled == ()
        assert tuple(third.decoded_tokens) == ("A", "B")
        assert calls == [(2, 1)]
        for state in engine.decode_states.values():
            assert state.request.sequence_length == cache_length(state.cache) + 1

        final = engine.step()
        assert final.finished_requests == ("A", "B")
        assert final.running_requests == ()
        assert engine.decode_states == {}
        assert not engine.has_unfinished_requests()
    finally:
        handle.remove()

    for request in (a, b):
        inputs = torch.tensor([request.prompt_token_ids])
        reference = engine.decode_runner.model.generate(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            max_new_tokens=request.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        assert request.generated_token_ids == reference[0, inputs.shape[1] :].tolist()


def test_capacity_reuse_and_prefill_finish(engine):
    """输入 Engine；无返回；让 B 首 token 结束，验证 C 下一轮复用槽位且无多余 decode。"""
    a = Request("A", [1, 2], max_new_tokens=3)
    b = Request("B", [1, 3], max_new_tokens=1)
    c = Request("C", [1, 4], max_new_tokens=1)
    for request in (a, b, c):
        engine.add_request(request)
    first = engine.step()
    assert first.finished_requests == ("B",)
    assert first.running_requests == ("A",)
    assert engine.scheduler.waiting_requests == (c,)
    assert set(engine.decode_states) == {"A"}
    second = engine.step()
    assert second.newly_prefilled == ("C",)
    assert tuple(second.decoded_tokens) == ("A",)
    assert second.finished_requests == ("C",)
    assert b.num_generated_tokens == c.num_generated_tokens == 1
    engine.step()
    assert engine.scheduler.finished_requests == (b, c, a)
    assert not engine.has_unfinished_requests()
    assert engine.decode_states == {}


@pytest.mark.parametrize("stop_at", [0, 1])
def test_eos_cleanup(engine, stop_at):
    """输入 Engine 与 EOS 位置；无返回；用参考输出指定 EOS，检查两阶段停止和 KV 回收。"""
    inputs = torch.tensor([[1, 2, 3]])
    reference = engine.decode_runner.model.generate(
        input_ids=inputs,
        attention_mask=torch.ones_like(inputs),
        max_new_tokens=2,
        do_sample=False,
    )[0, 3:].tolist()
    request = Request("EOS", [1, 2, 3], max_new_tokens=8)
    engine.add_request(request)
    if stop_at == 1:
        engine.step()
    engine.decode_runner.eos_token_ids = {reference[stop_at]}
    output = engine.step()
    assert output.finished_requests == ("EOS",)
    assert request.generated_token_ids == reference[: stop_at + 1]
    assert engine.decode_states == {}
    assert not engine.has_unfinished_requests()


def test_empty_step_and_missing_cache(engine):
    """输入 Engine；无返回；验证空轮安全，以及 KV 缺失时明确报错而非静默停滞。"""
    output = engine.step()
    assert output.newly_prefilled == output.running_requests == ()
    assert output.decoded_tokens == {}
    assert output.finished_requests == ()
    engine.add_request(Request("A", [1, 2], max_new_tokens=3))
    engine.step()
    engine.decode_states.clear()
    with pytest.raises(RuntimeError, match="Missing decode state: A"):
        engine.step()


def test_submission_time_and_failed_resubmission(engine, monkeypatch):
    """输入 Engine/时钟补丁；无返回；验证入队计时覆盖创建时间，失败提交不改时间。"""
    monkeypatch.setattr("miniserve.engine.perf_counter", lambda: 10.0)
    request = Request("A", [1], arrival_time=1.0)
    engine.add_request(request)
    assert request.arrival_time == 10.0
    monkeypatch.setattr("miniserve.engine.perf_counter", lambda: 20.0)
    with pytest.raises(ValueError):
        engine.add_request(request)
    assert request.arrival_time == 10.0


def test_real_generation_to_serving_report(engine):
    """输入 CPU Engine；无返回；执行真实生成并汇总，覆盖单 token 请求的缺省 TPOT。"""
    from time import perf_counter

    from miniserve.benchmark import summarize_serving

    requests = [
        Request("A", [1, 2], max_new_tokens=3),
        Request("B", [1, 3, 4], max_new_tokens=1),
    ]
    started = perf_counter()
    for request in requests:
        engine.add_request(request)
    for _ in range(3):
        engine.step()
    assert not engine.has_unfinished_requests()
    report = summarize_serving(
        requests, workload_start=started, workload_end=perf_counter()
    )
    assert report.num_requests == 2
    assert report.num_output_tokens == 4
    assert len(report.ttft_ms.samples_ms) == 2
    assert len(report.tpot_ms.samples_ms) == 1
    assert len(report.itl_ms.samples_ms) == 2
    assert all(r.finish_time == r.token_timestamps[-1] for r in requests)


def test_engine_metrics_and_decode_timestamp_after_readback(engine, monkeypatch):
    """输入真实 CPU Engine；无返回；检查 decode 先读回再计时，并汇总完整时间线。"""
    from time import perf_counter
    from types import SimpleNamespace

    from miniserve.benchmark import summarize_serving

    started = perf_counter()
    requests = [
        Request("A", [1, 2], max_new_tokens=2),
        Request("B", [1, 3, 4], max_new_tokens=2),
    ]
    for request in requests:
        engine.add_request(request)
    engine.step()
    events = []

    def readback():
        """输入无；返回假采样 token；记录读回事件，独立检测计时的先后顺序。"""
        events.append("readback")
        return [5, 6]

    def clock():
        """输入无；返回真实时间；记录时钟事件，验证 CPU 可用边界。"""
        events.append("timestamp")
        return perf_counter()

    monkeypatch.setattr(
        torch, "argmax", lambda *args, **kwargs: SimpleNamespace(tolist=readback)
    )
    monkeypatch.setattr(
        "miniserve.decode_batch.time", SimpleNamespace(perf_counter=clock)
    )
    engine.step()
    assert events == ["readback", "timestamp"]
    assert requests[0].token_timestamps[-1] == requests[1].token_timestamps[-1]
    report = summarize_serving(
        requests, workload_start=started, workload_end=perf_counter()
    )
    assert report.num_output_tokens == 4
    assert len(report.itl_ms.samples_ms) == 2
    assert all(len(r.token_timestamps) == r.num_generated_tokens for r in requests)
