from types import SimpleNamespace

from miniserve.engine import Engine
from miniserve.profiling import EngineProfiler
from miniserve.request import Request
from miniserve.scheduler import Scheduler


class Runner:
    """最小 runner；记录 phase 输入并推进 Request，隔离 profiling 测试与 Transformers。"""

    def prefill_request(self, request):
        """输入请求；返回假的 DecodeState；生成首 token 后进入 decode。"""
        request.append_generated_token(2)
        request.mark_prefill_completed()
        return SimpleNamespace(request=request)

    def decode_step(self, states):
        """输入状态列表；返回假的 batch 输出；每个请求追加一个 token。"""
        for state in states:
            state.request.append_generated_token(2)
            if state.request.reached_max_new_tokens:
                state.request.mark_finished()
        return SimpleNamespace(
            request_ids=tuple(s.request.request_id for s in states),
            token_ids=(2,) * len(states),
        )

    def release_state(self, state):
        """输入完成状态；无返回；fake state 不持有资源，满足 Engine runner 生命周期接口。"""
        del state


def test_engine_phase_profile_captures_work_and_sums_phases():
    """输入带 profiler 的 Engine；无返回；验证两轮 phase、工作量和累计记录。"""
    profiler = EngineProfiler()
    engine = Engine(
        scheduler=Scheduler(max_num_running=2, max_num_batched_tokens=8),
        decode_runner=Runner(),
        profiler=profiler,
    )
    request = Request("A", [1, 2, 3], max_new_tokens=2)
    engine.add_request(request)
    first = engine.step()
    assert first.newly_prefilled == ("A",)
    profile = engine.last_profile
    assert profile is not None
    assert profile.iteration == 0
    assert profile.num_prefill == 1
    assert profile.num_decode == 0
    assert profile.prefill_tokens == 3
    assert profile.decode_tokens == 0
    assert profile.scheduled_tokens == 3
    assert profile.total_seconds >= profile.scheduler_seconds
    assert profile.total_seconds >= profile.prefill_seconds
    second = engine.step()
    assert second.finished_requests == ("A",)
    assert len(profiler.records) == 2
    assert profiler.records[1].iteration == 1
    assert profiler.records[1].num_decode == 1
    assert profiler.records[1].decode_tokens == 1
    assert profiler.records[1].prefill_tokens == 0
    assert profiler.records[1].scheduled_tokens == 1


def test_profiling_disabled_keeps_no_records():
    """输入未配置 profiler 的 Engine；无返回；验证默认路径不创建 profile 记录。"""
    engine = Engine(
        scheduler=Scheduler(max_num_running=1),
        decode_runner=Runner(),
    )
    engine.add_request(Request("A", [1], max_new_tokens=1))
    engine.step()
    assert engine.last_profile is None
    assert engine.profiler is None
