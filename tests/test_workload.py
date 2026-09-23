from dataclasses import replace
from itertools import pairwise
from types import SimpleNamespace

import pytest

from miniserve.engine import Engine
from miniserve.scheduler import Scheduler
from miniserve.workload import RequestSpec, generate_workload, run_workload


class FakeClock:
    """可推进的单调时钟；测试无需真实 sleep。"""

    def __init__(self):
        """输入无；无返回；初始化时间和 sleep 记录，隔离测试状态。"""
        self.now = 100.0
        self.sleeps = []

    def __call__(self):
        """输入无；返回当前时间；只读，供 Engine/Request/驱动共享。"""
        return self.now

    def sleep(self, seconds):
        """输入等待秒数；无返回；推进模拟时间，用于验证空闲等待。"""
        self.sleeps.append(seconds)
        self.now += seconds


class FakeRunner:
    """每次 forward 消耗 0.1 秒，只模拟 token 与生命周期，真实 Scheduler/Engine 不替换。"""

    def __init__(self, clock):
        """输入共享时钟；无返回；保存依赖，便于确定性模拟阻塞 forward。"""
        self.clock = clock

    def prefill_request(self, request):
        """输入 RUNNING 请求；返回执行状态；推进时间并生成首 token，模拟完整 prefill。"""
        self.clock.now += 0.1
        request.append_generated_token(2, timestamp=self.clock())
        request.mark_prefill_completed()
        if request.reached_max_new_tokens:
            request.mark_finished(timestamp=self.clock())
        return SimpleNamespace(request=request)

    def decode_step(self, states):
        """输入 decode 状态；返回批量输出；推进一次 forward 并按长度结束请求。"""
        self.clock.now += 0.1
        for state in states:
            state.request.append_generated_token(2, timestamp=self.clock())
            if state.request.reached_max_new_tokens:
                state.request.mark_finished(timestamp=self.clock())
        return SimpleNamespace(
            request_ids=tuple(s.request.request_id for s in states),
            token_ids=(2,) * len(states),
        )

    def release_state(self, state):
        """输入完成状态；无返回；fake runner 无 KV 资源，只实现 Engine 回收契约。"""
        del state


@pytest.fixture
def system(monkeypatch):
    """输入时钟补丁；返回 Engine/Clock；让系统各模块共用时间，验证真实事件顺序。"""
    clock = FakeClock()
    monkeypatch.setattr("miniserve.engine.perf_counter", clock)
    monkeypatch.setattr("miniserve.request.perf_counter", clock)
    engine = Engine(
        scheduler=Scheduler(max_num_running=2, max_num_batched_tokens=8),
        decode_runner=FakeRunner(clock),
    )
    return engine, clock


def test_seeded_arrivals_and_independent_content():
    """输入无；无返回；验证可复现指数间隔、固定间隔和跨到达模式相同请求内容。"""
    options = {"num_requests": 8, "max_new_tokens": 4, "seed": 42, "request_rate": 10}
    poisson = generate_workload([[1], [1, 2, 3]], **options)
    assert poisson == generate_workload([[1], [1, 2, 3]], **options)
    assert poisson != generate_workload([[1], [1, 2, 3]], **{**options, "seed": 43})
    assert poisson[0].arrival_offset == 0
    assert all(b.arrival_offset > a.arrival_offset for a, b in pairwise(poisson))
    constant = generate_workload([[1], [1, 2, 3]], arrival="constant", **options)
    burst = generate_workload([[1], [1, 2, 3]], arrival="burst", **options)
    assert [s.arrival_offset for s in constant] == pytest.approx(
        [i / 10 for i in range(8)]
    )
    assert all(s.arrival_offset == 0 for s in burst)
    assert [(s.prompt_token_ids, s.max_new_tokens) for s in poisson] == [
        (s.prompt_token_ids, s.max_new_tokens) for s in constant
    ]


def test_catch_up_all_due_requests_without_hiding_delay(system):
    """输入系统；无返回；模拟 forward 期间两次到达，验证补交及两种 TTFT 的差额。"""
    engine, clock = system
    specs = (
        RequestSpec("A", 0, (1,), 2),
        RequestSpec("B", 0.05, (1,), 1),
        RequestSpec("C", 0.06, (1,), 1),
    )
    result = run_workload(engine, specs, clock=clock, sleep=clock.sleep)
    assert clock.sleeps == []
    assert [r.arrival_time - result.started for r in result.requests] == pytest.approx(
        [0, 0.1, 0.1]
    )
    assert result.dispatch_lag_ms.samples_ms == pytest.approx([0, 50, 40])
    for scheduled, submitted, lag in zip(
        result.scheduled_metrics.ttft_ms.samples_ms,
        result.submitted_metrics.ttft_ms.samples_ms,
        result.dispatch_lag_ms.samples_ms,
        strict=True,
    ):
        assert scheduled == pytest.approx(submitted + lag)
    assert result.iterations == 3
    assert result.submitted_metrics.num_output_tokens == 4
    assert result.scheduled_metrics.output_tokens_per_second == pytest.approx(10)
    assert engine.decode_states == {}


def test_idle_gap_and_no_request_state_reuse(system):
    """输入系统；无返回；验证中途完全空闲仍等后续请求，规格不会被运行修改。"""
    engine, clock = system
    specs = (RequestSpec("A", 0, (1,), 1), RequestSpec("B", 0.5, (1,), 1))
    result = run_workload(engine, specs, clock=clock, sleep=clock.sleep)
    assert sum(clock.sleeps) == pytest.approx(0.4)
    assert result.ended - result.started == pytest.approx(0.6)
    assert result.iterations == 2
    assert specs[1].arrival_offset == 0.5
    with pytest.raises(ValueError, match="fresh Engine"):
        run_workload(engine, specs, clock=clock, sleep=clock.sleep)
    other = Engine(
        scheduler=Scheduler(max_num_running=1, max_num_batched_tokens=8),
        decode_runner=FakeRunner(clock),
    )
    again = run_workload(other, specs, clock=clock, sleep=clock.sleep)
    assert again.requests[0] is not result.requests[0]
    assert (
        again.requests[0].generated_token_ids == result.requests[0].generated_token_ids
    )


@pytest.mark.parametrize(
    "change",
    [
        {"arrival_offset": -1},
        {"arrival_offset": float("nan")},
        {"prompt_token_ids": ()},
        {"prompt_token_ids": (1,) * 9},
        {"max_new_tokens": 0},
    ],
)
def test_validation_before_execution(system, change):
    """输入坏规格；无返回；验证先全量校验，后续坏请求不会导致前面请求执行一半。"""
    engine, clock = system
    first = RequestSpec("A", 0, (1,), 1)
    second = replace(RequestSpec("B", 0.2, (1,), 1), **change)
    with pytest.raises(ValueError):
        run_workload(engine, (first, second), clock=clock, sleep=clock.sleep)
    assert clock.now == 100
    assert not engine.has_unfinished_requests()


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf")])
def test_invalid_request_rate(rate):
    """输入非法请求率；无返回、不生成计划；拒绝无法解释的到达时间。"""
    with pytest.raises(ValueError):
        generate_workload([[1]], num_requests=2, max_new_tokens=1, request_rate=rate)


def test_duplicate_and_out_of_order_arrivals(system):
    """输入系统；无返回；验证 ID 唯一与时间排序，防止补交循环遗漏请求。"""
    engine, clock = system
    spec = RequestSpec("A", 0.2, (1,), 1)
    for specs in [(), (spec, spec), (spec, RequestSpec("B", 0.1, (1,), 1))]:
        with pytest.raises(ValueError):
            run_workload(engine, specs, clock=clock, sleep=clock.sleep)
