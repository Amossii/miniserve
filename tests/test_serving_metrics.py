import math

import pytest

from miniserve.benchmark import summarize_serving
from miniserve.request import Request


def completed_request(request_id, arrival, start, tokens, finish=None):
    """输入确定时间线；返回完成请求；只创建本地状态，让公式验证不依赖 sleep。"""
    request = Request(request_id, [1], arrival_time=arrival)
    request.mark_running(timestamp=start)
    for timestamp in tokens:
        request.append_generated_token(2, timestamp=timestamp)
    request.mark_finished(timestamp=tokens[-1] if finish is None else finish)
    return request


def test_request_metrics_and_distinct_population_weights():
    """输入无；无返回；构造课堂例子，验证公式、分位数和请求/间隔的不同权重。"""
    a = completed_request("A", 0.0, 0.1, [0.2, 0.3, 0.5])
    b = completed_request("B", 0.1, 0.2, [0.4, 0.9])
    assert a.queue_wait_seconds == pytest.approx(0.1)
    assert a.ttft_seconds == pytest.approx(0.2)
    assert a.itl_seconds == pytest.approx([0.1, 0.2])
    assert a.tpot_seconds == pytest.approx(0.15)
    assert b.e2e_latency_seconds == pytest.approx(0.8)
    report = summarize_serving([a, b], workload_start=0, workload_end=0.9)
    assert report.num_requests == 2
    assert report.num_output_tokens == 5
    assert report.output_tokens_per_second == pytest.approx(5 / 0.9)
    assert report.requests_per_second == pytest.approx(2 / 0.9)
    assert report.ttft_ms.samples_ms == pytest.approx([200, 300])
    assert report.ttft_ms.p50_ms == pytest.approx(250)
    assert report.ttft_ms.p99_ms == pytest.approx(299)
    assert report.tpot_ms.mean_ms == pytest.approx(325)
    assert report.itl_ms.mean_ms == pytest.approx(800 / 3)
    assert report.itl_ms.p99_ms == pytest.approx(494)
    assert len(report.itl_ms.samples_ms) == 3
    # 已汇总样本不会随 Request 后续被外部修改而变化。
    a.token_timestamps[-1] = 0.6
    assert report.tpot_ms.samples_ms == pytest.approx([150, 500])


def test_zero_and_one_token_metrics():
    """输入无；无返回；验证未生成及单 token 边界，无定义指标不能伪装成零延迟。"""
    waiting = Request("waiting", [1])
    assert waiting.queue_wait_seconds is None
    assert waiting.ttft_seconds is waiting.tpot_seconds is None
    assert waiting.itl_seconds == []
    single = completed_request("single", 0, 0, [0.2])
    assert single.tpot_seconds is None
    report = summarize_serving([single], workload_start=0, workload_end=1)
    assert report.itl_ms is report.tpot_ms is None
    assert report.ttft_ms.mean_ms == pytest.approx(200)
    empty = summarize_serving([], workload_start=0, workload_end=1)
    assert empty.num_output_tokens == empty.num_requests == 0
    assert empty.output_tokens_per_second == empty.requests_per_second == 0
    assert empty.ttft_ms is empty.e2e_ms is empty.queue_wait_ms is None


@pytest.mark.parametrize("bad_time", [0.1, math.nan, math.inf])
def test_bad_token_timestamp_is_atomic(bad_time):
    """输入非法 token 时间；无返回；验证追加失败同时保留 token 与时间列表。"""
    request = Request("A", [1], arrival_time=0)
    request.mark_running(timestamp=0)
    request.append_generated_token(2, timestamp=0.2)
    with pytest.raises(ValueError):
        request.append_generated_token(3, timestamp=bad_time)
    assert request.generated_token_ids == [2]
    assert request.token_timestamps == [0.2]
    assert request.first_token_time == 0.2


def test_equal_timestamps_and_finish_overhead():
    """输入无；无返回；允许时钟分辨率导致相同时间，TPOT 不包含末 token 后清理。"""
    request = completed_request("A", 0, 0, [0.2, 0.2], finish=0.5)
    assert request.tpot_seconds == 0
    assert request.itl_seconds == [0]
    assert request.e2e_latency_seconds == 0.5
    report = summarize_serving([request], workload_start=0, workload_end=1)
    assert report.tpot_ms.mean_ms == 0
    assert report.e2e_ms.mean_ms == 500


@pytest.mark.parametrize("start,end", [(0, 0), (1, 0), (0, math.inf), (math.nan, 1)])
def test_invalid_window(start, end):
    """输入非法窗口；无返回、不修改请求；验证吞吐分母必须有限且为正。"""
    with pytest.raises(ValueError, match="window"):
        summarize_serving([], workload_start=start, workload_end=end)


@pytest.mark.parametrize(
    "field,value",
    [
        ("arrival_time", -0.1),
        ("start_time", 0.4),
        ("finish_time", 2),
        ("finish_time", 0.1),
        ("finish_time", math.nan),
        ("start_time", None),
        ("token_timestamps", []),
        ("token_timestamps", [0.2, 0.1]),
        ("token_timestamps", [0.2]),
        ("first_token_time", 0.9),
    ],
)
def test_reject_corrupt_or_out_of_window_metrics(field, value):
    """输入损坏字段；无返回；修改测试请求，确保汇总拒绝不完整或越界数据。"""
    request = completed_request("A", 0, 0.1, [0.2, 0.3])
    setattr(request, field, value)
    with pytest.raises(ValueError):
        summarize_serving([request], workload_start=0, workload_end=1)


def test_reject_unfinished_and_duplicate_requests():
    """输入无；无返回；验证不能漏算未完成请求，也不能重复计算同一请求。"""
    with pytest.raises(ValueError, match="Incomplete"):
        summarize_serving([Request("A", [1])], workload_start=0, workload_end=1)
    request = completed_request("A", 0, 0, [0.2])
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_serving([request, request], workload_start=0, workload_end=1)
