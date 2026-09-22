from __future__ import annotations

import pytest

from miniserve.benchmark_report import render_markdown, summarize_payload


def _payload():
    """输入为空；返回两次重复的最小报告；用极端样本验证重新聚合分位数。"""
    runs = []
    for repeat, throughput, samples, peak in [
        (0, 10.0, [1.0, 2.0], 100),
        (1, 30.0, [3.0, 100.0], 200),
    ]:
        runs.append(
            {
                "policy": "continuous",
                "max_running": 2,
                "token_budget": 8,
                "repeat": repeat,
                "peak_allocated_bytes": peak,
                "scheduled_metrics": {
                    "output_tokens_per_second": throughput,
                    "ttft_ms": {"samples_ms": samples, "p99": -1},
                    "itl_ms": {"samples_ms": [repeat + 1.0], "p99": -1},
                },
            }
        )
    return {
        "config": {
            "arrival": "burst",
            "request_rate": 50,
            "num_requests": 4,
            "max_new_tokens": 2,
            "seed": 22,
        },
        "environment": {
            "model": "tiny",
            "device": "cuda",
            "cuda_device": "fake-gpu",
            "dtype": "float16",
        },
        "runs": runs,
    }


def test_summary_pools_raw_latency_samples_and_keeps_run_throughput():
    """输入两次重复；输出应重算原始样本分位数并保留吞吐重复，而非平均 P99。"""
    summary = summarize_payload(_payload())[0]

    assert summary.repeats == 2
    assert summary.output_tokens_per_second == [10.0, 30.0]
    assert summary.throughput_median == 20.0
    assert summary.ttft_ms.p50_ms == 2.5
    assert summary.ttft_ms.p99_ms == pytest.approx(97.09)
    assert summary.peak_allocated_bytes == 200


def test_markdown_is_traceable_and_labels_measurement_limits():
    """输入有效报告；输出包含原始来源、策略和限制，便于 reviewer 回查结论。"""
    report = render_markdown(_payload(), "raw.json")

    assert "Raw data: `raw.json`" in report
    assert "| continuous | 2 | 8 | 2 | 20.00 [10.00, 30.00]" in report
    assert "pooled raw samples" in report
    assert "offered request rate" in report


def test_summary_rejects_preaggregated_latency_without_raw_samples():
    """输入仅有 P99 的报告；输出为异常，防止再次聚合已聚合的统计量。"""
    payload = _payload()
    payload["runs"][0]["scheduled_metrics"]["ttft_ms"] = {"p99": 9.0}

    with pytest.raises(ValueError, match="samples_ms"):
        summarize_payload(payload)
