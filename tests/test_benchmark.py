import pytest

from miniserve.benchmark import (
    TimingStats,
    tokens_per_second,
)


def test_timing_stats():
    stats = TimingStats(
        [10.0, 20.0, 30.0, 40.0, 50.0]
    )

    assert stats.mean_ms == pytest.approx(30.0)
    assert stats.p50_ms == pytest.approx(30.0)
    assert stats.min_ms == pytest.approx(10.0)
    assert stats.max_ms == pytest.approx(50.0)


def test_tokens_per_second():
    throughput = tokens_per_second(
        num_tokens=100,
        elapsed_ms=2000,
    )

    assert throughput == pytest.approx(50.0)


def test_invalid_percentile():
    stats = TimingStats([1.0])

    with pytest.raises(ValueError):
        stats.percentile(101)