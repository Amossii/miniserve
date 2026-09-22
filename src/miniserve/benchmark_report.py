"""聚合 Step 22 benchmark JSON，并生成可追溯的 Markdown 表格。"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

from miniserve.benchmark import TimingStats


@dataclass(frozen=True)
class PolicySummary:
    """一个调度配置的跨重复摘要；样本来自原始 JSON，不保存或修改引擎状态。"""

    policy: str
    max_running: int
    token_budget: int
    repeats: int
    output_tokens_per_second: list[float]
    ttft_ms: TimingStats
    itl_ms: TimingStats | None
    peak_allocated_bytes: int | None

    @property
    def throughput_median(self) -> float:
        """输入当前摘要；返回各次完整 workload 吞吐的中位数；不合并不同时间窗口。"""
        return statistics.median(self.output_tokens_per_second)


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    """输入任意 JSON 值与字段名；返回字典；无内部状态，尽早拒绝损坏报告。"""
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object")
    return value


def _raw_samples(run: dict[str, Any], metric: str) -> list[float]:
    """输入单次运行与指标名；返回原始毫秒样本；拒绝缺失样本以免聚合已有 P99。"""
    scheduled = _require_mapping(run.get("scheduled_metrics"), "scheduled_metrics")
    value = _require_mapping(scheduled.get(metric), metric)
    samples = value.get("samples_ms")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{metric}.samples_ms must be a non-empty list")
    return [float(sample) for sample in samples]


def summarize_payload(payload: dict[str, Any]) -> list[PolicySummary]:
    """输入一份 benchmark payload；返回按策略聚合结果；只读并合并原始延迟样本。

    吞吐是每个完整 workload 的一个观测，所以保留重复样本并报告中位数。
    TTFT/ITL 则从所有重复的原始 request/token 样本重新计算 P50/P99。
    """
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("runs must be a non-empty list")

    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for raw_run in runs:
        run = _require_mapping(raw_run, "run")
        capacity = int(run["max_running"])
        policy = str(run.get("policy", "sequential" if capacity == 1 else "continuous"))
        key = (policy, capacity, int(run["token_budget"]))
        grouped.setdefault(key, []).append(run)

    summaries: list[PolicySummary] = []
    for (policy, capacity, budget), group in sorted(grouped.items()):
        throughputs: list[float] = []
        ttft_samples: list[float] = []
        itl_samples: list[float] = []
        peaks: list[int] = []
        for run in group:
            scheduled = _require_mapping(run["scheduled_metrics"], "scheduled_metrics")
            throughputs.append(float(scheduled["output_tokens_per_second"]))
            ttft_samples.extend(_raw_samples(run, "ttft_ms"))
            itl = scheduled.get("itl_ms")
            if isinstance(itl, dict):
                itl_samples.extend(float(value) for value in itl.get("samples_ms", []))
            peak = run.get("peak_allocated_bytes")
            if peak is not None:
                peaks.append(int(peak))
        summaries.append(
            PolicySummary(
                policy=policy,
                max_running=capacity,
                token_budget=budget,
                repeats=len(group),
                output_tokens_per_second=throughputs,
                ttft_ms=TimingStats(ttft_samples),
                itl_ms=TimingStats(itl_samples) if itl_samples else None,
                peak_allocated_bytes=max(peaks) if peaks else None,
            )
        )
    return summaries


def render_markdown(payload: dict[str, Any], source_name: str) -> str:
    """输入 benchmark payload 和来源名；返回 Markdown 报告；内容引用来源和环境元数据。"""
    environment = _require_mapping(payload.get("environment"), "environment")
    config = _require_mapping(payload.get("config"), "config")
    summaries = summarize_payload(payload)
    lines = [
        "# MiniServe Benchmark Report",
        "",
        f"Raw data: `{source_name}`",
        "",
        "## Experiment contract",
        "",
        f"- Model: `{environment.get('model')}`",
        f"- Device: `{environment.get('device')}` / `{environment.get('cuda_device')}`",
        f"- dtype: `{environment.get('dtype')}`",
        f"- Arrival: `{config.get('arrival')}`, request rate: `{config.get('request_rate')}` requests/s",
        f"- Requests: `{config.get('num_requests')}`, max new tokens: `{config.get('max_new_tokens')}`, seed: `{config.get('seed')}`",
        "",
        "## Results",
        "",
        "| Policy | Capacity | Token budget | Repeats | Output tok/s median [min, max] | TTFT P50 / P99 (ms) | ITL P50 / P99 (ms) | Peak allocated (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        throughput = summary.output_tokens_per_second
        itl = (
            f"{summary.itl_ms.p50_ms:.2f} / {summary.itl_ms.p99_ms:.2f}"
            if summary.itl_ms
            else "N/A"
        )
        peak = (
            f"{summary.peak_allocated_bytes / 2**20:.1f}"
            if summary.peak_allocated_bytes is not None
            else "N/A"
        )
        lines.append(
            f"| {summary.policy} | {summary.max_running} | {summary.token_budget} | "
            f"{summary.repeats} | {summary.throughput_median:.2f} "
            f"[{min(throughput):.2f}, {max(throughput):.2f}] | "
            f"{summary.ttft_ms.p50_ms:.2f} / {summary.ttft_ms.p99_ms:.2f} | "
            f"{itl} | {peak} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- Latency percentiles are recomputed from pooled raw samples; per-run P99 values are not averaged.",
            "- Throughput uses the median of complete-workload observations and keeps the observed range visible.",
            "- A low offered request rate can cap measured throughput before the engine reaches saturation.",
            "- Peak allocated memory includes model residency because the model remains loaded during measurement.",
            "- This synchronous in-process benchmark excludes network and tokenizer latency.",
            "",
        ]
    )
    return "\n".join(lines)
