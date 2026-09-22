# Step 18：Engine 请求级 Metrics

## 测量边界

本课测量进程内 Engine 请求：从 `Engine.add_request()` 调用入口，到 runner 将输出 token 读回 CPU。它不包含网络、客户端渲染，也不等于 `step()` 返回或流式发送时间。一次 batched decode 中全部输出读回后，共享一个可用时间戳。

`Engine.add_request()` 成功后覆盖 `Request.arrival_time`，避免提前创建对象污染排队时间。失败提交不改时间。直接使用 Scheduler 或 StaticBatchRunner 时，调用方必须自行在真实提交点设置 arrival_time。

所有时间采用同一个 `perf_counter()` 时钟。`mark_running(timestamp=...)` 和 token/finish 的 timestamp 参数用于确定性测试；不要将墙上时钟与单调时钟混用。

## 数据与公式

Request 保存 `arrival_time`、`start_time`、`token_timestamps`、`finish_time`。通过 `append_generated_token()` 同步追加输出 ID 与时间，时间必须有限且不倒退；相等时间合法。首 token 之后的输出不能覆盖 first_token_time。

| 指标 | 公式 | 无样本时 |
|---|---|---|
| queue_wait_seconds | start - arrival | 未接纳为 None |
| ttft_seconds | first token - arrival | 未输出为 None |
| itl_seconds | 每对相邻输出时间差 | 少于两个 token 为 [] |
| tpot_seconds | (last token - first token) / (N - 1) | 少于两个 token 为 None |
| e2e_latency_seconds | finish - arrival | 未结束为 None |

TPOT 排除最后一个 token 之后的完成开销；因此不直接用 `(E2E - TTFT)/(N-1)` 代替。输出计数包括实际保存的 EOS token，不包括 prompt 和 padding。

## GPU 时间戳

Prefill 在 `.item()` 取得首 token 后记录时间。Continuous decode 和 static batch 先用 `.tolist()` 取得整批 CPU token，再记录统一时间，避免在 CUDA 异步工作尚未产生可用结果时提前计时。不为指标在每个算子前后额外插入全设备 synchronize。

这是应用可见 token 时间，不是 kernel 时间；GPU kernel profiling 属于后续课程。

## 汇总 API

已有 Engine、runner 与 requests 时可以这样使用：

```python
from time import perf_counter
from miniserve.benchmark import summarize_serving

# 模型加载和 warmup 在此之前完成；使用新的请求和干净的 Engine。
workload_start = perf_counter()
for request in requests:
    engine.add_request(request)
while engine.has_unfinished_requests():
    engine.step()
workload_end = perf_counter()

report = summarize_serving(
    requests,
    workload_start=workload_start,
    workload_end=workload_end,
)
print(report.output_tokens_per_second)
if report.ttft_ms is not None:
    print(report.ttft_ms.p50_ms, report.ttft_ms.p99_ms)
```

此示例是一次性提交；后续 workload generator 将控制请求到达。汇总函数本身不执行模型。

吞吐分母是整个窗口的墙钟时间，包含空闲到达间隔、排队和执行，不能使用请求延迟之和。调用方必须传入窗口内全部请求，不能只筛选快请求；函数无法识别被调用方遗漏的请求。

该 API 仅支持完整排空 workload。它拒绝重复 ID、未完成或缺少 token 时间的请求，以及不满足以下顺序的数据：

```text
window_start <= arrival <= start <= token times <= finish <= window_end
```

空 workload 在正时间窗口中吞吐为 0，所有延迟分布为 None。单 token 请求有 TTFT/E2E，但不贡献 ITL/TPOT 样本。尚不支持取消、失败请求或稳态截断窗口的 goodput 统计。

## 统计口径

汇总延迟单位为毫秒，返回现有 `TimingStats`，支持 mean/min/max/P50/P95/P99。没有样本时返回 None。每个分布的样本数为 `len(stats.samples_ms)`。

- queue wait、TTFT、TPOT、E2E：每个有定义的请求贡献一个样本。
- ITL：汇集所有输出间隔；长输出请求贡献更多样本。
- 分位数对排序样本进行线性插值，位置为 `(N-1)*p/100`。小样本 P99 仅用于验证统计计算，不是性能结论。

课堂例子：A 在 0 到达、0.1 接纳、于 0.2/0.3/0.5 输出；B 在 0.1 到达、0.2 接纳、于 0.4/0.9 输出；两者均在最后输出时结束。

窗口 [0, 0.9] 的输出吞吐为 5/0.9 tokens/s。请求级 TPOT 均值为 325 ms，汇集 ITL 的均值为 800/3 ms，两者权重不同。这是人工时间线，不是真实性能数据。

## 验证与限制

```bash
.venv/bin/python -m pytest tests/test_serving_metrics.py tests/test_engine.py tests/test_static_batch.py -q
```

测试覆盖确定性公式、空/单 token 边界、损坏时间线、时间戳追加的原子性、Engine 提交边界，以及两个 batch runner 先读回再计时的顺序。CPU 小型 Llama 验证了真实 Engine 到报告的路径；没有据此得出 GPU 性能结论。

保存所有 token 时间戳的内存开销随输出 token 数增长，这是 Phase A 为可解释的 ITL 分布采用的取舍。

参考：[vLLM metrics design](https://docs.vllm.ai/en/latest/design/metrics/)。跨框架比较前必须对齐事件边界、TPOT 定义和样本单位。
