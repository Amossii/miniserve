# Step 22：GPU Benchmark Matrix 与性能分析

## 1. 本节目标

本课把 Step 19 的单次 benchmark 扩展成可复核的实验链路：固定实验条件，遍历调度策略，保留每次原始样本，记录 CUDA 峰值显存，再从 JSON 生成 Markdown 报告。

```text
固定 workload + 环境元数据
        ↓
sequential / continuous 策略矩阵
        ↓
每个配置独立预热与重复测量
        ↓
schema v2 原始 JSON
        ↓
重新聚合原始 TTFT / ITL 样本
        ↓
可追溯 Markdown 表格
```

## 2. 为什么需要

单次 tokens/s 不能回答调度策略是否稳定，也不能显示吞吐与尾延迟的交换关系。低请求率实验还可能由输入负载限制：引擎处理完一个请求后等待下一次到达，此时吞吐接近 offered load，不代表 GPU 已经饱和。

报告工具遵守两个统计边界：

- 吞吐是一次完整 workload 的属性，跨重复报告中位数和 `[min, max]`，保留波动。
- TTFT 和 ITL 是请求或 token 间隔样本，跨重复合并原始 `samples_ms` 后重新计算 P50/P99，禁止平均各次 P99。

## 3. 策略边界

`max_running=1` 标为 `sequential`；更大的容量标为 `continuous`。两者使用同一个 Engine、相同到达计划、指标窗口和生成语义，因此可以直接比较。

Step 11 的 `StaticBatchRunner` 只接收一个同时到达的固定 batch，没有 waiting queue，也没有按到达时间提交请求。把它直接放入 serving 表会改变 TTFT 和吞吐的测量边界。本课保留 static batch 作为 kernel/executor baseline；需要公平比较时，应先为它定义相同的到达与组批策略。

## 4. 运行 GPU 实验矩阵

示例使用本地模型，固定模型、GPU、dtype、请求内容、到达模式与种子，只改变容量和 token budget：

```bash
.venv/bin/python scripts/benchmark_serving.py \
  --model /path/to/model \
  --device cuda \
  --arrival burst \
  --num-requests 64 \
  --max-new-tokens 64 \
  --max-running 1 2 4 8 \
  --token-budgets 128 256 512 \
  --repeats 5 \
  --seed 22 \
  --check-reference \
  --output benchmarks/results/step22_gpu_burst.json
```

随后生成报告：

```bash
.venv/bin/python scripts/analyze_benchmark.py \
  benchmarks/results/step22_gpu_burst.json \
  --output benchmarks/reports/step22_gpu_burst.md
```

对负载曲线，应另外固定策略，分别运行多个 `--request-rate` 文件。低速区观察 latency 和 dispatch lag；逐步加压，直到吞吐不再随 offered load 增长且排队延迟上升。不同请求率使用单独 JSON，避免把不同 workload 聚合成一个策略重复。

## 5. 峰值显存口径

每次正式 workload 前调用 `torch.cuda.reset_peak_memory_stats()`，运行结束后同步并记录：

- `peak_allocated_bytes`：PyTorch tensor allocator 的峰值已分配显存。
- `peak_reserved_bytes`：PyTorch caching allocator 的峰值保留显存。

模型在重置统计前已经加载且保持驻留，因此峰值包含模型权重。它不等于设备全部显存，也不覆盖其他进程或所有 CUDA driver 开销。报告表显示跨重复最大的 allocated peak，避免隐藏最坏内存需求。

## 6. 当前验证结果

GPU 实验已在 NVIDIA GeForce RTX 4070 Laptop GPU 上完成，使用 Qwen2.5-0.5B-Instruct、BF16、12 个 burst 请求、最多 8 个输出 token、3 次重复。原始数据位于 `benchmarks/results/step22_qwen_gpu_burst.json`，聚合表位于 `benchmarks/reports/step22_qwen_gpu_burst.md`；HF reference 和跨策略输出一致性均通过。

| 策略 | Capacity | Budget | 吞吐中位数 [min, max] (tok/s) | TTFT P50 / P99 (ms) | ITL P50 / P99 (ms) | Peak allocated (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| sequential | 1 | 64 | 20.13 [18.81, 23.51] | 1280.92 / 2582.11 | 41.79 / 76.27 | 969.5 |
| continuous | 2 | 64 | 28.91 [26.79, 31.20] | 769.57 / 1693.66 | 55.82 / 151.01 | 970.0 |
| continuous | 4 | 64 | 37.92 [36.65, 42.68] | 514.85 / 1112.74 | 86.77 / 174.76 | 970.4 |
| continuous | 4 | 128 | 45.99 [27.06, 48.80] | 457.08 / 1266.62 | 86.55 / 293.65 | 970.9 |

这组数据支持一个有限结论：在该 burst workload 下，增加连续批处理容量提高了输出吞吐并降低了排队造成的 TTFT；同时单请求 token 间隔的尾部延迟变高。`capacity=4, budget=128` 的重复范围很宽，不能据此断言 budget 128 稳定优于 64。约 1 MiB 的容量间峰值差异来自当前短上下文、小 batch 的 KV 和临时 tensor；约 970 MiB 总峰值主要是模型常驻显存。

本机 GPU 在受限工具沙箱内不可见，需要在允许设备访问的执行环境运行 CUDA 命令。这是工具进程的设备隔离，不是本机或项目 `.venv` 缺少 CUDA。

正式 GPU 实验至少检查：

1. JSON 中模型、GPU、dtype、软件版本、seed 和 workload 是否一致。
2. `correctness.cross_run_equal` 与 HF reference 是否通过。
3. 重复范围是否过大；若过大，先排查预热、温度、频率和后台负载。
4. 吞吐增长是否伴随 TTFT/ITL 尾延迟和峰值显存上升。
5. 结合 Step 21 trace 判断瓶颈来自 model forward、KV pack/unpack 还是 CPU launch gap。

## 7. Checkpoint

- 为什么不能平均五次 P99 得到“总 P99”？
- 为什么低 request rate 下 tokens/s 低，不能说明 GPU 性能差？
- `max_memory_allocated` 与 `max_memory_reserved` 分别反映什么？
- 为什么当前 StaticBatchRunner 不能与时间驱动 continuous serving 直接比较 TTFT？

## 8. Git commit

建议提交信息：

```text
feat: add reproducible benchmark report pipeline
```

## 9. 面试价值

这部分证明你不仅会“跑一个 benchmark”，还理解实验控制、测量窗口、尾延迟聚合、负载受限与系统饱和的区别，以及 GPU allocator 指标的边界。面试中应先说清实验合同，再展示数字和瓶颈结论。
