# MiniServe Phase A Report

## 实现结果

Phase A 完成了从 Request 到 Metrics 的连续批处理推理链路：不同 prompt/context length 的请求可以共同 decode；新请求可在旧请求运行期间进入 prefill；完成请求释放 slot，下一轮立即接纳 waiting 请求。调度同时受最大运行请求数和每轮输入 token budget 约束。

正确性通过状态机/invariant 单元测试、跨策略逐 token 对比以及 Hugging Face `generate()` reference 验证。当前回归基线为 94 个测试。

## GPU benchmark

实验环境：RTX 4070 Laptop GPU、Qwen2.5-0.5B-Instruct、BF16、eager attention、12 个 burst 请求、最多 8 个输出 token、3 次重复。完整原始数据见 `benchmarks/results/step22_qwen_gpu_burst.json`。

| Policy | Capacity | Budget | Output tok/s median [min, max] | TTFT P50 / P99 (ms) | ITL P50 / P99 (ms) | Peak allocated (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| sequential | 1 | 64 | 20.13 [18.81, 23.51] | 1280.92 / 2582.11 | 41.79 / 76.27 | 969.5 |
| continuous | 2 | 64 | 28.91 [26.79, 31.20] | 769.57 / 1693.66 | 55.82 / 151.01 | 970.0 |
| continuous | 4 | 64 | 37.92 [36.65, 42.68] | 514.85 / 1112.74 | 86.77 / 174.76 | 970.4 |
| continuous | 4 | 128 | 45.99 [27.06, 48.80] | 457.08 / 1266.62 | 86.55 / 293.65 | 970.9 |

在这个有限 burst workload 中，capacity 4 / budget 64 相对 sequential / budget 64 的吞吐中位数提高约 88%，TTFT P50 降低约 60%，同时 ITL P99 增加。capacity 4 / budget 128 的吞吐范围较宽，不能据此认为更大 budget 稳定更快。这组数据用于展示吞吐、排队延迟和单请求 token 间隔之间的取舍，不代表生产 serving 性能。

## GPU profiling

PyTorch CUDA profiler 使用同一 Qwen 模型、capacity 4、budget 128，采集 8 个 Engine iteration。metadata 与 operator table 位于 `benchmarks/profiles/step21_qwen_cuda/`；82 MiB Chrome trace 保留在本地并由 Git 忽略，可用文档命令重新生成。

可复核证据：

- `miniserve::prefill_model_forward`、`decode_model_forward`、`kv_pack` 和 `kv_unpack` 均能在 trace 中独立定位。
- `aten::cat` 出现 3035 次，累计约 7.82 ms self CUDA time，并报告约 59.24 MiB CUDA allocation。
- 高频、小粒度 cat/elementwise kernel 与当前逐层 padding/拼接 KV 的实现一致。

这说明 KV pack/unpack 是明确的内存流量与 kernel launch 优化候选。该 trace 带有 shape/memory profiler 开销，用户区间会重叠，不能把区间百分比相加，也不能用其时长替代正式 benchmark。是否 memory-bound 仍需结合 Nsight Systems/Compute 的带宽和 stall 指标判断。

## Phase B 方向

下一阶段优先实现 block allocator、block table 和 slot mapping，再把 decode executor 改成 paged KV 访问。目标是消除当前每轮全量 pack/unpack，而不是先写一个与已测瓶颈无关的 kernel。
