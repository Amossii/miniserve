# Step 21：PyTorch Profiler 与 GPU Profiling

## 目标

Step 20 回答 Engine 一轮中 prefill/decode/control plane 各占多少时间。Step 21 继续下钻：哪些 PyTorch operator、CUDA runtime 调用和 GPU kernel 构成这些阶段。

MiniServe 在显式开启 `annotate_profiler` 时添加以下用户区间：

```text
miniserve::scheduler
miniserve::prefill
  miniserve::prefill_model_forward
miniserve::decode
  miniserve::kv_pack
  miniserve::decode_model_forward
  miniserve::kv_unpack
miniserve::reclaim
```

默认 Engine 不注册这些 `record_function` 区间，但仍经过轻量的 Python 空上下文。正式 benchmark 应保持 annotation 关闭。

## PyTorch Profiler

CPU smoke trace：

```bash
.venv/bin/python scripts/profile_torch.py \
  --steps 6 \
  --output-dir benchmarks/profiles/step21_cpu
```

CUDA：

```bash
.venv/bin/python scripts/profile_torch.py \
  --model /home/henry/project/models/Qwen2.5-0.5B-Instruct \
  --device cuda --steps 8 \
  --output-dir benchmarks/profiles/step21_cuda
```

输出：

- `trace.json`：Chrome/Perfetto 时间线。
- `operator_table.txt`：按 self CPU 或 self CUDA time 排序的 operator 表。
- `metadata.json`：环境、活动类型、实际采集轮数和配置。

脚本先在 profiler 外预热，再采集有限 iteration。`record_shapes` 与 `profile_memory` 会增加开销，所以结果用于定位热点，不作为最终吞吐数字。打开 trace 后，先找到 `miniserve::*` 区间，再观察其中的 ATen operator、CUDA runtime 和 kernel。

PyTorch 官方文档说明 `ProfilerActivity.CPU` 记录 CPU operator 和用户 label，`ProfilerActivity.CUDA` 记录 CUDA kernel/runtime；Chrome trace 可用 trace viewer 查看。参考：<https://docs.pytorch.org/docs/stable/profiler.html>。

## Nsight Systems

有可用 NVIDIA GPU 时：

```bash
bash scripts/profile_nsys.sh \
  --model /home/henry/project/models/Qwen2.5-0.5B-Instruct \
  --max-new-tokens 8 --steps 8
```

生成 `benchmarks/profiles/nsys/miniserve.nsys-rep`。重点检查：

- CPU 是否持续向 GPU 发射工作，还是存在 launch gap。
- `.item()` / `.tolist()` 附近是否出现同步。
- prefill 与 decode 的 kernel 组成和时长是否不同。
- KV pack/unpack 是否产生大量小 kernel、copy 或 allocation。
- GPU 是否存在长时间空闲。

Nsight Systems 用于系统时间线。只有当时间线已经定位到关键 kernel，才使用 Nsight Compute 分析该 kernel 的 memory throughput、occupancy、warp stall 等指标。不要一开始对所有 kernel 全量采集 NCU 指标。

## 当前 GPU trace 结论

已在 NVIDIA GeForce RTX 4070 Laptop GPU 上采集 Qwen2.5-0.5B-Instruct BF16 trace：capacity 4、token budget 128、8 个 Engine iteration。metadata 与 operator table 位于 `benchmarks/profiles/step21_qwen_cuda/`；大型 Chrome trace 默认由 Git 忽略并保留在本地。

- `decode_model_forward`、`prefill_model_forward`、`kv_pack` 和 `kv_unpack` 用户区间均可定位。
- `aten::cat` 出现 3035 次，约占 7.82 ms self CUDA time，并报告约 59.24 MiB CUDA allocation。
- 大量 cat 与小粒度 elementwise kernel 符合当前逐层 KV padding、pack、unpack 的执行方式，因此它是 Phase B block/paged KV 的直接优化候选。

Profiler 开启了 shape 和 memory 记录，会显著增加开销；用户区间可以嵌套和重叠，不能把表中的区间百分比相加。这份证据可以定位数据搬运问题，但不足以单独证明整个 decode memory-bound。后续用 Nsight Systems 检查 launch gap，再针对关键 kernel 用 Nsight Compute 检查带宽、occupancy 和 stall。
