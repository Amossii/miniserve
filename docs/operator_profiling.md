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

## 当前环境结论

当前 PyTorch 是 CUDA build，但 `torch.cuda.is_available()` 为 False；`nsys` 和 `ncu` 命令存在。因此本课完成了 CPU operator trace 和全部 CUDA profiling 入口，尚未生成可信的 GPU trace。

本次 CPU 小模型 trace 中：

- `decode_model_forward` 与 `prefill_model_forward` 是主要用户区间。
- `kv_pack` 和 `kv_unpack` 已能从模型 forward 中独立识别。
- 这只证明标注和导出路径正确，不支持关于 GPU utilization、memory-bound 或 compute-bound 的结论。

在 CUDA 可用机器上完成 PyTorch CUDA trace 与 Nsight Systems trace 后，Step 21 才达到完整验收标准。
