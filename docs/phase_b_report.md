# Step 30：Profiler-Driven Optimization 与 Phase B 总结

## 1. 本节目标

根据已有 GPU profiler 证据选择一个优化候选，完成 correctness、GPU benchmark、结果分析和是否保留实现的工程决策，并总结 Phase B。

## 2. 为什么需要

性能工程不能以“代码看起来更快”结束。一次完整优化必须回答：证据指向哪里、改动减少了什么、输出是否一致、端到端指标是否改善，以及失败时是否应该回退。

## 3. 在系统中的位置

已有 Qwen2.5-0.5B CUDA trace 在 dynamic KV decode 路径观察到 3035 次 `aten::cat`、约 7.82 ms self CUDA time 和约 59.24 MiB allocator activity。热点位于：

```text
per-request DynamicCache
    ↓
left padding + torch.cat
    ↓
batched model forward
    ↓
slice + clone
```

## 4. 核心原理

候选实现把每层的“逐请求 pad，再 cat”替换为“一次分配最终 batched K/V，再把每个请求复制到右对齐区域”。预期收益是减少临时 padding tensor 和 `aten::cat`；代价是 Python 循环内产生逐层、逐请求 `copy_`。

定向测试确认两种实现的 cache 内容和最终 token 一致。端到端 GPU benchmark 没有显示收益，因此最终代码回退到原实现。这里的结论是该替换方案失败，不代表 `aten::cat` 不是热点，也不代表 fused packing kernel 没有价值。

## 5. 修改文件与实验产物

- `benchmarks/results/step30_pack_before.json`：原始实现的五次 GPU 结果。
- `benchmarks/results/step30_pack_after.json`：direct-copy 候选的五次 GPU 结果。
- `docs/phase_b_report.md`：实验合同、结果和 Phase B 总结。
- `PROJECT_STATE.md`、`README.md`、`docs/architecture.md`、`docs/interview_guide.md`：更新最终状态与项目边界。

## 6. 实验设计

固定环境为 RTX 4070 Laptop GPU、Qwen2.5-0.5B-Instruct、BF16 eager attention。workload 为 12 个 burst 请求、每个最多生成 8 token、capacity 4、token budget 128、seed 30。before/after 各运行 5 次，均通过 Hugging Face greedy reference。

命令结构为：

```bash
python scripts/benchmark_serving.py \
  --model /home/henry/project/models/Qwen2.5-0.5B-Instruct \
  --device cuda --num-requests 12 --max-new-tokens 8 \
  --arrival burst --seed 30 --max-running 4 --token-budgets 128 \
  --repeats 5 --kv-backend dynamic --check-reference
```

## 7. 测试与 Benchmark

| 指标（5 次中位数） | Before | Direct-copy candidate | 变化 |
|---|---:|---:|---:|
| Output tokens/s | 50.44 | 40.34 | -20.0% |
| TTFT P50 | 332.84 ms | 356.10 ms | +7.0% |
| ITL P99 | 172.24 ms | 256.96 ms | +49.2% |
| Peak allocated | 970.44 MiB | 970.44 MiB | 0% |

after 的吞吐范围为 19.88–49.35 token/s，波动明显；before 也有一次较低的 34.01 token/s。两组是先后运行而非进程内交错实验，因此温度、功耗和系统负载可能影响结果。即使考虑这一限制，现有数据也没有支持保留 direct-copy 方案。

## 8. Checkpoint

Phase B 完成了 block allocator、block table/slot mapping、paged KV storage adapter、dynamic chunked prefill，以及 paged preemption/recompute。它也证明了两个重要负结果：仅把 KV 改成 paged storage、或把 `pad + cat` 改成许多 `copy_`，都不会自动带来端到端加速。

当前组合边界为 dynamic + chunked prefill，以及 paged + preemption。HF attention 仍要求连续 cache，因此 paged backend decode 仍有 gather/pack。下一阶段若继续优化，应先建立交错 A/B harness，再考虑一个能融合 padding/copy 的 Triton pack kernel，或改 attention 路径以直接消费 block table。

## 9. Git Commit

建议提交信息：

```text
docs: complete profiler-driven Phase B evaluation
```

## 10. 面试价值

这次实验展示的重点不是一个正收益数字，而是完整性能工程闭环：从 trace 提出假设，以 reference 守住 correctness，用固定 workload 做 A/B，发现端到端回退后拒绝合入候选实现，并解释 operator 数量、kernel launch 和整体模型耗时之间的关系。
