# Step 27：Paged KV Cache Execution

## 1. 本节目标

让真实 Engine 使用 BlockAllocator、BlockTable 和预分配 physical K/V tensors 保存长期 cache，并与 Phase A per-request DynamicCache 路径做逐 token correctness 和 GPU A/B benchmark。

## 2. 系统位置与简化方案

vLLM/SGLang 的 paged attention kernel 能根据 block table 直接读取离散 KV blocks。MiniServe 当前使用未修改的 Hugging Face attention，它要求连续 `DynamicCache`，不能直接消费 physical slot metadata。

本课采用教学版 adapter：

```text
长期状态：PagedKVStorage + BlockTable
                  ↓ gather
临时连续、left-padded DynamicCache
                  ↓ HF attention forward
只取本轮 input token 的新 K/V
                  ↓ slot write
长期状态：PagedKVStorage + BlockTable
```

它真实改变了 KV ownership：`PagedDecodeState` 不再持有 per-request DynamicCache，历史 K/V 常驻共享 block pool。它仍保留 forward 前 gather/cat，因此不是 fused PagedAttention。

## 3. Prefill、Decode 与生命周期

Prefill 前先检查 block 容量；模型产生 prompt cache 后，BlockTable 分配 slots，PagedKVStorage 将每层 K/V 写入 physical blocks。输出是 `PagedDecodeState(request, block_table)`。

Decode 输入多个 active paged states。执行器 gather history，完成一次 batched forward，为每个请求追加一个 slot，只把 output cache 最后一列写入该 slot，再更新 Request。下一轮继续满足：

```text
request.sequence_length == block_table.num_tokens + 1
```

请求结束时 Engine 调用 `release_state()`，清零对应 blocks、释放 BlockTable，physical slots 随即可被复用。

## 4. 与真实 PagedAttention 的差距

生产系统把 block table、sequence length 和 slot mapping 传给专用 kernel，直接寻址离散 blocks。MiniServe 为兼容 HF 模型仍会 gather 每请求历史 KV、left-pad、`torch.cat` 并创建临时 DynamicCache。

本课消除了 forward 后把完整历史 clone 回每请求 cache 的长期 ownership，但没有消除 forward 前的数据整理。真正消除 gather 需要自定义 PyTorch attention、Triton 或 CUDA kernel。

## 5. 正确性验证

```bash
.venv/bin/python scripts/run_engine.py \
  --kv-backend paged --num-kv-blocks 32 --kv-block-size 2 \
  --max-new-tokens 4 --check-reference
```

相同 tiny Llama/workload 下 dynamic 与 paged 输出逐 token 一致；四请求 Engine demo 通过 HF reference；排空后 allocator 恢复全空；paged state 不持有 per-request DynamicCache。

## 6. GPU A/B Benchmark

环境：RTX 4070 Laptop GPU、Qwen2.5-0.5B-Instruct、BF16、12 个 burst 请求、最多 8 个输出 token、capacity 4、token budget 128、seed 27、3 次重复。Paged pool 为 64 blocks × 16 tokens。

| Backend | Output tok/s median [min, max] | TTFT P50 (ms) | ITL P99 (ms) | Peak allocated (MiB) |
|---|---:|---:|---:|---:|
| dynamic | 79.66 [77.02, 79.86] | 234.65 | 90.30 | 970.9 |
| paged HF adapter | 70.65 [69.18, 70.72] | 257.29 | 102.87 | 981.5 |

Paged adapter 吞吐中位数下降约 11.3%，TTFT P50 和 ITL P99 均变差，峰值 allocated 显存增加约 10.7 MiB。预分配 pool 增加常驻内存，而 gather/cat 仍存在，所以当前实现没有获得 PagedAttention 性能优势。

原始结果位于 `benchmarks/results/step27_dynamic_gpu.json` 和 `step27_paged_gpu.json`。第一次 paged GPU 实验发现 inference tensor 不能在普通 reclaim context 中原地清零；修复为 inference-mode 回收。随后又发现 warmup 与正式测量使用不同 runner，导致 pool 初始化落入测量窗口；修正为复用已排空 runner、重建 Scheduler/Request 后，重新运行了两组最终实验。

## 7. Checkpoint

- 为什么 BlockTable 已存在，HF attention 仍不能直接读取 paged KV？
- Paged state 与 Phase A DecodeState 的 KV ownership 有何不同？
- 为什么只写 output cache 最后一列就足够？
- 为什么预分配 block pool可能提高峰值显存？
- 这次负性能结果说明下一步需要什么？

## 8. Git Commit

```text
feat: add paged KV storage with HF gather adapter
```

## 9. 面试价值

这次实现区分了分页内存管理和 PagedAttention kernel。Block allocator/table 不会自动提速；attention API 仍要求连续 cache 时，gather 成本依然存在，真实 A/B 结果验证了这一点。
