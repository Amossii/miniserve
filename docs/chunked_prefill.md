# Step 28：Chunked Prefill

## 1. 本节目标

让超过单轮 token budget 的 prompt 跨多个 Engine iteration 完成 prefill，并让已有 decode 请求优先获得预算。完整 prefill 保持默认行为，使用 `--chunked-prefill` 显式开启。

## 2. 为什么需要

完整 prefill 要求 `prompt_length <= max_num_batched_tokens`。生产系统通常把长 prefill 切成 chunks，让 decode token 插在 chunks 之间，避免一次长 forward 持续阻塞所有 decode 请求。vLLM/SGLang 会把它与完整 token scheduler、KV block 管理和执行器结合；MiniServe 简化为同步 Engine、decode-priority budget 和 DynamicCache partial state。

## 3. 系统位置

```text
Request.num_prefilled_tokens
          ↓
Scheduler: PrefillChunk(request, start, end)
          ↓
Engine 固定本轮 decode/chunk 集合
          ↓
DecodeBatchRunner.prefill_chunk()
          ↓
Partial DynamicCache 跨轮保存
          ↓
最后一个 chunk 产生 first token
```

本课只支持 dynamic KV backend。Paged backend 明确拒绝该组合，因为 block storage 的 chunk 写入与容量失败回滚需要单独设计。

## 4. 核心原理

每轮先为 active decode 请求各预留 1 token，再推进已有 partial prefill，最后用剩余 slot/budget 按 FIFO 接纳新请求。中间 chunk 只推进 cursor/cache，不生成 token；最后一个 chunk 才从最后一个 prompt token 的 logits 产生首 token并切换到 DECODE。

```text
budget = 3
A decode: 1 token
B prefill: prompt[0:2]
total: 3
```

下一次调用必须满足 chunk start 等于 Request cursor，partial cache length 也等于 cursor，避免重复或跳过 prompt token。Decode priority 控制单轮阻塞，但更多小 forward 会增加 launch 和 Python 开销。

## 5. 修改文件

- `request.py`：prefill cursor 与推进规则。
- `scheduler.py`：PrefillChunk 和 decode-priority policy。
- `decode_batch.py`：增量 prefill 与 partial cache。
- `engine.py`：执行 chunk metadata并记录真实工作量。
- `runtime.py`：backend/feature 组合验证。
- CLI 与 benchmark：显式开关及 trace。

## 6. 完整代码

核心实现位于 `src/miniserve/request.py`、`scheduler.py`、`decode_batch.py` 和 `engine.py`。正式运行：

```bash
.venv/bin/python scripts/run_engine.py \
  --chunked-prefill --token-budget 3 --max-running 2 \
  --max-new-tokens 4 --check-reference
```

## 7. 测试验证

9-token prompt 在 budget 4 下按 `[0,4) → [4,8) → [8,9)` 执行，与完整 HF greedy generation 一致。四请求 demo 中 partial prefill 与 decode 共存，所有 iteration 均不超预算。

GPU：RTX 4070、Qwen2.5-0.5B、BF16、12 个 burst 请求、capacity 4、seed 27、3 次重复。完整 prefill 使用 budget 128；chunked 使用 budget 32，所以这是配置取舍实验，不是严格等条件算法对比。

| 配置 | Output tok/s median [min, max] | TTFT P50 / P99 (ms) | ITL P99 (ms) | Peak MiB |
|---|---:|---:|---:|---:|
| Full, budget 128 | 79.66 [77.02, 79.86] | 234.65 / 462.29 | 90.30 | 970.86 |
| Chunked, budget 32 | 47.59 [41.43, 48.60] | 459.02 / 944.17 | 187.92 | 968.41 |

本 workload 的 prompt 只有几十 tokens，拆分增加小 forward，却没有足够长的 prefill 阻塞可缓解；吞吐和延迟均变差，峰值只降低约 2.45 MiB。不能据此推出 chunked prefill 普遍有害，目标场景需要长 prompt、并发 decode 和多组 chunk size。

原始数据：`benchmarks/results/step28_chunked_gpu.json`。

## 8. Checkpoint

- 为什么中间 chunk 不能产生首 token？
- 为什么 decode 要先预留 token budget？
- Cursor 与 partial cache length 为什么必须一致？
- 为什么 chunk 越小不一定越好？
- 为什么本次结果不能解释为严格算法加速对比？

## 9. Git Commit

```text
feat: add decode-priority chunked prefill
```

## 10. 面试价值

Chunked prefill 不减少 prompt 总计算量，而是把长 prefill 拆成可调度单位。收益取决于 prompt 长度、chunk size、并发 decode 和 kernel efficiency；短 prompt 上更多小 forward 可能更慢。
