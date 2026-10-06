# HF Static Batch 与 MiniServe Continuous Benchmark

## 实验目标

在完全相同的变长 burst workload 下，对比固定大小的 Hugging Face static batching
与 MiniServe continuous batching。HF 使用 `model.generate()` 和逐 token streamer 提供
TTFT/TPOT 时间戳；MiniServe 使用现有 Engine 和 dynamic KV backend。

## 实验合同

- 所有请求在同一 workload 起点到达，TTFT/E2E 包含排队时间。
- 每个请求的输入长度从 `{128, 256, 512}` 中按 seed 抽样。
- 每个请求的输出目标从 `{16, 32, 64, 128}` 中按 seed 抽样。
- 两端均使用 greedy decoding、相同模型和 dtype，并关闭 EOS 提前停止。
- HF static batch size 为 4；一批执行到该批最长输出结束，期间不替换已完成 row。
- MiniServe `max_concurrency=4`；请求完成后释放 slot，下一轮可以接纳新请求。
- 默认 token budget 为 `max(input_lengths) * max_concurrency = 2048`。
- 吞吐按完整 workload 墙钟时间计算；延迟分位数从所有请求的原始样本重新计算。
- tokenizer、模型加载和 warmup 不进入正式测量窗口；网络传输不在实验范围内。

## 运行方式

T4 上可运行：

```bash
.venv/bin/python scripts/benchmark_hf_vs_miniserve.py \
  --model /path/to/Qwen2.5-0.5B-Instruct \
  --device cuda \
  --input-lengths 128 256 512 \
  --output-lengths 16 32 64 128 \
  --num-requests 80 \
  --max-concurrency 4 \
  --token-budget 2048 \
  --repeats 3 \
  --output benchmarks/results/hf_vs_miniserve_t4.json
```

脚本输出并保存：

- total output throughput（output tokens/s）
- request throughput（requests/s）
- TTFT P50/P99
- TPOT P50/P99
- E2E P50/P99
- 每次重复的原始指标和延迟样本
- 模型、dtype、GPU、PyTorch 与 Transformers 版本
- HF 与 MiniServe 的 exact-request/token-position match rate

## 指标解释

HF static batching 中，后续 batch 的 TTFT 包含等待前序 batch 完成的时间。较短请求达到
自己的输出目标后视为完成，但其 row 直到该批最长请求完成后才可复用。MiniServe 的 TTFT
包含 Scheduler 排队、prefill 和首 token 时间。TPOT 是单请求首末输出 token 时间差除以
`num_output_tokens - 1`，不包含首 token 之前的排队和 prefill。E2E 从共同 burst 到达时刻
计算到请求完成。

两条路径使用不同的 batch shape、padding 和浮点归约路径；当候选 logits 非常接近时，
greedy `argmax` 也可能分叉。因此脚本要求每个后端跨重复输出稳定，同时记录跨后端的
exact-request 和 token-position match rate，但不会因少量数值分叉丢弃性能样本。

只有在目标 GPU 上实际运行并保留 JSON 后，才能把结果写入简历；不要预填性能数字。
