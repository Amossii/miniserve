# MiniServe 面试表达与简历材料

## 30 秒项目介绍

MiniServe 是我从零实现的简化 LLM continuous-batching inference engine。它支持 Request 状态机、prefill/decode 共存、不同 context length 的 batched decode、per-request KV Cache、FIFO admission 和 token-budget scheduling。我还建立了 TTFT/ITL/TPOT 指标、时间驱动 workload、GPU benchmark 与 PyTorch profiler 链路，并用 Hugging Face reference 验证跨策略 token 一致性。

## 简历 bullet

- 从零实现 continuous-batching LLM inference engine，拆分 Scheduler、Engine 与 Model Runner，支持动态请求接纳、prefill/decode 共存、异长 context batched decode 和 token-budget scheduling。
- 构建 per-request KV Cache pack/unpack 生命周期与完整 correctness suite，通过 Hugging Face greedy generation 和跨调度策略逐 token 对照验证，项目回归覆盖 94 个测试。
- 建立 TTFT、ITL、TPOT、E2E、吞吐与峰值显存 benchmark；在 RTX 4070 / Qwen2.5-0.5B burst workload 上，capacity 4 相对串行基线将吞吐中位数提高约 88%、TTFT P50 降低约 60%。
- 使用 PyTorch Profiler 标注 prefill、decode、KV pack/unpack，观察到 3035 次 `aten::cat` 和约 59 MiB allocator activity，据此规划 block/paged KV 优化路径。

数字必须与 `docs/phase_a_report.md` 的固定实验配置一起陈述，不能泛化成所有模型、输入长度或 GPU 上的收益。

## 高频问题

### 为什么 decode 常被称为 memory-bound？

单个 decode step 每个请求只产生一个 token，矩阵乘法的 M 维很小，但仍要读取大量模型权重和历史 KV；算术强度低于大 prompt prefill。是否在某台 GPU 上实际 memory-bound 仍应通过 profiler/roofline 验证。

### KV Cache 为什么降低计算量？

不使用 cache 时，每一步都会重新计算历史 token 的 K/V。缓存后只计算新 token 的 Q/K/V，并让 attention 读取已有 K/V；代价是 KV 显存随层数、KV heads、head dimension 和 sequence length 线性增长。

### Continuous batching 比 static batching 好在哪里？

固定 batch 中已完成请求仍占 row，且新请求必须等整批结束。Continuous batching 每轮重建成员，完成请求释放 slot 后可立刻接纳新请求，从而减少空槽和队列等待。

### Token budget 与 max running 有什么区别？

`max_running` 限制活跃 sequence 数；token budget 限制本轮送入模型的 input token 数。一个空闲 sequence slot 不代表剩余预算足够容纳完整 prompt。

### 当前最明显的架构瓶颈是什么？

异长 decode 通过每轮 padding/cat 把 per-request KV 组成 batch，forward 后再 slice/unpack。它正确且易懂，但产生额外 copy、allocation 和小 kernel；CUDA trace 已观察到高频 `aten::cat`。Phase B 用 block table 和 paged KV 解决这一点。

### 这个 benchmark 有什么局限？

它是同步、单进程、进程内 workload，不含网络和 tokenizer；模型仅 0.5B，请求数和输出长度较小。报告保留原始样本和波动范围，并将结论限制在固定实验合同内。
