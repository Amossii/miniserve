# MiniServe Architecture

## 请求到 token 的完整路径

```mermaid
flowchart LR
    W[Workload / Request] --> S[Scheduler]
    S -->|SchedulePlan| E[Engine.step]
    E --> P[Prefill batch]
    E --> D[Decode batch]
    P --> M[HF Causal LM]
    D --> K1[KV pack + padding]
    K1 --> M
    M --> K2[KV unpack]
    K2 --> R[Per-request DecodeState]
    P --> R
    R --> Q[Request state + timestamps]
    Q --> S
    Q --> X[Metrics / Result]
```

## Control plane 与 execution plane

Scheduler 是 control plane，只读取 Request 元数据并产生一轮不可变的调度计划。它管理 waiting/running 队列、FIFO admission、并发容量和 token budget，不接触模型 tensor。

DecodeBatchRunner 是 execution plane，负责构造 tensor、执行模型 forward 和管理每请求 KV。Engine 位于两者之间：先固定本轮 prefill/decode 集合，再依次执行，最后回收完成请求。这个边界保证本轮新 prefill 的请求不会同时进入 decode。

```mermaid
stateDiagram-v2
    [*] --> WAITING
    WAITING --> RUNNING: Scheduler admission
    state RUNNING {
        [*] --> PREFILL
        PREFILL --> DECODE: first token + KV created
    }
    RUNNING --> FINISHED: EOS or max_new_tokens
    FINISHED --> [*]: reclaim KV/state
```

## 一轮 Engine iteration

1. 回收上一轮已经完成的请求。
2. 为现有 decode 请求预留一个 token 的预算。
3. 在剩余容量和预算内，按 FIFO 接纳完整 prompt。
4. 固定本轮 `prefill_requests` 和 `decode_requests`。
5. Prefill 新请求，生成首 token 和初始 KV。
6. Pack 不同 context length 的 decode KV，执行一个 batched forward，再 unpack。
7. 更新时间戳和状态；完成请求在下一轮开始时回收。

## 当前 KV 数据流

```text
Request A KV [layers, heads, 20, dim] ─┐
Request B KV [layers, heads, 50, dim] ─┼─ pad + cat ─ batched forward
Request C KV [layers, heads, 18, dim] ─┘                  │
                                                        ↓
                                     slice/unpack → per-request KV
```

这种实现易于验证 heterogeneous decode correctness，但每轮会产生 padding、copy、`aten::cat` 和新 tensor allocation。Paged backend 已把长期 ownership 改为 block pool，但由于 HF attention 仍要求连续 cache，forward 前仍需 gather/pack；它是教学 adapter，不是 fused PagedAttention。

当前支持两条独立的高级路径：dynamic backend 支持 chunked prefill；paged backend 支持 block capacity preemption 与 recompute。runtime 暂不组合 paged KV 和 chunked prefill。

## 关键 invariant

- Request ID 全局唯一；正常路径为 WAITING → RUNNING → FINISHED，paged victim 可从 RUNNING/DECODE 返回 WAITING/RECOMPUTE。
- 一轮调度输入 token 数不超过 `max_num_batched_tokens`。
- Full-prefill 模式要求 prompt 能放入预算；dynamic chunked 模式允许跨轮推进 prefill cursor。
- 每个 running decode 请求必须存在且只存在一份 DecodeState。
- 新 prefill 请求在下一轮才能参与 decode。
- Request 完成后，其调度槽位和 KV state 都必须回收。
