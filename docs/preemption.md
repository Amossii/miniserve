# Step 29：Preemption 与 Recompute

## 1. 本节目标

当 paged KV 容量不足时，选择一个运行中的 decode 请求作为 victim，释放其物理 blocks，并在之后根据逻辑 token 历史重建 KV cache。

## 2. 为什么需要

请求的 KV cache 会随 decode 持续增长。即使调度器的并发槽位尚未用完，物理 KV blocks 也可能先耗尽。如果没有抢占策略，系统只能让分配失败或拒绝新请求。

生产系统通常会在 reject、swap 和 recompute 之间做选择。MiniServe 采用 recompute：它容易验证，且能清楚展示逻辑请求状态与物理 KV 状态的分离，代价是恢复时要重复执行历史 token 的 forward。

## 3. 在系统中的位置

```text
KV pressure
    ↓
Engine 选择 victim
    ↓
Scheduler: RUNNING → WAITING/RECOMPUTE
    ↓
Paged runner 释放 block table
    ↓
Scheduler 再次 admission
    ↓
Runner 从 token history 重建 KV
    ↓
下一轮继续 decode
```

## 4. 核心原理

Request 保留 `prompt_token_ids` 和已经生成的 token，因此 KV 可以丢弃后重建。恢复上下文是：

```text
prompt + generated[:-1]
```

最后一个 generated token 是下一次 decode 的输入，尚未写入 cache；把它也用于 recompute 会导致重复缓存。

本课使用 LIFO victim policy：优先抢占 running queue 尾部请求。这是确定、易测试的简化策略。victim 被放到 waiting queue 尾部，让已经等待的新请求先获得机会；`max_preemptions` 限制同一请求被反复抢占。

## 5. 修改文件

- `src/miniserve/request.py`：增加 RECOMPUTE phase、抢占次数和重计算 token 计数。
- `src/miniserve/scheduler.py`：增加 victim 状态迁移、recompute admission 和预算核算。
- `src/miniserve/engine.py`：增加单 victim/目标空闲 blocks 抢占，并协调恢复执行。
- `src/miniserve/decode_batch.py`：根据 token history 重建 DynamicCache。
- `src/miniserve/paged_kv.py`：重建 paged block table 并写回共享 storage。
- `tests/test_preemption.py`：覆盖资源压力、LIFO、队尾重入、输出一致性、次数上限和 block 回收。

## 6. 实现设计

`Engine.preempt_until_free()` 是 paged backend 专用的显式容量水位接口：调用方给出下一次执行前需要的空闲 block 数，Engine 反复选择可抢占请求，直到达到水位。Dynamic backend 没有 block allocator，因此拒绝调用抢占接口。

当前 `dynamic + chunked prefill` 与 `paged + preemption` 是两条独立实验路径；runtime 暂不支持 `paged + chunked prefill`。虽然生产系统可以组合这两项能力，本阶段不在 chunked scheduler 中提前实现 recompute，以免形成无法通过正式入口执行和验证的死逻辑。

恢复在 scheduler token budget 中按完整 recompute context 计费。一次恢复只重建 cache，不生成新 token；请求在下一轮才进入 decode。这避免把两个不同语义的 forward 合并成一个状态变化。

## 7. 运行验证

压力测试使用 8 个物理 blocks、block size 2。A/B 完成 prefill 后只剩 4 个空闲 blocks；当目标水位为 6 时，B 作为 LIFO victim 被抢占并释放两个 blocks。C 排在 B 前面执行，之后 B 重计算 3 个历史 token并继续生成。

验证项包括：

- 三个请求最终全部完成且无死锁。
- B 已生成的 token 和首 token 时间戳在抢占时保持不变。
- B 最终输出与独立 Hugging Face `generate()` 一致。
- 所有请求完成后 allocator 的 blocks 全部归还。
- 达到 `max_preemptions` 后不再选择同一 victim。

## 8. Checkpoint

完成本课后，MiniServe 已具备逻辑状态保存、物理 KV 回收、waiting queue 重入和 cache 重建的完整生命周期。当前限制是抢占触发仍由显式水位调用完成，victim policy 也尚未考虑优先级、已计算工作量和公平性。

## 9. Git Commit

建议提交信息：

```text
feat: add KV preemption and recompute lifecycle
```

## 10. 面试价值

可以用这一实现解释：KV 容量与 scheduler 并发限制为什么是两类资源约束；为什么 recompute context 不包含最后一个待 decode token；抢占如何影响延迟、公平性和重复计算量；以及生产系统为什么需要更成熟的 victim scoring 和 starvation protection。
