# Step 25：Block Allocator

## 1. 本节目标

把 KV Cache 的容量管理从“每个请求拥有一个不断增长的 tensor”抽象为固定大小 physical block 池。本课只实现 metadata allocator，不改变现有模型执行路径，也不提前绑定模型层数、KV heads、dtype 或 GPU tensor layout。

```text
total physical blocks
        ↓
free block heap
        ↓ allocate(request_id, n)
request → [physical block IDs]
        ↓ release_request(request_id)
free block heap
```

## 2. 为什么固定大小 block

假设 block size 为 4 tokens，一条 10-token sequence 需要 3 个 blocks：

```text
logical block 0: tokens 0..3   full
logical block 1: tokens 4..7   full
logical block 2: tokens 8..9   2/4 used
```

只有最后一个 block 可能存在内部碎片，本例浪费 2 个 token slots。所有 physical blocks 等大，因此请求 B 释放的中间 block 可以直接交给请求 D，不存在必须寻找连续大块的外部碎片。

## 3. Logical 与 Physical

`LogicalBlock` 描述某请求序列中的第几个块以及已经使用多少 token slots。它不知道数据放在哪个 physical block。

`PhysicalBlock` 是全局池中的固定槽位，`block_id` 在 allocator 生命周期内保持稳定。`BlockAllocator` 单独维护 `block_id → request_id` 所有权，避免把可变 owner 放进不可变 block identity。

下一课的 BlockTable 会建立：

```text
(request_id, logical_block_index)
                ↓
         physical_block_id
```

## 4. 分配与释放语义

- 分配采用最小 block ID 优先的 heap，结果确定，便于测试和调试。
- 一次申请多个 blocks 是原子的；容量不足不会分配一部分。
- 一次释放先验证所有 ID，再修改状态；混入非法 ID 不会造成部分释放。
- 重复释放、池外 ID 和未知 request 回收都会明确失败。
- `release_request()` 回收该请求跨多次分配得到的所有 blocks。

核心 invariant：

```text
free_ids ∩ allocated_ids = ∅
free_ids ∪ allocated_ids = all_block_ids
len(free_ids) + len(allocated_ids) = num_blocks
```

## 5. 为什么暂时不分配真实 KV tensor

Allocator 回答“哪些 token slots 属于哪个请求”；KV storage 回答“每层 K/V 数据如何在 GPU tensor 中布局”。两者分离后，allocator 的耗尽、回收和碎片语义可以用纯 CPU 单元测试验证。Step 26 加入 block table/slot mapping，Step 27 再决定真实 tensor layout 并接入 decode executor。

## 6. 运行验证

```bash
.venv/bin/python -m pytest tests/test_kv_blocks.py -q
```

测试覆盖：

- logical block 追加与容量边界。
- 多请求所有权和确定性 ID。
- allocator 耗尽时的原子失败。
- 释放中间洞后立即复用。
- 同一请求跨多次分配后的完整回收。
- 非法释放、重复释放和无效池配置。

## 7. Checkpoint

- 固定大小 block 为什么没有外部碎片？
- 为什么最后一个 logical block 仍可能有内部碎片？
- 为什么多 block allocation 必须原子失败？
- 为什么 allocator 不应该依赖 PyTorch、模型层数或 CUDA？
- Request 结束时只释放最后一个 block 会造成什么问题？

## 8. Git commit

建议提交信息：

```text
feat: add fixed-size KV block allocator
```

## 9. 面试价值

这部分对应 vLLM PagedAttention 的内存管理基础。你应能解释 logical/physical block 的区别、free list、内部与外部碎片、原子分配和请求结束时的资源生命周期，而不只会描述“分页可以省显存”。
