# Step 26：Block Table 与 Slot Mapping

## 1. 本节目标

把请求的连续逻辑 token positions 映射到可能离散的 physical KV blocks，并生成执行器可以消费的 batch slot metadata。

```text
logical position
      ↓ divmod(block_size)
logical block index + block offset
      ↓ BlockTable
physical block ID + block offset
      ↓
physical slot = physical_block_id × block_size + block_offset
```

本课建立地址转换层，尚未把 K/V tensor 写入这些 slots。Step 27 将根据 slot mapping 构造真实 paged KV storage 和执行路径。

## 2. 映射示例

block size 为 4，Request A 的 block table 为：

```text
logical block 0 → physical block 3
logical block 1 → physical block 7
```

逻辑 position 5：

```text
logical_block_index = 5 // 4 = 1
block_offset        = 5 % 4  = 1
physical_block_id   = block_table[1] = 7
physical_slot       = 7 * 4 + 1 = 29
```

Position 0..7 在逻辑上连续，但物理 block 3 和 7 不需要相邻。这是 paged KV 避免连续大块分配的基础。

## 3. 跨 block 追加

`BlockTable.append_tokens(n)` 先计算追加后的总 block 数，再一次性向 allocator 申请所有缺少的 blocks。只有申请成功后，才提交 physical block 列表、logical block 使用量和 sequence token 总数。

例如 block size 为 4，当前有 3 tokens，再追加 6 tokens，需要从 1 个 block 增长到 3 个 blocks。如果 allocator 只剩 1 个 block，整次追加失败，原来的 3-token table 不变。

## 4. SlotMapping 与 batch metadata

每个 `SlotMapping` 保存 request ID、logical position、logical block index、block offset、physical block ID 和扁平 physical slot。冗余字段让测试与 trace 容易解释；真实 kernel 最终通常只需要压缩后的 block table、sequence lengths 和 slot mapping tensors。

`BatchSlotMapping` 保持调用方给出的 row 顺序，不按 request 或 physical block 排序。Batch row 必须继续与 input token、position ID 和输出 logits 一一对应。

## 5. 生命周期与 invariant

请求结束时 `BlockTable.release()` 释放本表全部 physical blocks、清空映射并永久关闭 table。Released table 不允许查询旧地址或重新追加，因为对应 blocks 可能已经属于其他请求。

```text
len(logical_blocks) == len(physical_blocks)
sum(logical_block.num_tokens) == table.num_tokens
除最后一个 logical block 外，其余 block 全满
physical block owner == table.request_id
released table 不持有任何 block 或 token
```

## 6. 运行验证

```bash
.venv/bin/python -m pytest tests/test_block_table.py -q
```

测试覆盖跨 block 边界、逻辑连续但物理离散、容量不足联合原子失败、多请求 batch row 顺序、释放后地址失效以及 physical block 复用。

## 7. Checkpoint

- 为什么 logical blocks 连续，而 physical blocks 不需要连续？
- 为什么 `physical_slot` 可以用 block ID 和 offset 计算？
- 为什么 batch slot mapping 不能按 physical slot 排序？
- 为什么释放后的 BlockTable 必须失效？
- 跨 block 追加失败时，哪些状态必须保持不变？

## 8. Git commit

建议提交信息：

```text
feat: add request block tables and slot mappings
```

## 9. 面试价值

你现在可以具体说明 PagedAttention 中的地址转换：请求看到连续的逻辑 sequence，内存管理器提供离散 physical blocks，执行器通过 block table 和 offset 找到真实 K/V slot。
