from __future__ import annotations

import pytest

from miniserve.block_table import BlockTable, build_batch_slot_mapping
from miniserve.kv_blocks import BlockAllocator, BlockCapacityError


def test_append_crosses_block_boundary_and_returns_new_slots():
    """输入 3+3 tokens；输出跨边界 slots；验证 logical blocks 的占用与物理映射。"""
    allocator = BlockAllocator(num_blocks=4, block_size=4)
    table = BlockTable("A", allocator)

    first = table.append_tokens(3)
    second = table.append_tokens(3)

    assert tuple(item.physical_slot for item in first) == (0, 1, 2)
    assert tuple(item.physical_slot for item in second) == (3, 4, 5)
    assert table.physical_block_ids == (0, 1)
    assert tuple(block.num_tokens for block in table.logical_blocks) == (4, 2)
    assert table.num_tokens == 6


def test_noncontiguous_physical_blocks_preserve_contiguous_logical_positions():
    """输入被其他请求隔开的分配；输出连续逻辑位置和离散 physical slot。"""
    allocator = BlockAllocator(num_blocks=5, block_size=2)
    a = BlockTable("A", allocator)
    b = BlockTable("B", allocator)
    a.append_tokens(2)
    b.append_tokens(2)
    a.append_tokens(3)

    assert a.physical_block_ids == (0, 2, 3)
    assert tuple(item.logical_position for item in a.mappings()) == (0, 1, 2, 3, 4)
    assert tuple(item.physical_slot for item in a.mappings()) == (0, 1, 4, 5, 6)


def test_failed_growth_keeps_table_and_allocator_unchanged():
    """输入超过剩余 blocks 的追加；输出容量异常；table 和 allocator 都不出现部分提交。"""
    allocator = BlockAllocator(num_blocks=2, block_size=4)
    table = BlockTable("A", allocator)
    table.append_tokens(3)
    allocator.allocate("B")

    with pytest.raises(BlockCapacityError):
        table.append_tokens(6)

    assert table.num_tokens == 3
    assert table.physical_block_ids == (0,)
    assert tuple(block.num_tokens for block in table.logical_blocks) == (3,)
    assert allocator.allocated_block_ids("A") == (0,)
    assert allocator.allocated_block_ids("B") == (1,)


def test_batch_mapping_keeps_row_order_across_requests():
    """输入不同请求和 position 顺序；输出平行 batch metadata；不按物理 ID 重排 rows。"""
    allocator = BlockAllocator(num_blocks=4, block_size=4)
    a, b = BlockTable("A", allocator), BlockTable("B", allocator)
    a.append_tokens(5)
    b.append_tokens(2)

    batch = build_batch_slot_mapping([(b, 1), (a, 4), (a, 0)])

    assert batch.request_ids == ("B", "A", "A")
    assert batch.logical_positions == (1, 4, 0)
    assert batch.physical_slots == (9, 4, 0)


def test_release_invalidates_table_and_allows_physical_reuse():
    """输入请求完成；输出全部释放 blocks；旧地址失效且新请求复用最小 IDs。"""
    allocator = BlockAllocator(num_blocks=3, block_size=2)
    table = BlockTable("A", allocator)
    table.append_tokens(5)

    released = table.release()

    assert tuple(block.block_id for block in released) == (0, 1, 2)
    assert table.is_released
    assert allocator.num_free_blocks == 3
    with pytest.raises(RuntimeError, match="released"):
        table.mapping_for(0)
    with pytest.raises(RuntimeError, match="released"):
        table.append_tokens()
    assert allocator.allocate("B", 2)[0].block_id == 0


def test_empty_table_release_and_invalid_positions_fail_explicitly():
    """输入空表与非法位置；输出明确错误；空请求 release 不触发 allocator 非法 free。"""
    allocator = BlockAllocator(num_blocks=2, block_size=4)
    table = BlockTable("A", allocator)

    with pytest.raises(IndexError):
        table.mapping_for(0)
    with pytest.raises(ValueError, match="cannot be empty"):
        build_batch_slot_mapping([])
    assert table.release() == ()
    assert allocator.num_free_blocks == 2
