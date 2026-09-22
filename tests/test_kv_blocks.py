from __future__ import annotations

import pytest

from miniserve.kv_blocks import (
    BlockAllocator,
    BlockCapacityError,
    LogicalBlock,
    PhysicalBlock,
)


def test_logical_block_tracks_slots_without_physical_mapping():
    """输入两次追加；输出连续块内 slots；验证逻辑容量不依赖 physical ID。"""
    block = LogicalBlock(block_index=2, block_size=4)

    assert list(block.append(3)) == [0, 1, 2]
    assert block.remaining_tokens == 1
    assert not block.is_full
    assert list(block.append()) == [3]
    assert block.is_full
    with pytest.raises(BlockCapacityError):
        block.append()


def test_allocator_assigns_stable_blocks_and_tracks_owners():
    """输入两个请求的分配；输出确定性 block ID；验证 free/allocated 完整分区。"""
    allocator = BlockAllocator(num_blocks=4, block_size=16)

    a = allocator.allocate("A", 2)
    b = allocator.allocate("B")

    assert a == (PhysicalBlock(0, 16), PhysicalBlock(1, 16))
    assert b == (PhysicalBlock(2, 16),)
    assert allocator.allocated_block_ids("A") == (0, 1)
    assert allocator.owner_of(2) == "B"
    assert allocator.free_block_ids == (3,)
    assert allocator.num_allocated_blocks + allocator.num_free_blocks == 4


def test_exhaustion_is_atomic_and_recovery_reuses_hole():
    """输入超额分配及中间释放；失败不改状态，释放的等长洞由下个请求复用。"""
    allocator = BlockAllocator(num_blocks=3, block_size=4)
    allocator.allocate("A")
    middle = allocator.allocate("B")
    allocator.allocate("C")

    with pytest.raises(BlockCapacityError, match="only 0"):
        allocator.allocate("D", 2)
    assert allocator.allocated_block_ids("D") == ()
    assert allocator.num_free_blocks == 0

    allocator.free(middle)
    assert allocator.allocate("D")[0].block_id == 1
    assert allocator.owner_of(1) == "D"


def test_release_request_reclaims_all_noncontiguous_blocks():
    """输入跨多次分配的同一 owner；输出全部 blocks；验证一次请求结束完整回收。"""
    allocator = BlockAllocator(num_blocks=5, block_size=8)
    allocator.allocate("A", 2)
    allocator.allocate("B")
    allocator.allocate("A", 2)

    released = allocator.release_request("A")

    assert tuple(block.block_id for block in released) == (0, 1, 3, 4)
    assert allocator.allocated_block_ids("A") == ()
    assert allocator.allocated_block_ids("B") == (2,)
    assert allocator.free_block_ids == (0, 1, 3, 4)


def test_invalid_free_is_atomic_and_double_free_is_rejected():
    """输入混合合法/非法和重复释放；输出异常；验证校验在任何状态修改之前完成。"""
    allocator = BlockAllocator(num_blocks=3, block_size=4)
    blocks = allocator.allocate("A", 2)

    with pytest.raises(ValueError, match="not allocated"):
        allocator.free([blocks[0], 2])
    assert allocator.allocated_block_ids("A") == (0, 1)

    allocator.free([blocks[0]])
    with pytest.raises(ValueError, match="not allocated"):
        allocator.free([blocks[0]])
    assert allocator.allocated_block_ids("A") == (1,)


@pytest.mark.parametrize("num_blocks,block_size", [(0, 4), (2, 0), (True, 4), (2, 1.5)])
def test_allocator_rejects_invalid_pool_shape(num_blocks, block_size):
    """输入非法池维度；输出异常；避免零容量、bool 或非整数破坏 allocator invariant。"""
    with pytest.raises(ValueError):
        BlockAllocator(num_blocks=num_blocks, block_size=block_size)


def test_request_release_and_foreign_physical_block_fail_fast():
    """输入未知 request 与池外 block；输出异常；避免静默隐藏资源生命周期错误。"""
    allocator = BlockAllocator(num_blocks=2, block_size=4)
    allocator.allocate("A")

    with pytest.raises(KeyError, match="no allocated blocks"):
        allocator.release_request("missing")
    with pytest.raises(ValueError, match="not allocated"):
        allocator.free([PhysicalBlock(9, 4)])
    assert allocator.allocated_block_ids("A") == (0,)
