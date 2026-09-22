"""固定大小 KV block 的元数据与 allocator；Step 25 暂不分配真实 K/V tensor。"""

from __future__ import annotations

import heapq
from collections.abc import Iterable
from dataclasses import dataclass


class BlockCapacityError(RuntimeError):
    """请求的 physical block 超过当前 free list 时抛出，allocator 状态保持不变。"""


@dataclass(frozen=True, order=True)
class PhysicalBlock:
    """一个固定容量的物理槽；ID 在池生命周期内稳定，所有权由 allocator 单独维护。"""

    block_id: int
    block_size: int

    def __post_init__(self) -> None:
        """输入构造字段；无返回；验证物理 ID 和 token 容量，实例保持不可变。"""
        if isinstance(self.block_id, bool) or not isinstance(self.block_id, int):
            raise TypeError("block_id must be an integer")
        if self.block_id < 0:
            raise ValueError("block_id must be non-negative")
        if (
            isinstance(self.block_size, bool)
            or not isinstance(self.block_size, int)
            or self.block_size <= 0
        ):
            raise ValueError("block_size must be a positive integer")


@dataclass
class LogicalBlock:
    """请求序列中的逻辑块；只记录 token 使用量，不决定映射到哪个 physical block。"""

    block_index: int
    block_size: int
    num_tokens: int = 0

    def __post_init__(self) -> None:
        """输入构造字段；无返回；验证逻辑顺序和当前占用不超过固定容量。"""
        if (
            isinstance(self.block_index, bool)
            or not isinstance(self.block_index, int)
            or self.block_index < 0
        ):
            raise ValueError("block_index must be a non-negative integer")
        if (
            isinstance(self.block_size, bool)
            or not isinstance(self.block_size, int)
            or self.block_size <= 0
        ):
            raise ValueError("block_size must be a positive integer")
        if (
            isinstance(self.num_tokens, bool)
            or not isinstance(self.num_tokens, int)
            or not 0 <= self.num_tokens <= self.block_size
        ):
            raise ValueError("num_tokens must be between zero and block_size")

    @property
    def remaining_tokens(self) -> int:
        """输入自身；返回剩余 token 槽位；只读，供下一课的 block table 追加逻辑位置。"""
        return self.block_size - self.num_tokens

    @property
    def is_full(self) -> bool:
        """输入自身；返回逻辑块是否填满；只读，不涉及 physical block 所有权。"""
        return self.num_tokens == self.block_size

    def append(self, num_tokens: int = 1) -> range:
        """输入追加 token 数；返回块内 slot 范围；更新占用并拒绝跨越当前 block。"""
        if (
            isinstance(num_tokens, bool)
            or not isinstance(num_tokens, int)
            or num_tokens <= 0
        ):
            raise ValueError("num_tokens must be a positive integer")
        if num_tokens > self.remaining_tokens:
            raise BlockCapacityError("Logical block does not have enough free token slots")
        start = self.num_tokens
        self.num_tokens += num_tokens
        return range(start, self.num_tokens)


class BlockAllocator:
    """固定大小 physical block 池；以最小 ID free heap 实现确定性分配与洞复用。"""

    def __init__(self, *, num_blocks: int, block_size: int) -> None:
        """输入池大小和每块 token 数；无返回；创建全空 free list 与稳定 block 元数据。"""
        if (
            isinstance(num_blocks, bool)
            or not isinstance(num_blocks, int)
            or num_blocks <= 0
        ):
            raise ValueError("num_blocks must be a positive integer")
        if (
            isinstance(block_size, bool)
            or not isinstance(block_size, int)
            or block_size <= 0
        ):
            raise ValueError("block_size must be a positive integer")
        self._blocks = tuple(
            PhysicalBlock(block_id=block_id, block_size=block_size)
            for block_id in range(num_blocks)
        )
        self._free_block_ids = list(range(num_blocks))
        heapq.heapify(self._free_block_ids)
        self._owners: dict[int, str] = {}
        self._assert_invariants()

    @property
    def block_size(self) -> int:
        """输入自身；返回每个 block 的 token 容量；只读，整个池固定不变。"""
        return self._blocks[0].block_size

    @property
    def num_blocks(self) -> int:
        """输入自身；返回池的 physical block 总数；只读，供容量规划。"""
        return len(self._blocks)

    @property
    def num_free_blocks(self) -> int:
        """输入自身；返回当前 free list 大小；只读，反映可立即分配容量。"""
        return len(self._free_block_ids)

    @property
    def num_allocated_blocks(self) -> int:
        """输入自身；返回已有 owner 的 block 数；只读，与 free 数之和必须等于总数。"""
        return len(self._owners)

    @property
    def free_block_ids(self) -> tuple[int, ...]:
        """输入自身；返回排序后的 free ID 快照；不暴露内部 heap，避免外部修改状态。"""
        return tuple(sorted(self._free_block_ids))

    def can_allocate(self, num_blocks: int) -> bool:
        """输入请求 block 数；返回当前容量是否足够；只读，非法数量直接拒绝。"""
        self._validate_allocation_size(num_blocks)
        return num_blocks <= self.num_free_blocks

    def allocate(self, request_id: str, num_blocks: int = 1) -> tuple[PhysicalBlock, ...]:
        """输入 owner 与数量；返回新 blocks；原子更新 free/owner，容量不足时不改状态。"""
        self._validate_request_id(request_id)
        self._validate_allocation_size(num_blocks)
        if not self.can_allocate(num_blocks):
            raise BlockCapacityError(
                f"Requested {num_blocks} blocks, only {self.num_free_blocks} are free"
            )

        block_ids = [heapq.heappop(self._free_block_ids) for _ in range(num_blocks)]
        for block_id in block_ids:
            self._owners[block_id] = request_id
        self._assert_invariants()
        return tuple(self._blocks[block_id] for block_id in block_ids)

    def free(self, blocks: Iterable[PhysicalBlock | int]) -> None:
        """输入 block 或 ID 集合；无返回；整批校验后原子释放，拒绝重复或外部 block。"""
        block_ids = tuple(
            block.block_id if isinstance(block, PhysicalBlock) else block
            for block in blocks
        )
        if not block_ids:
            raise ValueError("At least one block must be freed")
        if any(isinstance(block_id, bool) or not isinstance(block_id, int) for block_id in block_ids):
            raise TypeError("Every block ID must be an integer")
        if len(set(block_ids)) != len(block_ids):
            raise ValueError("A block cannot appear twice in one free operation")
        invalid = [
            block_id
            for block_id in block_ids
            if block_id < 0 or block_id >= self.num_blocks or block_id not in self._owners
        ]
        if invalid:
            raise ValueError(f"Blocks are not allocated by this allocator: {invalid}")

        for block_id in block_ids:
            del self._owners[block_id]
            heapq.heappush(self._free_block_ids, block_id)
        self._assert_invariants()

    def release_request(self, request_id: str) -> tuple[PhysicalBlock, ...]:
        """输入 request ID；返回被回收 blocks；释放该 owner 全部资源，未知 owner 明确失败。"""
        self._validate_request_id(request_id)
        block_ids = tuple(
            sorted(block_id for block_id, owner in self._owners.items() if owner == request_id)
        )
        if not block_ids:
            raise KeyError(f"Request has no allocated blocks: {request_id}")
        blocks = tuple(self._blocks[block_id] for block_id in block_ids)
        self.free(block_ids)
        return blocks

    def allocated_block_ids(self, request_id: str) -> tuple[int, ...]:
        """输入 request ID；返回其 block ID 排序快照；只读，未知请求返回空 tuple。"""
        self._validate_request_id(request_id)
        return tuple(
            sorted(block_id for block_id, owner in self._owners.items() if owner == request_id)
        )

    def owner_of(self, block_id: int) -> str | None:
        """输入池内 block ID；返回 owner 或 None；只读，池外 ID 明确失败。"""
        if (
            isinstance(block_id, bool)
            or not isinstance(block_id, int)
            or not 0 <= block_id < self.num_blocks
        ):
            raise ValueError("block_id is outside this allocator")
        return self._owners.get(block_id)

    @staticmethod
    def _validate_request_id(request_id: str) -> None:
        """输入 request ID；无返回；无状态，拒绝空值以防资源变成匿名 owner。"""
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a non-empty string")

    @staticmethod
    def _validate_allocation_size(num_blocks: int) -> None:
        """输入分配数量；无返回；无状态，只接受正整数且拒绝 bool。"""
        if (
            isinstance(num_blocks, bool)
            or not isinstance(num_blocks, int)
            or num_blocks <= 0
        ):
            raise ValueError("num_blocks must be a positive integer")

    def _assert_invariants(self) -> None:
        """输入自身；无返回；检查 free/allocated 完整分区，发现内部损坏立即失败。"""
        free_ids = set(self._free_block_ids)
        allocated_ids = set(self._owners)
        all_ids = set(range(self.num_blocks))
        if (
            len(free_ids) != len(self._free_block_ids)
            or free_ids & allocated_ids
            or free_ids | allocated_ids != all_ids
        ):
            raise RuntimeError("Block allocator invariant violated")
