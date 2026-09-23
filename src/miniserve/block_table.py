"""请求级 BlockTable、逻辑位置到物理 slot 的映射与 batch metadata。"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from miniserve.kv_blocks import BlockAllocator, LogicalBlock, PhysicalBlock


@dataclass(frozen=True)
class SlotMapping:
    """一个逻辑 token 的完整地址；字段不可变，可安全传给后续 KV executor。"""

    request_id: str
    logical_position: int
    logical_block_index: int
    block_offset: int
    physical_block_id: int
    physical_slot: int


@dataclass(frozen=True)
class BatchSlotMapping:
    """一批 token 地址的结构化快照；保持调用顺序，不引用可变 BlockTable。"""

    entries: tuple[SlotMapping, ...]

    @property
    def request_ids(self) -> tuple[str, ...]:
        """输入自身；返回 batch row 对应 request IDs；只读，顺序与 entries 一致。"""
        return tuple(entry.request_id for entry in self.entries)

    @property
    def logical_positions(self) -> tuple[int, ...]:
        """输入自身；返回每个 batch row 的逻辑位置；只读，供 position metadata 使用。"""
        return tuple(entry.logical_position for entry in self.entries)

    @property
    def physical_slots(self) -> tuple[int, ...]:
        """输入自身；返回扁平 physical slots；只读，供下一课构造 tensor index。"""
        return tuple(entry.physical_slot for entry in self.entries)


class BlockTable:
    """单个请求的 logical block → physical block 映射及 token 使用进度。"""

    def __init__(self, request_id: str, allocator: BlockAllocator) -> None:
        """输入 request ID 与共享 allocator；无返回；创建空表但暂不占用 physical block。"""
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a non-empty string")
        if not isinstance(allocator, BlockAllocator):
            raise TypeError("allocator must be a BlockAllocator")
        self.request_id = request_id
        self._allocator = allocator
        self._logical_blocks: list[LogicalBlock] = []
        self._physical_blocks: list[PhysicalBlock] = []
        self._num_tokens = 0
        self._released = False

    @property
    def block_size(self) -> int:
        """输入自身；返回共享 allocator 的 block token 容量；只读且生命周期内固定。"""
        return self._allocator.block_size

    @property
    def num_tokens(self) -> int:
        """输入自身；返回已映射逻辑 token 数；只读，等于所有 logical block 占用之和。"""
        return self._num_tokens

    @property
    def num_blocks(self) -> int:
        """输入自身；返回 logical/physical 映射条目数；只读，两侧长度始终相等。"""
        return len(self._physical_blocks)

    @property
    def is_released(self) -> bool:
        """输入自身；返回资源是否已结束；只读，released table 不允许重新追加。"""
        return self._released

    @property
    def physical_block_ids(self) -> tuple[int, ...]:
        """输入自身；返回按 logical block 顺序排列的 physical IDs；只读映射快照。"""
        return tuple(block.block_id for block in self._physical_blocks)

    @property
    def logical_blocks(self) -> tuple[LogicalBlock, ...]:
        """输入自身；返回 logical block 副本；避免调用方修改 table 内部 token 计数。"""
        return tuple(
            LogicalBlock(block.block_index, block.block_size, block.num_tokens)
            for block in self._logical_blocks
        )

    def append_tokens(self, num_tokens: int = 1) -> tuple[SlotMapping, ...]:
        """输入新增 token 数；返回新增地址；需要时原子申请 blocks，再一次性提交 table 状态。"""
        self._ensure_active()
        if (
            isinstance(num_tokens, bool)
            or not isinstance(num_tokens, int)
            or num_tokens <= 0
        ):
            raise ValueError("num_tokens must be a positive integer")

        old_total = self._num_tokens
        new_total = old_total + num_tokens
        required_blocks = (new_total + self.block_size - 1) // self.block_size
        additional_blocks = required_blocks - self.num_blocks

        # allocator.allocate() 在容量不足时保证原子失败，因此 table 在这里仍未改变。
        allocated = (
            self._allocator.allocate(self.request_id, additional_blocks)
            if additional_blocks
            else ()
        )
        new_physical_blocks = [*self._physical_blocks, *allocated]
        new_logical_blocks = [
            LogicalBlock(
                block_index=index,
                block_size=self.block_size,
                num_tokens=min(
                    self.block_size,
                    max(0, new_total - index * self.block_size),
                ),
            )
            for index in range(required_blocks)
        ]

        # 上面的所有校验/分配成功后才提交三份关联状态，调用方不会看到半张表。
        self._physical_blocks = new_physical_blocks
        self._logical_blocks = new_logical_blocks
        self._num_tokens = new_total
        self._assert_invariants()
        return tuple(self.mapping_for(position) for position in range(old_total, new_total))

    def mapping_for(self, logical_position: int) -> SlotMapping:
        """输入已有逻辑 token position；返回 physical 地址；只读并拒绝越界或已释放表。"""
        self._ensure_active()
        if (
            isinstance(logical_position, bool)
            or not isinstance(logical_position, int)
            or not 0 <= logical_position < self.num_tokens
        ):
            raise IndexError("logical_position is outside the allocated sequence")
        logical_block_index, block_offset = divmod(logical_position, self.block_size)
        physical_block_id = self._physical_blocks[logical_block_index].block_id
        return SlotMapping(
            request_id=self.request_id,
            logical_position=logical_position,
            logical_block_index=logical_block_index,
            block_offset=block_offset,
            physical_block_id=physical_block_id,
            physical_slot=physical_block_id * self.block_size + block_offset,
        )

    def mappings(self) -> tuple[SlotMapping, ...]:
        """输入自身；返回全部 token 地址快照；只读，按逻辑 position 递增排列。"""
        self._ensure_active()
        return tuple(self.mapping_for(position) for position in range(self.num_tokens))

    def release(self) -> tuple[PhysicalBlock, ...]:
        """输入自身；返回已释放 physical blocks；清空映射并永久关闭 table，防止重用旧地址。"""
        self._ensure_active()
        released = tuple(self._physical_blocks)
        if released:
            self._allocator.free(released)
        self._physical_blocks.clear()
        self._logical_blocks.clear()
        self._num_tokens = 0
        self._released = True
        self._assert_invariants()
        return released

    def _ensure_active(self) -> None:
        """输入自身；无返回；拒绝 release 后读取或追加，避免使用已被复用的 physical 地址。"""
        if self._released:
            raise RuntimeError(f"BlockTable has been released: {self.request_id}")

    def _assert_invariants(self) -> None:
        """输入自身；无返回；检查表长、占用、owner 和末块规则，发现损坏立即失败。"""
        if self._released:
            if self._logical_blocks or self._physical_blocks or self._num_tokens:
                raise RuntimeError("Released BlockTable still owns state")
            return
        if len(self._logical_blocks) != len(self._physical_blocks):
            raise RuntimeError("Logical and physical block counts differ")
        if sum(block.num_tokens for block in self._logical_blocks) != self.num_tokens:
            raise RuntimeError("Logical block token counts differ from table total")
        for index, (logical, physical) in enumerate(
            zip(self._logical_blocks, self._physical_blocks, strict=True)
        ):
            if logical.block_index != index:
                raise RuntimeError("Logical block indices are not contiguous")
            if index < self.num_blocks - 1 and not logical.is_full:
                raise RuntimeError("Only the final logical block may be partially filled")
            if self._allocator.owner_of(physical.block_id) != self.request_id:
                raise RuntimeError("Physical block owner differs from BlockTable request")


def build_batch_slot_mapping(
    selections: Iterable[tuple[BlockTable, int]],
) -> BatchSlotMapping:
    """输入有序 table/position 对；返回 batch 地址快照；不修改 table 或 allocator。"""
    entries = tuple(table.mapping_for(position) for table, position in selections)
    if not entries:
        raise ValueError("Batch slot mapping cannot be empty")
    return BatchSlotMapping(entries)
