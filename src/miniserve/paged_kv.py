"""教学版 paged KV storage 与 HF gather adapter；长期 KV 使用 physical block tensor。"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from transformers import DynamicCache

from miniserve.block_table import BlockTable
from miniserve.decode_batch import (
    DecodeBatchRunner,
    DecodeState,
    DecodeStepOutput,
    build_decode_attention_mask,
    cache_length,
    pack_dynamic_caches,
)
from miniserve.kv_blocks import BlockAllocator, BlockCapacityError
from miniserve.request import Request


@dataclass
class PagedDecodeState:
    """Paged decode 请求状态；Request 保存逻辑生命周期，BlockTable 保存历史 KV 地址。"""

    request: Request
    block_table: BlockTable


class PagedKVStorage:
    """所有请求共享的预分配 K/V block tensors；首次看到模型 cache 时确定 layout。"""

    def __init__(self, allocator: BlockAllocator) -> None:
        """输入 metadata allocator；无返回；创建尚未确定 heads/head_dim 的延迟 storage。"""
        self.allocator = allocator
        self._layers: list[tuple[torch.Tensor, torch.Tensor]] = []

    @property
    def is_initialized(self) -> bool:
        """输入自身；返回是否已经根据模型 cache 建立 tensor pool；只读。"""
        return bool(self._layers)

    @property
    def num_layers(self) -> int:
        """输入自身；返回 storage layer 数；初始化前为零。"""
        return len(self._layers)

    def initialize_from(self, cache: DynamicCache) -> None:
        """输入一次真实模型 cache；无返回；为每层一次性分配全部 physical K/V blocks。"""
        if self.is_initialized:
            self._validate_cache_layout(cache)
            return
        if not cache.layers:
            raise ValueError("Cannot initialize paged storage from an empty cache")
        for layer in cache.layers:
            if layer.keys.shape[0] != 1 or layer.values.shape != layer.keys.shape:
                raise ValueError(
                    "Paged storage expects single-request K/V with equal shapes"
                )
            _, num_heads, _, head_dim = layer.keys.shape
            shape = (
                self.allocator.num_blocks,
                num_heads,
                self.allocator.block_size,
                head_dim,
            )
            self._layers.append(
                (
                    torch.empty(
                        shape, dtype=layer.keys.dtype, device=layer.keys.device
                    ),
                    torch.empty(
                        shape, dtype=layer.values.dtype, device=layer.values.device
                    ),
                )
            )

    def write_cache(self, table: BlockTable, cache: DynamicCache) -> None:
        """输入已分配 table 与完整单请求 cache；无返回；按 logical blocks 写入离散 slots。"""
        self.initialize_from(cache)
        self._validate_cache_layout(cache)
        if cache_length(cache) != table.num_tokens:
            raise ValueError("Cache length must equal BlockTable token count")
        cursor = 0
        for logical, physical_id in zip(
            table.logical_blocks, table.physical_block_ids, strict=True
        ):
            count = logical.num_tokens
            for layer_index, layer in enumerate(cache.layers):
                keys, values = self._layers[layer_index]
                keys[physical_id, :, :count, :].copy_(
                    layer.keys[0, :, cursor : cursor + count, :]
                )
                values[physical_id, :, :count, :].copy_(
                    layer.values[0, :, cursor : cursor + count, :]
                )
            cursor += count

    def gather_cache(self, table: BlockTable, *, model_config=None) -> DynamicCache:
        """输入 request block table；返回连续单请求 DynamicCache；只读 storage 供 HF attention。"""
        if not self.is_initialized or table.num_tokens <= 0:
            raise RuntimeError("Paged storage and table must contain KV before gather")
        per_layer_data: list[tuple[torch.Tensor, torch.Tensor]] = []
        for keys, values in self._layers:
            key_parts: list[torch.Tensor] = []
            value_parts: list[torch.Tensor] = []
            for logical, physical_id in zip(
                table.logical_blocks, table.physical_block_ids, strict=True
            ):
                count = logical.num_tokens
                key_parts.append(keys[physical_id, :, :count, :])
                value_parts.append(values[physical_id, :, :count, :])
            per_layer_data.append(
                (
                    torch.cat(key_parts, dim=-2).unsqueeze(0),
                    torch.cat(value_parts, dim=-2).unsqueeze(0),
                )
            )
        return DynamicCache(ddp_cache_data=per_layer_data, config=model_config)

    def write_batched_last_token(
        self,
        table: BlockTable,
        batched_cache: DynamicCache,
        batch_index: int,
    ) -> None:
        """输入已追加一个 slot 的 table 和 forward cache row；无返回；只写本轮新增 K/V。"""
        mapping = table.mapping_for(table.num_tokens - 1)
        for layer_index, layer in enumerate(batched_cache.layers):
            keys, values = self._layers[layer_index]
            keys[mapping.physical_block_id, :, mapping.block_offset, :].copy_(
                layer.keys[batch_index, :, -1, :]
            )
            values[mapping.physical_block_id, :, mapping.block_offset, :].copy_(
                layer.values[batch_index, :, -1, :]
            )

    @torch.inference_mode()
    def clear_blocks(self, block_ids: tuple[int, ...]) -> None:
        """输入即将释放的 physical IDs；无返回；清零 K/V，避免复用时保留跨请求旧数据。"""
        if not self.is_initialized:
            return
        for keys, values in self._layers:
            for block_id in block_ids:
                # 单个 basic index 返回 view；不能使用 advanced-index 临时副本清零。
                keys[block_id].zero_()
                values[block_id].zero_()

    def _validate_cache_layout(self, cache: DynamicCache) -> None:
        """输入模型 cache；无返回；检查 layer/head/dtype/device 与已分配 storage 一致。"""
        if len(cache.layers) != self.num_layers:
            raise ValueError("Cache layer count differs from paged storage")
        for layer, (keys, values) in zip(cache.layers, self._layers, strict=True):
            expected = (keys.shape[1], keys.shape[3])
            actual = (layer.keys.shape[1], layer.keys.shape[3])
            if (
                layer.keys.shape[0] != 1
                or actual != expected
                or layer.keys.dtype != keys.dtype
                or layer.keys.device != keys.device
                or layer.values.dtype != values.dtype
            ):
                raise ValueError("Cache layout differs from paged storage")


class PagedDecodeBatchRunner(DecodeBatchRunner):
    """使用 block storage 保存长期 KV、通过 gather adapter 调用未修改 HF attention 的执行器。"""

    def __init__(
        self,
        *,
        model,
        device: torch.device,
        eos_token_ids: set[int],
        num_blocks: int,
        block_size: int,
        annotate_profiler: bool = False,
    ) -> None:
        """输入模型、设备和 KV 池容量；无返回；建立共享 allocator/storage，不分配请求 blocks。"""
        super().__init__(
            model=model,
            device=device,
            eos_token_ids=eos_token_ids,
            annotate_profiler=annotate_profiler,
        )
        self.allocator = BlockAllocator(num_blocks=num_blocks, block_size=block_size)
        self.storage = PagedKVStorage(self.allocator)

    @torch.inference_mode()
    def prefill_request(self, request: Request) -> PagedDecodeState:
        """输入 RUNNING/PREFILL request；返回 paged state；预检容量后执行 prefill 并写入 block pool。"""
        required_blocks = (
            request.prompt_length + self.allocator.block_size - 1
        ) // self.allocator.block_size
        if not self.allocator.can_allocate(required_blocks):
            raise BlockCapacityError("Insufficient KV blocks for request prefill")
        dynamic_state = super().prefill_request(request)
        table = BlockTable(request.request_id, self.allocator)
        table.append_tokens(cache_length(dynamic_state.cache))
        self.storage.write_cache(table, dynamic_state.cache)
        return PagedDecodeState(request=request, block_table=table)

    @torch.inference_mode()
    def decode_step(self, states: list[PagedDecodeState]) -> DecodeStepOutput:
        """输入 active paged states；返回每请求一个新 token；gather 历史，forward 后只写新增 KV。"""
        if not states:
            raise ValueError("decode_step requires at least one state")
        for state in states:
            request = state.request
            if (
                request.is_finished
                or not request.is_running
                or not request.needs_decode
            ):
                raise RuntimeError(
                    "Every paged decode request must be active in DECODE"
                )
            if request.sequence_length != state.block_table.num_tokens + 1:
                raise RuntimeError(
                    f"Paged decode invariant violated for {request.request_id}"
                )

        needed_blocks = sum(
            int(state.block_table.num_tokens % self.allocator.block_size == 0)
            for state in states
        )
        if needed_blocks and not self.allocator.can_allocate(needed_blocks):
            raise BlockCapacityError("Insufficient KV blocks for this decode batch")

        with self._profile_scope("miniserve::kv_gather"):
            transient_states = [
                DecodeState(
                    request=state.request,
                    cache=self.storage.gather_cache(
                        state.block_table, model_config=self.model.config
                    ),
                )
                for state in states
            ]
            batched_cache, logical_lengths, physical_length = pack_dynamic_caches(
                transient_states, model_config=self.model.config
            )

        input_ids = torch.tensor(
            [[state.request.generated_token_ids[-1]] for state in states],
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = build_decode_attention_mask(
            logical_lengths, physical_length=physical_length, device=self.device
        )
        position_ids = torch.tensor(
            logical_lengths, dtype=torch.long, device=self.device
        ).unsqueeze(1)
        cache_position = torch.tensor(
            [physical_length], dtype=torch.long, device=self.device
        )
        with self._profile_scope("miniserve::decode_model_forward"):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                past_key_values=batched_cache,
                use_cache=True,
            )

        with self._profile_scope("miniserve::kv_slot_write"):
            for batch_index, state in enumerate(states):
                state.block_table.append_tokens()
                self.storage.write_batched_last_token(
                    state.block_table, outputs.past_key_values, batch_index
                )

        produced_token_ids: list[int] = torch.argmax(
            outputs.logits[:, -1, :], dim=-1
        ).tolist()
        timestamp = time.perf_counter()
        for state, token_id in zip(states, produced_token_ids, strict=True):
            state.request.append_generated_token(token_id, timestamp=timestamp)
            if token_id in self.eos_token_ids or state.request.reached_max_new_tokens:
                state.request.mark_finished(timestamp=timestamp)
            elif state.request.sequence_length != state.block_table.num_tokens + 1:
                raise RuntimeError("Paged decode invariant failed after token update")
        return DecodeStepOutput(
            request_ids=tuple(state.request.request_id for state in states),
            token_ids=tuple(produced_token_ids),
            logical_cache_lengths_before=tuple(logical_lengths),
            physical_cache_length=physical_length,
            input_shape=tuple(input_ids.shape),
        )

    @torch.inference_mode()
    def release_state(self, state: PagedDecodeState) -> None:
        """输入完成请求的 paged state；无返回；清零并归还其全部 physical blocks。"""
        block_ids = state.block_table.physical_block_ids
        self.storage.clear_blocks(block_ids)
        state.block_table.release()
