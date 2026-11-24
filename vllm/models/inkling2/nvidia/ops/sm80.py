# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling 2 Mini small-batch fast paths for SM80."""

import json
from typing import Any

import torch

from vllm import envs
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import get_device_name_as_file_name, num_compute_units

from .ll_gemm_sm120 import MAX_M, _ll_gemm
from .moe import _maximum


def _is_full_a100_pcie(device_id: int) -> bool:
    return (
        get_device_name_as_file_name(device_id) == "NVIDIA_A100_80GB_PCIe"
        and num_compute_units(device_id) == 108
    )


def use_loop_down(pair: torch.Tensor, down: torch.Tensor) -> bool:
    return (
        down.shape == (3, 64, 4608)
        and 1 < pair.shape[1] <= MAX_M
        and not envs.VLLM_BATCH_INVARIANT
        and _is_full_a100_pcie(pair.get_device())
    )


def sm80_bf16_loop_down(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    out = x.new_empty((*x.shape[:2], weight.shape[1]))
    _ll_gemm(x, weight, None, out, enable_pdl=False)
    return out


def use_grouped_qkv(
    x_q: torch.Tensor, x_k: torch.Tensor, x_v: torch.Tensor, w: torch.Tensor
) -> bool:
    return (
        w.shape in ((8600, 2304), (4304, 2304))
        and x_q.shape[1] == w.shape[1]
        and x_q.device == x_k.device == x_v.device == w.device
        and not envs.VLLM_BATCH_INVARIANT
        and current_platform.is_device_capability(80, device_id=x_q.get_device())
    )


def use_three_stage_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    max_query_len: int,
    num_splits: int,
    window: int,
    extent: int,
) -> bool:
    page, num_kv_heads = key_cache.shape[1:3]
    return (
        max_query_len == 1
        and num_splits == 3
        and window == extent == 1024
        and q.shape[2] == 128
        and q.shape[1] // num_kv_heads == 3
        and num_kv_heads in (4, 8)
        and page == 64
        and q.dtype == key_cache.dtype == value_cache.dtype == torch.bfloat16
        and _is_full_a100_pcie(q.get_device())
    )


@triton.jit
def _align_small_routing(
    Ids,
    SortedIds,
    ExpertIds,
    TotalPadded,
    SLOTS: tl.constexpr,
    EXPERTS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_SLOTS: tl.constexpr,
):
    pos = tl.arange(0, BLOCK_SLOTS)
    ids = tl.load(Ids + pos, pos < SLOTS, -1)
    ids = tl.where((ids >= 0) & (ids < EXPERTS), ids, EXPERTS)
    key = tl.sort(ids * BLOCK_SLOTS + pos, descending=False)
    expert = key // BLOCK_SLOTS
    valid = expert < EXPERTS
    prev = tl.gather(expert, tl.maximum(pos - 1, 0), axis=0)
    next_expert = tl.gather(expert, tl.minimum(pos + 1, BLOCK_SLOTS - 1), axis=0)
    first = valid & ((pos == 0) | (prev != expert))
    last = valid & ((pos == BLOCK_SLOTS - 1) | (next_expert != expert))
    blocks = tl.cumsum(first.to(tl.int32), 0) - 1
    start = tl.associative_scan(tl.where(first, pos, 0), 0, _maximum)
    slot = pos - start
    count = tl.sum(first.to(tl.int32), 0)

    tl.store(TotalPadded, count * BLOCK_M)
    tl.store(ExpertIds + blocks, expert, first)
    tl.store(ExpertIds + pos, -1, (pos < SLOTS) & (pos >= count))
    tl.store(SortedIds + blocks * BLOCK_M + slot, key % BLOCK_SLOTS, valid)
    pad = tl.arange(0, BLOCK_M)
    tl.store(
        SortedIds + blocks[:, None] * BLOCK_M + pad[None, :],
        SLOTS,
        last[:, None] & (pad[None, :] > slot[:, None]),
    )
    remaining = tl.arange(0, BLOCK_SLOTS * BLOCK_M)
    tl.store(
        SortedIds + remaining,
        SLOTS,
        (remaining >= count * BLOCK_M) & (remaining < SLOTS * BLOCK_M),
    )


def _align_indexed_experts(
    ids: torch.Tensor, block_size: int, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens, topk = ids.shape
    # select_experts returns distinct experts per token, so each expert fits
    # into one block when the number of tokens is at most the block size.
    if (
        0 < tokens <= block_size == 8
        and topk == 8
        and num_experts in (64, 128)
        and ids.dtype == torch.int32
        and ids.is_contiguous()
        and ids.is_cuda
        and current_platform.is_device_capability(80, device_id=ids.get_device())
        and not envs.VLLM_BATCH_INVARIANT
    ):
        slots = tokens * topk
        sorted_ids = ids.new_empty((slots * block_size,))
        expert_ids = ids.new_empty((slots,))
        total_padded = ids.new_empty((1,))
        _align_small_routing[(1,)](
            ids,
            sorted_ids,
            expert_ids,
            total_padded,
            slots,
            num_experts,
            block_size,
            triton.next_power_of_2(slots),
            num_warps=1,
        )
        return sorted_ids, expert_ids, total_padded
    return moe_align_block_size(
        ids, block_size, num_experts, ignore_invalid_experts=True
    )


def humming_decode_tuning(
    weight: torch.Tensor,
    shapes: tuple[tuple[int, int], tuple[int, int]],
    nvfp4: bool,
    bf16_activations: bool,
    tuning: list[tuple[int, int, dict[str, Any]]],
) -> dict[int, str]:
    num_experts = weight.shape[0]
    if not (
        nvfp4
        and bf16_activations
        and shapes == ((2304, 2304), (2304, 1152))
        and num_experts in (64, 128)
        and weight.is_cuda
        and not envs.VLLM_BATCH_INVARIANT
        and current_platform.is_device_capability(80, device_id=weight.get_device())
    ):
        return {}

    # Two CTAs hide the decode weight-load latency at 5–8 tokens on full A100.
    # Keep the stock block size and K reduction.
    result = {}
    for tokens in range(5, 9):
        rows = tokens * 8 * num_experts // 128
        for lower, upper, schedule in tuning:
            if (
                lower < rows <= upper
                and tuple(schedule["block_shape"]) in ((8, 256, 64), (8, 256, 128))
                and schedule.get("num_ctas_per_sm") == 1
                and schedule.get("num_sms") == 108
                and schedule.get("num_stages") in (4, 5)
            ):
                result[rows] = json.dumps(
                    dict(schedule, num_ctas_per_sm=2, num_stages=4)
                )
                break
    return result
