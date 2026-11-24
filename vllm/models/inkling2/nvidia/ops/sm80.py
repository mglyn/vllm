# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling 2 Mini small-batch fast paths for SM80."""

import json
from typing import Any

import torch

from vllm import envs
from vllm.platforms import current_platform
from vllm.utils.platform_utils import get_device_name_as_file_name, num_compute_units

from .ll_gemm_sm120 import MAX_M, _ll_gemm


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
