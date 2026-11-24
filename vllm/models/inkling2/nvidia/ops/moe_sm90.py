# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 routed experts on Hopper: Humming expert GEMMs (bf16
activations, NVFP4 / MXFP8 weights) in place of the TRT-LLM-gen ones.

Rows stay in the flat (token * topk + slot) order: the gate/up GEMM
gathers each row's token, the down GEMM reads and writes row by row, and
`ops.moe.combine` sums them with the identity `src2dst` (-1: not local).
"""

import json
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.triton_utils import tl, triton

from .activation import _situ
from .linear import k_group_partials
from .moe import _load_bias, _local_experts, _maximum, _select_experts

if TYPE_CHECKING:
    from vllm.utils.humming import HummingInputSchema

# Row block size when no Humming tuning range covers the row count.
_DEFAULT_BLOCK_M = 64


@triton.jit
def _select_kernel(
    Partials,
    Bias,
    ExpertBounds,
    IsPadding,
    Ids,
    Weights,
    Src2Dst,
    T,
    route_scale,
    E: tl.constexpr,
    E_POW2: tl.constexpr,
    GROUPS: tl.constexpr,
    TOPK: tl.constexpr,
    SORT: tl.constexpr,
):
    """Local expert ids (-1: not local), routing weights and identity
    `src2dst` of the flat rows of one token."""
    token = tl.program_id(0) + tl.arange(0, 1)
    bias = _load_bias(Bias, E, E_POW2)
    ids, weights = _select_experts(
        Partials, bias, token, T, route_scale, E, E_POW2, GROUPS, TOPK, SORT
    )
    padding = None
    if IsPadding is not None:
        padding = tl.load(IsPadding + token)
    local = _local_experts(
        ids, tl.load(ExpertBounds), tl.load(ExpertBounds + 1), padding
    )
    rows = token[:, None] * TOPK + tl.arange(0, TOPK)[None, :]
    tl.store(Ids + rows, local)
    tl.store(Weights + rows, weights)
    tl.store(Src2Dst + rows, tl.where(local >= 0, rows, -1))


def select_experts(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    route_scale: float,
    expert_bounds: torch.Tensor,
    is_padding: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The router and top-k of `ops.moe.route`, without its layout.

    Returns the local expert id of each token's slots `[T, topk]` (-1: not
    one of the local experts `expert_bounds`, or a padding token), their
    routing weights `[T * topk]` and the identity `src2dst` `[T * topk]`
    of the local rows (-1 elsewhere).
    """
    tokens = x.shape[0]
    experts = gate_weight.shape[0]
    partials = k_group_partials(x, gate_weight)
    ids = torch.empty((tokens, topk), dtype=torch.int32, device=x.device)
    weights = torch.empty(tokens * topk, dtype=torch.float32, device=x.device)
    src2dst = torch.empty(tokens * topk, dtype=torch.int32, device=x.device)
    _select_kernel[(tokens,)](
        partials,
        bias,
        expert_bounds,
        is_padding,
        ids,
        weights,
        src2dst,
        tokens,
        route_scale,
        E=experts,
        E_POW2=triton.next_power_of_2(experts),
        GROUPS=partials.shape[0],
        TOPK=topk,
        SORT=tokens <= 128,
        num_warps=4 if tokens <= 256 else 1,
    )
    return ids, weights, src2dst


@triton.jit
def _swiglu_kernel(
    GateUp,
    Weights,
    Src2Dst,
    Out,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    GATE_CAP: tl.constexpr,
    UP_CAP: tl.constexpr,
):
    """bf16 softcapped SwiGLU of a local row times its routing weight."""
    row = tl.program_id(0)
    if tl.load(Src2Dst + row) < 0:
        return
    start = tl.program_id(1) * BLOCK
    gamma = tl.load(Weights + row)
    values = _situ(GateUp, row, start, gamma, N, BLOCK, GATE_CAP, UP_CAP)
    tl.store(Out + row * N + start + tl.arange(0, BLOCK), values.to(tl.bfloat16))


def swiglu(
    gate_up: torch.Tensor,
    weights: torch.Tensor,
    src2dst: torch.Tensor,
    gate_cap: float,
    up_cap: float,
) -> torch.Tensor:
    """`activation.swiglu_quantize`'s SiTU of the interleaved (gate, up)
    columns times the routing weights, in bf16, for the local rows (the
    others are left unset)."""
    rows, n = gate_up.shape[0], gate_up.shape[1] // 2
    block = 256 if n % 256 == 0 else 128
    assert n % block == 0
    out = torch.empty((rows, n), dtype=torch.bfloat16, device=gate_up.device)
    _swiglu_kernel[(rows, n // block)](
        gate_up,
        weights,
        src2dst,
        out,
        N=n,
        BLOCK=block,
        GATE_CAP=gate_cap,
        UP_CAP=up_cap,
        num_warps=2,
    )
    return out


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


class HummingExperts:
    """Humming tensors and kernel configs of the two expert projections.

    Args:
        weights: Per projection (`w13`, `w2`), the row-major checkpoint
            weights `[E, N, K / pack]`, block scales `[E, N, K / group]`
            and, for NVFP4, the per-expert global scales `[E]` (else None).
        shapes: `(N, K)` of each projection.
        nvfp4: NVFP4 (else MXFP8) weights.
        input_schema: Quantized input contract, or None for BF16 activations.

    """

    def __init__(
        self,
        weights: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]],
        shapes: tuple[tuple[int, int], tuple[int, int]],
        nvfp4: bool,
        input_schema: "HummingInputSchema | None" = None,
    ) -> None:
        from vllm.utils.humming import (
            GemmType,
            HummingInputSchema,
            ModeloptMxfp8WeightSchema,
            ModeloptNvfp4WeightSchema,
            get_heuristics_config,
            prepare_and_transform_humming_tensors,
        )

        self.configs = {}
        self.tensors: dict[str, dict[str, torch.Tensor]] = {}
        self.tunings: dict[str, list] = {}
        self.tuning_strs: dict[str, str] = {}
        for name, (n, k) in zip(("w13", "w2"), shapes):
            weight, scale, global_scale = weights[name]
            num_experts = weight.shape[0]
            tensors = {"weight": weight, "weight_scale": scale}
            if nvfp4:
                source = ModeloptNvfp4WeightSchema()
                tensors["weight_scale_2"] = global_scale
            else:
                source = ModeloptMxfp8WeightSchema()
            schema, tensors = source.convert_humming(
                tensors=tensors,
                shape_n_stacks=[n],
                shape_k_stacks=[k],
                param_dtype=torch.bfloat16,
                num_experts=num_experts,
            )
            config, tensors = prepare_and_transform_humming_tensors(
                tensors=tensors,
                shape_n=n,
                shape_k=k,
                pad_n_to_multiple=256,
                pad_k_to_multiple=128,
                input_schema=input_schema or HummingInputSchema(),
                weight_schema=schema,
                num_experts=num_experts,
                torch_dtype=torch.bfloat16,
                device=weight.device,
            )
            assert config.shape_n == n and config.shape_k == k, (n, k)
            tuning = get_heuristics_config(
                layer_config=config,
                use_f16_accum=False,
                use_batch_invariant=envs.VLLM_BATCH_INVARIANT,
                gemm_type=GemmType.INDEXED,
            )
            # Stream-K adds the K slices into the output with bf16 atomics,
            # whose order (and so the result) changes from run to run.
            for _, _, kernel_config in tuning:
                kernel_config["use_stream_k"] = False
            self.configs[name] = config
            self.tensors[name] = tensors
            self.tunings[name] = tuning
            self.tuning_strs[name] = json.dumps(tuning)
        self.num_experts = num_experts
        self.compute_config = json.dumps(
            {
                "use_batch_invariant": envs.VLLM_BATCH_INVARIANT,
                "use_f16_accum": False,
                "gemm_type": GemmType.INDEXED.value,
            }
        )
        self.locks = torch.zeros(2048, dtype=torch.int32, device=weight.device)

    def _block_m(self, name: str, rows: int) -> int:
        for lower, upper, config in self.tunings[name]:
            if lower < rows <= upper:
                return config["block_shape"][0]
        return _DEFAULT_BLOCK_M

    def _gemm(
        self,
        name: str,
        inputs: torch.Tensor,
        rows: int,
        alignment: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        top_k: int,
        valid_rows: int,
    ) -> torch.Tensor:
        from vllm.utils.humming import humming_forward

        config = self.configs[name]
        tensors = self.tensors[name]
        out = torch.empty(
            (rows, config.shape_n), dtype=torch.bfloat16, device=inputs.device
        )
        sorted_ids, expert_ids, num_tokens_padded = alignment
        return humming_forward(
            config,
            inputs=inputs,
            weight=tensors["weight"],
            weight_scale=tensors.get("weight_scale"),
            weight_scale_2=tensors.get("weight_scale_2"),
            outputs=out,
            locks=self.locks,
            sorted_ids=sorted_ids,
            expert_ids=expert_ids,
            num_tokens_padded=num_tokens_padded,
            top_k=top_k,
            valid_shape_m=valid_rows,
            compute_config=self.compute_config,
            tuning_config=self.tuning_strs[name],
        )

    def __call__(
        self,
        x: torch.Tensor,
        ids: torch.Tensor,
        weights: torch.Tensor,
        src2dst: torch.Tensor,
        num_global_experts: int,
        gate_cap: float,
        up_cap: float,
    ) -> torch.Tensor:
        """bf16 `[T * topk, H]` down-projection rows of the local slots of
        `ids` (`select_experts`); the other rows are left unset."""
        tokens, topk = ids.shape
        rows = tokens * topk
        # Expected local rows, which pick the kernel configs.
        valid_rows = -(-rows * self.num_experts // num_global_experts)
        block_m = {name: self._block_m(name, valid_rows) for name in ("w13", "w2")}
        alignments = {
            size: _align_indexed_experts(ids, size, self.num_experts)
            for size in set(block_m.values())
        }
        gate_up = self._gemm(
            "w13", x, rows, alignments[block_m["w13"]], topk, valid_rows
        )
        activated = swiglu(gate_up, weights, src2dst, gate_cap, up_cap)
        del gate_up
        return self._gemm(
            "w2", activated, rows, alignments[block_m["w2"]], 1, valid_rows
        )
