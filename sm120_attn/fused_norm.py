"""Fused Gemma-style RMSNorm (x * (1 + w)) with optional residual add, in one Triton kernel.

vLLM 0.30.0 runs Qwen3.5's `GemmaRMSNorm.forward_cuda` through `ir.ops.rms_norm` / `fused_add_rms_norm`, which
are plain torch ops (fp32 cast, pow, mean, rsqrt, mul, mul, cast: 6-10 launches and several HBM round trips per
norm). This kernel reproduces those semantics exactly (fp32 math, residual returned as bf16 of the fp32 sum,
weight applied in fp32 as (w + 1)) in a single memory pass per row.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gemma_rms_norm_kernel(x_ptr, res_ptr, w_ptr, out_ptr, res_out_ptr, N, eps,
                           stride_x, stride_res, stride_out, stride_res_out,
                           HAS_RES: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < N
    x = tl.load(x_ptr + row * stride_x + cols, mask=m, other=0.0).to(tl.float32)
    if HAS_RES:
        r = tl.load(res_ptr + row * stride_res + cols, mask=m, other=0.0).to(tl.float32)
        x = x + r
        tl.store(res_out_ptr + row * stride_res_out + cols, x.to(res_out_ptr.dtype.element_ty), mask=m)
    var = tl.sum(x * x, 0) / N
    y = x * tl.rsqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32) + 1.0
    tl.store(out_ptr + row * stride_out + cols, (y * w).to(out_ptr.dtype.element_ty), mask=m)


def gemma_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float, residual: torch.Tensor | None = None):
    """x [..., N] (bf16/fp16/fp32), weight [N] (the raw Gemma weight, +1 applied here), optional residual [..., N].
    Returns out, or (out, new_residual) when residual is given; same dtypes as vLLM's torch implementation."""
    shape = x.shape
    N = shape[-1]
    x2 = x.reshape(-1, N)
    if x2.stride(1) != 1:
        x2 = x2.contiguous()
    rows = x2.shape[0]
    out = torch.empty_like(x2)
    if residual is not None:
        r2 = residual.reshape(-1, N)
        if r2.stride(1) != 1:
            r2 = r2.contiguous()
        res_out = torch.empty_like(r2)
    else:
        r2 = x2
        res_out = out
    BLOCK = triton.next_power_of_2(N)
    num_warps = 4 if BLOCK <= 2048 else (8 if BLOCK <= 8192 else 16)
    _gemma_rms_norm_kernel[(rows,)](x2, r2, weight, out, res_out, N, float(eps),
                                     x2.stride(0), r2.stride(0), out.stride(0), res_out.stride(0),
                                     HAS_RES=residual is not None, BLOCK=BLOCK, num_warps=num_warps)
    out = out.view(shape)
    if residual is None:
        return out
    return out, res_out.view(shape)
