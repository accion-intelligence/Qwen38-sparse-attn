"""Fused block selection for the FA4 (SM90) path: one Triton kernel per (query head, 128-row q block).

Replaces the torch chain softmax -> sort -> cumsum -> scatter -> any -> argsort x2 of SparsePrefill.block_masks
+ _pack_block_lists with a single kernel that, per row of block scores:
  1. masked softmax over the causally visible key blocks,
  2. finds the smallest attention-mass prefix >= tau (threshold found by bisection on the probability; this is
     the sorted-prefix rule of block_masks up to ties),
  3. forces the diagonal (overlapping) key blocks and block 0 (sink),
  4. splits selected blocks into 'full' (entirely before the q block: no masking needed) and 'mask' blocks,
  5. writes FA4's packed ascending block lists and counts directly.
Inputs are the fp32 scores logits[h, m, j] = scale * mean(Q block m, head h) . mean(K block j, kv head of h).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _select_kernel(logits_ptr, mcnt_ptr, midx_ptr, fcnt_ptr, fidx_ptr,
                   MB, NB, q_start, tau,
                   stride_lh, stride_lm, stride_ih, stride_im,
                   QB: tl.constexpr, KB: tl.constexpr, SUB: tl.constexpr, BLOCK_N: tl.constexpr, ITERS: tl.constexpr):
    """One program per (head h, kernel q block m of QB rows). Scores are given per sub-block of QB // SUB rows
    (SUB score rows per kernel block, row index m * SUB + s); each sub-block selects its own top-mass set and the
    kernel block keeps the union, so a finer selection granularity does not change the kernel tiles."""
    pid = tl.program_id(0)
    h = pid // MB
    m = pid % MB
    j = tl.arange(0, BLOCK_N)
    first_pos = q_start + m * QB
    gi_first = first_pos // KB
    gi_last = (first_pos + QB - 1) // KB
    valid = (j < NB) & (j <= gi_last)
    sel = j < 0
    for s_ in tl.static_range(SUB):
        sub_first = first_pos + s_ * (QB // SUB)
        sub_last_blk = (sub_first + QB // SUB - 1) // KB
        vsub = (j < NB) & (j <= sub_last_blk)
        x = tl.load(logits_ptr + h * stride_lh + (m * SUB + s_) * stride_lm + j, mask=vsub, other=float('-inf'))
        mx = tl.max(x, 0)
        e = tl.where(vsub, tl.exp(x - mx), 0.0)
        p = e / tl.sum(e, 0)
        lo = tl.zeros((), dtype=tl.float32)
        hi = tl.max(p, 0)
        # largest threshold t with mass(p >= t) >= tau  ==  smallest top-mass prefix reaching tau
        for _ in tl.static_range(ITERS):
            mid = (lo + hi) * 0.5
            mass = tl.sum(tl.where(p >= mid, p, 0.0), 0)
            ok = mass >= tau
            lo = tl.where(ok, mid, lo)
            hi = tl.where(ok, hi, mid)
        sel = sel | ((p >= lo) & vsub)
    sel = sel | ((j >= gi_first) & (j <= gi_last)) | (j == 0)
    sel = sel & valid
    full = sel & ((KB * j + KB - 1) <= first_pos)
    part = sel & (full == 0)
    fi = tl.cumsum(full.to(tl.int32), 0) - 1
    pi = tl.cumsum(part.to(tl.int32), 0) - 1
    base = h * stride_ih + m * stride_im
    tl.store(fidx_ptr + base + fi, j, mask=full)
    tl.store(midx_ptr + base + pi, j, mask=part)
    tl.store(fcnt_ptr + h * MB + m, tl.sum(full.to(tl.int32), 0))
    tl.store(mcnt_ptr + h * MB + m, tl.sum(part.to(tl.int32), 0))


def fused_select(logits: torch.Tensor, q_start: int, tau: float, qb: int = 128, kb: int = 64, iters: int = 24,
                 sub: int = 1):
    """logits [H, MB * sub, NB] fp32 (unit stride along NB): one score row per (qb // sub)-row query sub-block.
    Returns FA4 lists for qb-row kernel blocks (mask_cnt [1,H,MB], mask_idx [1,H,MB,NB], full_cnt, full_idx),
    all int32, selected key blocks packed first in ascending order; a kernel block keeps the union of its
    sub-blocks' selections."""
    assert logits.dtype == torch.float32 and logits.stride(2) == 1
    H, MBS, NB = logits.shape
    assert MBS % sub == 0
    MB = MBS // sub
    dev = logits.device
    midx = torch.empty(1, H, MB, NB, dtype=torch.int32, device=dev)
    fidx = torch.empty(1, H, MB, NB, dtype=torch.int32, device=dev)
    mcnt = torch.empty(1, H, MB, dtype=torch.int32, device=dev)
    fcnt = torch.empty(1, H, MB, dtype=torch.int32, device=dev)
    BLOCK_N = max(128, triton.next_power_of_2(NB))
    _select_kernel[(H * MB,)](logits, mcnt, midx, fcnt, fidx, MB, NB, int(q_start), float(tau),
                               logits.stride(0), logits.stride(1), midx.stride(1), midx.stride(2),
                               QB=qb, KB=kb, SUB=sub, BLOCK_N=BLOCK_N, ITERS=iters,
                               num_warps=8 if BLOCK_N >= 2048 else 4)
    return mcnt, midx, fcnt, fidx


def block_scores(q: torch.Tensor, km: torch.Tensor, num_kv_heads: int, scale: float, qb: int = 128,
                 valid_rows: int | None = None) -> torch.Tensor:
    """Scores for fused_select: q [n, H, D] (n multiple of qb; rows >= valid_rows are padding), km [Hk, NB, D] fp32.
    Returns logits [H, MB, NB] fp32 with head index h = kv_head * (H // Hk) + i, as SparsePrefill.block_masks."""
    n, H, D = q.shape
    MB = n // qb
    g = H // num_kv_heads
    if valid_rows is None or valid_rows >= n:
        qm = q.float().view(MB, qb, H, D).mean(1)                                   # [MB, H, D]
    else:
        real = (torch.arange(n, device=q.device) < valid_rows).float()
        counts = real.view(MB, qb).sum(1).clamp_min(1.0)
        qm = (q.float() * real[:, None, None]).view(MB, qb, H, D).sum(1) / counts[:, None, None]
    qg = qm.view(MB, num_kv_heads, g, D).permute(1, 2, 0, 3).reshape(num_kv_heads, g * MB, D)   # row = i * MB + m
    logits = torch.matmul(qg, km.transpose(1, 2)) * scale                           # [Hk, g*MB, NB]
    return logits.view(H, MB, logits.shape[-1])
