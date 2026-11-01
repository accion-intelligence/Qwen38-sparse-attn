"""Fused block selection for a100_attn: masked softmax + top-tau mass + diagonal/sink + union + list compaction.

Replaces the sort/cumsum/scatter/argsort chain of SparsePrefillA100.block_masks + _plan with one Triton kernel
(plus the q block means and one einsum for the logits, which stay in PyTorch/cuBLAS).

Selection rule (same as the PyTorch path up to ties): per query head and 64-row query block, keep the smallest
set of key blocks whose softmax mass (over causal blocks, from block-mean logits) reaches tau. Instead of a
sort, the kernel bisects for the largest probability threshold t with sum_{p >= t} p >= tau and keeps p >= t.
Then the diagonal block and block 0 (sink) are added, blocks >= nb_real are dropped, and for 128-row query
blocks the two 64-row selections are OR-ed. Output: ascending block lists (diagonal block last) + counts.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _select_kernel(logits_ptr, counts_ptr, lists_ptr,
                   MB, NB, G, list_stride, gi0, nb_real, tau,
                   SUB: tl.constexpr, BLOCK_N: tl.constexpr, ITERS: tl.constexpr):
    # program = output row r = head * MBo + mo  (head = hk * G + gidx; MBo = MB // SUB)
    r = tl.program_id(0)
    MBo = MB // SUB
    head = r // MBo
    mo = r - head * MBo
    hk = head // G
    gidx = head - hk * G
    j = tl.arange(0, BLOCK_N)
    in_range = j < NB
    sel_any = tl.zeros([BLOCK_N], dtype=tl.int1)
    for sub in tl.static_range(SUB):
        m = mo * SUB + sub
        gi = gi0 + m                                                   # diagonal key block of this 64-row block
        # logits layout [Hk, MB, G, NB] (contiguous)
        base = ((hk * MB + m) * G + gidx).to(tl.int64) * NB
        causal = in_range & (j <= gi)
        x = tl.load(logits_ptr + base + j, mask=causal, other=float('-inf'))
        mx = tl.max(x, axis=0)
        p = tl.where(causal, tl.exp(x - mx), 0.0)
        p = p / tl.sum(p, axis=0)
        # bisection: largest t with sum(p[p >= t]) >= tau  (S(t) is non-increasing in t)
        lo = tl.zeros([], dtype=tl.float32)
        hi = tl.max(p, axis=0)
        for _ in tl.static_range(ITERS):
            mid = (lo + hi) * 0.5
            s = tl.sum(tl.where(p >= mid, p, 0.0), axis=0)
            ok = s >= tau
            lo = tl.where(ok, mid, lo)
            hi = tl.where(ok, hi, mid)
        sel = (p >= lo) & causal & (p > 0.0)
        sel = sel | (j == gi) | (j == 0)                               # diagonal + sink
        sel = sel & (j <= gi) & (j < nb_real) & in_range
        sel_any = sel_any | sel
    # compact ascending indices: position = exclusive prefix count
    sel_i = sel_any.to(tl.int32)
    pos = tl.cumsum(sel_i, axis=0) - sel_i
    tl.store(lists_ptr + r.to(tl.int64) * list_stride + pos, j, mask=sel_any)
    tl.store(counts_ptr + r, tl.sum(sel_i, axis=0))


def fused_select(qm: torch.Tensor, km: torch.Tensor, scale: float, q_start_block: int, nb_real: int | None,
                 tau: float, m_block: int, num_kv_heads: int, iters: int = 20):
    """qm [MB, H, D] fp32 query-block means (MB 64-row blocks, absolute first block q_start_block);
    km [Hk, NB, D] fp32 key-block means with NB >= q_start_block + MB. Returns (counts int32 [H * MB/SUB],
    lists int32 [H * MB/SUB, NB]) for SUB = m_block // 64; lists rows ascend, entries past counts[r] undefined."""
    MB, H, D = qm.shape
    Hk = num_kv_heads
    G = H // Hk
    NB = km.shape[1]
    SUB = m_block // 64
    assert MB % SUB == 0 and NB >= q_start_block + MB
    # logits [Hk, MB * G, NB] == [Hk, MB, G, NB]: one batched fp32 GEMM writing the kernel's layout, scale folded in
    a = qm.view(MB, Hk, G, D).permute(1, 0, 2, 3).reshape(Hk, MB * G, D)
    logits = torch.empty(Hk, MB * G, NB, dtype=torch.float32, device=qm.device)
    torch.baddbmm(logits, a, km.transpose(1, 2), beta=0.0, alpha=scale, out=logits)
    rows = H * (MB // SUB)
    counts = torch.empty(rows, dtype=torch.int32, device=qm.device)
    lists = torch.empty(rows, NB, dtype=torch.int32, device=qm.device)
    BLOCK_N = max(triton.next_power_of_2(NB), 16)
    num_warps = 4 if BLOCK_N <= 1024 else 8
    _select_kernel[(rows,)](logits, counts, lists, MB, NB, G, NB, q_start_block,
                            NB if nb_real is None else nb_real, float(tau),
                            SUB=SUB, BLOCK_N=BLOCK_N, ITERS=iters, num_warps=num_warps)
    return counts, lists
