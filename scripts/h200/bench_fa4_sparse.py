"""FlashAttention-4 (CuTe DSL, wgmma) on SM90: dense vs block-sparse with this repository's block selection.

    source scripts/h200/env.sh && CUDA_VISIBLE_DEVICES=4 python scripts/h200/bench_fa4_sparse.py

Shape as tests/*.py --big: q = last `n` rows of a `kv`-token context (chunked prefill), head_dim 256, GQA 24/4.
Sparse lists are per query head and per 128-row q block over 64-token KV blocks (FA4 SM90 fits tile_n to 64):
'mask' blocks intersect the causal boundary, 'full' blocks are entirely visible.
"""
import argparse
import time

import torch

from sm120_attn.sparse_prefill import BLOCK, SparsePrefill

H, Hk, D = 24, 4, 256
QB = 128  # q block rows for FA4 SM90 (tile_m)


def timeit(fn, iters=5, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1000


def make(n, kv, dev, peaky=3.0):
    """Synthetic 'hot block' data (as sparse_prefill.py self-test): NHD k/v [kv, Hk, D]."""
    q = torch.randn(n, H, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(kv, Hk, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(kv, Hk, D, device=dev, dtype=torch.bfloat16)
    hot = (torch.rand(kv // BLOCK + 1, device=dev) < 0.1).repeat_interleave(BLOCK)[:kv]
    u = torch.randn(Hk, 1, D, device=dev)
    k = (k.float() * 0.5 + hot[:, None, None] * u.transpose(0, 1) * peaky).to(torch.bfloat16)
    q = (q.float() + u.repeat_interleave(H // Hk, 0).transpose(0, 1) * 1.0).to(torch.bfloat16)
    return q, k, v


def lists_from_masks(masks64, q_start, n, kv):
    """masks64: [H, MB64, NB] bool over 64-row q blocks (q_start multiple of 64, n multiple of 128).
    Returns FA4 BlockSparseTensorsTorch for 128-row q blocks x 64-token KV blocks, plus kept fraction."""
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
    Hh, MB64, NB = masks64.shape
    MB = n // QB
    m = masks64.view(Hh, MB, 2, NB).any(2)                                   # [H, MB, NB]
    dev = masks64.device
    j = torch.arange(NB, device=dev)
    first_q = q_start + QB * torch.arange(MB, device=dev)                     # first query position per q block
    full = m & ((BLOCK * j + BLOCK - 1)[None, None, :] <= first_q[None, :, None])
    future = (BLOCK * j)[None, None, :] > (first_q + QB - 1)[None, :, None]
    m &= ~future
    mask_b = m & ~full
    def pack(sel):
        cnt = sel.sum(-1).to(torch.int32)
        idx = torch.argsort((~sel).to(torch.int8), dim=-1, stable=True).to(torch.int32)   # selected first, ascending
        return cnt[None], idx[None]
    mc, mi = pack(mask_b)
    fc, fi = pack(full)
    causal_total = sum(min(NB, (q_start + QB * (b + 1) + BLOCK - 1) // BLOCK) for b in range(MB)) * Hh
    kept = int(m.sum().item()) / causal_total
    return BlockSparseTensorsTorch(mc, mi, fc, fi, block_size=(QB, BLOCK)), kept, m


def reference_sparse(q, k, v, q_start, m128):
    """fp32 reference with per-head 128x64 block masks m128 [H, MB, NB] and causality."""
    n = q.shape[0]; kv = k.shape[0]; g = H // Hk
    pos_q = q_start + torch.arange(n, device=q.device); pos_k = torch.arange(kv, device=q.device)
    allowed = pos_k[None, :] <= pos_q[:, None]
    outs = []
    for h in range(H):
        bm = m128[h].repeat_interleave(QB, 0)[:n].repeat_interleave(BLOCK, 1)[:, :kv]
        s = (q[:, h].float() @ k[:, h // g].float().T) * D ** -0.5
        s.masked_fill_(~(allowed & bm), float('-inf'))
        outs.append(torch.softmax(s, -1) @ v[:, h // g].float())
    return torch.stack(outs, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=8192)
    ap.add_argument('--kv', type=int, default=65536)
    ap.add_argument('--densities', default='0.5,0.4,0.25')
    ap.add_argument('--taus', default='0.95,0.9')
    ap.add_argument('--pack-gqa', action='store_true')
    args = ap.parse_args()
    dev = 'cuda'
    torch.manual_seed(0)
    from flash_attn.cute import flash_attn_func as fa4_func
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    print('device:', torch.cuda.get_device_name(0), '| pack_gqa', args.pack_gqa)

    # --- correctness on a small case ---
    n, kv = 1024, 8192
    q, k, v = make(n, kv, dev)
    sp = SparsePrefill(H, Hk, D, dev, tau=0.9, min_kv=0, min_q=0, kernel='sm120')
    masks64 = sp.block_masks(q, k, kv - n, union=False)                     # [H, MB64, NB]
    bst, kept, m128 = lists_from_masks(masks64, kv - n, n, kv)
    o = fa4_func(q[None], k[None], v[None], causal=True, pack_gqa=args.pack_gqa, block_sparse_tensors=bst)[0]
    ref = reference_sparse(q, k, v, kv - n, m128)
    err = (o.float() - ref).abs().max().item() / ref.abs().max().item()
    o_d = fa4_func(q[None], k[None], v[None], causal=True, pack_gqa=args.pack_gqa)[0]
    ref_d = reference_sparse(q, k, v, kv - n, torch.ones_like(m128))
    err_d = (o_d.float() - ref_d).abs().max().item() / ref_d.abs().max().item()
    print(f'  correctness n={n} kv={kv}: dense FA4 max rel err {err_d:.2e} | sparse FA4 (kept {kept:.3f}) max rel err vs masked ref {err:.2e}')

    # --- timing ---
    n, kv = args.n, args.kv
    q, k, v = make(n, kv, dev)
    fl = 4 * H * D * n * (kv - n + (n + 1) / 2)
    cu_q = torch.tensor([0, n], dtype=torch.int32, device=dev); cu_k = torch.tensor([0, kv], dtype=torch.int32, device=dev)
    ms3 = timeit(lambda: flash_attn_varlen_func(q=q, k=k, v=v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=n,
                                                max_seqlen_k=kv, softmax_scale=D ** -0.5, causal=True, fa_version=3))
    print(f'\n  dense FA3 (vllm C++)        {ms3:7.2f} ms  {fl / ms3 / 1e9:6.1f} TFLOPS')
    ms4 = timeit(lambda: fa4_func(q[None], k[None], v[None], causal=True, pack_gqa=args.pack_gqa))
    print(f'  dense FA4 (CuTe)            {ms4:7.2f} ms  {fl / ms4 / 1e9:6.1f} TFLOPS')
    # all-blocks list through the sparse path = overhead of the sparse machinery
    ones = torch.ones(H, n // BLOCK, kv // BLOCK, dtype=torch.bool, device=dev)
    bst, kept, _ = lists_from_masks(ones, kv - n, n, kv)
    ms = timeit(lambda: fa4_func(q[None], k[None], v[None], causal=True, pack_gqa=args.pack_gqa, block_sparse_tensors=bst))
    print(f'  FA4 sparse path, all blocks {ms:7.2f} ms  {fl / ms / 1e9:6.1f} TFLOPS  (kept {kept:.3f})')
    for tau in (float(x) for x in args.taus.split(',')):
        sp = SparsePrefill(H, Hk, D, dev, tau=tau, min_kv=0, min_q=0, kernel='sm120')
        t_sel = timeit(lambda: sp.block_masks(q, k, kv - n, union=False))
        masks64 = sp.block_masks(q, k, kv - n, union=False)
        bst, kept, _ = lists_from_masks(masks64, kv - n, n, kv)
        t_list = timeit(lambda: lists_from_masks(masks64, kv - n, n, kv))
        ms = timeit(lambda: fa4_func(q[None], k[None], v[None], causal=True, pack_gqa=args.pack_gqa, block_sparse_tensors=bst))
        print(f'  FA4 sparse tau={tau:4.2f} kept={kept:.3f}: attn {ms:7.2f} ms ({fl * kept / ms / 1e9:6.1f} TFLOPS on kept) | select {t_sel:5.2f} + lists {t_list:5.2f} ms | vs FA3 dense {ms3 / ms:4.2f}x (attn only)')
    for dens in (float(x) for x in args.densities.split(',')):
        rnd = torch.rand(H, n // BLOCK, kv // BLOCK, device=dev) < dens
        rnd[:, :, 0] = True
        bst, kept, _ = lists_from_masks(rnd, kv - n, n, kv)
        ms = timeit(lambda: fa4_func(q[None], k[None], v[None], causal=True, pack_gqa=args.pack_gqa, block_sparse_tensors=bst))
        print(f'  FA4 sparse random kept={kept:.3f}: attn {ms:7.2f} ms ({fl * kept / ms / 1e9:6.1f} TFLOPS on kept) | vs FA3 dense {ms3 / ms:4.2f}x')


if __name__ == '__main__':
    main()
