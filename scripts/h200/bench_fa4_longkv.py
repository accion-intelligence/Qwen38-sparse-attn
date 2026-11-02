"""FA4 block-sparse efficiency at long KV (one 8192-row chunk against 64K / 128K / 256K keys) with realistic
kept fractions, plus the cost of the fused selection at NB = 1024 / 2048 / 4096 key blocks.

    source scripts/h200/env.sh && CUDA_VISIBLE_DEVICES=4 python scripts/h200/bench_fa4_longkv.py
"""
import argparse
import time

import torch

from sm120_attn.select_fused import block_scores, fused_select
from sm120_attn.sparse_prefill import BLOCK, FA4_QB, _pack_block_lists

H, Hk, D = 24, 4, 256


def timeit(fn, iters=5, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=8192)
    ap.add_argument('--kvs', default='65536,131072,262144')
    ap.add_argument('--densities', default='0.55,0.4,0.3')
    ap.add_argument('--kb', type=int, default=BLOCK, help='key block size for the FA4 lists (64 default; 80 = dense hdim-256 tile)')
    ap.add_argument('--skip-dense', action='store_true')
    ap.add_argument('--sched', default='static,sorted', help='comma list of FA4 tile schedules to time: static (FA4 grid) and/or sorted (persistent, longest list first; needs patch 4)')
    ap.add_argument('--skew', type=float, default=0.0, help='per-head density spread: head h gets density * (1 + skew * u_h), u_h uniform in [-1, 1]')
    args = ap.parse_args()
    dev = 'cuda'
    torch.manual_seed(0)
    from flash_attn.cute import flash_attn_func as fa4_func
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    n = args.n
    print('device:', torch.cuda.get_device_name(0))
    KB = args.kb
    for kv in (int(x) for x in args.kvs.split(',')):
        q = torch.randn(n, H, D, device=dev, dtype=torch.bfloat16)
        k = torch.randn(kv, Hk, D, device=dev, dtype=torch.bfloat16)
        v = torch.randn(kv, Hk, D, device=dev, dtype=torch.bfloat16)
        q_start = kv - n
        NB = -(-kv // KB)
        MB = n // FA4_QB
        fl = 4 * H * D * n * (kv - n + (n + 1) / 2)
        cu_q = torch.tensor([0, n], dtype=torch.int32, device=dev); cu_k = torch.tensor([0, kv], dtype=torch.int32, device=dev)
        ms3 = timeit(lambda: flash_attn_varlen_func(q=q, k=k, v=v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=n,
                                                    max_seqlen_k=kv, softmax_scale=D ** -0.5, causal=True, fa_version=3))
        ms4 = float('nan') if args.skip_dense else timeit(lambda: fa4_func(q[None], k[None], v[None], causal=True, pack_gqa=False))
        print(f'\n== {n} x {kv} (key block {KB}): dense FA3 {ms3:7.2f} ms ({fl / ms3 / 1e9:4.0f} TFLOPS) | dense FA4 {ms4:7.2f} ms ({fl / ms4 / 1e9:4.0f} TFLOPS)')
        if KB == BLOCK:
            # selection cost at this NB (block means precomputed as the hook's cache does)
            km = k.view(NB, BLOCK, Hk, D).float().mean(1).transpose(0, 1)
            t_scores = timeit(lambda: block_scores(q, km, Hk, D ** -0.5, qb=FA4_QB))
            logits = block_scores(q, km, Hk, D ** -0.5, qb=FA4_QB)
            t_sel = timeit(lambda: fused_select(logits, q_start, 0.95, qb=FA4_QB, kb=BLOCK))
            print(f'   selection at NB={NB}: block scores {t_scores:5.2f} ms + fused select {t_sel:5.2f} ms')
        j = torch.arange(NB, device=dev)
        first_pos = q_start + FA4_QB * torch.arange(MB, device=dev)
        for dens in (float(x) for x in args.densities.split(',')):
            if args.skew > 0:
                dens_h = (dens * (1 + args.skew * (2 * torch.rand(H, device=dev) - 1))).clamp(0.05, 1.0)
                m = torch.rand(H, MB, NB, device=dev) < dens_h[:, None, None]
            else:
                m = torch.rand(H, MB, NB, device=dev) < dens
            m[:, :, 0] = True
            m |= ((j[None, :] >= (first_pos // KB)[:, None]) & (j[None, :] <= ((first_pos + FA4_QB - 1) // KB)[:, None]))[None]
            m &= (j[None, None, :] <= ((first_pos + FA4_QB - 1) // KB)[None, :, None])
            full = m & ((KB * j + KB - 1)[None, None, :] <= first_pos[None, :, None])
            mc, mi = _pack_block_lists(m & ~full); fc, fi = _pack_block_lists(full)
            causal_total = H * sum(min(NB, (q_start + FA4_QB * (b + 1) + KB - 1) // KB) for b in range(MB))
            kept = m.sum().item() / causal_total
            lens = (mc + fc)[0].reshape(-1).float()
            skew_txt = f' (list max/mean {lens.max().item() / lens.mean().item():.2f})'
            for sched in args.sched.split(','):
                order = torch.argsort(lens, descending=True).to(torch.int32) if sched == 'sorted' else None
                bst = BlockSparseTensorsTorch(mc, mi, fc, fi, block_size=(FA4_QB, KB), cu_block_idx_offsets=order)
                ms = timeit(lambda: fa4_func(q[None], k[None], v[None], causal=True, pack_gqa=False, block_sparse_tensors=bst))
                print(f'   FA4 sparse [{sched:6s}] kept={kept:.3f}{skew_txt}: {ms:7.2f} ms  ({fl * kept / ms / 1e9:4.0f} TFLOPS on kept)  = {ms3 / ms:4.2f}x dense FA3; time ratio {ms / ms3:.3f}')


if __name__ == '__main__':
    main()
