"""Driver for Nsight Compute: one FA4 block-sparse forward (realistic 55% kept lists) and one dense FA3 forward
at 8192 x KV, head_dim 256, GQA 24/4, bf16. Each kernel is launched once after a warm-up (JIT + caches).

    source scripts/h200/env.sh && CUDA_VISIBLE_DEVICES=4 ncu --set full -k regex:'FlashAttentionForwardSm90|flash_fwd' \
        --launch-skip 0 --launch-count 2 -o $H200_WORK/ncu-attn python scripts/h200/ncu_attn_driver.py --kv 262144
"""
import argparse

import torch

from sm120_attn.sparse_prefill import BLOCK, FA4_QB, _pack_block_lists

H, Hk, D = 24, 4, 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=8192)
    ap.add_argument('--kv', type=int, default=262144)
    ap.add_argument('--density', type=float, default=0.55)
    ap.add_argument('--sched', default='static', help='static | sorted (persistent sorted tile order, patch 4)')
    ap.add_argument('--which', default='sparse,dense', help='comma list: sparse (FA4 lists), dense (FA3), fa4dense')
    ap.add_argument('--structured', type=float, default=0.0,
                    help='fraction of blocks that are "hot" for every head and q block (shared, like sinks / salient passages); '
                         'the rest of --density is per-row random. 0 = fully random lists')
    args = ap.parse_args()
    dev = 'cuda'
    torch.manual_seed(0)
    from flash_attn.cute import flash_attn_func as fa4_func
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    n, kv = args.n, args.kv
    q = torch.randn(n, H, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(kv, Hk, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(kv, Hk, D, device=dev, dtype=torch.bfloat16)
    q_start = kv - n
    NB, MB = kv // BLOCK, n // FA4_QB
    j = torch.arange(NB, device=dev)
    first_pos = q_start + FA4_QB * torch.arange(MB, device=dev)
    if args.structured > 0:
        hot = torch.rand(NB, device=dev) < args.structured                            # shared across heads and rows
        rest = max(0.0, (args.density - args.structured) / max(1e-6, 1 - args.structured))
        m = hot[None, None, :] | (torch.rand(H, MB, NB, device=dev) < rest)
    else:
        m = torch.rand(H, MB, NB, device=dev) < args.density
    m[:, :, 0] = True
    m |= ((j[None, :] >= (first_pos // BLOCK)[:, None]) & (j[None, :] <= ((first_pos + FA4_QB - 1) // BLOCK)[:, None]))[None]
    m &= (j[None, None, :] <= ((first_pos + FA4_QB - 1) // BLOCK)[None, :, None])
    full = m & ((BLOCK * j + BLOCK - 1)[None, None, :] <= first_pos[None, :, None])
    mc, mi = _pack_block_lists(m & ~full); fc, fi = _pack_block_lists(full)
    order = torch.argsort((mc + fc)[0].reshape(-1).float(), descending=True).to(torch.int32) if args.sched == 'sorted' else None
    bst = BlockSparseTensorsTorch(mc, mi, fc, fi, block_size=(FA4_QB, BLOCK), cu_block_idx_offsets=order)
    cu_q = torch.tensor([0, n], dtype=torch.int32, device=dev); cu_k = torch.tensor([0, kv], dtype=torch.int32, device=dev)
    which = args.which.split(',')
    runs = []
    if 'sparse' in which:
        runs.append(('fa4-sparse', lambda: fa4_func(q[None], k[None], v[None], causal=True, pack_gqa=False, block_sparse_tensors=bst)))
    if 'fa4dense' in which:
        runs.append(('fa4-dense', lambda: fa4_func(q[None], k[None], v[None], causal=True, pack_gqa=False)))
    if 'dense' in which:
        runs.append(('fa3-dense', lambda: flash_attn_varlen_func(q=q, k=k, v=v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=n,
                                                               max_seqlen_k=kv, softmax_scale=D ** -0.5, causal=True, fa_version=3)))
    for name, fn in runs:            # warm-up (JIT, descriptors) outside the profiled launches
        fn()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for name, fn in runs:
        fn()
        torch.cuda.synchronize()
        print('launched', name, flush=True)
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == '__main__':
    main()
