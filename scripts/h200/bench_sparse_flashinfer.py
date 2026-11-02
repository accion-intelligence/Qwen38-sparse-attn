"""Block-sparse prefill attention on this GPU with FlashInfer's BSR kernels (fa3 = SM90 wgmma, fa2 = mma.sync),
driven by the repository's block selection (SparsePrefill, kernel='flashinfer', per-KV-head union masks),
versus dense FlashAttention-3. Same synthetic 'hot block' data as sm120_attn/sparse_prefill.py's self-test.

    source scripts/h200/env.sh && CUDA_VISIBLE_DEVICES=4 python scripts/h200/bench_sparse_flashinfer.py
"""
import argparse
import time

import torch

from sm120_attn.sparse_prefill import BLOCK, SparsePrefill, reference

H, Hk, D = 24, 4, 256


def make(n, kv_len, dev, peaky=3.0):
    q = torch.randn(n, H, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(Hk, kv_len, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(Hk, kv_len, D, device=dev, dtype=torch.bfloat16)
    hot = (torch.rand(kv_len // BLOCK + 1, device=dev) < 0.1).repeat_interleave(BLOCK)[:kv_len]
    u = torch.randn(Hk, 1, D, device=dev)
    k = (k.float() * 0.5 + hot[None, :, None] * u * peaky).to(torch.bfloat16)
    q = (q.float() + u.repeat_interleave(H // Hk, 0).transpose(0, 1) * 1.0).to(torch.bfloat16)
    return q, k, v


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
    ap.add_argument('--kv', type=int, default=65536)
    ap.add_argument('--taus', default='1.0,0.95,0.9')
    args = ap.parse_args()
    dev = 'cuda'
    torch.manual_seed(0)
    print('device:', torch.cuda.get_device_name(0))
    # correctness (small) per backend
    for backend in ('fa3', 'fa2'):
        try:
            sp = SparsePrefill(H, Hk, D, dev, tau=0.9, min_kv=0, min_q=0, kernel='flashinfer', fi_backend=backend)
            n, kv = 1000, 4024
            q, k, v = make(n, kv, dev)
            o = sp.forward(q, k, v, kv - n)
            qp = torch.cat([q.new_zeros((kv - n) % BLOCK, H, D), q, q.new_zeros((-(((kv - n) % BLOCK) + n)) % BLOCK, H, D)])
            kpad = (-kv) % BLOCK
            kp, vp = torch.nn.functional.pad(k, (0, 0, 0, kpad)), torch.nn.functional.pad(v, (0, 0, 0, kpad))
            masks = sp.block_masks(qp, kp, (kv - n) - (kv - n) % BLOCK, valid=((kv - n) % BLOCK, n))
            ref = reference(qp, kp, vp, (kv - n) - (kv - n) % BLOCK, masks)[(kv - n) % BLOCK:(kv - n) % BLOCK + n]
            err = (o.float() - ref).abs().max().item() / ref.abs().max().item()
            print(f'  {backend}: n={n} kv={kv} tau=0.9 kept={sp.stats["blocks_kept"] / sp.stats["blocks_total"]:.3f}  max rel err vs masked ref {err:.2e}')
        except Exception as e:
            print(f'  {backend}: FAILED: {str(e)[:200]}')
    n, kv = args.n, args.kv
    q, k, v = make(n, kv, dev)
    fl = 4 * H * D * n * (kv - n + (n + 1) / 2)
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    cu_q = torch.tensor([0, n], dtype=torch.int32, device=dev)
    cu_k = torch.tensor([0, kv], dtype=torch.int32, device=dev)
    k_nhd, v_nhd = k.transpose(0, 1).contiguous(), v.transpose(0, 1).contiguous()
    ms = timeit(lambda: flash_attn_varlen_func(q=q, k=k_nhd, v=v_nhd, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=n,
                                               max_seqlen_k=kv, softmax_scale=D ** -0.5, causal=True, fa_version=3))
    print(f'\n dense FA3 {n}x{kv}: {ms:7.2f} ms  {fl / ms / 1e9:6.1f} TFLOPS')
    for backend in ('fa3', 'fa2'):
        for tau in (float(x) for x in args.taus.split(',')):
            try:
                sp = SparsePrefill(H, Hk, D, dev, tau=tau, min_kv=0, min_q=0, kernel='flashinfer', fi_backend=backend)
                sp.forward(q, k, v, kv - n)
                sp.timing = True
                sp.stats.update(plan_ms=0.0, run_ms=0.0, select_ms=0.0, blocks_total=0, blocks_kept=0)
                iters = 5
                for _ in range(iters):
                    sp.forward(q, k, v, kv - n)
                st = sp.stats
                kept = st['blocks_kept'] / st['blocks_total']
                tot = (st['select_ms'] + st['plan_ms'] + st['run_ms']) / iters
                print(f' flashinfer-{backend} tau={tau:4.2f} kept={kept:.3f}: select {st["select_ms"] / iters:6.2f} + plan {st["plan_ms"] / iters:6.2f} + run {st["run_ms"] / iters:7.2f} = {tot:7.2f} ms'
                      f'  (run: {fl * kept / (st["run_ms"] / iters) / 1e9:6.1f} TFLOPS on kept blocks; vs dense FA3 {ms / tot:4.2f}x)')
            except Exception as e:
                print(f' flashinfer-{backend} tau={tau}: FAILED: {str(e)[:200]}')


if __name__ == '__main__':
    main()
