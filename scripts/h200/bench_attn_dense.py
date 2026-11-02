"""Dense causal prefill-attention reference timings on this GPU: vLLM's FlashAttention-3 / -2 (what stock vLLM
runs on SM90) versus the SM120 CuTe kernels of this repository, same shape as tests/*.py --big.

    source scripts/h200/env.sh && CUDA_VISIBLE_DEVICES=4 python scripts/h200/bench_attn_dense.py [--n 8192 --kv 65536]

The query block is the last `n` rows of a `kv`-token context (q_start = kv - n), head_dim 256, GQA 24/4,
bf16 operands, non-paged. TFLOPS use the same causal-area formula as tests/test_fa_bf16.py.
"""
import argparse
import time

import torch

H, Hk, D = 24, 4, 256


def timeit(fn, iters=5, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=8192)
    ap.add_argument('--kv', type=int, default=65536)
    ap.add_argument('--skip-sm120', action='store_true')
    args = ap.parse_args()
    n, kv = args.n, args.kv
    dev = 'cuda'
    torch.manual_seed(0)
    print('device:', torch.cuda.get_device_name(0), '| shape: n=%d kv=%d H=%d Hk=%d D=%d' % (n, kv, H, Hk, D))
    q = torch.randn(n, H, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(kv, Hk, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(kv, Hk, D, device=dev, dtype=torch.bfloat16)
    fl = 4 * H * D * n * (kv - n + (n + 1) / 2)

    from vllm.vllm_flash_attn import flash_attn_varlen_func
    from vllm.vllm_flash_attn.flash_attn_interface import is_fa_version_supported
    cu_q = torch.tensor([0, n], dtype=torch.int32, device=dev)
    cu_k = torch.tensor([0, kv], dtype=torch.int32, device=dev)
    outs = {}
    for ver in (3, 2):
        if not is_fa_version_supported(ver):
            print(f'  FA{ver}: not supported on this device/build')
            continue
        def run(ver=ver):
            return flash_attn_varlen_func(q=q, k=k, v=v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=n,
                                          max_seqlen_k=kv, softmax_scale=D ** -0.5, causal=True, fa_version=ver)
        outs[ver] = run()
        ms = timeit(run)
        print(f'  vLLM FlashAttention-{ver} (varlen, non-paged): {ms:8.2f} ms  {fl / ms / 1e9:6.1f} TFLOPS')
    if 3 in outs and 2 in outs:
        d = (outs[3].float() - outs[2].float()).norm() / outs[2].float().norm()
        print(f'  FA3 vs FA2 rel L2: {d.item():.2e}')

    if not args.skip_sm120:
        from sm120_attn.kernels.kai_fa_bf16 import attention_sm120
        ms = timeit(lambda: attention_sm120(q, k, v))
        o = attention_sm120(q, k, v)
        ref = outs.get(3, outs.get(2))
        d = (o.float() - ref.float()).norm() / ref.float().norm() if ref is not None else float('nan')
        print(f'  sm120 kai_fa_bf16 (unchanged, mma.sync):     {ms:8.2f} ms  {fl / ms / 1e9:6.1f} TFLOPS  rel L2 vs FA: {d:.2e}')
        try:
            from sm120_attn.kernels.kai_fa_fp8_tma import attention_sm120_fp8_tma
            ms = timeit(lambda: attention_sm120_fp8_tma(q, k, v, kv - n, m_block=128))
            print(f'  sm120 kai_fa_fp8_tma (unchanged, fp8 mma.sync): {ms:8.2f} ms  {fl / ms / 1e9:6.1f} TFLOPS')
        except Exception as e:
            print('  sm120 kai_fa_fp8_tma: skipped:', str(e)[:120])


if __name__ == '__main__':
    main()
