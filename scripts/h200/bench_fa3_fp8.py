"""Stage-4 feasibility: FP8 (e4m3) dense attention with vLLM's FA3 on SM90 vs BF16 FA3, one 8192-row chunk.
Per-tensor descales (q/k/v) as the FA3 FP8 path expects; output bf16. Also reports the FP8 error vs the BF16 result.

    source scripts/h200/env.sh && CUDA_VISIBLE_DEVICES=4 python scripts/h200/bench_fa3_fp8.py
"""
import argparse
import time

import torch

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
    ap.add_argument('--kvs', default='65536,262144')
    args = ap.parse_args()
    dev = 'cuda'
    torch.manual_seed(0)
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    print('device:', torch.cuda.get_device_name(0))
    for kv in (int(x) for x in args.kvs.split(',')):
        n = args.n
        q = torch.randn(n, H, D, device=dev, dtype=torch.bfloat16)
        k = torch.randn(kv, Hk, D, device=dev, dtype=torch.bfloat16)
        v = torch.randn(kv, Hk, D, device=dev, dtype=torch.bfloat16)
        fl = 4 * H * D * n * (kv - n + (n + 1) / 2)
        cu_q = torch.tensor([0, n], dtype=torch.int32, device=dev); cu_k = torch.tensor([0, kv], dtype=torch.int32, device=dev)
        common = dict(cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=n, max_seqlen_k=kv, softmax_scale=D ** -0.5, causal=True, fa_version=3)
        o_bf16 = flash_attn_varlen_func(q=q, k=k, v=v, **common)
        ms = timeit(lambda: flash_attn_varlen_func(q=q, k=k, v=v, **common))
        print(f'\n== {n} x {kv}: FA3 bf16 {ms:7.2f} ms ({fl / ms / 1e9:4.0f} TFLOPS)')
        # per-tensor e4m3 scales (amax / 448), as the hook's FP8 path would do per request and layer
        def q8(t):
            s = t.float().abs().amax().clamp_min(1e-6) / 448.0
            return (t.float() / s).to(torch.float8_e4m3fn), s.reshape(1, 1).expand(1, Hk).contiguous()   # (batch, num_heads_k)
        q_f8, sq = q8(q); k_f8, sk = q8(k); v_f8, sv = q8(v)
        try:
            o_fp8 = flash_attn_varlen_func(q=q_f8, k=k_f8, v=v_f8, q_descale=sq, k_descale=sk, v_descale=sv, **common)
            ms8 = timeit(lambda: flash_attn_varlen_func(q=q_f8, k=k_f8, v=v_f8, q_descale=sq, k_descale=sk, v_descale=sv, **common))
            err = (o_fp8.float() - o_bf16.float()).norm().item() / o_bf16.float().norm().item()
            print(f'   FA3 fp8 e4m3 (per-tensor descales): {ms8:7.2f} ms ({fl / ms8 / 1e9:4.0f} TFLOPS) = {ms / ms8:.2f}x bf16 | rel L2 err vs bf16 {err:.3e} | out dtype {o_fp8.dtype}')
        except Exception as e:
            print('   FA3 fp8 failed:', str(e)[:300])


if __name__ == '__main__':
    main()
