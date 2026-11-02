"""Per-layer GEMM shapes of Qwen3.8-27B at one 8192-token prefill chunk: BF16 (cuBLAS via torch.matmul) vs FP8
W8A8 (torch._scaled_mm, per-tensor scales) on this GPU. Shows the headroom of stage 4 and the BF16 efficiency.

    source scripts/h200/env.sh && CUDA_VISIBLE_DEVICES=4 python scripts/h200/bench_gemm.py [--m 8192]
"""
import argparse
import time

import torch

# (name, K, N, count per 64-layer model); hidden 5120, inter 17408, attn heads 24x256 (+ output gate 24x256),
# kv 4x256, GDN q/k 16x128, v/z 48x128, b/a 48 each (see config.json; TP=1)
SHAPES = [
    ('gdn qkvz proj', 5120, 2 * 16 * 128 + 2 * 48 * 128, 48),
    ('gdn ba proj', 5120, 96, 48),
    ('gdn out proj', 48 * 128, 5120, 48),
    ('attn qkv(+gate) proj', 5120, 24 * 256 * 2 + 2 * 4 * 256, 16),
    ('attn o proj', 24 * 256, 5120, 16),
    ('mlp gate_up', 5120, 2 * 17408, 64),
    ('mlp down', 17408, 5120, 64),
]


def timeit(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--m', type=int, default=8192)
    args = ap.parse_args()
    dev = 'cuda'
    M = args.m
    print('device:', torch.cuda.get_device_name(0), '| M =', M)
    tot_bf16 = tot_fp8 = 0.0
    print(f'{"gemm":22s} {"K":>6s} {"N":>6s} {"x/model":>7s} | {"bf16 ms":>8s} {"TFLOPS":>7s} | {"fp8 ms":>8s} {"TFLOPS":>7s} | per-chunk total bf16 / fp8 (ms)')
    for name, K, N, cnt in SHAPES:
        a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
        fl = 2.0 * M * K * N
        ms = timeit(lambda: torch.matmul(a, w.t()))
        a8 = a.to(torch.float8_e4m3fn)
        w8 = w.to(torch.float8_e4m3fn)
        sa = torch.tensor(1.0, device=dev); sw = torch.tensor(1.0, device=dev)
        try:
            ms8 = timeit(lambda: torch._scaled_mm(a8, w8.t(), scale_a=sa, scale_b=sw, out_dtype=torch.bfloat16))
        except Exception as e:  # noqa: BLE001
            ms8 = float('nan'); print('   fp8 failed:', str(e)[:100])
        tot_bf16 += ms * cnt; tot_fp8 += ms8 * cnt
        print(f'{name:22s} {K:6d} {N:6d} {cnt:7d} | {ms:8.3f} {fl / ms / 1e9:7.0f} | {ms8:8.3f} {fl / ms8 / 1e9:7.0f} | {ms * cnt:8.1f} / {ms8 * cnt:8.1f}')
    print(f'\nall GEMMs of one {M}-token chunk: bf16 {tot_bf16:.1f} ms, fp8 {tot_fp8:.1f} ms ({tot_bf16 / tot_fp8:.2f}x); '
          f'x8 chunks (64K prefill): {tot_bf16 * 8 / 1000:.2f} s vs {tot_fp8 * 8 / 1000:.2f} s')


if __name__ == '__main__':
    main()
