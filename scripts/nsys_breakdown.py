"""Bucket the kernels of an nsys cuda_gpu_kern_sum CSV (attention / GEMM / GDN / small ops) for Qwen3.8 prefill.

    python scripts/nsys_breakdown.py report_cuda_gpu_kern_sum.csv [--top 40]
"""
import csv
import re
import sys
from collections import defaultdict

# first match wins; order matters
RULES = [
    ('full_attn (FA2)', r'flash_fwd|flash::|FlashAttn|fmha|flash_attn'),
    ('full_attn (a100_attn sparse)', r'FlashAttentionForward|a100_attn'),
    ('GDN core (chunked delta rule)', r'chunk_gated_delta|chunk_fwd|fwd_kernel_o|recompute_w_u|chunk_scaled_dot|solve_tril|'
                                      r'merge_16x16|chunk_local_cumsum|l2norm|gated_delta|kkt|fused_recurrent|chunk_'),
    ('GDN conv1d', r'causal_conv1d|conv1d'),
    ('GEMM', r'gemm|cutlass|sm80_xmma|ampere_.*(gemm|s16816)|cublas|nvjet|xmma'),
    ('KV cache write', r'reshape_and_cache|cache_kernel|concat_and_cache'),
    ('rope', r'rotary|rope'),
    ('norm', r'rms_norm|layer_norm|rmsnorm|norm_kernel'),
    ('triton fused (inductor)', r'^triton_|triton_poi|triton_red|triton_per'),
    ('activation', r'act_and_mul|silu|swiglu|gelu'),
    ('copy / elementwise (aten)', r'elementwise|vectorized|unrolled|copy|fill|index|gather|scatter|cat|reduce'),
]


def bucket(name):
    for b, pat in RULES:
        if re.search(pat, name, re.I):
            return b
    return 'other'


def main():
    path = sys.argv[1]
    top = int(sys.argv[sys.argv.index('--top') + 1]) if '--top' in sys.argv else 30
    rows = list(csv.DictReader(open(path)))
    tcol = next(c for c in rows[0] if c.startswith('Total Time'))
    total = sum(float(r[tcol]) for r in rows)
    agg = defaultdict(lambda: [0.0, 0])
    for r in rows:
        b = bucket(r['Name'])
        agg[b][0] += float(r[tcol])
        agg[b][1] += int(r['Instances'])
    print(f'GPU kernel time total: {total / 1e9:.2f} s')
    for b, (t, n) in sorted(agg.items(), key=lambda x: -x[1][0]):
        print(f'  {b:34s} {t / 1e9:8.2f} s  {100 * t / total:5.1f}%  {n:8d} launches')
    print(f'\nTop {top} kernels:')
    for r in sorted(rows, key=lambda r: -float(r[tcol]))[:top]:
        print(f'  {float(r[tcol]) / 1e9:7.2f} s {100 * float(r[tcol]) / total:5.1f}% {int(r["Instances"]):7d}x  '
              f'[{bucket(r["Name"])}] {r["Name"][:110]}')


if __name__ == '__main__':
    main()
