"""Fused Gemma RMSNorm vs vLLM's torch implementation: numerics and time ([8192, 5120] bf16 = one prefill chunk)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sm120_attn.fused_norm import gemma_rms_norm  # noqa: E402
from vllm import ir  # noqa: E402


def timeit(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e3


def main():
    torch.manual_seed(0)
    dev = 'cuda'
    rows, N, eps = 8192, 5120, 1e-6
    x = torch.randn(rows, N, device=dev, dtype=torch.bfloat16) * 3
    res = torch.randn(rows, N, device=dev, dtype=torch.bfloat16)
    w = (torch.randn(N, device=dev) * 0.05).to(torch.bfloat16)      # Gemma weight is stored around 0
    wf = w.float() + 1.0
    ref = ir.ops.rms_norm(x, wf, eps)
    out = gemma_rms_norm(x, w, eps)
    d = (out.float() - ref.float()).abs()
    print(f'rms_norm:            max abs diff {d.max().item():.3e}  mismatching bf16 elements {(d > 0).float().mean().item() * 100:.3f}%')
    ref2, rres = ir.ops.fused_add_rms_norm(x, res, wf, eps)
    out2, ores = gemma_rms_norm(x, w, eps, res)
    d2 = (out2.float() - ref2.float()).abs(); dr = (ores.float() - rres.float()).abs()
    print(f'fused_add_rms_norm:  out max abs diff {d2.max().item():.3e} ({(d2 > 0).float().mean().item() * 100:.3f}% differ) | residual max abs diff {dr.max().item():.3e}')
    assert d.max().item() < 1e-1 and d2.max().item() < 1e-1 and dr.max().item() == 0.0
    t_ref = timeit(lambda: ir.ops.rms_norm(x, wf, eps)); t_new = timeit(lambda: gemma_rms_norm(x, w, eps))
    t_ref2 = timeit(lambda: ir.ops.fused_add_rms_norm(x, res, wf, eps)); t_new2 = timeit(lambda: gemma_rms_norm(x, w, eps, res))
    print(f'time [8192 x 5120] bf16: rms_norm torch {t_ref:.3f} ms -> fused {t_new:.3f} ms | fused_add torch {t_ref2:.3f} ms -> fused {t_new2:.3f} ms')
    print('OK')


if __name__ == '__main__':
    main()
