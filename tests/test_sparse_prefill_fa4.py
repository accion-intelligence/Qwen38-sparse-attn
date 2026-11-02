"""Correctness and timing of the SM90 path: SparsePrefill(kernel='fa4') = FlashAttention-4 CuTe block-sparse forward.

    CUDA_VISIBLE_DEVICES=4 python tests/test_sparse_prefill_fa4.py [--big]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sm120_attn.sparse_prefill import BLOCK, FA4_QB, SparsePrefill  # noqa: E402

H, Hk, D = 24, 4, 256


def make(n, kv, dev, peaky=3.0):
    q = torch.randn(n, H, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(kv, Hk, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(kv, Hk, D, device=dev, dtype=torch.bfloat16)
    hot = (torch.rand(kv // BLOCK + 1, device=dev) < 0.1).repeat_interleave(BLOCK)[:kv]
    u = torch.randn(Hk, 1, D, device=dev)
    k = (k.float() * 0.5 + hot[:, None, None] * u.transpose(0, 1) * peaky).to(torch.bfloat16)
    q = (q.float() + u.repeat_interleave(H // Hk, 0).transpose(0, 1) * 1.0).to(torch.bfloat16)
    return q, k, v


def reference(q, k, v, q_start, masks=None):
    """fp32 causal attention; q [n,H,D] at positions q_start.., k/v NHD [kv,Hk,D]; masks [H, MB(FA4_QB rows), NB] bool."""
    n = q.shape[0]
    kv = k.shape[0]
    g = H // Hk
    pos_q = q_start + torch.arange(n, device=q.device)
    pos_k = torch.arange(kv, device=q.device)
    allowed = pos_k[None, :] <= pos_q[:, None]
    outs = []
    for h in range(H):
        m = allowed.clone()
        if masks is not None:
            m &= masks[h].repeat_interleave(FA4_QB, 0)[:n].repeat_interleave(BLOCK, 1)[:, :kv]
        s = (q[:, h].float() @ k[:, h // g].float().T) * D ** -0.5
        s.masked_fill_(~m, float('-inf'))
        outs.append(torch.softmax(s, -1) @ v[:, h // g].float())
    return torch.stack(outs, 1)


def masks_from_lists(lists, NB):
    """FA4 packed lists -> bool masks [H, MB, NB] (union of mask and full lists)."""
    mc, mi, fc, fi = (t[0] for t in lists)
    Hh, MB, _ = mi.shape
    dev = mi.device
    r = torch.arange(mi.shape[-1], device=dev)
    out = torch.zeros(Hh, MB, NB + 1, dtype=torch.int32, device=dev)      # column NB collects the padding entries
    for cnt, idx in ((mc, mi), (fc, fi)):
        sel = r[None, None, :] < cnt[:, :, None]
        out.scatter_add_(2, torch.where(sel, idx.long().clamp(0, NB - 1), NB), sel.to(torch.int32))
    return out[:, :, :NB] > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--big', action='store_true', help='also time 8192 x 65536')
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = 'cuda'
    print('device:', torch.cuda.get_device_name(0))
    print('--- correctness: FA4 block-sparse vs fp32 masked reference (and vs dense)')
    worst = 0.0
    for n, kv, tau in ((256, 1024, 1.0), (1000, 4024, 0.9), (64, 4096, 0.9), (2048, 2048, 0.95), (777, 3001, 0.95), (1024, 8192, 0.9)):
      q, k, v = make(n, kv, dev)
      for fused in (False, True):
        sp = SparsePrefill(H, Hk, D, dev, tau=tau, min_kv=0, min_q=0, kernel='fa4', fused_select=fused)
        q_start = kv - n
        o = sp.forward(q, k, v, q_start)
        masks = sp.last_masks if sp.last_masks is not None else masks_from_lists(sp.last_lists, -(-kv // BLOCK))
        ref_masked = reference(q, k, v, q_start, masks)
        ref_dense = reference(q, k, v, q_start)
        e_mask = (o.float() - ref_masked).abs().max().item() / ref_masked.abs().max().item()
        e_dense = (o.float() - ref_dense).norm().item() / ref_dense.norm().item()
        worst = max(worst, e_mask)
        print(f'  n={n:5d} kv={kv:5d} q_start={q_start:5d} tau={tau} fused={int(fused)}: kept {sp.kept_fraction():.3f} | max rel err vs masked ref {e_mask:.2e} | rel L2 vs dense {e_dense:.3e}')
    # padded-history input (T > kv_len) must give the same result
    n, kv = 1000, 4024
    sp = SparsePrefill(H, Hk, D, dev, tau=0.9, min_kv=0, min_q=0, kernel='fa4')
    q, k, v = make(n, kv, dev)
    o1 = sp.forward(q, k, v, kv - n)
    kp, vp = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, 300)), torch.nn.functional.pad(v, (0, 0, 0, 0, 0, 300))
    o2 = sp.forward(q, kp, vp, kv - n)
    print(f'  padded history (T = kv_len + 300): max abs diff {(o1 - o2).abs().max().item():.2e}')
    assert worst < 2e-2, f'FA4 sparse path error too large: {worst}'
    print('--- fused Triton selection vs torch selection (same lists?)')
    mism = 0
    for n, kv, tau, q_start in ((1000, 4024, 0.9, 3024), (2048, 8192, 0.95, 6144), (640, 4096, 0.9, 1000), (8192, 65536, 0.95, 57344)):
        q, k, v = make(n, kv, dev)
        a = SparsePrefill(H, Hk, D, dev, tau=tau, min_kv=0, min_q=0, kernel='fa4', fused_select=False)
        b = SparsePrefill(H, Hk, D, dev, tau=tau, min_kv=0, min_q=0, kernel='fa4', fused_select=True)
        oa = a.forward(q, k, v, q_start)
        ob = b.forward(q, k, v, q_start)
        def sets(lists):
            mc, mi, fc, fi = (t[0] for t in lists)
            Hh, MB, _ = mi.shape
            out = []
            for h in range(Hh):
                for m in range(MB):
                    out.append((frozenset(mi[h, m, :mc[h, m]].tolist()), frozenset(fi[h, m, :fc[h, m]].tolist())))
            return out
        sa, sb = sets(a.last_lists), sets(b.last_lists)
        diff = sum(1 for x, y in zip(sa, sb) if x != y)
        extra = sum(len((x[0] | x[1]) ^ (y[0] | y[1])) for x, y in zip(sa, sb))
        mism += diff
        print(f'  n={n:5d} kv={kv:5d} q_start={q_start:5d} tau={tau}: rows differing {diff}/{len(sa)} (blocks differing {extra}) | '
              f'kept torch {a.kept_fraction():.4f} fused {b.kept_fraction():.4f} | max abs out diff {(oa - ob).abs().max().item():.2e}')
    print('OK' if worst < 2e-2 else 'FAIL')
    if args.big:
        from vllm.vllm_flash_attn import flash_attn_varlen_func
        n, kv = 8192, 65536
        q, k, v = make(n, kv, dev)
        fl = 4 * H * D * n * (kv - n + (n + 1) / 2)
        cu_q = torch.tensor([0, n], dtype=torch.int32, device=dev)
        cu_k = torch.tensor([0, kv], dtype=torch.int32, device=dev)
        def dense():
            return flash_attn_varlen_func(q=q, k=k, v=v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=n, max_seqlen_k=kv,
                                          softmax_scale=D ** -0.5, causal=True, fa_version=3)
        for _ in range(3): dense()
        torch.cuda.synchronize(); t = time.time()
        for _ in range(5): dense()
        torch.cuda.synchronize(); ms_d = (time.time() - t) / 5 * 1000
        print(f'--- timing 8192 x 65536: dense FA3 {ms_d:.2f} ms ({fl / ms_d / 1e9:.0f} TFLOPS)')
        for tau, fused in ((0.95, False), (0.95, True), (0.9, True)):
            sp = SparsePrefill(H, Hk, D, dev, tau=tau, min_kv=0, min_q=0, kernel='fa4', fused_select=fused)
            for _ in range(2): sp.forward(q, k, v, kv - n)
            sp.timing = True
            sp.stats.update(run_ms=0.0, select_ms=0.0, blocks_total=0, calls=0); sp._kept_acc = None
            for _ in range(5): sp.forward(q, k, v, kv - n)
            st = sp.stats
            kept = sp.kept_fraction()
            print(f'  fa4 tau={tau} fused_select={fused}: kept {kept:.3f} | select+lists {st["select_ms"] / 5:.2f} ms, attention {st["run_ms"] / 5:.2f} ms'
                  f' ({fl * kept / (st["run_ms"] / 5) / 1e9:.0f} TFLOPS on kept) | total {(st["select_ms"] + st["run_ms"]) / 5:.2f} ms = {ms_d / ((st["select_ms"] + st["run_ms"]) / 5):.2f}x dense FA3')


if __name__ == '__main__':
    main()
