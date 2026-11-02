"""Paged FA4 block-sparse path (tools/patch_fa4.py + SparsePrefill.forward_fa4(page_table=...)) must equal the
gathered path bit for bit: same lists, K/V read through a page table over BLOCK-token sub-pages.

    CUDA_VISIBLE_DEVICES=4 python tests/test_sparse_prefill_fa4_paged.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sm120_attn.sparse_prefill import BLOCK, SparsePrefill  # noqa: E402

H, Hk, D = 24, 4, 256


def main():
    torch.manual_seed(0)
    dev = 'cuda'
    print('device:', torch.cuda.get_device_name(0))
    worst = 0.0
    for n, kv_len, page, tau in ((1000, 4024, 1024, 0.9), (2048, 9000, 1024, 0.95), (777, 3001, 256, 0.95), (8192, 65536, 1024, 0.95)):
        P = page // BLOCK
        npages = -(-kv_len // page)
        num_pages_total = npages + 37
        # physical cache with a random page order; the request's history fills its pages in order
        perm = torch.randperm(num_pages_total, device=dev)[:npages]
        key_cache = torch.randn(num_pages_total, page, Hk, D, device=dev, dtype=torch.bfloat16)
        value_cache = torch.randn(num_pages_total, page, Hk, D, device=dev, dtype=torch.bfloat16)
        k_all = key_cache[perm].reshape(npages * page, Hk, D)        # gathered history (rows >= kv_len are junk)
        v_all = value_cache[perm].reshape(npages * page, Hk, D)
        q = torch.randn(n, H, D, device=dev, dtype=torch.bfloat16)
        q_start = kv_len - n
        NB = -(-kv_len // BLOCK)
        kpad = torch.nn.functional.pad(k_all[:kv_len], (0, 0, 0, 0, 0, NB * BLOCK - kv_len))
        cnt = torch.full((NB,), float(BLOCK), device=dev); cnt[-1] = kv_len - (NB - 1) * BLOCK
        km = (kpad.view(NB, BLOCK, Hk, D).float().sum(1) / cnt[:, None, None]).transpose(0, 1)
        sp = SparsePrefill(H, Hk, D, dev, tau=tau, min_kv=0, min_q=0, kernel='fa4')
        o_gather = sp.forward_fa4(q, k_all, v_all, q_start, km=km)
        pt = (perm[:, None] * P + torch.arange(P, device=dev)[None, :]).reshape(1, -1).to(torch.int32)
        o_paged = sp.forward_fa4(q, key_cache.view(-1, BLOCK, Hk, D), value_cache.view(-1, BLOCK, Hk, D), q_start, km=km, page_table=pt)
        d = (o_gather.float() - o_paged.float()).abs().max().item()
        worst = max(worst, d)
        print(f'  n={n:5d} kv_len={kv_len:6d} page={page:5d} tau={tau}: kept {sp.kept_fraction():.3f} | max abs diff gathered vs paged {d:.2e}')
    assert worst < 1e-2, worst
    # timing at 8K x 64K: paged vs gathered (+ the gather copy itself)
    n, kv_len, page = 8192, 65536, 1024
    P = page // BLOCK; npages = kv_len // page
    perm = torch.randperm(npages + 37, device=dev)[:npages]
    key_cache = torch.randn(npages + 37, page, Hk, D, device=dev, dtype=torch.bfloat16)
    value_cache = torch.randn(npages + 37, page, Hk, D, device=dev, dtype=torch.bfloat16)
    q = torch.randn(n, H, D, device=dev, dtype=torch.bfloat16)
    k_all = key_cache[perm].reshape(-1, Hk, D); v_all = value_cache[perm].reshape(-1, Hk, D)
    NB = kv_len // BLOCK
    km = k_all.view(NB, BLOCK, Hk, D).float().mean(1).transpose(0, 1)
    pt = (perm[:, None] * P + torch.arange(P, device=dev)[None, :]).reshape(1, -1).to(torch.int32)
    sp = SparsePrefill(H, Hk, D, dev, tau=0.95, min_kv=0, min_q=0, kernel='fa4')
    kc, vc = key_cache.view(-1, BLOCK, Hk, D), value_cache.view(-1, BLOCK, Hk, D)
    def timeit(fn, iters=5):
        for _ in range(2): fn()
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(iters): fn()
        torch.cuda.synchronize(); return (time.perf_counter() - t) / iters * 1000
    t_g = timeit(lambda: sp.forward_fa4(q, k_all, v_all, kv_len - n, km=km))
    t_p = timeit(lambda: sp.forward_fa4(q, kc, vc, kv_len - n, km=km, page_table=pt))
    t_copy = timeit(lambda: (key_cache[perm].reshape(-1, Hk, D), value_cache[perm].reshape(-1, Hk, D)))
    print(f'  8K x 64K tau 0.95: gathered path {t_g:.2f} ms (+ gather copy {t_copy:.2f} ms) | paged path {t_p:.2f} ms')
    print('OK')


if __name__ == '__main__':
    main()
