"""Correctness of a100_attn.SparsePrefillA100 (A100 / SM80).

1. gather path: output vs an fp32 reference using the block lists the planner produced, for m_block 64/128,
   with the fused Triton selection and with the PyTorch reference selection;
2. paged path: forward_paged on a shuffled, NaN-poisoned paged cache in vLLM's FlashAttention layout
   ([pages, page, Hk, 2D] buffer, K/V halves) must match the gather path (keys past kv_len never read).

    python tests/test_a100_sparse_prefill.py
"""
import torch

from a100_attn.sparse_prefill import BLOCK, SparsePrefillA100

H, HK, D = 24, 4, 256


def reference(q, k, v, q_start, masks64):
    """q [n,H,D] at positions q_start..; k/v [kv,Hk,D]; masks64 [H, MB, NB] over 64-row blocks from row block q_start//64."""
    n, kv = q.shape[0], k.shape[0]
    g = H // HK
    pos_q = q_start + torch.arange(n, device=q.device)
    pos_k = torch.arange(kv, device=q.device)
    allowed = pos_k[None, :] <= pos_q[:, None]
    rb = (pos_q // BLOCK) - (q_start // BLOCK)
    outs = []
    for h in range(H):
        s = (q[:, h].float() @ k[:, h // g].float().T) * D ** -0.5
        m = allowed & masks64[h][rb].repeat_interleave(BLOCK, 1)[:, :kv]
        s.masked_fill_(~m, float('-inf'))
        outs.append(torch.softmax(s, -1) @ v[:, h // g].float())
    return torch.stack(outs, 1)


def masks_from_lists(counts, lists, sub):
    """[H, MB64, NB] bool from kernel lists (rows = H * MBo, 64-key blocks); sub = m_block // 64."""
    rows, L = lists.shape
    keep = torch.arange(L, device=lists.device)[None, :] < counts[:, None]
    idx = lists.long()
    NB = int(idx[keep].max().item()) + 1
    m = torch.zeros(rows, NB, dtype=torch.bool, device=lists.device)
    m[torch.arange(rows, device=lists.device)[:, None].expand_as(idx)[keep], idx[keep]] = True
    return m.view(H, rows // H, NB).repeat_interleave(sub, 1)


def make_inputs(n, kv, seed):
    """Random q/k/v with 10% "hot" key blocks so that selection is non-trivial."""
    torch.manual_seed(seed)
    q = torch.randn(n, H, D, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(kv, HK, D, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(kv, HK, D, device='cuda', dtype=torch.bfloat16)
    hot = (torch.rand(kv // BLOCK + 1, device='cuda') < 0.1).repeat_interleave(BLOCK)[:kv]
    u = torch.randn(1, HK, D, device='cuda')
    k = (k.float() * 0.5 + hot[:, None, None] * u * 3).to(torch.bfloat16)
    return q, k, v


def run(n, kv, tau, m_block, fused):
    q_start = kv - n
    q, k, v = make_inputs(n, kv, n + kv)
    sp = SparsePrefillA100(H, HK, D, tau=tau, min_kv=0, min_q=0, m_block=m_block, fused_select=fused)
    captured = {}
    orig = sp._plan

    def spy(*a, **kw):
        captured['r'] = orig(*a, **kw)
        return captured['r']
    sp._plan = spy
    o = sp.forward(q, k, v, q_start)
    _, qs, front, counts, lists = captured['r']
    masks = masks_from_lists(counts, lists, m_block // BLOCK)[:, front // BLOCK:]   # rows from q_start's block
    nbk = -(-kv // BLOCK)
    if masks.shape[2] < nbk:
        masks = torch.nn.functional.pad(masks, (0, nbk - masks.shape[2]))
    ref = reference(q, k, v, q_start, masks)
    err = (o.float() - ref).abs().max().item() / ref.abs().max().item()
    ok = err < 1e-2
    print(f'gather n={n:5d} kv={kv:6d} q_start%64={q_start % 64:2d} tau={tau} m{m_block} '
          f'{"fused" if fused else "torch"}: kept {sp.kept_fraction():.3f} max rel err {err:.2e} '
          f'{"OK" if ok else "FAIL"}')
    return ok


def run_paged(n, kv, tau, m_block, page=832):
    q_start = kv - n
    q, k, v = make_inputs(n, kv, 7 * n + kv)
    npages = -(-kv // page)
    total = npages + 5
    perm = torch.randperm(total, device='cuda')[:npages]
    flat = torch.full((npages * page, HK, 2 * D), float('nan'), device='cuda', dtype=torch.bfloat16)
    flat[:kv, :, :D], flat[:kv, :, D:] = k, v           # slots past kv stay NaN (stale-page stand-in)
    buf = torch.full((total, page, HK, 2 * D), float('nan'), device='cuda', dtype=torch.bfloat16)
    buf[perm] = flat.view(npages, page, HK, 2 * D)
    kc, vc = buf.split(D, dim=-1)                       # vLLM FlashAttention views [pages, page, Hk, D]
    block_row = torch.cat([perm, torch.zeros(64, dtype=perm.dtype, device='cuda')]).to(torch.int32)
    nb = -(-kv // BLOCK)
    km = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, nb * BLOCK - kv)).view(nb, BLOCK, HK, D).float().mean(1).transpose(0, 1)
    ref = SparsePrefillA100(H, HK, D, tau=tau, min_kv=0, min_q=0, m_block=m_block).forward(q, k, v, q_start, km=km)
    o = SparsePrefillA100(H, HK, D, tau=tau, min_kv=0, min_q=0, m_block=m_block).forward_paged(
        q, kc, vc, block_row, q_start, km)
    finite = bool(torch.isfinite(o).all())
    err = (o.float() - ref.float()).abs().max().item() / ref.float().abs().max().item() if finite else float('inf')
    ok = finite and err < 1e-2
    print(f'paged  n={n:5d} kv={kv:6d} kv%64={kv % 64:2d} m{m_block}: finite={finite} '
          f'max rel err vs gather path {err:.2e} {"OK" if ok else "FAIL"}')
    return ok


if __name__ == '__main__':
    ok = True
    for fused in (True, False):
        for mb in (128, 64):
            for n, kv, tau in ((1024, 8192, 0.95), (1000, 9001, 0.95), (2048, 16384, 0.9), (777, 12345, 1.0)):
                ok &= run(n, kv, tau, mb, fused)
    for mb in (128, 64):
        for n, kv in ((1024, 8192), (1000, 9001), (2048, 16384 + 20), (777, 12345), (1500, 9000 + 40)):
            ok &= run_paged(n, kv, 0.97, mb)
    print('ALL OK' if ok else 'SOME FAILED')
