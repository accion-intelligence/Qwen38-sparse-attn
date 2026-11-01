"""Dynamic block-sparse causal prefill attention for long contexts on A100 (SM80), BF16 only.

Per full-attention layer and prefill chunk (Qwen3.8-27B: H=24, Hk=4, D=256):
  1. block selection (a100_attn.select_triton): 64-row query-block means are scored against cached 64-key
     block means of K; each query head keeps the smallest set of key blocks carrying `tau` of the softmax mass,
     plus the diagonal and the sink block; for 128-row query blocks the two 64-row selections are OR-ed.
     One Triton kernel builds the ascending block lists directly (no sort, no host sync);
  2. the SM80 BF16 FlashAttention kernel (a100_attn.kernels.fa_bf16_sm80) walks each (head, query block)'s
     list, reading K/V in place from vLLM's paged cache (page size a multiple of 64, i.e. vLLM
     --block-size 64), longest rows first.

Environment (or JSON file named by A100_SPARSE_CONFIG):
  A100_SPARSE_ATTN (1)  A100_SPARSE_TAU (0.97)  A100_SPARSE_MIN_Q (1024)  A100_SPARSE_MIN_KV (8192)
  A100_SPARSE_M_BLOCK (128; 64 also supported)  A100_SPARSE_PAGED (1)  A100_SPARSE_FUSED_SELECT (1)
  A100_SPARSE_DEBUG (0: number of diagnostic lines to print)
"""
from __future__ import annotations

import json
import os

import torch

BLOCK = 64                                     # key block = kernel n-block = selection granularity
_THREADS = {64: 128, 128: 256}                 # m_block -> threads (4 / 8 MMA warps)


def paged_layout_ok(key_cache, value_cache) -> bool:
    """True for vLLM FlashAttention cache views [num_pages, page, Hk, D] that are the K and V halves of one
    [num_pages, page, Hk, 2D] buffer (vLLM 0.30 on A100): token stride Hk*2D, head stride 2D. The kernel then
    reads them in place as [num_pages * page, Hk, D]; otherwise the hook falls back to a gather copy."""
    if key_cache.shape != value_cache.shape or key_cache.stride() != value_cache.stride():
        return False
    num_pages, page, Hk, D = key_cache.shape
    s0, s1, s2, s3 = key_cache.stride()
    return s3 == 1 and s2 >= D and s2 % 8 == 0 and s1 == Hk * s2 and s0 == page * s1


class SparsePrefillA100:
    kernel = 'sm80'

    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int, tau: float = 0.97,
                 min_kv: int = 8192, min_q: int = 1024, m_block: int = 128, fused_select: bool = True):
        self.H, self.Hk, self.D = num_heads, num_kv_heads, head_dim
        self.g = num_heads // num_kv_heads
        self.tau, self.min_kv, self.min_q = tau, min_kv, min_q
        assert m_block in _THREADS, m_block
        self.m_block = m_block
        self.threads = _THREADS[m_block]
        self.fused_select = fused_select     # Triton selection kernel; False: PyTorch reference (tests)
        self.scale = head_dim ** -0.5
        self.stats = {'calls': 0, 'blocks_total': 0}
        self._kept_acc = None

    def _q_means(self, q, valid):
        """fp32 means [MB, H, D] of the real rows (valid = (first, count)) of each 64-row block of q [M, H, D].
        Rows outside `valid` are the zero padding added by _plan, so a plain block sum equals the sum over real
        rows; the sum accumulates in fp32 straight from bf16 (no fp32 copy of q)."""
        MB = q.shape[0] // BLOCK
        sums = q.view(MB, BLOCK, self.H, self.D).sum(1, dtype=torch.float32)
        first, count = valid
        starts = torch.arange(MB, device=q.device) * BLOCK
        real = (torch.minimum(starts + BLOCK, torch.full_like(starts, first + count))
                - torch.maximum(starts, torch.full_like(starts, first))).clamp(min=1)
        return sums / real[:, None, None].float()

    @torch.no_grad()
    def block_masks(self, q: torch.Tensor, q_start: int, km: torch.Tensor, valid) -> torch.Tensor:
        """PyTorch reference of the selection (sort-based). q [M, H, D] zero-padded to whole 64-row blocks at
        absolute position q_start (multiple of 64); km [Hk, NB, D] fp32 key-block means; valid = (first real
        row, real rows). Returns per-query-head masks [H, MB, NB] (bool)."""
        M = q.shape[0]
        MB, NB = M // BLOCK, km.shape[1]
        dev = q.device
        qm = self._q_means(q, valid)
        gi = (q_start // BLOCK) + torch.arange(MB, device=dev)                       # diagonal block per q block
        j = torch.arange(NB, device=dev)
        future = j[None, :] > gi[:, None]                                            # [MB, NB]
        logits = torch.einsum('mhgd,hnd->hmgn', qm.view(MB, self.Hk, self.g, self.D), km) * self.scale
        logits.masked_fill_(future[None, :, None, :], float('-inf'))
        probs = torch.softmax(logits, dim=-1)                                        # [Hk, MB, g, NB]
        srt, order = torch.sort(probs, dim=-1, descending=True)
        keep = (srt.cumsum(-1) < self.tau).sum(-1, keepdim=True) + 1
        sel = torch.zeros_like(probs, dtype=torch.bool).scatter_(-1, order, j[None, None, None, :] < keep)
        masks = sel.permute(0, 2, 1, 3).reshape(self.H, MB, NB)                      # head = kvh * g + i
        masks[:, torch.arange(MB, device=dev), gi] = True                            # diagonal
        masks[:, :, 0] = True                                                        # sink
        masks &= ~future[None]
        return masks

    def _plan(self, q, q_start, km, nb_real=None):
        """Pad q to whole m-blocks and build the kernel's block lists.

        q [n, H, D] at absolute positions q_start..; km [Hk, >= nb_real, D] fp32 key-block means.
        nb_real: when given, 64-key blocks >= nb_real (only reachable from padded query rows) are dropped so
        the kernel never reads keys past the logical end (required for the in-place paged path: stale pages
        of the hybrid KV/mamba pool may hold NaN bit patterns).
        Returns (q_padded, qs, front, counts [rows] int32, lists [rows, >= max count] int32 ascending).
        """
        BM = self.m_block
        n = q.shape[0]
        front = q_start % BM
        end = (-(front + n)) % BM
        if front or end:
            q = torch.cat([q.new_zeros(front, self.H, self.D), q, q.new_zeros(end, self.H, self.D)])
        M = q.shape[0]
        qs = q_start - front
        nb_need = (qs + M) // BLOCK                                  # diagonal blocks of every padded row
        if km.shape[1] < nb_need:
            km = torch.nn.functional.pad(km, (0, 0, 0, nb_need - km.shape[1]))
        if self.fused_select:
            from .select_triton import fused_select
            counts, lists = fused_select(self._q_means(q, (front, n)), km.contiguous(), self.scale, qs // BLOCK,
                                         nb_real, self.tau, BM, self.Hk)
        else:
            masks = self.block_masks(q, qs, km, valid=(front, n))    # [H, M/64, NB]
            if nb_real is not None and masks.shape[2] > nb_real:
                masks[:, :, nb_real:] = False
            if BM == 128:
                masks = masks.view(self.H, M // 128, 2, -1).any(2)
            counts = masks.sum(2).to(torch.int32).flatten()
            lists = torch.argsort((~masks).to(torch.int8), dim=-1, stable=True).to(torch.int32).reshape(counts.numel(), -1)
        MB64 = M // BLOCK
        self.stats['calls'] += 1
        self.stats['blocks_total'] += int(self.H * ((qs // BLOCK) * MB64 + MB64 * (MB64 + 1) // 2))
        kept = counts.sum() * (BM // BLOCK)
        self._kept_acc = kept if self._kept_acc is None else self._kept_acc + kept
        return q, qs, front, counts, lists

    @torch.no_grad()
    def forward(self, q, k, v, q_start, out=None, km=None):
        """Gather-copy path. q [n, H, D] bf16 at absolute positions q_start..; k, v NHD [T, Hk, D] bf16 with
        T >= q_start + n; km: block means [Hk, >= ceil((q_start+n)/64), D] fp32 (computed here when None)."""
        from .kernels.fa_bf16_sm80 import attention_sm80
        n = q.shape[0]
        kv_len = q_start + n
        if km is None:
            nb = -(-kv_len // BLOCK)
            kk = k[:kv_len] if kv_len % BLOCK == 0 else torch.nn.functional.pad(k[:kv_len], (0, 0, 0, 0, 0, nb * BLOCK - kv_len))
            km = kk.view(nb, BLOCK, self.Hk, self.D).float().mean(1).transpose(0, 1)
        qp, qs, front, counts, lists = self._plan(q, q_start, km)
        need = max(-(-k.shape[0] // BLOCK) * BLOCK, qs + qp.shape[0])   # keys must cover the padded query range
        if need > k.shape[0]:
            k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, need - k.shape[0]))
            v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, need - v.shape[0]))
        o = attention_sm80(qp, k, v, qs, m_block=self.m_block, n_block=BLOCK, num_threads=self.threads,
                           counts=counts, lists=lists)[front:front + n]
        if out is None:
            return o
        out.copy_(o)
        return out

    @torch.no_grad()
    def forward_paged(self, q, key_cache, value_cache, block_row, q_start, km, out=None):
        """Same as forward, reading K/V in place from vLLM's paged cache (no gather copy).

        q [n, H, D] bf16 at absolute positions q_start..; key_cache/value_cache [num_pages, page, Hk, D]
        (see paged_layout_ok; page a multiple of 64); block_row [>= ceil((q_start+n)/page)] physical pages of
        this request; km [Hk, >= ceil((q_start+n)/64), D] fp32 block means. Keys >= q_start + n are never read.
        """
        from .kernels.fa_bf16_sm80 import attention_sm80
        num_pages, page, Hk, D = key_cache.shape
        assert page % BLOCK == 0 and paged_layout_ok(key_cache, value_cache), (page, key_cache.stride())
        n = q.shape[0]
        kv_len = q_start + n
        qp, qs, front, counts, lists = self._plan(q, q_start, km, nb_real=-(-kv_len // BLOCK))
        # logical 64-key tile l -> physical tile of the [num_pages * page, Hk, D] view: (page_id * page + off) / 64
        tok = torch.arange(-(-(qs + qp.shape[0]) // BLOCK), device=q.device, dtype=torch.int64) * BLOCK
        pg = block_row.long()[(tok // page).clamp_max(block_row.numel() - 1)]
        tile_map = ((pg * page + tok % page) // BLOCK).to(torch.int32)
        s1, s2 = key_cache.stride(1), key_cache.stride(2)
        k_rows = key_cache.as_strided((num_pages * page, Hk, D), (s1, s2, 1))
        v_rows = value_cache.as_strided((num_pages * page, Hk, D), (s1, s2, 1))
        o = attention_sm80(qp, k_rows, v_rows, qs, m_block=self.m_block, n_block=BLOCK, num_threads=self.threads,
                           counts=counts, lists=lists, kv_len=kv_len, tile_map=tile_map)[front:front + n]
        if out is None:
            return o
        out.copy_(o)
        return out

    def kept_fraction(self):
        """Approximate kept / causal 64x64 blocks over all calls so far (one device sync)."""
        if self._kept_acc is None or not self.stats['blocks_total']:
            return None
        return int(self._kept_acc.item()) / self.stats['blocks_total']


# ---------------------------------------------------------------------------
# vLLM glue
# ---------------------------------------------------------------------------
_CONFIG = None
_STATE: dict = {}
_DEBUG_LEFT = [int(os.environ.get('A100_SPARSE_DEBUG', '0'))]


def _debug(msg):
    if _DEBUG_LEFT[0] > 0:
        _DEBUG_LEFT[0] -= 1
        print('[a100-sparse] ' + msg, flush=True)


def _flag(v):
    return str(v) not in ('0', 'false', 'False')


def config():
    global _CONFIG
    if _CONFIG is None:
        file_cfg = {}
        path = os.environ.get('A100_SPARSE_CONFIG')
        if path:
            with open(path) as f:
                file_cfg = json.load(f)

        def pick(env, key, default, cast):
            if env in os.environ:
                return cast(os.environ[env])
            return cast(file_cfg.get(key, default))
        _CONFIG = {
            'enabled': pick('A100_SPARSE_ATTN', 'enabled', 1, _flag),
            'tau': pick('A100_SPARSE_TAU', 'tau', 0.97, float),
            'min_kv': pick('A100_SPARSE_MIN_KV', 'min_kv', 8192, int),
            'min_q': pick('A100_SPARSE_MIN_Q', 'min_q', 1024, int),
            'm_block': pick('A100_SPARSE_M_BLOCK', 'm_block', 128, int),
            'paged': pick('A100_SPARSE_PAGED', 'paged', 1, _flag),
            'fused_select': pick('A100_SPARSE_FUSED_SELECT', 'fused_select', 1, _flag),
        }
        print('[a100-sparse] config %r' % (_CONFIG,), flush=True)
    return _CONFIG


def _state(num_heads, num_kv_heads, head_dim, device):
    key = (num_heads, num_kv_heads, head_dim, str(device))
    if key not in _STATE:
        cfg = config()
        _STATE[key] = SparsePrefillA100(num_heads, num_kv_heads, head_dim, tau=cfg['tau'], min_kv=cfg['min_kv'],
                                        min_q=cfg['min_q'], m_block=cfg['m_block'],
                                        fused_select=cfg['fused_select'])
    return _STATE[key]


_KMEAN_CACHE: dict = {}
_KMEAN_MAX_ENTRIES = 128


def _cached_block_means(layer_key, first_page, key_chunk, q_start, kv_len, key_all=None):
    """Per-(layer, request) K block means [Hk, NB, D] (fp32), updated incrementally from each chunk's keys.

    A request is identified by its first physical page; an entry is trusted only when its last chunk ended
    exactly at q_start, otherwise it is rebuilt from the full key history `key_all` [kv_len, Hk, D] (a tensor,
    or a callable returning it, so the paged path gathers the history only when a rebuild is needed).
    (Same logic as sm120_attn.sparse_prefill._cached_block_means.)
    """
    n, Hk, D = key_chunk.shape
    cache_key = (layer_key, first_page)
    entry = _KMEAN_CACHE.get(cache_key)
    b0, b1 = q_start // BLOCK, -(-(q_start + n) // BLOCK)
    if entry is None or q_start == 0 or entry['nb'] < b0 or entry.get('last_kv') != q_start:
        if len(_KMEAN_CACHE) >= _KMEAN_MAX_ENTRIES:
            _KMEAN_CACHE.pop(next(iter(_KMEAN_CACHE)))
        cap = max(4096, b1)
        entry = {'km': torch.zeros(Hk, cap, D, dtype=torch.float32, device=key_chunk.device), 'nb': 0, 'last_kv': 0}
        _KMEAN_CACHE[cache_key] = entry
        if q_start > 0:
            if key_all is None:
                raise RuntimeError('stale block-mean cache entry and no key history to rebuild it')
            if callable(key_all):
                key_all = key_all()
            hist = key_all[:q_start]
            full, rem = q_start // BLOCK, q_start % BLOCK
            if full:
                entry['km'][:, :full] = hist[:full * BLOCK].float().view(full, BLOCK, Hk, D).mean(1).transpose(0, 1)
            if rem:
                entry['km'][:, full] = hist[full * BLOCK:].float().sum(0) / rem
            entry['nb'] = full + (1 if rem else 0)
            key_chunk = key_chunk if rem == 0 else torch.cat([hist[full * BLOCK:], key_chunk], 0)
            if rem:
                q_start, n = full * BLOCK, n + rem
                b0 = full
    if entry['km'].shape[1] < b1:
        grown = torch.zeros(Hk, max(b1, 2 * entry['km'].shape[1]), D, dtype=torch.float32, device=key_chunk.device)
        grown[:, :entry['km'].shape[1]] = entry['km']
        entry['km'] = grown
    front = q_start - b0 * BLOCK
    rows = (b1 - b0) * BLOCK
    padded = key_chunk.new_zeros(rows, Hk, D, dtype=torch.float32)
    padded[front:front + n] = key_chunk.float()
    valid = torch.zeros(rows, device=key_chunk.device, dtype=torch.float32)
    valid[front:front + n] = 1.0
    cnt = valid.view(b1 - b0, BLOCK).sum(1).clamp_min(1.0)
    entry['km'][:, b0:b1] = (padded.view(b1 - b0, BLOCK, Hk, D).sum(1) / cnt[:, None, None]).transpose(0, 1)
    entry['nb'] = b1
    entry['last_kv'] = kv_len
    return entry['km'][:, :b1]


@torch.no_grad()
def try_sparse_forward(impl, query, key, key_cache, value_cache, output, attn_metadata) -> bool:
    """Block-sparse attention for the long prefill chunks of this batch; other requests of the batch get a
    per-request dense FlashAttention call. Returns False (nothing written) when no request qualifies."""
    cfg = config()
    if not cfg['enabled']:
        return False
    window = getattr(impl, 'sliding_window', None)
    windowed = window is not None and any(int(w) >= 0 for w in window)
    if attn_metadata.causal is not True or windowed:
        _debug('fallback: causal=%r window=%r' % (attn_metadata.causal, window))
        return False
    if attn_metadata.max_query_len < cfg['min_q'] or attn_metadata.max_seq_len < cfg['min_kv']:
        return False
    if torch.cuda.is_current_stream_capturing():
        return False
    qsl = attn_metadata.query_start_loc.tolist()
    seq_lens = attn_metadata.seq_lens.tolist()
    nreq = len(seq_lens)
    eligible = [r for r in range(nreq) if qsl[r + 1] - qsl[r] >= cfg['min_q'] and seq_lens[r] >= cfg['min_kv']]
    if not eligible:
        return False
    H, D = query.shape[1], query.shape[2]
    Hk = key_cache.shape[2]
    sp = _state(H, Hk, D, query.device)
    page = key_cache.shape[1]
    bt = attn_metadata.block_table
    # in-place paged read needs whole 64-key blocks per page (vLLM --block-size 64 -> page 832 for Qwen3.8)
    # and vLLM's FlashAttention cache layout; otherwise fall back to the gather copy
    paged_ok = cfg['paged'] and page % BLOCK == 0 and paged_layout_ok(key_cache, value_cache)
    _debug('sparse path: %d/%d requests, q_lens=%s kv_lens=%s page=%d paged=%s'
           % (len(eligible), nreq, [qsl[i + 1] - qsl[i] for i in range(nreq)], seq_lens, page, paged_ok))
    first_pages = bt[:nreq, 0].tolist()
    for r in range(nreq):
        kv_len = seq_lens[r]
        n = qsl[r + 1] - qsl[r]
        q_r = query[qsl[r]:qsl[r + 1]]
        o_r = output[qsl[r]:qsl[r + 1]]
        if r not in eligible:
            _dense_request(impl, q_r, key_cache, value_cache, o_r, bt[r], n, kv_len)
            continue
        npages = -(-kv_len // page)
        pages = bt[r, :npages].long()
        if paged_ok:
            gather = lambda: key_cache[pages].reshape(npages * page, Hk, D)   # noqa: E731
            km = _cached_block_means(id(impl), first_pages[r], key[qsl[r]:qsl[r + 1]], kv_len - n, kv_len,
                                     key_all=gather)
            sp.forward_paged(q_r, key_cache, value_cache, bt[r], kv_len - n, km, out=o_r)
            continue
        k = key_cache[pages].reshape(npages * page, Hk, D)      # one gather copy, NHD
        v = value_cache[pages].reshape(npages * page, Hk, D)
        km = _cached_block_means(id(impl), first_pages[r], key[qsl[r]:qsl[r + 1]], kv_len - n, kv_len, key_all=k)
        sp.forward(q_r, k, v, kv_len - n, out=o_r, km=km)
    return True


def _dense_request(impl, query, key_cache, value_cache, output, bt_row, n, kv_len):
    """Dense paged FlashAttention for one request of a mixed batch (same call as the backend)."""
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    dev = query.device
    cu = torch.tensor([0, n], dtype=torch.int32, device=dev)
    su = torch.tensor([kv_len], dtype=torch.int32, device=dev)
    window = getattr(impl, 'sliding_window', None)
    flash_attn_varlen_func(q=query, k=key_cache, v=value_cache, out=output, cu_seqlens_q=cu, max_seqlen_q=n,
                           seqused_k=su, max_seqlen_k=kv_len, softmax_scale=impl.scale, causal=True,
                           alibi_slopes=getattr(impl, 'alibi_slopes', None),
                           window_size=list(window) if window is not None else None,
                           block_table=bt_row.unsqueeze(0), softcap=getattr(impl, 'logits_soft_cap', 0.0),
                           fa_version=getattr(impl, 'vllm_flash_attn_version', 2))
