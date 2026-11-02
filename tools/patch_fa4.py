"""Apply the H200-branch change to the installed FlashAttention-4 CuTe package (flash-attn-4 4.0.0b37):
block-sparse lists over a *paged* KV cache on SM90.

Upstream `flash_fwd_sm90.py` loads block-sparse K/V tiles with the TMA closures directly ("not paged"), so
the vLLM hook had to gather each request's whole KV history into a contiguous buffer per layer and chunk.
This patch maps every listed key block through the page table when one is given (page_size == tile_n), so
the kernel reads vLLM's cache in place.

    python tools/patch_fa4.py [--check]           # patches the flash_attn.cute package of the current python
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ANCHOR = '''                    if is_kv_load_warp:
                        kv_producer_state = produce_block_sparse_loads(
                            blocksparse_tensors,
                            batch_idx,
                            head_idx,
                            m_block,
                            seqlen,
                            kv_producer_state,
                            tma_load_K_fn,
                            tma_load_V_fn,
'''
PATCHED = '''                    # KAI H200: block-sparse lists over a paged KV cache (page_size == tile_n): map each listed
                    # key block to its physical page through the page table instead of gathering K/V first.
                    if const_expr(mPageTable is not None):
                        def sparse_load_K_fn(src_idx, producer_state):
                            tma_load_K_fn(src_idx=mPageTable[batch_idx, src_idx], producer_state=producer_state)

                        def sparse_load_V_fn(src_idx, producer_state):
                            tma_load_V_fn(src_idx=mPageTable[batch_idx, src_idx], producer_state=producer_state)
                    else:
                        sparse_load_K_fn, sparse_load_V_fn = tma_load_K_fn, tma_load_V_fn
                    if is_kv_load_warp:
                        kv_producer_state = produce_block_sparse_loads(
                            blocksparse_tensors,
                            batch_idx,
                            head_idx,
                            m_block,
                            seqlen,
                            kv_producer_state,
                            sparse_load_K_fn,
                            sparse_load_V_fn,
'''


# --- patch 2: producer-side prefetch of block-list entries (block_sparse_utils.load_block_list) ---------------
# NCU (docs/validation/h200-ncu-fa4sparse-vs-dense.md): the sparse producer issues one dependent global load of
# the list entry before every TMA; with divergent lists (62% L2 hit) that latency sits in front of each block.
# Load the *next* entry before blocking in producer_acquire, so the wait hides it (software pipelining).
ANCHOR2 = '''    if block_count > 0:
        total_blocks = block_count * kv_subtile_factor
        if const_expr(not intra_wg_overlap):
            for offset in cutlass.range(total_blocks):
                n_block = sparse_physical_n_block_forward(
                    block_indices,
                    total_blocks - 1 - offset,
                    kv_subtile_factor,
                )
                pipeline_k.producer_acquire(kv_producer_state)
                load_K(src_idx=n_block, producer_state=kv_producer_state)
                pipeline_v.producer_acquire(kv_producer_state)
                load_V(src_idx=n_block, producer_state=kv_producer_state)
                kv_producer_state.advance()
        else:
            n_block_first = sparse_physical_n_block_forward(
                block_indices,
                total_blocks - 1,
                kv_subtile_factor,
            )
            if const_expr(not first_block_preloaded):
                pipeline_k.producer_acquire(kv_producer_state)
                load_K(src_idx=n_block_first, producer_state=kv_producer_state)

            for idx in cutlass.range(total_blocks - 1, unroll=1):
                n_block_prev = sparse_physical_n_block_forward(
                    block_indices,
                    total_blocks - 1 - idx,
                    kv_subtile_factor,
                )
                n_block = sparse_physical_n_block_forward(
                    block_indices,
                    total_blocks - 1 - (idx + 1),
                    kv_subtile_factor,
                )
                kv_producer_state_prev = kv_producer_state.clone()
                kv_producer_state.advance()
                pipeline_k.producer_acquire(kv_producer_state)
                load_K(src_idx=n_block, producer_state=kv_producer_state)
                pipeline_v.producer_acquire(kv_producer_state_prev)
                load_V(src_idx=n_block_prev, producer_state=kv_producer_state_prev)

    return kv_producer_state
'''
PATCHED2 = '''    if block_count > 0:
        total_blocks = block_count * kv_subtile_factor
        # KAI H200: the list entry for the next block is loaded before producer_acquire blocks, so its global-
        # memory latency overlaps the pipeline wait instead of preceding every TMA issue.
        if const_expr(not intra_wg_overlap):
            n_block = sparse_physical_n_block_forward(block_indices, total_blocks - 1, kv_subtile_factor)
            for offset in cutlass.range(total_blocks, unroll=1):
                n_block_next = sparse_physical_n_block_forward(
                    block_indices,
                    cutlass.max(total_blocks - 2 - offset, 0),
                    kv_subtile_factor,
                )
                pipeline_k.producer_acquire(kv_producer_state)
                load_K(src_idx=n_block, producer_state=kv_producer_state)
                pipeline_v.producer_acquire(kv_producer_state)
                load_V(src_idx=n_block, producer_state=kv_producer_state)
                kv_producer_state.advance()
                n_block = n_block_next
        else:
            n_block_first = sparse_physical_n_block_forward(
                block_indices,
                total_blocks - 1,
                kv_subtile_factor,
            )
            if const_expr(not first_block_preloaded):
                pipeline_k.producer_acquire(kv_producer_state)
                load_K(src_idx=n_block_first, producer_state=kv_producer_state)
            n_block_cur = n_block_first                                   # K in flight, V still to issue
            n_block_nxt = sparse_physical_n_block_forward(
                block_indices,
                cutlass.max(total_blocks - 2, 0),
                kv_subtile_factor,
            )
            for idx in cutlass.range(total_blocks - 1, unroll=1):
                n_block_nxt2 = sparse_physical_n_block_forward(
                    block_indices,
                    cutlass.max(total_blocks - 3 - idx, 0),
                    kv_subtile_factor,
                )
                kv_producer_state_prev = kv_producer_state.clone()
                kv_producer_state.advance()
                pipeline_k.producer_acquire(kv_producer_state)
                load_K(src_idx=n_block_nxt, producer_state=kv_producer_state)
                pipeline_v.producer_acquire(kv_producer_state_prev)
                load_V(src_idx=n_block_cur, producer_state=kv_producer_state_prev)
                n_block_cur = n_block_nxt
                n_block_nxt = n_block_nxt2

    return kv_producer_state
'''


# --- patch 3: no mask function on 'full' blocks in the sparse consumer -----------------------------------------
# flash_fwd_sm90 builds mask_fn = partial(apply_mask, mask_causal=is_causal, ...). The dense path passes mask_fn=None
# for interior blocks, but consume_block_sparse_loads calls mask_fn(mask_mod=None, ...) on every 'full' block, so a
# causal kernel runs the causal-mask code (plus the dependent list-index load) on blocks that need no masking.
# 'full' blocks are entirely visible and never cross seqlen_k by construction (sm120_attn lists), so None is exact.
FULL_BLOCK_MASK_FNS = [
    'mask_fn=partial(mask_fn, mask_seqlen=True),',
    'mask_fn=partial(mask_fn, mask_seqlen=False),',
    'mask_fn=partial(mask_fn, mask_mod=None, mask_seqlen=True),',
    'mask_fn=partial(mask_fn, mask_mod=None, mask_seqlen=False),',
]
FULL_BLOCK_MARK = '# KAI H200: full blocks skip the mask function\n'


def _rewrite_full_block_masks(body: str) -> tuple[str, int]:
    """Replace the full-block mask_fn partials by None, except inside process_first_half_block(...) calls:
    first_half_block_overlap calls mask_fn unconditionally (seqlen masking of the first block)."""
    out, n, in_first = [], 0, False
    for line in body.split('\n'):
        if 'process_first_half_block(' in line:
            in_first = True
        if not in_first:
            for x in FULL_BLOCK_MASK_FNS:
                if x in line:
                    line = line.replace(x, 'mask_fn=None,  # KAI H200: full block, no masking')
                    n += 1
        elif line.strip() == ')':
            in_first = False
        out.append(line)
    return '\n'.join(out), n


def _apply_full_block_mask(path: Path, check: bool) -> str:
    text = path.read_text()
    start = text.find('def consume_block_sparse_loads(')
    end = text.find('\n@cute.jit', start + 10)
    if start < 0 or end < 0:
        return 'ANCHOR NOT FOUND (consume_block_sparse_loads)'
    body = text[start:end]
    if FULL_BLOCK_MARK in text:
        # repair an earlier version of this patch that also nulled the process_first_half_block sites
        fixed, in_first, k = [], False, 0
        for line in body.split('\n'):
            if 'process_first_half_block(' in line:
                in_first = True
            if in_first and 'mask_fn=None,  # KAI H200' in line:
                line = line.replace('mask_fn=None,  # KAI H200: full block, no masking', 'mask_fn=partial(mask_fn, mask_mod=None, mask_seqlen=True),')
                k += 1
            if in_first and line.strip() == ')':
                in_first = False
            fixed.append(line)
        if k and not check:
            path.write_text(text[:start] + '\n'.join(fixed) + text[end:])
            for pyc in (path.parent / '__pycache__').glob(path.stem + '.*.pyc'):
                pyc.unlink()
            return f'already patched (repaired {k} first-block sites)'
        return 'already patched'
    body, n = _rewrite_full_block_masks(body)
    if n == 0:
        return 'ANCHOR NOT FOUND (no full-block mask_fn partials)'
    if check:
        return f'would patch ({n} call sites)'
    body = body.replace('def consume_block_sparse_loads(', FULL_BLOCK_MARK + 'def consume_block_sparse_loads(', 1)
    path.write_text(text[:start] + body + text[end:])
    for pyc in (path.parent / '__pycache__').glob(path.stem + '.*.pyc'):
        pyc.unlink()
    return f'patched ({n} call sites)'


# --- patch 4: sorted persistent tile scheduler for block-sparse tiles (stage 3, step 1) -----------------------
# Real prompts give (head, q-block) lists whose longest is 1.6-1.9x the mean; FA4's static grid order ignores that.
# A persistent grid (one CTA per SM) walks a host-sorted order carried in blocksparse_tensors.cu_block_idx_offsets
# (unused by the fixed-length sparse path), longest lists first.
SCHED_ANCHOR = '''class SingleTileLPTScheduler:'''
SCHED_CLASS = (Path(__file__).resolve().parent / 'fa4_sorted_scheduler.py').read_text()
SM90_IMPORT_ANCHOR = '''    SingleTileVarlenScheduler,
)'''
SM90_IMPORT_PATCHED = '''    SingleTileVarlenScheduler,
    SortedPersistentTileScheduler,
)'''
SM90_SELECT_ANCHOR = '''        if const_expr(mCuSeqlensQ is not None or mSeqUsedQ is not None):
            # TODO: dispatch to DynamicPersistentVarlenScheduler when appropriate
            TileScheduler = SingleTileVarlenScheduler
        else:
            TileScheduler = (
                SingleTileScheduler
                if const_expr(not self.is_causal or self.is_local)
                else SingleTileLPTScheduler
            )
'''
SM90_SELECT_PATCHED = '''        # KAI H200: block-sparse tiles with a host-sorted order -> persistent sorted scheduler
        self.use_sorted_schedule = cutlass.const_expr(
            self.use_block_sparsity
            and blocksparse_tensors.cu_block_idx_offsets is not None
            and blocksparse_tensors.cu_total_m_blocks is None
            and mCuSeqlensQ is None
            and mSeqUsedQ is None
        )
        if const_expr(self.use_sorted_schedule):
            TileScheduler = SortedPersistentTileScheduler
        elif const_expr(mCuSeqlensQ is not None or mSeqUsedQ is not None):
            # TODO: dispatch to DynamicPersistentVarlenScheduler when appropriate
            TileScheduler = SingleTileVarlenScheduler
        else:
            TileScheduler = (
                SingleTileScheduler
                if const_expr(not self.is_causal or self.is_local)
                else SingleTileLPTScheduler
            )
'''
SM90_PARAMS_ANCHOR = '''        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
'''
SM90_PARAMS_PATCHED = '''        if const_expr(self.use_sorted_schedule):
            tile_sched_params = TileScheduler.to_underlying_arguments(
                tile_sched_args, blocksparse_tensors.cu_block_idx_offsets
            )
        else:
            tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
'''
SM90_SEQLEN_ANCHOR = '''            mCuBlockIdxOffsets=(
                blocksparse_tensors.cu_block_idx_offsets
                if blocksparse_tensors is not None
                else None
            ),'''
SM90_SEQLEN_PATCHED = '''            mCuBlockIdxOffsets=(
                blocksparse_tensors.cu_block_idx_offsets
                if (blocksparse_tensors is not None and not self.use_sorted_schedule)
                else None
            ),'''


# 4b: with a persistent schedule the consumer must keep Q (sO aliases sQ) until the epilogue's TMA store has read it
SM90_QREL_ANCHOR = '''                # Release Q pipeline so the producer can load the next tile's Q
                pipeline_q.consumer_release_w_index(0)

                # Handle empty case (when no blocks to process)
                if not processed_any:'''
SM90_QREL_PATCHED = '''                # KAI H200: with the persistent sorted schedule Q is released after the epilogue instead
                # (sO aliases sQ and the next tile's Q load must not overwrite the O tile being stored)
                if const_expr(not self.use_sorted_schedule):
                    pipeline_q.consumer_release_w_index(0)

                # Handle empty case (when no blocks to process)
                if not processed_any:'''
SM90_EPI_ANCHOR = '''                batch_idx,
            )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def first_half_block_overlap('''
SM90_EPI_PATCHED = '''                batch_idx,
            )
            if const_expr(self.use_block_sparsity and self.use_sorted_schedule):
                # KAI H200: the epilogue has waited for the O TMA store, so sQ may now receive the next tile's Q
                pipeline_q.consumer_release_w_index(0)

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def first_half_block_overlap('''


def _apply_sorted_scheduler(sched_path: Path, sm90_path: Path, check: bool) -> str:
    out = []
    t = sched_path.read_text()
    if 'class SortedPersistentTileScheduler' in t:
        # replace the installed copy of the class with the current side file (idempotent refresh)
        i = t.index('class SortedPersistentTileScheduler'); j = t.index(SCHED_ANCHOR)
        if t[i:j] == SCHED_CLASS:
            out.append('scheduler class: already patched')
        elif not check:
            sched_path.write_text(t[:i] + SCHED_CLASS + t[j:])
            for pyc in (sched_path.parent / '__pycache__').glob(sched_path.stem + '.*.pyc'):
                pyc.unlink()
            out.append('scheduler class: refreshed')
        else:
            out.append('scheduler class: would refresh')
    elif SCHED_ANCHOR not in t:
        return 'ANCHOR NOT FOUND (tile_scheduler.py)'
    elif not check:
        sched_path.write_text(t.replace(SCHED_ANCHOR, SCHED_CLASS + SCHED_ANCHOR, 1))
        for pyc in (sched_path.parent / '__pycache__').glob(sched_path.stem + '.*.pyc'):
            pyc.unlink()
        out.append('scheduler class: patched')
    else:
        out.append('scheduler class: would patch')
    t = sm90_path.read_text()
    if 'use_sorted_schedule' in t:
        out.append('sm90 selection: already patched')
        return '; '.join(out)
    for a in (SM90_IMPORT_ANCHOR, SM90_SELECT_ANCHOR, SM90_PARAMS_ANCHOR, SM90_SEQLEN_ANCHOR):
        if a not in t:
            return 'ANCHOR NOT FOUND (flash_fwd_sm90.py scheduler selection)'
    if check:
        out.append('sm90 selection: would patch')
        return '; '.join(out)
    t = t.replace(SM90_IMPORT_ANCHOR, SM90_IMPORT_PATCHED, 1).replace(SM90_SELECT_ANCHOR, SM90_SELECT_PATCHED, 1)
    t = t.replace(SM90_PARAMS_ANCHOR, SM90_PARAMS_PATCHED, 1).replace(SM90_SEQLEN_ANCHOR, SM90_SEQLEN_PATCHED, 1)
    sm90_path.write_text(t)
    for pyc in (sm90_path.parent / '__pycache__').glob(sm90_path.stem + '.*.pyc'):
        pyc.unlink()
    out.append('sm90 selection: patched')
    return '; '.join(out)


def _apply(path: Path, anchor: str, patched: str, check: bool) -> str:
    text = path.read_text()
    if patched in text:
        return 'already patched'
    if anchor not in text:
        return 'ANCHOR NOT FOUND (flash-attn-4 version mismatch?)'
    if check:
        return 'would patch'
    path.write_text(text.replace(anchor, patched, 1))
    for pyc in (path.parent / '__pycache__').glob(path.stem + '.*.pyc'):
        pyc.unlink()
    return 'patched'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--no-prefetch', action='store_true', help='skip patch 2 (producer list prefetch)')
    ap.add_argument('--no-full-mask', action='store_true', help='skip patch 3 (no mask function on full blocks)')
    ap.add_argument('--no-sorted-sched', action='store_true', help='skip patch 4 (sorted persistent tile scheduler)')
    args = ap.parse_args()
    import flash_attn.cute.block_sparse_utils as bsu
    import flash_attn.cute.flash_fwd_sm90 as m
    r1 = _apply(Path(m.__file__), ANCHOR, PATCHED, args.check)
    print(f'{m.__file__} (paged block-sparse loads): {r1}')
    if not args.no_prefetch:
        r2 = _apply(Path(bsu.__file__), ANCHOR2, PATCHED2, args.check)
        print(f'{bsu.__file__} (producer list prefetch): {r2}')
    r3 = 'skipped'
    if not args.no_full_mask:
        r3 = _apply_full_block_mask(Path(bsu.__file__), args.check)
        print(f'{bsu.__file__} (no mask_fn on full blocks): {r3}')
    r4 = 'skipped'
    if not args.no_sorted_sched:
        import flash_attn.cute.tile_scheduler as ts
        r4 = _apply_sorted_scheduler(Path(ts.__file__), Path(m.__file__), args.check)
        print(f'{ts.__file__} + flash_fwd_sm90.py (sorted persistent scheduler): {r4}')
    if 'NOT FOUND' in r1 or (not args.no_prefetch and 'NOT FOUND' in r2) or 'NOT FOUND' in r3 or 'NOT FOUND' in r4:
        sys.exit(1)


if __name__ == '__main__':
    main()
