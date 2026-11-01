"""Insert (or remove) the a100_attn hooks in an installed vLLM v0.30.0 package.

    python tools/patch_vllm_a100.py --target $(python -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
    python tools/patch_vllm_a100.py --target ... --check     # report only
    python tools/patch_vllm_a100.py --target ... --revert    # restore the stock code

Hooks (anchor-based, idempotent):
  flash_attn  v1/attention/backends/flash_attn.py: block-sparse prefill attention before the ordinary
              flash_attn_varlen_func call; returns False for batches without a long prefill chunk.
              Runtime switch: A100_SPARSE_ATTN=0.
  gdn_kkt     third_party/flash_linear_attention/ops/chunk.py: fused kkt + solve_tril (a100_attn.gdn_intra)
              in the GDN chunked prefill. Runtime switch: A100_GDN_FUSED_KKT=0.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PATCHES = {
    'flash_attn': ('v1/attention/backends/flash_attn.py',
                   '''            else:
                causal = attn_metadata.causal
                is_dynamic_causal = isinstance(causal, torch.Tensor)
''',
                   '''            else:
                # A100 dynamic block-sparse attention for long prefill chunks (a100_attn)
                from a100_attn.sparse_prefill import try_sparse_forward
                if try_sparse_forward(self, query[:num_actual_tokens], key[:num_actual_tokens], key_cache,
                                      value_cache, output[:num_actual_tokens], attn_metadata):
                    return output
                causal = attn_metadata.causal
                is_dynamic_causal = isinstance(causal, torch.Tensor)
'''),
    'gdn_kkt': ('third_party/flash_linear_attention/ops/chunk.py',
                '''    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        beta=beta,
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        output_dtype=torch.float32,
    )
    A = solve_tril(
        A=A, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, output_dtype=k.dtype
    )
''',
                '''    # A100: fused kkt + solve_tril (a100_attn.gdn_intra); A100_GDN_FUSED_KKT=0 restores the stock path
    from a100_attn.gdn_intra import fused_kkt_enabled, fused_kkt_solve
    if fused_kkt_enabled() and chunk_indices is not None:
        A = fused_kkt_solve(k, g, beta, cu_seqlens, chunk_indices, output_dtype=k.dtype)
    else:
        A = chunk_scaled_dot_kkt_fwd(
            k=k,
            beta=beta,
            g=g,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            output_dtype=torch.float32,
        )
        A = solve_tril(
            A=A, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, output_dtype=k.dtype
        )
'''),
}


def apply(path: Path, anchor: str, hook: str, revert: bool, check: bool) -> str:
    text = path.read_text()
    old, new = (hook, anchor) if revert else (anchor, hook)
    if new in text:
        return 'already ' + ('reverted' if revert else 'patched')
    if old not in text:
        return 'ANCHOR NOT FOUND (vLLM version mismatch, or patched by another hook?)'
    if check:
        return 'would ' + ('revert' if revert else 'patch')
    path.write_text(text.replace(old, new, 1))
    return 'reverted' if revert else 'patched'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', type=Path, required=True, help='directory of the vllm package')
    ap.add_argument('--only', choices=sorted(PATCHES), action='append', help='apply only these hooks (repeatable)')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--revert', action='store_true')
    args = ap.parse_args()
    root = args.target.resolve()
    for name in args.only or sorted(PATCHES):
        rel, anchor, hook = PATCHES[name]
        path = root / rel
        if not path.exists():
            sys.exit(f'{path} not found: --target must be the vllm package directory')
        print(f'{name:10s} {rel}: {apply(path, anchor, hook, args.revert, args.check)}')
    try:
        import a100_attn  # noqa: F401
        print('a100_attn importable:', Path(a100_attn.__file__).parent)
    except ImportError:
        print('NOTE: a100_attn is not importable; run `pip install -e .` in the repository root')


if __name__ == '__main__':
    main()
