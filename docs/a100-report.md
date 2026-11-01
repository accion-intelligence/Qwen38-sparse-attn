# Qwen3.8-27B long-context TTFT on A100 (SM80)

Goal: minimum **time to first token (TTFT)** for 64K / 128K / 256K-token inputs, one request, on a single
**A100-SXM4-80GB**. Constraints: BF16 only (no INT8 / FP8), vLLM 0.30.0. The `A100` branch ports the dynamic
block-sparse prefill of this repository to A100 as a separate package, `a100_attn/`; `sm120_attn/` is unchanged.

## 1. Results

TTFT, one request, each side at its best prefill chunk (stock vLLM: 8192; this port: 16384):

| Input length | Stock vLLM | A100 port | Speedup |
|---|---:|---:|---:|
| 64K | 19.77 s | **16.71 s** | **1.18×** |
| 128K | 51.02 s | **38.67 s** | **1.32×** |
| 256K (260096) | 145.89 s | **90.97 s** | **1.60×** |

Stock vLLM gets slower with a 16384 chunk (19.91 / 51.80 / 147.63 s), so its 8192 numbers are the baseline.

Output quality (teacher-forced scoring of a 256-token continuation of the BF16 dense output; needle retrieval):

| Prompt | Length | Blocks kept | ΔNLL | Top-1 agreement | Mean \|Δlogprob\| | Needle |
|---|---|---:|---:|---:|---:|---|
| Code docs (vLLM source + markdown) | 64K / 128K / 256K | 66% / 62% / 51% | −0.0007 / −0.0016 / +0.0045 | 99.6% / 99.2% / 96.1% | 0.008 / 0.011 / 0.036 | pass |
| CUDA manuals #0 | 64K / 128K / 256K | 66% / 63% / 62% | +0.0012 / +0.0012 / −0.0074 | 99.2% / 99.2% / 98.4% | 0.012 / 0.012 / 0.019 | pass |
| CUDA manuals #1 | 64K / 128K / 256K | 59% / 62% / 61% | +0.0020 / +0.0017 / +0.0262 | 98.4% / 99.2% / 94.9% | 0.010 / 0.009 / 0.054 | pass |

64K and 128K are essentially lossless. At 256K the error is in the range of the SM120 release at 256K (sparse BF16:
ΔNLL +0.001, top-1 97.3%, mean |Δlogprob| 0.053), with a large spread between prompts.

## 2. Where stock vLLM spends the time (nsys)

| Category | 64K | 128K | 256K |
|---|---:|---:|---:|
| Full attention (vLLM FA2) | 5.99 s (30%) | 23.52 s (46%) | **91.62 s (63%)** |
| GEMM (cuBLAS) | 12.02 s (61%) | 24.03 s (47%) | 47.55 s (33%) |
| GDN (FLA Triton) | 0.75 s | 1.51 s | 2.98 s |
| Other small kernels | 0.94 s | 1.88 s | 3.73 s |

- On the paged KV cache vLLM runs FA2's `flash_fwd_splitkv_kernel<256, 64, 64, 4>`, which reaches only
  **~146 TFLOPS (47% of peak)** at head_dim 256; non-paged FA2 on the same shapes reaches ~196 TFLOPS.
- cuBLAS GEMMs already run at **~89% of peak** (~278 TFLOPS); without lower precision there is no headroom there.
- Everything other than GEMM and attention is 4.6% (256K) to 8.6% (64K) of the time: the upper bound for any
  operator fusion.

## 3. Method

### 3.1 Dynamic block-sparse BF16 prefill attention (≈ −47 s at 256K)

Block selection follows the SM120 release: per full-attention layer and prefill chunk, 64-row query-block means are
scored against per-request cached 64-key block means; each query head keeps the smallest set of key blocks that
carries τ of the softmax mass, plus the diagonal and sink blocks. A100 settings:

- **τ = 0.97** (SM120 release: 0.95). On code-heavy documents τ = 0.95 keeps only 35% of the blocks at 256K and
  costs ΔNLL +0.03; τ = 0.97 restores the SM120-release accuracy at about +6% 256K TTFT.
- **128-row query blocks, 8 warps** (m128 / n64 / 256 threads). The union of two 64-row selections keeps ~7%
  more blocks than 64-row blocks, but the kernel is faster, so end-to-end time and accuracy both improve.

The kernel (`a100_attn/kernels/fa_bf16_sm80.py`) keeps the CuTe DSL Ampere FA2 structure (cp.async + mma.sync
m16n8k16) and the block-list walk of the SM120 BF16 kernel, with two A100-specific changes:

1. **Reads vLLM's paged KV cache in place**, removing the per-chunk gather copy of the whole K/V history
   (≈ −1.2 s at 256K).
   - Needs `--block-size 64`. Qwen3.8 is an attention/mamba hybrid and vLLM picks a 784-token page by default
     (not a multiple of 64); with 64 alignment the page becomes 832 = 13 × 64 tokens, so every 64-key block lies
     in one page.
   - vLLM 0.30 on A100 allocates the cache as `[pages, page, Hk, 2D]` (K and V interleaved per head). The kernel
     reads K and V as `[pages·page, Hk, D]` views with token stride 2048 (static strides compiled in, so the
     compiler can prove 16-byte alignment for cp.async) and maps logical to physical blocks through a tile map.
   - KV pages and mamba-state pages share one block pool, so a recycled page can hold NaN bit patterns. All
     length checks use the logical `kv_len`, and block lists drop blocks entirely past `kv_len`, so keys past
     the end never enter the softmax (tested with shuffled, NaN-filled pages).
2. **Longest-first CTA scheduling.** After selection, work per (head, query block) row is very uneven (256K: max
   3524 vs mean 1408 blocks), and launching in grid order leaves a 7–14% tail. In sparse mode the grid is 1-D and
   the host passes the rows sorted by block count, descending. +7–13% in single-layer replays; total attention
   time at 256K 37.4 s → 35.5 s. On real workloads the kernel runs at **189–194 TFLOPS** (dense FA2 with the
   same tile: 197).

### 3.2 Fused block-selection Triton kernel (`a100_attn/select_triton.py`, ≈ −1 s at 256K)

Selection used to be ~15 small PyTorch ops (einsum, softmax, sort, cumsum, scatter, argsort, ...) costing
1.4–2.8 ms per call and growing with context length. Now:

- query-block means accumulate in fp32 straight from bf16 (padding rows are zero, so no masked fp32 copy);
- one `baddbmm` writes the logits in the kernel's layout with the scale folded in;
- one Triton kernel does the causal mask, softmax, top-τ selection **by bisection on a probability threshold
  (20 iterations) instead of a sort**, diagonal / sink blocks, clipping past `kv_len`, the 128-row union, and a
  prefix-sum compaction that writes ascending block lists and counts directly.

0.32–0.64 ms per call (**3.4–4.4×**); selections differ from the sort-based reference in ≤0.06% of blocks
(ties at the threshold, biased towards keeping more).

### 3.3 Fused GDN intra-chunk kkt + solve_tril (`a100_attn/gdn_intra.py`)

On A100 vLLM's GDN prefill can only use the FLA Triton path (the FlashInfer and CuTe DSL backends are
SM90 / SM100 / SM120 only). vLLM's copy of FLA runs `chunk_scaled_dot_kkt_fwd` and `solve_tril` as two kernels
with an fp32 round trip of the A matrix. The fused kernel
`chunk_gated_delta_rule_fwd_kkt_solve_kernel` is ported from upstream flash-linear-attention (MIT, attribution
kept). Upstream works in the log2 domain and vLLM in natural log, so g is scaled by 1/ln 2 once; the downstream
`recompute_w_u`, `h` and `o` kernels are unchanged. 0.70 ms → 0.26 ms per call (**2.7×**), bf16-rounding-level
difference (1.95e-3).

### 3.4 Prefill chunk 16384

Larger chunks improve GEMM and sparse-attention wave utilization: 8192 → 16384 saves another ~0.8%; 32768 is
slower again and leaves 18 GiB for the KV cache.

### Contribution of each step (TTFT, seconds)

| Step | 64K | 128K | 256K |
|---|---:|---:|---:|
| Stock vLLM | 19.77 | 51.02 | 145.89 |
| + sparse BF16 attention, in-place paged reads (τ 0.97, m128) | 17.18 | 39.71 | 94.56 |
| + longest-first kernel scheduling | 17.16 | 39.59 | 92.76 |
| + fused block selection | 17.02 | 39.25 | 91.81 |
| + fused GDN kkt + solve_tril | 16.87 | 38.87 | 91.60 |
| + prefill chunk 16384 (final) | **16.71** | **38.67** | **90.97** |

## 4. Tried and dropped

| Approach | Result | Reason |
|---|---|---|
| gate_up GEMM + SwiGLU fusion (CUTLASS DualGemm, SM80) | best 247 TFLOPS, **1–3% slower** than cuBLAS (261) + SwiGLU | two accumulators double register use; warp tile limited to 64×32 |
| Q kept in registers in the attention kernel | **~10% slower** | 253 → 255 registers plus a 40-byte stack spill |
| 32-key n-blocks | +2%, not significant | occupancy is not the limit |
| 64-row query blocks (4 warps) | ~12% slower than m128 (dense) | one 4-warp CTA per SM |
| τ = 0.95 | ~6% faster, 256K ΔNLL +0.03 | accuracy loss too large |
| W8A16, megakernel-style whole-layer fusion | not implemented | A100 has no INT8×BF16 MMA, so a weight-only GEMM is slower in compute-bound prefill; everything outside GEMM and attention is 5–9% of the time |

## 5. Remaining headroom (256K, ~91 s)

- GEMM ~48 s: cuBLAS is at 89% of peak; going further needs lower precision or 2:4 structured sparsity
  (accuracy cost).
- Sparse attention ~35 s: the kernel uses 253 of 255 registers with one 8-warp CTA per SM, close to the limit of
  this design on A100; further gains need a lower τ (accuracy cost).
- Everything else ~6 s: the remaining fusion candidates (conv1d + post-conv, copies on the GDN path, gated
  norm) add up to ~0.5 s.

## 6. Usage

```bash
pip install -e .                       # installs a100_attn (same package as sm120_attn)
pip install vllm==0.30.0
python tools/patch_vllm_a100.py --target "$(python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')"
#   --check reports only, --revert restores; hooks: flash_attn (sparse attention), gdn_kkt (GDN fusion)

vllm serve Qwen/Qwen3.8-27B --dtype bfloat16 --max-model-len 262144 \
    --block-size 64 --max-num-batched-tokens 16384 --max-num-seqs 1 \
    --enable-chunked-prefill --no-enable-prefix-caching --language-model-only
```

| Variable | Default | Meaning |
|---|---|---|
| `A100_SPARSE_ATTN` | 1 | enable the sparse attention hook |
| `A100_SPARSE_TAU` | 0.97 | attention mass kept per head and query block |
| `A100_SPARSE_M_BLOCK` | 128 | query-block rows (128 or 64) |
| `A100_SPARSE_MIN_Q` / `_MIN_KV` | 1024 / 8192 | minimum chunk / context length for the sparse path |
| `A100_SPARSE_PAGED` | 1 | read the paged KV cache in place when the page size is a multiple of 64; otherwise gather |
| `A100_SPARSE_FUSED_SELECT` | 1 | Triton block selection (0: PyTorch reference) |
| `A100_GDN_FUSED_KKT` | 1 | fused GDN kkt + solve_tril kernel |
| `A100_SPARSE_DEBUG` | 0 | diagnostic lines to print |

Tests and reproduction:

```bash
python tests/test_a100_sparse_prefill.py   # gather and paged (shuffled, NaN-filled pages) paths vs fp32 reference
python tests/test_a100_gdn_intra.py        # fused kkt + solve_tril vs vLLM's two kernels
python scripts/bench_ttft.py --model $MODEL --prompts prompts.json --output ttft-a100 --concurrencies 1 \
    --max-num-seqs 1 --max-num-batched-tokens 16384 --serve-args "--block-size 64"
A100_SPARSE_ATTN=0 python scripts/validate_generation.py ...   # dense reference, then teacher_forced_eval.py
VLLM_ENABLE_V1_MULTIPROCESSING=0 nsys profile --capture-range=cudaProfilerApi ... python scripts/profile_prefill.py ...
python scripts/nsys_breakdown.py <report>_cuda_gpu_kern_sum.csv
```

## 7. Measurement setup

- A100-SXM4-80GB, CUDA 13.0, Python 3.12, torch 2.13.0+cu130, vLLM 0.30.0, nvidia-cutlass-dsl 4.7.0.
- Accuracy prompts: code docs (vLLM source and markdown) and CUDA manuals (Programming Guide, PTX ISA, Nsight
  Compute), built with `scripts/build_prompts.py` as exact-length prompts with a planted needle.
