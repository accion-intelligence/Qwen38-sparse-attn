# Qwen3.8-27B long-prompt prefill on NVIDIA H200 (SM90)

An open-source optimization case study from [Accion Intelligence](https://accion-intelligence.com/): minimum
time to first token (TTFT) for a single long request (64K / 128K / 256K input tokens) of **Qwen3.8-27B** on
**one NVIDIA H200 SXM**. This branch (`h200`) re-targets the project at Hopper; the RTX PRO 6000 Blackwell
(SM120) release stays on `main`.

The result combines dynamic block-sparse prefill attention, built on the FlashAttention-4 CuTe DSL SM90 kernel
with four kernel patches, with vLLM's FP8 W8A8 GEMM. Every number below is reproducible with the scripts in
`scripts/h200/`; the raw records are in `docs/validation/` (files prefixed `h200-`).

[Summary](docs/h200-summary.md) · [Full measurement log](docs/h200.md) · [Kernel analysis (Nsight Compute)](docs/validation/h200-ncu-fa4sparse-vs-dense.md)

## Measured performance

TTFT for one request on one H200, prefill chunk 8192, mean of two measured runs after one warm-up, four
configurations run back to back ([raw](docs/validation/h200-matrix-gpu4-20261101.md)). The first generated
token is identical to stock BF16 vLLM in every configuration and run.

| Input length | Stock vLLM 0.30.0 (BF16, FlashAttention-3) | Sparse attention (tau 0.95) | Dense + FP8 GEMM | Sparse + FP8 GEMM |
|---|---:|---:|---:|---:|
| 64K | 6666 ms | 6323 ms (-5.1%) | 5043 ms (-24.4%) | **4657 ms (-30.1%)** |
| 128K | 16040 ms | 13789 ms (-14.0%) | 12679 ms (-21.0%) | **10426 ms (-35.0%)** |
| 256K | 42259 ms | 35557 ms (-15.9%) | 35550 ms (-15.9%) | **28789 ms (-31.9%)** |

- The sorted tile dispatch (now the default, see below) adds -2.2% at 256K in a separate A/B.
- Speed operating point, tau 0.9 + FP8 GEMM: 4529 / 9652 / 25197 ms (-32% / -40% / -40%).
- FP8 GEMM is vLLM's own CUTLASS SM90 kernel (`--quantization fp8`); the sparse attention path is this
  branch's work. The two gains stack almost additively.

Output quality, teacher-forced 256-token continuation of the BF16 dense output ([raw](docs/validation/)):

| Configuration | Blocks kept 64K / 128K / 256K | ΔNLL (nat) | Top-1 agreement | Needle |
|---|---|---|---|---|
| Sparse tau 0.95 | 0.62 / 0.47 / 0.56 | +0.0007 / +0.0071 / +0.0105 | 0.992 / 0.977 / 0.973 | retrieved |
| Sparse tau 0.95 + FP8 GEMM | same | +0.0006 / +0.0065 / +0.0110 | 0.992 / 0.977 / 0.965 | retrieved |
| Sparse tau 0.9 | 0.48 / 0.35 / 0.42 | +0.0021 / +0.0105 / +0.0265 | 0.996 / 0.977 / 0.957 | retrieved |
| Dense + FP8 GEMM | - | +0.0003 / -0.0071 / -0.0006 | 1.000 / 0.992 / 0.992 | retrieved |

## What this branch adds

- **Sparse prefill attention on SM90.** `SparsePrefill(kernel='fa4')` drives FlashAttention-4's CuTe DSL
  SM90 forward (wgmma, TMA, warp specialisation) with this repository's block selection: per-query-head lists
  over 128-row query blocks and 64-token KV blocks, mask/full split at the causal boundary, cached key block
  means, 64-row selection granularity. Dense FA3 runs the tensor pipe at 99.7%; the sparse kernel runs at
  90.7% on the kept blocks (629 TFLOPS at 8K x 256K, 55% kept) after the patches below.
- **Fused selection kernel** (`sm120_attn/select_fused.py`, Triton): masked softmax, top-mass threshold,
  diagonal/sink forcing and FA4 list packing in one kernel (1.27 -> 0.63 ms per call at 8K x 64K).
- **FA4 kernel patches** (`tools/patch_fa4.py`, applied to the installed `flash_attn/cute`):
  1. block-sparse loads through a page table, so the kernel reads vLLM's paged cache in place;
  2. software-pipelined list-index loads in the producer warp;
  3. no mask function on fully visible blocks (-6.6% kernel time; dense FA4 already skips it);
  4. `SortedPersistentTileScheduler`: tiles launched longest-list-first (-15% kernel time under the per-head
     skew of real prompts, -2.2% TTFT at 256K).
- **vLLM integration** for vLLM 0.30.0 through the same idempotent hook as `main` (`tools/patch_vllm.py`).
- **Measurement and quality tooling**: back-to-back TTFT matrix, teacher-forced quality pipeline,
  compiled-mode kernel breakdown, kernel microbenchmarks, Nsight Compute driver and report.
- **Two-GPU serving (experimental)**: prefill/decode disaggregation over NVLink with vLLM's NIXL connector,
  prefill on the sparse stack, decode on a second H200 (`scripts/h200/pd_serve.sh`, `pd_client.py`,
  `pd_bench.sh`).

What was measured and not adopted is recorded in [docs/h200.md](docs/h200.md): FlashInfer's block-sparse
kernel (230 TFLOPS), the SM120 `mma.sync` kernels on Hopper (209-319 TFLOPS), a fused RMSNorm (no gain under
torch.compile), 80-token KV blocks (1-7% but breaks page alignment), a persistent tile scheduler (register
spills on SM90), and vLLM's FA3 FP8 forward at head_dim 256 (2x slower than BF16).

## Requirements

H200 (CC 9.0) with a CUDA 12.9 driver; Python 3.12; PyTorch 2.13.0+cu129; vLLM 0.30.0+cu129; flash-attn-4
4.0.0b37; nvidia-cutlass-dsl 4.7.1; Triton 3.7.1; nixl 1.5.2 (two-GPU serving only). `scripts/h200/setup_env.sh`
installs all of it into a venv (on local disk; `scripts/h200/env.sh` holds the paths).

## Quick start

```bash
bash scripts/h200/setup_env.sh                      # venv, vLLM 0.30.0+cu129, flash-attn-4 + kernel patches, NIXL
source scripts/h200/env.sh
python scripts/h200/check_env.py --jit              # device, FA3/FA4, CuTe JIT, patch anchors, model files
python tools/patch_vllm.py --target "$(python -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')"

export SM120_SPARSE_ATTN=1 SM120_SPARSE_KERNEL=fa4 SM120_SPARSE_TAU=0.95
vllm serve Qwen/Qwen3.8-27B --dtype bfloat16 --quantization fp8 \
    --max-model-len 262144 --max-num-batched-tokens 8192 --max-num-seqs 4 \
    --enable-chunked-prefill --no-enable-prefix-caching --language-model-only
```

Environment variables (the `SM120_` prefix is kept for compatibility with `main`):

| Variable | Default | Meaning |
|---|---|---|
| `SM120_SPARSE_ATTN` | 1 | enable the sparse attention hook |
| `SM120_SPARSE_KERNEL` | `fa4` on SM90 | attention core (`fa4`; the SM120 kernels remain selectable) |
| `SM120_SPARSE_TAU` | 0.95 | attention mass retained per head and query block |
| `SM120_SPARSE_SELECT_QB` | 64 | query rows per selection block (64 or 128) |
| `SM120_SPARSE_SCHED` | `sorted` | tile dispatch: longest lists first, or FA4's `static` grid |
| `SM120_SPARSE_PAGED` | 1 | read the paged KV cache in place when pages are 64-token aligned (`--block-size 1024`) |
| `SM120_SPARSE_FUSED_SELECT` | 1 | fused Triton selection (0: torch ops) |
| `SM120_SPARSE_MIN_Q` / `_MIN_KV` | 1024 / 8192 | minimum chunk / context length for the sparse path |

## Reproduce

```bash
bash scripts/h200/build_prompts.sh                            # 64K / 128K / 256K needle prompts from local markdown
CUDA_VISIBLE_DEVICES=4 TAU=0.95 bash scripts/h200/final_matrix.sh   # stock / sparse / FP8 / sparse+FP8, back to back
TAU=0.95 bash scripts/h200/quality.sh                         # needle retrieval + teacher-forced scores
python scripts/h200/profile_prefill.py --no-enforce-eager ... # per-kernel-category breakdown (serving config)
python tests/test_sparse_prefill_fa4.py --big                 # kernel correctness vs fp32 reference, timing
```

## Repository layout (additions on this branch)

| Path | Contents |
|---|---|
| `sm120_attn/sparse_prefill.py` | block selection, caches, vLLM hook; `forward_fa4` (SM90 path) |
| `sm120_attn/select_fused.py` | fused Triton selection kernel |
| `tools/patch_fa4.py`, `tools/fa4_sorted_scheduler.py` | FlashAttention-4 kernel patches and the sorted tile scheduler |
| `scripts/h200/` | environment, benchmarks, quality pipeline, profiling, two-GPU serving |
| `tests/test_sparse_prefill_fa4*.py` | SM90 path tests (reference, fused vs torch lists, paged vs gathered) |
| `docs/h200-summary.md`, `docs/h200.md`, `docs/validation/h200-*` | summary, full log, raw records |

## Status and next steps

Done: block-sparse prefill attention on SM90 with a patched FA4 kernel, fused selection, sorted dispatch, FP8
GEMM evaluation, quality validation, two-GPU PD serving path. Open: an FP8 attention kernel for head_dim 256
on SM90 (no existing kernel is faster than BF16 here), cross-SM cooperation on the decode side, and a fused
gate-up GEMM with SwiGLU epilogue (~2-3% of TTFT). Details in [docs/h200-summary.md](docs/h200-summary.md).

## License and acknowledgments

Released under [Apache-2.0](LICENSE). The SM90 attention core is FlashAttention-4 (CuTe DSL, BSD-3-Clause),
patched in place by `tools/patch_fa4.py`; the SM120 kernels on `main` build on NVIDIA CUTLASS examples
(BSD-3-Clause); vLLM is Apache-2.0. See [third-party notices](THIRD_PARTY_NOTICES.md).
