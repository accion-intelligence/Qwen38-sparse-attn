# Qwen3.8-27B long-prompt prefill on one NVIDIA H200 (SM90): results and components

Branch `h200`. Goal: minimum time to first token (TTFT) for a single long request (64K / 128K / 256K input
tokens) on one H200 SXM, same prompts, serving flags and evaluation scripts as the SM120 work on `main`
([results.md](results.md)). Detailed status log with every measurement: [h200.md](h200.md); raw records:
[validation/](validation/) (files prefixed `h200-`).

## Setup

| component | value |
|---|---|
| GPU | H200 SXM 141 GB, CC 9.0, 132 SMs; driver 575.57.08 (CUDA 12.9), `/usr/local/cuda-12.9` |
| software | Python 3.12, torch 2.13.0+cu129, vLLM 0.30.0+cu129, flash-attn-4 4.0.0b37 (CuTe DSL), nvidia-cutlass-dsl 4.7.1, Triton 3.7.1 |
| model | Qwen/Qwen3.8-27B BF16 (64 layers: 48 GatedDeltaNet + 16 full attention, GQA 24/4, head_dim 256, 262144 context) |
| method | `scripts/bench_ttft.py`: one request, prefill chunk 8192, greedy, no prefix caching, one warm-up + two measured runs; prompts from `scripts/build_prompts.py` with a planted needle |
| quality | `scripts/teacher_forced_eval.py`: 256-token continuation of the BF16 dense output scored under each configuration (ΔNLL, top-1 agreement); needle retrieval |

## Results

Reference matrix, one GPU, four configurations run back to back under identical conditions
([validation/h200-matrix-gpu4-20261101.md](validation/h200-matrix-gpu4-20261101.md)). The first generated
token is identical to stock BF16 in every configuration and run.

| input | stock vLLM (BF16, FA3) | sparse attention (tau 0.95) | dense + FP8 GEMM | sparse + FP8 GEMM |
|---|---:|---:|---:|---:|
| 64K | 6666 ms | 6323 ms (-5.1%) | 5043 ms (-24.4%) | **4657 ms (-30.1%)** |
| 128K | 16040 ms | 13789 ms (-14.0%) | 12679 ms (-21.0%) | **10426 ms (-35.0%)** |
| 256K | 42259 ms | 35557 ms (-15.9%) | 35550 ms (-15.9%) | **28789 ms (-31.9%)** |

- Sorted tile dispatch (see below, now the default) adds -2.2% at 256K in a separate A/B (35.7 -> 34.9 s).
- Speed operating point, tau 0.9 + sorted dispatch + FP8 GEMM: 4529 / 9652 / 25197 ms (-32% / -40% / -40%).
- FP8 GEMM is vLLM's own CUTLASS SM90 W8A8 kernel (`--quantization fp8`); the sparse attention path is this
  branch's work. The two gains stack almost additively.

Output quality (teacher-forced vs the BF16 dense output; needle retrieved in every configuration):

| configuration | kept blocks 64K / 128K / 256K | ΔNLL (nat) | top-1 agreement | max abs Δlogprob (256K) |
|---|---|---|---|---:|
| sparse tau 0.95, 64-row selection | 0.62 / 0.47 / 0.56 | +0.0007 / +0.0071 / +0.0105 | 0.992 / 0.977 / 0.973 | 0.87 |
| sparse tau 0.95 + FP8 GEMM | same | +0.0006 / +0.0065 / +0.0110 | 0.992 / 0.977 / 0.965 | 0.79 |
| sparse tau 0.9 | 0.48 / 0.35 / 0.42 | +0.0021 / +0.0105 / +0.0265 | 0.996 / 0.977 / 0.957 | 2.04 |
| sparse tau 0.85 | 0.40 / 0.27 / 0.34 | +0.0052 / +0.0167 / +0.0399 | 0.988 / 0.969 / 0.941 | 2.51 |
| dense + FP8 GEMM | - | +0.0003 / -0.0071 / -0.0006 | 1.000 / 0.992 / 0.992 | 0.26 |

For reference, the SM120 release reports 6677 / 15203 / 36156 ms for its best configuration on an RTX PRO 6000
Blackwell; the H200 stock baseline alone is 6666 / 16040 / 42259 ms.

## Where the prefill time goes (torch.compile on, the serving configuration)

| share of GPU kernel time | stock 64K | stock 256K | sparse 64K | sparse 256K |
|---|---:|---:|---:|---:|
| GEMM (cuBLAS BF16) | 69.8% | 43.4% | 73.8% | 53.6% |
| attention | 20.7% (FA3) | 50.7% (FA3) | 14.2% (FA4 sparse) | 35.9% (FA4 sparse) |
| norm / RoPE / activation | 4.4% | 2.8% | 4.7% | 3.4% |
| GDN linear attention | 3.7% | 2.3% | 3.9% | 2.9% |
| elementwise / copies | 0.5% | 0.3% | 2.2% | 3.3% |

cuBLAS BF16 runs at 680-805 TFLOPS on the model's GEMM shapes (`scripts/h200/bench_gemm.py`), i.e. at the
H200's sustained rate; FP8 W8A8 is 1.86x faster on the same shapes. Under torch.compile Inductor already
fuses the norm / gating / copy chains, so hand-fused elementwise kernels give nothing end to end (measured:
a fused Gemma RMSNorm kernel, 10x faster in isolation, 0 to -4% in serving).

## Components (all in this branch)

| component | file(s) | what it does |
|---|---|---|
| FA4 block-sparse attention path | `sm120_attn/sparse_prefill.py` (`SparsePrefill.forward_fa4`, kernel `fa4`, default on SM90) | FlashAttention-4 CuTe DSL SM90 forward (wgmma, TMA, warp specialisation) driven by this repo's block selection: per-query-head lists over 128-row q blocks x 64-token KV blocks, mask/full split at the causal boundary, exact `kv_len` slicing, cached block means, per-request scales unchanged |
| selection granularity | `SM120_SPARSE_SELECT_QB` (default 64) | 64-row scoring unioned into the 128-row kernel blocks; halves the 256K error vs 128-row scoring for ~6% more blocks |
| fused selection kernel | `sm120_attn/select_fused.py` (Triton) | masked softmax, top-mass threshold by bisection, diagonal/sink forcing, mask/full split, packed FA4 lists in one kernel: 1.27 -> 0.63 ms per call at 8K x 64K, lists identical to the torch path |
| FA4 kernel patches | `tools/patch_fa4.py`, `tools/fa4_sorted_scheduler.py` (applied to the installed `flash_attn/cute`) | (1) block-sparse loads through a page table (reads vLLM's paged cache in place, `--block-size 1024`); (2) producer software-pipelines the list-index loads; (3) no mask function on 'full' blocks (dense FA4 passes None on interior blocks, the sparse consumer called the causal mask on every block): -6.6% kernel time at 8K x 256K, tensor-pipe activity 86.3% -> 90.7%; (4) `SortedPersistentTileScheduler`: one CTA per tile launched longest-list-first (-15% kernel time with real-like per-head skew at 256K) |
| vLLM hook | `tools/patch_vllm.py` (unchanged anchor for vLLM 0.30.0), `sm120_attn/vllm_fusions.py` | hook dispatches 'fa4'; optional import-time fusions (`SM120_FUSED_NORM`, default off) |
| paged FA4 path | `SM120_SPARSE_PAGED` (default on when pages are 64-aligned) | bit-identical to the gathered path; removes the per-call KV history copies (1 GB per layer-chunk at 256K) |
| environment | `scripts/h200/{env,setup_env}.sh`, `scripts/h200/check_env.py` | CUDA 12.9 venvs on local disk, FA4 + patches, NIXL; verification of device, FA3/FA4, CuTe JIT, patch anchors |
| measurement | `scripts/h200/{bench_stock,bench_sparse,final_matrix,quality}.sh`, `profile_prefill.py`, `bench_attn_dense.py`, `bench_fa4_sparse.py`, `bench_fa4_longkv.py`, `bench_gemm.py`, `bench_fa3_fp8.py`, `ncu_attn_driver.py` | TTFT matrix, quality pipeline, compiled-mode kernel breakdown, kernel microbenchmarks, Nsight Compute driver |
| tests | `tests/test_sparse_prefill_fa4.py`, `tests/test_sparse_prefill_fa4_paged.py`, `tests/test_fused_norm.py` | fp32 masked reference, fused vs torch lists, paged == gathered |

## Kernel findings (Nsight Compute, 8K x 256K, 55% of blocks kept; [validation/h200-ncu-fa4sparse-vs-dense.md](validation/h200-ncu-fa4sparse-vs-dense.md))

| kernel | tensor pipe active | L2 hit | DRAM read |
|---|---:|---:|---:|
| FA3 dense (vLLM C++) | 99.7% | 93.6% | 4 GB |
| FA4 dense (CuTe) | 99.9% | 93.6% | 13 GB |
| FA4 sparse, upstream | 86.3% | 62% | 59 GB |
| FA4 sparse, patches 1-3 | 90.7% | 71% | 37 GB |

- The sparse kernel's deficit is per-block fixed overhead, not memory: structured lists that raise the L2 hit
  rate to 75% leave the time unchanged, and the producer prefetch of list indices changes nothing; removing the
  mask function on full blocks cut 12% of the instructions.
- The remaining ~9 points match the 64-key tile forced by 64-token KV blocks versus the 80-key tile dense
  hdim-256 uses; 80-token blocks are worth 1-7% more but break 1024-token page alignment (not enabled).
- A persistent (multi-tile) grid on this kernel makes the 240-register consumer warpgroups spill (1.36 G local
  loads and stores, 6x slower); the sorted dispatch therefore keeps one CTA per tile and only orders the grid.
- Real prompts give per-CTA list lengths with max/mean 1.6-1.9 and per-head CV 0.15-0.24; that imbalance,
  not bandwidth, is what the sorted dispatch removes.
- The three SM120 kernels (`mma.sync`, cp.async / TMA) run unchanged on the H200 but at 209-319 TFLOPS versus
  FA3's 720-770 (FP8 `mma.sync` is slower than BF16 on Hopper); FlashInfer's BSR kernel reaches only ~230
  TFLOPS on kept blocks. Both were rejected as the attention core.
- vLLM's FA3 FP8 forward is 2x slower than its BF16 forward at head_dim 256 (362-387 TFLOPS); FA4's FP8 path is
  SM100-only. FP8 attention on this model requires a new hdim-256 FP8 kernel.

## Status against the four-stage plan

1. BF16 sparse attention with a custom kernel: done (above).
2. Megakernel-style fusion with wgmma pipelines: not built. Measured ceiling in BF16 after Inductor fusion is
   below 5% of TTFT; the candidate with a measurable gain is a gate-up GEMM with fused SwiGLU epilogue (saves
   the 570 MB intermediate per chunk, ~2-3%).
3. Cross-SM fusion (Meld): the scheduling part is implemented (sorted dispatch, -15% kernel / -2.2% TTFT at
   256K); DSMEM cooperation and just-in-time reduction were not implemented because single-request prefill is
   compute-bound with mild imbalance (remaining tail ~2%). Their natural target is decode and concurrent
   serving (see next steps).
4. FP8: FP8 GEMM is in (vLLM's kernel, -24..-21% alone, -30..-35% with sparsity); FP8 attention is not.

## Reproduce

```bash
bash scripts/h200/setup_env.sh                       # venvs on /local1, vLLM 0.30.0+cu129, flash-attn-4 + patches, NIXL
source scripts/h200/env.sh && python scripts/h200/check_env.py --jit
python tools/patch_vllm.py --target "$(python -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')"
bash scripts/h200/build_prompts.sh                   # 64K / 128K / 256K needle prompts from local markdown
CUDA_VISIBLE_DEVICES=4 TAU=0.95 bash scripts/h200/final_matrix.sh   # stock / sparse / FP8 / sparse+FP8
TAU=0.95 bash scripts/h200/quality.sh                # needle + teacher-forced scores
```

Environment variables of the hook: `SM120_SPARSE_ATTN`, `SM120_SPARSE_KERNEL=fa4`, `SM120_SPARSE_TAU`,
`SM120_SPARSE_SELECT_QB`, `SM120_SPARSE_SCHED=sorted|static`, `SM120_SPARSE_PAGED`, `SM120_SPARSE_FUSED_SELECT`.

## Next steps

- Two-GPU serving with concurrency (GPUs 4 and 5, NVLink NV18): disaggregated prefill/decode with vLLM's NIXL
  connector (hybrid GDN state transfer supported in 0.30.0). Prefill keeps this stack; decode is where the
  Meld-style split-KV with DSMEM just-in-time combine, block-sparse decode attention reusing the prefill block
  means, FP8 KV cache and a persistent decode megakernel apply. Launcher and client:
  `scripts/h200/pd_serve.sh`, `scripts/h200/pd_client.py`, `scripts/h200/pd_bench.sh`.
- FP8 sparse attention kernel for head_dim 256 on SM90 (K-major FP8 wgmma operands, V transposed on chip or at
  gather time, P in registers); it must beat 720 TFLOPS BF16 to matter and would address the 36% attention
  share of the sparse 256K prefill.
- Upstream candidates: FA4 full-block mask fix (patch 3) and the paged block-sparse loads (patch 1).
