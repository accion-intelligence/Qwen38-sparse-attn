# H200 results: Qwen3.8-27B

Single-request prefill on one H200 SXM. [Configuration and reproduction](h200.md) ·
[Nsight Compute analysis](validation/h200-ncu-fa4sparse-vs-dense.md).

## TTFT

Prefill chunk 8192, greedy decoding, no prefix caching; one warm-up + two measured runs.
Sparse: tau 0.95, 64-row selection, FA4 patches 1–3 with static dispatch.
FP8: vLLM W8A8 GEMM. First generated tokens matched stock BF16 in every configuration and run.
[Raw matrix](validation/h200-matrix-gpu4-20261101.md).

| input | stock BF16 / FA3 | sparse | dense + FP8 GEMM | sparse + FP8 GEMM |
|---|---:|---:|---:|---:|
| 64K | 6666 ms | 6323 ms (-5.1%) | 5043 ms (-24.4%) | **4657 ms (-30.1%)** |
| 128K | 16040 ms | 13789 ms (-14.0%) | 12679 ms (-21.0%) | **10426 ms (-35.0%)** |
| 256K | 42259 ms | 35557 ms (-15.9%) | 35550 ms (-15.9%) | **28789 ms (-31.9%)** |

Sorted dispatch, now the default, adds -2.2% at 256K in a separate A/B.
Tau 0.9 + sorted dispatch + FP8 GEMM: 4529 / 9652 / 25197 ms (-32% / -40% / -40%).

## Quality

Teacher-forced 256-token continuation of the BF16 dense output. Values are for 64K / 128K / 256K;
needle retrieval passed in every configuration. Raw scores: `validation/h200-tfe-*.json`.

| configuration | kept blocks | ΔNLL (nat) | top-1 agreement |
|---|---|---|---|
| sparse tau 0.95 | 0.62 / 0.47 / 0.56 | +0.0007 / +0.0071 / +0.0105 | 0.992 / 0.977 / 0.973 |
| sparse tau 0.95 + FP8 GEMM | same | +0.0006 / +0.0065 / +0.0110 | 0.992 / 0.977 / 0.965 |
| sparse tau 0.9 | 0.48 / 0.35 / 0.42 | +0.0021 / +0.0105 / +0.0265 | 0.996 / 0.977 / 0.957 |
| sparse tau 0.85 | 0.40 / 0.27 / 0.34 | +0.0052 / +0.0167 / +0.0399 | 0.988 / 0.969 / 0.941 |
| dense + FP8 GEMM | — | +0.0003 / -0.0071 / -0.0006 | 1.000 / 0.992 / 0.992 |

## Implementation

- FA4 CuTe SM90 forward with per-query-head sparse lists, 128-row query tiles and 64-token KV blocks.
- Fused Triton selection: 1.27 → 0.63 ms per call at 8K × 64K. Default 64-row scoring improves
  256K ΔNLL from +0.0176 to +0.0105 over 128-row scoring.
- Four FA4 patches: paged KV reads, producer list prefetch, no mask on fully visible blocks, and
  longest-list-first dispatch. Mask removal saves 6.6% kernel time; sorted dispatch saves 15% on
  skewed lists at 256K.
- BF16 sparse attention with vLLM's FP8 GEMM. FP8 GEMM is 1.86× faster on the model's shapes;
  FA3 FP8 attention at head_dim 256 is 2× slower than BF16 and is not enabled.
- Compiled serving already fuses norm and gating chains. The optional fused RMSNorm has no end-to-end
  gain and is disabled by default.

## Two-GPU serving

GPU 4 runs sparse prefill + FP8 GEMM; GPU 5 runs dense decode + FP8 GEMM. NIXL/UCX transfers KV and
GDN state over NVLink. With 1 / 2 / 4 simultaneous requests, mean TPOT stays at 11–13 ms and the
maximum per-request p99 stays at or below 26 ms. Single-request TTFT is 4.87 s at 64K and 29.3 s at 256K;
prefills queue serially on P. [Full PD measurements and launch commands](h200.md#two-gpu-prefilldecode-serving).
