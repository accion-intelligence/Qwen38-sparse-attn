# SM120 attention kernels

The kernels in `sm120_attn/kernels/` target RTX PRO 6000 Blackwell.
[Serving configuration](deployment.md) · [End-to-end results](results.md).

## BF16

`kai_fa_bf16.py` derives from the CUTLASS CuTe DSL Ampere FlashAttention-2 example
(`cute/ampere/kernel/attention/flash_attention_v2.py`, BSD-3-Clause). It uses `mma.sync` m16n8k16,
cp.async, GQA head mapping, absolute query offsets for chunked causal prefill, and optional
per-query-head sparse block lists.

`attention_sm120` takes contiguous NHD tensors: Q `[n, H, D]`, K/V `[kv_len, Hk, D]`.
The vLLM hook selects it with `SM120_SPARSE_KERNEL=sm120`.

## FP8

`kai_fa_fp8.py` uses e4m3 operands with m16n8k32 MMA and FP32 accumulation. Q/K are NHD;
V is transposed to `[Hk, D, T]` with contiguous keys for the K-major MMA operand.
Quantization scales fold into `softmax_scale * s_q * s_k` and `s_v / 448`.

The kernel processes 128-row tiles with 256 threads. Each sparse list is the union of two
64-row selection lists. P remains in registers: `VT_KEY_PERM` permutes keys within each 32-key
block so the S accumulator layout matches the FP8 A operand. Callers constructing `vT8` must use
`permute_vt_keys()`.

Compile-time masked/unmasked loop specialization avoids register spills. Error against an FP32
reference on the same e4m3 operands is 1–2e-2 from P rounding; output error on real activations
is 1.1–2.5%.

## TMA pipeline

`kai_fa_fp8_tma.py` uses 384 threads: eight MMA warps and four producer warps, with one warp issuing
TMA. K and Vᵀ use two-stage mbarrier pipelines; consumer warps release stages independently.
Only pipeline initialization uses a CTA-wide barrier.

`min_blocks_per_mp=1` is required for ptxas to honor `setmaxnreg`. The register split balances
`4 × (168 - 24) == 8 × (240 - 168)`. Q/K/Vᵀ use `(Sq, D, H)`, `(Sk, D, Hk)`, `(D, Sk, Hk)` views;
output is BF16 `(Sq, H, D)`. TMA zero-fills out-of-bounds rows and keys.

## Performance

Dense causal attention, 8192 query rows × 65536 keys, head_dim 256, GQA 24/4.

| kernel | time | TFLOPS |
|---|---:|---:|
| `kai_fa_bf16`, 64-row tiles | 36.6 ms | 340 |
| `kai_fa_fp8`, double-buffered K, P in registers | 19.47 ms | 635 |
| `kai_fa_fp8_tma` | 17.98 ms | 688 |

TMA versus cp.async: 37.85 / 40.41 ms at 8K × 128K; 41.79 / 45.41 ms at 4K × 256K.
At 40% block density: 11.5 / 13.0 ms, 24.3 / 27.1 ms and 26.8 / 30.1 ms for 64K / 128K / 256K keys.

## Scheduling

An 8192-row chunk launches 1536 CTAs over 188 SMs (8.17 waves), costing about 5% kernel time
in the final partial wave. A 12032-row chunk launches 2256 CTAs (12 complete waves) and reduces
single-request TTFT by 1.9% / 1.9% / 2.6% at 64K / 128K / 256K.

Warp-group ping-pong, explicit ldmatrix prefetch and S double buffering gave no gain;
these options remain disabled. TMA-kernel tensor-pipe activity measured 84% at 8K × 16K.

## Validation

```bash
python tests/test_fa_bf16.py --big
python tests/test_fa_fp8.py --big
python tests/test_fa_fp8_tma.py --big
```
