# H200 stock vLLM baseline: single request TTFT (2026-11-01)

One NVIDIA H200 SXM (GPU 5 of a shared node, driver 575.57.08 / CUDA 12.9, 700 W, SM clock cap 1980 MHz),
vLLM 0.30.0+cu129, torch 2.13.0+cu129, Qwen3.8-27B BF16 weights, branch `h200` at `d704d93` + working tree.
Attention backend FLASH_ATTN with FlashAttention-3; GDN prefill backend FlashInfer (JIT); attention page 784 tokens.

Server (launched by `scripts/bench_ttft.py` through `scripts/h200/bench_stock.sh`):

```
vllm serve <Qwen3.8-27B> --dtype bfloat16 --max-model-len 262144 --no-enable-prefix-caching --generation-config vllm
  --seed 20261015 --max-num-seqs 4 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.9
  --enable-chunked-prefill --language-model-only        (env SM120_SPARSE_ATTN=0; the hook is not installed)
```

Prompts: `scripts/build_prompts.py` over the local KernelWiki markdown corpus (docs_needle, two per length);
one warmup + two measured runs per case; greedy; first generated token checked across runs (all identical).

| case | warmup | rep 1 | rep 2 | mean of measured |
|---|---:|---:|---:|---:|
| 64K (65536 in), 1 request | 6897 ms | 6765 ms | 6754 ms | **6760 ms** |
| 128K (131072 in), 1 request | 16236 ms | 16280 ms | 16276 ms | **16278 ms** |
| 256K (260096 in), 1 request | 42498 ms | 42430 ms | 42449 ms | **42439 ms** |

Against the RTX PRO 6000 Blackwell numbers in [results.md](../results.md) (same prompts recipe, chunk 8192):

| input | H200 stock (this run) | SM120 stock | SM120 best (C: FP8 sparse attn + FP8 GEMM) |
|---|---:|---:|---:|
| 64K | 6.76 s | 12.06 s | 6.68 s |
| 128K | 16.28 s | 30.17 s | 15.20 s |
| 256K | 42.44 s | 81.66 s | 36.16 s |

Scaling: 64K -> 128K is 2.41x, 128K -> 256K is 2.61x (pure O(N) work would be 2.0x), so at 256K a sizeable
share of the time is already quadratic (attention) even with FA3. Raw files: `/local1/li004074/exp/ttft-stock-c1/`
(`results.json`, `server.log`, `summary.md`).
