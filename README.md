# Qwen-3.8-27B long-context prefill on A100 (SM80)

**An open-source optimization case study from [Accion Intelligence](https://accion-intelligence.com/).**

This branch ports the dynamic block-sparse prefill of this repository from RTX PRO 6000 Blackwell (SM120) to a
single **NVIDIA A100-SXM4-80GB (SM80)** and optimizes **Qwen3.8-27B** long-context time to first token on it.
A100 has no FP8 tensor cores, so everything runs in BF16. The A100 code lives in `a100_attn/`; the original
SM120 release (`sm120_attn/`, FP8 kernels) is unchanged and documented on the `main` branch.

At 256K input length, time to first token drops from **145.89 s to 90.97 s (1.60×)** with stock vLLM 0.30.0's
BF16 GEMMs unchanged.

[A100 report](docs/a100-report.md) · [About Accion Intelligence](#about-accion-intelligence)

## Measured performance

Time to first token (TTFT) for **Qwen3.8-27B on one A100-SXM4-80GB**, one request, BF16, each configuration
at its best prefill chunk (stock vLLM: 8192; A100 port: 16384):

| Input length | Stock vLLM | A100 port | Speedup vs stock |
|---|---:|---:|---:|
| 64K | 19.77 s | 16.71 s | **1.18×** |
| 128K | 51.02 s | 38.67 s | **1.32×** |
| 256K | 145.89 s | 90.97 s | **1.60×** |

In teacher-forced evaluation against the BF16 dense reference (three prompts: code docs and two CUDA-manual
prompts), 64K and 128K are essentially lossless (|ΔNLL| ≤ 0.002 nat, top-1 agreement ≥ 98.4%); at 256K, ΔNLL is
−0.007 to +0.026 nat and top-1 agreement 94.9–98.4%. Needle retrieval passed at 64K, 128K and 256K for every
prompt. See the [A100 report](docs/a100-report.md) for per-prompt numbers, the profile and the contribution of
each step.

## What the A100 port contains

- **SM80 block-sparse BF16 attention kernel** (CuTe DSL, cp.async + mma.sync). It walks per-head,
  per-query-block lists of 64-key blocks that carry τ = 0.97 of the attention mass, reads K/V **in place from
  vLLM's paged KV cache**, and issues (head, query-block) rows **longest first**. On real 256K workloads it runs
  at 189–194 TFLOPS of kept work, versus ~146 TFLOPS for the paged FA2 kernel vLLM uses.
- **Fused block selection** (Triton). Masked softmax, top-mass selection by threshold bisection (no sort), and
  list compaction in one kernel, 3.4–4.4× faster than the PyTorch implementation.
- **Fused GDN kkt + solve_tril** (Triton, ported from flash-linear-attention) for the 48 linear-attention layers,
  2.7× faster than vLLM's two-kernel path.
- **vLLM integration.** `tools/patch_vllm_a100.py` adds two idempotent, revertible hooks to vLLM 0.30.0. The port
  targets prefill; decode uses vLLM's existing kernels.

## Requirements

The measured environment uses **A100-SXM4-80GB (CC 8.0)**, CUDA 13.0, PyTorch 2.13 (cu130),
`nvidia-cutlass-dsl` 4.7, vLLM 0.30.0, and Python 3.12. Kernels JIT-compile on first use; no C++ build is
needed.

## Quick start

### Install

```bash
git clone -b A100 <this repo> && cd Qwen38-SM120-sparse-attn
pip install -e .
pip install vllm==0.30.0
python tools/patch_vllm_a100.py --target "$(python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')"
```

The patch is idempotent. Add `--check` to inspect its status without changing files, or `--revert` to restore
stock vLLM.

### Serve

```bash
vllm serve Qwen/Qwen3.8-27B --dtype bfloat16 --max-model-len 262144 \
    --block-size 64 --max-num-batched-tokens 16384 --max-num-seqs 1 \
    --enable-chunked-prefill --no-enable-prefix-caching --language-model-only
```

`--block-size 64` makes vLLM use 832-token KV pages (a multiple of 64) so the kernel can read the cache in
place; without it the hook falls back to copying the K/V history every chunk.

### Configure

| Variable | Default | Meaning |
|---|---|---|
| `A100_SPARSE_ATTN` | 1 | enable the sparse attention hook |
| `A100_SPARSE_TAU` | 0.97 | target attention mass retained per head and query block |
| `A100_SPARSE_M_BLOCK` | 128 | query-block rows (128 or 64) |
| `A100_SPARSE_MIN_Q` / `_MIN_KV` | 1024 / 8192 | minimum chunk / context length for the sparse path |
| `A100_SPARSE_PAGED` | 1 | read the paged KV cache in place when the page size is a multiple of 64 |
| `A100_SPARSE_FUSED_SELECT` | 1 | Triton block selection (0: PyTorch reference) |
| `A100_GDN_FUSED_KKT` | 1 | fused GDN kkt + solve_tril kernel |
| `A100_SPARSE_DEBUG` | 0 | diagnostic lines to print |

Settings can also come from a JSON file named by `A100_SPARSE_CONFIG`.

## Reproduce the results

Install the evaluation dependencies with `pip install -e '.[eval]'`, set `MODEL` to your model name or local
path, and provide a corpus for prompt construction:

```bash
python scripts/build_prompts.py --model $MODEL --corpus path/to/docs --lengths 65536,131072,260096 --concurrency 1 --output prompts.json
python scripts/bench_ttft.py --model $MODEL --prompts prompts.json --output ttft-stock --concurrencies 1 --max-num-seqs 1 \
    --env A100_SPARSE_ATTN=0 A100_GDN_FUSED_KKT=0
python scripts/bench_ttft.py --model $MODEL --prompts prompts.json --output ttft-a100 --concurrencies 1 --max-num-seqs 1 \
    --max-num-batched-tokens 16384 --serve-args "--block-size 64"
A100_SPARSE_ATTN=0 A100_GDN_FUSED_KKT=0 python scripts/validate_generation.py --model $MODEL --prompts prompts.json --output gen-dense.json
A100_SPARSE_ATTN=0 A100_GDN_FUSED_KKT=0 python scripts/teacher_forced_eval.py --model $MODEL --prompts prompts.json --reference gen-dense.json --output tfe-dense.json
python scripts/validate_generation.py --model $MODEL --prompts prompts.json --output gen-a100.json
python scripts/teacher_forced_eval.py --model $MODEL --prompts prompts.json --reference gen-dense.json --reference-scores tfe-dense.json --output tfe-a100.json
```

Kernel-level profile of one prefill:

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
    --trace=cuda,nvtx -o prefill-256k python scripts/profile_prefill.py --model $MODEL --prompts prompts.json \
    --length 260096 --chunk 16384 --block-size 64
nsys stats --report cuda_gpu_kern_sum --format csv -o prefill-256k prefill-256k.nsys-rep
python scripts/nsys_breakdown.py prefill-256k_cuda_gpu_kern_sum.csv
```

Run the correctness checks from the repository root on an A100:

```bash
python tests/test_a100_sparse_prefill.py
python tests/test_a100_gdn_intra.py
```

## Repository layout

| Path | Contents |
|---|---|
| `a100_attn/kernels/fa_bf16_sm80.py` | SM80 BF16 block-sparse attention kernel (paged reads, longest-first rows) |
| `a100_attn/sparse_prefill.py` | Block-mean caches, planning, and the vLLM entry point |
| `a100_attn/select_triton.py` | Fused block-selection Triton kernel |
| `a100_attn/gdn_intra.py` | Fused GDN kkt + solve_tril Triton kernel |
| `tools/patch_vllm_a100.py` | vLLM 0.30.0 hooks (FlashAttention backend, FLA GDN chunk) |
| `tests/test_a100_*.py` | A100 correctness checks |
| `scripts/` | Prompt construction, TTFT benchmark, generation and teacher-forced evaluation, profiling |
| `docs/a100-report.md` | A100 method, measurements, accuracy, and dropped approaches |
| `sm120_attn/`, `extras/`, `tools/patch_vllm.py`, other `docs/` | Original SM120 release (see `main`) |

## About Accion Intelligence

At **[Accion Intelligence](https://accion-intelligence.com/)**, we build software to unlock the performance
of AI hardware. This project makes the implementation and measured results available for the community to
evaluate and build on.

To explore optimization for your own GPU workloads, **[request a demo](https://accion-intelligence.com/demo)**
or contact us at [hello@accion-intelligence.com](mailto:hello@accion-intelligence.com).

## License and acknowledgments

Released under [Apache-2.0](LICENSE). The kernels build on NVIDIA CUTLASS examples under BSD-3-Clause, and
`a100_attn/gdn_intra.py` contains a kernel from flash-linear-attention under MIT, with the original notices
retained. vLLM is included as a submodule under Apache-2.0. See [third-party notices](THIRD_PARTY_NOTICES.md)
for attribution and license details.
