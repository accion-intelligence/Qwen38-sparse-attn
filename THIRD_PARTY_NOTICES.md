# Third-party notices

- **NVIDIA CUTLASS** (BSD-3-Clause). `sm120_attn/kernels/kai_fa_bf16.py` (and through it `kai_fa_fp8.py` and
  `kai_fa_fp8_tma.py`) is derived from the CuTe DSL example
  `examples/python/CuTeDSL/cute/ampere/kernel/attention/flash_attention_v2.py` at CUTLASS commit
  78e46228a1bfc5c972e2f9a634223af784cd6949. `extras/bf16_mlp_fusion/kai_swiglu_vendor.py` is derived from the
  `blackwell_geforce` dense GEMM example at the same commit. The BSD-3-Clause notice is retained in those files.
- **NVIDIA CUTLASS** (BSD-3-Clause), A100 port. `a100_attn/kernels/fa_bf16_sm80.py` is derived from
  `sm120_attn/kernels/kai_fa_bf16.py` and therefore from the same CuTe DSL `flash_attention_v2.py` example;
  the BSD-3-Clause notice is retained.
- **flash-linear-attention** (MIT). `a100_attn/gdn_intra.py` contains the kernel
  `chunk_gated_delta_rule_fwd_kkt_solve_kernel` from `fla/ops/gated_delta_rule/chunk_fwd.py`
  (Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li); the MIT notice is retained in that file.
- **vLLM** (Apache-2.0). Included as the git submodule `vllm/` at tag `v0.30.0`, unmodified; this project
  patches one file of an installed copy (`patches/0001-sparse-attention-hook.md`; on A100,
  `tools/patch_vllm_a100.py` patches the FlashAttention backend and the bundled FLA `chunk.py`) and optionally the Qwen3.5
  model file (`extras/bf16_mlp_fusion`).
- **FlashInfer** (Apache-2.0). Optional; only the retired `kernel: flashinfer` reference path imports it.

This project itself is licensed under the Apache License 2.0 (see `LICENSE`).
