"""Import-time vLLM patches for the H200 branch (installed by tools/patch_vllm.py into qwen3_5.py).

SM120_FUSED_NORM (default 0: no end-to-end gain under torch.compile, see docs/h200.md): replace GemmaRMSNorm.forward_cuda (torch-op chain) with the fused Triton kernel.
"""
from __future__ import annotations

import os

_ENABLED = os.environ.get('SM120_FUSED_NORM', '0') in ('1', 'true', 'True')


def _install_fused_gemma_norm():
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm
    from .fused_norm import gemma_rms_norm

    def forward_cuda(self, x, residual=None):
        return gemma_rms_norm(x, self.weight, self.variance_epsilon, residual)

    forward_cuda._kai_fused = True
    GemmaRMSNorm.forward_cuda = forward_cuda
    print('[kai-h200] GemmaRMSNorm.forward_cuda -> fused Triton kernel (SM120_FUSED_NORM=0 to disable)', flush=True)


if _ENABLED:
    _install_fused_gemma_norm()
