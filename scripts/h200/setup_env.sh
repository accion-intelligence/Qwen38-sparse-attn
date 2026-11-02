#!/usr/bin/env bash
# Idempotent environment setup for the H200 (SM90) branch.
#
#   bash scripts/h200/setup_env.sh            # full install (vLLM + kernels + eval + TIRx-harness)
#   H200_WITH_TIRX=0 bash scripts/h200/setup_env.sh   # skip TIRx-harness
#
# Differences from the SM120 recipe in README.md (CUDA 13 / torch cu130 / cuda-python 13):
# this node's driver supports CUDA <= 12.9, so every GPU wheel must be a cu12x build.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
source "$HERE/env.sh"

VLLM_VERSION=0.30.0      # same version as the SM120 work: tools/patch_vllm.py anchors on its flash_attn.py
VLLM_WHL="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"

mkdir -p "$H200_USER_ROOT/envs" "$UV_CACHE_DIR" "$H200_WORK"
UV_BOOT=$(command -v uv || true)
[ -n "$UV_BOOT" ] || UV_BOOT=/mnt/data1/$USER/miniforge3/bin/uv
[ -x "$UV_BOOT" ] || { echo "uv not found; install uv first (https://docs.astral.sh/uv/)"; exit 1; }

PY="$H200_ENV_DIR/bin/python"
if [ ! -x "$PY" ]; then
  echo "== creating venv $H200_ENV_DIR"
  "$UV_BOOT" venv --python 3.12 "$H200_ENV_DIR"
fi
"$UV_BOOT" pip install --python "$PY" -q "uv>=0.13"   # --torch-backend needs a recent uv
UV="$H200_ENV_DIR/bin/uv"
pipi() { "$UV" pip install --python "$PY" "$@"; }

echo "== vLLM ${VLLM_VERSION}+cu129 (pulls torch +cu129 from the PyTorch cu129 index)"
pipi "vllm @ ${VLLM_WHL}" --torch-backend=cu129

echo "== CuTe DSL (cu12 libs are the default extra) + cuda-python 12.9"
pipi "nvidia-cutlass-dsl==4.7.1" "cuda-python==12.9.*"

echo "== this repository (editable) + evaluation dependencies"
pipi -e "${REPO}[eval]"

echo "== FlashAttention-4 (CuTe DSL) for the SM90 block-sparse kernel, then the H200 kernel patches"
pipi "flash-attn-4==4.0.0b37"
"$PY" "${REPO}/tools/patch_fa4.py"

echo "== NIXL (UCX transport) for disaggregated prefill/decode across GPUs"
pipi "nixl[cu12]==1.5.2"

if [ "${H200_WITH_TIRX:-1}" = 1 ]; then
  # TIRx-harness (mlc-ai) lives in its own venv: it pins apache-tvm-ffi 0.1.14.post0 and numpy>=2.5.3 while
  # vLLM 0.30.0's compiled deps (flashinfer, quack-kernels, tilelang, cudnn-frontend) ship against
  # apache-tvm-ffi 0.1.11, and its CUDA 13 pins (nvidia-cutlass-dsl[cu13]==4.7.0, nvidia-cuda-nvrtc 13.2,
  # nvidia-cuda-nvdisasm 13.3) cannot run on this driver. Kernels exported from TIRx (tvm-ffi) are consumed
  # by the vLLM venv, which already has tvm-ffi.
  echo "== TIRx-harness venv $TIRX_ENV_DIR (CUDA 12.9 substitutes for its CUDA 13 pins)"
  TP="$TIRX_ENV_DIR/bin/python"
  [ -x "$TP" ] || "$UV" venv --python 3.12 "$TIRX_ENV_DIR"
  tpipi() { "$UV" pip install --python "$TP" "$@"; }
  tpipi "torch==2.13.0" --torch-backend=cu129
  tpipi kcoral "tirx-kernels>=0.1.2.post1"
  tpipi --no-deps tirx-harness
  tpipi "apache-tvm==0.27.0.post1" "apache-tvm-ffi==0.1.14.post0" "numpy>=2.5.3" ml-dtypes \
        "nvidia-cutlass-dsl==4.7.1" "cuda-python==12.9.*" nvidia-cuda-nvrtc-cu12 "nvidia-cuda-nvdisasm==13.3.73" \
        "pypandoc-binary>=1.17" "pyyaml>=6.0.3" "threadpoolctl>=3.6,<4" torch-c-dlpack-ext
  if [ ! -x "$CARGO_HOME/bin/cargo" ]; then
    echo "== Rust toolchain for TIRx NumSim -> $CARGO_HOME"
    curl -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --profile minimal
  fi
  "$TP" -c "import tvm.tirx, tvm_ffi, tirx_kernels.tirx_lite, tirx_harness; print('TIRx core imports OK', tirx_harness.__version__)"
fi

echo "== done. Next: source scripts/h200/env.sh && python scripts/h200/check_env.py --jit"
