# Source this file. Environment for the H200 (SM90) branch of Qwen38 sparse-attn.
#
#   source scripts/h200/env.sh
#
# Everything heavy lives on the node's local RAID (/local1) because $HOME is on
# NFS with ~10 ms per-op latency (a torch+vLLM import from NFS takes ~10 min).
# Override any variable before sourcing.

export H200_USER_ROOT=${H200_USER_ROOT:-/local1/$USER}
export H200_ENV_DIR=${H200_ENV_DIR:-$H200_USER_ROOT/envs/qwen38-h200}   # serving/benchmark venv (vLLM)
export TIRX_ENV_DIR=${TIRX_ENV_DIR:-$H200_USER_ROOT/envs/tirx}           # kernel-authoring venv (TIRx-harness)
export H200_WORK=${H200_WORK:-$H200_USER_ROOT/exp}                       # prompts, results, logs
export UV_CACHE_DIR=${UV_CACHE_DIR:-$H200_USER_ROOT/.cache/uv}

# Qwen3.8-27B BF16 weights (HF cache layout, already on local disk).
export QWEN38_MODEL=${QWEN38_MODEL:-/local1/vllm_models/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

# CUDA 12.9 toolkit on local disk (the driver on this node, 575.57.08, caps at CUDA 12.9).
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.9}

# Rust toolchain for TIRx-harness NumSim (installed by setup_env.sh).
export CARGO_HOME=${CARGO_HOME:-$H200_USER_ROOT/.cargo}
export RUSTUP_HOME=${RUSTUP_HOME:-$H200_USER_ROOT/.rustup}

# One H200 per experiment. RULE (user, 2026-11-01): experiments may use ONLY GPU 4 and GPU 5 on this node.
# GPU 5 is the default (this user's conventional slot, port 3000X <-> GPU X); use GPU 4 for a second job.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}
case ",$CUDA_VISIBLE_DEVICES," in *,4,*|*,5,*) ;; *) echo "env.sh: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES is outside the allowed GPUs {4,5}" >&2 ;; esac

# JIT / compile caches on local disk too (defaults are under the NFS $HOME: vLLM torch.compile cache,
# Triton, FlashInfer JIT, inductor).
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$H200_USER_ROOT/.cache/vllm}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$H200_USER_ROOT/.cache/triton}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-$H200_USER_ROOT/.cache/flashinfer}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$H200_USER_ROOT/.cache/torchinductor}

export PATH="$H200_ENV_DIR/bin:$CARGO_HOME/bin:$CUDA_HOME/bin:$PATH"
export PYTHONUNBUFFERED=1
mkdir -p "$H200_WORK" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE" "$TORCHINDUCTOR_CACHE_DIR" 2>/dev/null || true

# Switch the shell to the TIRx-harness venv (kernel authoring / checks / profiling): `use_tirx`
use_tirx() { export PATH="$TIRX_ENV_DIR/bin:$PATH"; hash -r; echo "python -> $(command -v python)"; }
