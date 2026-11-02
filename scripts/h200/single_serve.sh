#!/usr/bin/env bash
# One co-located vLLM instance (prefill + decode on the same GPU) with the same flags as the PD prefill side,
# for comparing TTFT/TPOT under concurrency against the two-GPU PD stack (scripts/h200/pd_serve.sh).
#   bash scripts/h200/single_serve.sh start|wait|stop      # GPU ${GPU:-4}, port ${PORT:-30004}, tmux session "single"
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh"
GPU=${GPU:-4}; PORT=${PORT:-30004}
QUANT=${QUANT:-fp8}; TAU=${TAU:-0.95}; BLOCK_SIZE=${BLOCK_SIZE:-1024}; MAX_LEN=${MAX_LEN:-262144}
MAX_SEQS=${MAX_SEQS:-64}; CHUNK=${CHUNK:-8192}
LOGD=$H200_WORK/pd; mkdir -p "$LOGD"
COMMON="--served-model-name model --dtype bfloat16 --max-model-len $MAX_LEN --block-size $BLOCK_SIZE --no-enable-prefix-caching \
 --language-model-only --gpu-memory-utilization 0.9 --generation-config vllm --seed 20261015 ${QUANT:+--quantization $QUANT} \
 --max-num-seqs $MAX_SEQS --max-num-batched-tokens $CHUNK --enable-chunked-prefill"
ENVS="SM120_SPARSE_ATTN=${SPARSE:-1} SM120_SPARSE_KERNEL=fa4 SM120_SPARSE_TAU=$TAU SM120_SPARSE_SELECT_QB=64 SM120_SPARSE_SCHED=sorted SM120_SPARSE_DEBUG=1"
case "${1:-start}" in
  start)
    tmux kill-session -t single 2>/dev/null || true
    tmux new-session -d -s single "cd $LOGD && source $HERE/env.sh && export CUDA_VISIBLE_DEVICES=$GPU $ENVS && vllm serve $QWEN38_MODEL --host 127.0.0.1 --port $PORT $COMMON 2>&1 | tee $LOGD/single.log; sleep 1d"
    echo "started tmux session 'single' (GPU $GPU, port $PORT); log $LOGD/single.log" ;;
  wait) until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; do sleep 5; done; echo "port $PORT healthy" ;;
  stop) tmux kill-session -t single 2>/dev/null || true; pkill -TERM -f "vllm serve.*--port $PORT" || true; echo stopped ;;
esac
