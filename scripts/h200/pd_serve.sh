#!/usr/bin/env bash
# Disaggregated prefill/decode serving of Qwen3.8-27B on two H200s (NIXL connector over NVLink/UCX):
#   P = GPU 4 (sparse FA4 attention hook + FP8 GEMM, our prefill stack), D = GPU 5 (stock FA3 + FP8 GEMM),
#   proxy on $PROXY_PORT routes each request P -> D.
#
#   bash scripts/h200/pd_serve.sh start      # launch P, D, proxy in tmux session "pd"; logs in $H200_WORK/pd/
#   bash scripts/h200/pd_serve.sh wait       # block until both engines report healthy
#   bash scripts/h200/pd_serve.sh stop
# Knobs: P_SPARSE=1|0 (hook on P), QUANT=fp8|"" , TAU, BLOCK_SIZE (default 1024, same on both sides),
#        MAX_LEN (default 262144), P_MAX_SEQS (default 4), D_MAX_SEQS (default 64), P_GPU=4, D_GPU=5.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
source "$HERE/env.sh"
P_GPU=${P_GPU:-4}; D_GPU=${D_GPU:-5}
P_PORT=${P_PORT:-30014}; D_PORT=${D_PORT:-30015}; PROXY_PORT=${PROXY_PORT:-30010}
P_SIDE=${P_SIDE:-5559}; D_SIDE=${D_SIDE:-5659}
QUANT=${QUANT:-fp8}; TAU=${TAU:-0.95}; BLOCK_SIZE=${BLOCK_SIZE:-1024}; MAX_LEN=${MAX_LEN:-262144}
P_MAX_SEQS=${P_MAX_SEQS:-4}; D_MAX_SEQS=${D_MAX_SEQS:-64}; CHUNK=${CHUNK:-8192}
LOGD=$H200_WORK/pd; mkdir -p "$LOGD"
PROXY=/local1/li004074/exp/pd/toy_proxy_server.py
COMMON="--served-model-name model --dtype bfloat16 --max-model-len $MAX_LEN --block-size $BLOCK_SIZE --no-enable-prefix-caching \
 --language-model-only --gpu-memory-utilization 0.9 --generation-config vllm --seed 20261015 \
 ${QUANT:+--quantization $QUANT}"
KV_P='{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'
KV_D='{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}'
P_ENV="SM120_SPARSE_ATTN=${P_SPARSE:-1} SM120_SPARSE_KERNEL=fa4 SM120_SPARSE_TAU=$TAU SM120_SPARSE_SELECT_QB=64 SM120_SPARSE_SCHED=sorted SM120_SPARSE_DEBUG=1"
case "${1:-start}" in
  start)
    tmux kill-session -t pd 2>/dev/null || true
    tmux new-session -d -s pd -n prefill "cd $LOGD && source $HERE/env.sh && export CUDA_VISIBLE_DEVICES=$P_GPU VLLM_SSM_CONV_STATE_LAYOUT=DS UCX_TLS=all UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT=$P_SIDE $P_ENV && \
      vllm serve $QWEN38_MODEL --host 127.0.0.1 --port $P_PORT $COMMON --max-num-seqs $P_MAX_SEQS --max-num-batched-tokens $CHUNK --enable-chunked-prefill \
      --kv-transfer-config '$KV_P' 2>&1 | tee $LOGD/prefill.log; sleep 1d"
    tmux new-window -t pd -n decode "cd $LOGD && source $HERE/env.sh && export CUDA_VISIBLE_DEVICES=$D_GPU VLLM_SSM_CONV_STATE_LAYOUT=DS UCX_TLS=all UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT=$D_SIDE SM120_SPARSE_ATTN=0 && \
      vllm serve $QWEN38_MODEL --host 127.0.0.1 --port $D_PORT $COMMON --max-num-seqs $D_MAX_SEQS \
      --kv-transfer-config '$KV_D' 2>&1 | tee $LOGD/decode.log; sleep 1d"
    tmux new-window -t pd -n proxy "cd $LOGD && source $HERE/env.sh && python $PROXY --port $PROXY_PORT --prefiller-hosts 127.0.0.1 --prefiller-ports $P_PORT --decoder-hosts 127.0.0.1 --decoder-ports $D_PORT 2>&1 | tee $LOGD/proxy.log; sleep 1d"
    echo "started tmux session 'pd' (P :$P_PORT on GPU $P_GPU, D :$D_PORT on GPU $D_GPU, proxy :$PROXY_PORT); logs in $LOGD" ;;
  wait)
    for p in $P_PORT $D_PORT; do until curl -sf "http://127.0.0.1:$p/health" >/dev/null; do sleep 5; done; echo "port $p healthy"; done
    until curl -sf "http://127.0.0.1:$PROXY_PORT/healthcheck" >/dev/null 2>&1 || curl -sf "http://127.0.0.1:$PROXY_PORT/v1/models" >/dev/null 2>&1; do sleep 2; done; echo "proxy up" ;;
  stop)
    tmux kill-session -t pd 2>/dev/null || true; pkill -f "toy_proxy_server.py" || true; pkill -TERM -f "vllm serve.*--port $P_PORT" || true; pkill -TERM -f "vllm serve.*--port $D_PORT" || true; echo stopped ;;
esac
