#!/usr/bin/env bash
# Stock vLLM time-to-first-token baseline on one H200 (dense attention, BF16 GEMM, sparse hook off).
#
#   bash scripts/h200/bench_stock.sh [output-dir] [extra bench_ttft.py args...]
#   LENGTHS=65536 CONC=1 REPS=1 bash scripts/h200/bench_stock.sh        # quick single case
#
# Same methodology as docs/results.md: chunked prefill 8192, no prefix caching, greedy, one warmup.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
source "$HERE/env.sh"
# BLOCK_SIZE=1024 makes vLLM pages a multiple of the 64-token KV block so the FA4 path reads the cache in place
SERVE_ARGS=${SERVE_ARGS:-}${BLOCK_SIZE:+ --block-size $BLOCK_SIZE}
OUT=${1:-$H200_WORK/ttft-stock-$(date +%Y%m%d-%H%M%S)}
shift || true
python "$REPO/scripts/bench_ttft.py" --model "$QWEN38_MODEL" --prompts "${PROMPTS:-$H200_WORK/prompts.json}" \
    --output "$OUT" --lengths "${LENGTHS:-65536,131072,260096}" --concurrencies "${CONC:-1}" \
    --repetitions "${REPS:-2}" --max-num-batched-tokens "${CHUNK:-8192}" \
    --gpu-memory-utilization "${GPU_UTIL:-0.9}" --serve-args "$SERVE_ARGS" --env SM120_SPARSE_ATTN=0 "$@"
python "$REPO/scripts/summarize_ttft.py" --runs "stock=$OUT" --output "$OUT/summary.json" || true
