#!/usr/bin/env bash
# Sparse-attention TTFT on one H200: vLLM + hook (tools/patch_vllm.py must be applied to the installed vLLM),
# FA4 block-sparse kernel, first-token check against a stock run.
#
#   TAU=0.95 REF=$H200_WORK/ttft-stock-c1/results.json bash scripts/h200/bench_sparse.sh [output-dir] [extra args]
#   LENGTHS=65536 REPS=1 bash scripts/h200/bench_sparse.sh                   # quick single case
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
source "$HERE/env.sh"
# BLOCK_SIZE=1024 makes vLLM pages a multiple of the 64-token KV block so the FA4 path reads the cache in place
SERVE_ARGS=${SERVE_ARGS:-}${BLOCK_SIZE:+ --block-size $BLOCK_SIZE}
TAU=${TAU:-0.95}
KERNEL=${KERNEL:-fa4}
OUT=${1:-$H200_WORK/ttft-sparse-${KERNEL}-tau${TAU}-$(date +%Y%m%d-%H%M%S)}
shift || true
REFARG=()
[ -n "${REF:-}" ] && REFARG=(--reference "$REF")
python "$REPO/scripts/bench_ttft.py" --model "$QWEN38_MODEL" --prompts "${PROMPTS:-$H200_WORK/prompts.json}" \
    --output "$OUT" --lengths "${LENGTHS:-65536,131072,260096}" --concurrencies "${CONC:-1}" \
    --repetitions "${REPS:-2}" --max-num-batched-tokens "${CHUNK:-8192}" \
    --gpu-memory-utilization "${GPU_UTIL:-0.9}" --serve-args "$SERVE_ARGS" "${REFARG[@]}" \
    --env SM120_SPARSE_ATTN=1 "SM120_SPARSE_KERNEL=$KERNEL" "SM120_SPARSE_TAU=$TAU" "SM120_SPARSE_DEBUG=${DEBUG:-4}" "$@"
python "$REPO/scripts/summarize_ttft.py" --runs "stock=${REF%/results.json}" "sparse=$OUT" --output "$OUT/summary.json" 2>/dev/null || true
