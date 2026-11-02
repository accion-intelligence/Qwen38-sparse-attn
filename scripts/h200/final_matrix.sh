#!/usr/bin/env bash
# Back-to-back TTFT matrix on one GPU (same conditions for every row): stock BF16, sparse FA4 (tau, 64-row
# selection), dense + vLLM FP8 GEMM, sparse + FP8 GEMM. Results in $H200_WORK/matrix-<date>/.
#
#   CUDA_VISIBLE_DEVICES=5 TAU=0.95 bash scripts/h200/final_matrix.sh
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
source "$HERE/env.sh"
TAU=${TAU:-0.95}; REPS=${REPS:-2}
OUT=${OUT:-$H200_WORK/matrix-$(date +%Y%m%d-%H%M%S)}; mkdir -p "$OUT"
echo "matrix -> $OUT (GPU $CUDA_VISIBLE_DEVICES, tau $TAU)"
REPS=$REPS bash "$HERE/bench_stock.sh" "$OUT/stock-bf16" > "$OUT/stock-bf16.log" 2>&1
REF=$OUT/stock-bf16/results.json
REPS=$REPS TAU=$TAU DEBUG=1 REF=$REF bash "$HERE/bench_sparse.sh" "$OUT/sparse-bf16" SM120_SPARSE_SELECT_QB=64 > "$OUT/sparse-bf16.log" 2>&1
REPS=$REPS SERVE_ARGS="--quantization fp8" bash "$HERE/bench_stock.sh" "$OUT/dense-fp8" > "$OUT/dense-fp8.log" 2>&1
REPS=$REPS TAU=$TAU DEBUG=1 REF=$REF SERVE_ARGS="--quantization fp8" bash "$HERE/bench_sparse.sh" "$OUT/sparse-fp8" SM120_SPARSE_SELECT_QB=64 > "$OUT/sparse-fp8.log" 2>&1
python "$REPO/scripts/summarize_ttft.py" --runs "stock=$OUT/stock-bf16" "sparse=$OUT/sparse-bf16" "fp8=$OUT/dense-fp8" "sparse+fp8=$OUT/sparse-fp8" --output "$OUT/summary.json"
