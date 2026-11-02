#!/usr/bin/env bash
# Concurrency sweep against a running endpoint (PD proxy or a single instance) with pd_client.py.
#   URL=http://127.0.0.1:30010 TAG=pd bash scripts/h200/pd_bench.sh
#   LENGTHS=65536,260096 CONCS=1,2,4 MAX_TOKENS=256 REPS=1
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh"
URL=${URL:-http://127.0.0.1:30010}; TAG=${TAG:-pd}
OUT=$H200_WORK/pd/bench-$TAG-$(date +%H%M%S); mkdir -p "$OUT"
for L in ${LENGTHS//,/ }; do for C in ${CONCS//,/ }; do
  python "$HERE/pd_client.py" --url "$URL" --model model --prompts "${PROMPTS:-$H200_WORK/prompts.json}" --lengths "$L" \
      --concurrency "$C" --max-tokens "${MAX_TOKENS:-256}" --repetitions "${REPS:-1}" --warmup "${WARMUP:-1}" --output "$OUT/len$L-c$C.json" | tee -a "$OUT/summary.txt"
done; done
echo "results: $OUT"
