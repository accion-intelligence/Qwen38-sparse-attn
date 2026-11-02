#!/usr/bin/env bash
# Output-quality pipeline for one sparse configuration vs the BF16 dense reference (in-process vLLM, one GPU):
#   dense reference generations -> dense teacher-forced scores -> sparse generations (needle) -> sparse scores.
#
#   TAU=0.95 KERNEL=fa4 bash scripts/h200/quality.sh            # results in $H200_WORK/quality/
#   STEPS=sparse bash scripts/h200/quality.sh                    # skip the (cached) dense reference steps
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
source "$HERE/env.sh"
TAU=${TAU:-0.95}; KERNEL=${KERNEL:-fa4}; TAG=${TAG:-${KERNEL}-tau${TAU}}
Q=$H200_WORK/quality; mkdir -p "$Q"
P=${PROMPTS:-$H200_WORK/prompts.json}
LENGTHS=${LENGTHS:-65536,131072,260096}
COMMON=(--model "$QWEN38_MODEL" --prompts "$P" --lengths "$LENGTHS" ${BLOCK_SIZE:+--block-size $BLOCK_SIZE} ${QUANT:+--quantization $QUANT})   # QUANT=fp8 for the FP8 GEMM variants
if [ "${STEPS:-all}" = all ]; then
  [ -f "$Q/gen-dense.json" ] || SM120_SPARSE_ATTN=0 python "$REPO/scripts/validate_generation.py" "${COMMON[@]}" --output "$Q/gen-dense.json" --max-tokens "${MAX_TOKENS:-384}"
  [ -f "$Q/tfe-dense.json" ] || SM120_SPARSE_ATTN=0 python "$REPO/scripts/teacher_forced_eval.py" "${COMMON[@]}" --reference "$Q/gen-dense.json" --output "$Q/tfe-dense.json"
fi
export SM120_SPARSE_ATTN=1 SM120_SPARSE_KERNEL=$KERNEL SM120_SPARSE_TAU=$TAU
python "$REPO/scripts/validate_generation.py" "${COMMON[@]}" --output "$Q/gen-$TAG.json" --max-tokens "${MAX_TOKENS:-384}"
python "$REPO/scripts/teacher_forced_eval.py" "${COMMON[@]}" --reference "$Q/gen-dense.json" --reference-scores "$Q/tfe-dense.json" --output "$Q/tfe-$TAG.json"
python "$REPO/scripts/compare_generations.py" --dense "$Q/gen-dense.json" --sparse "$Q/gen-$TAG.json" --output "$Q/compare-$TAG.json" || true
python - "$Q/tfe-$TAG.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
for row in r['rows']:
    print(f"{row['length']:>7}: dNLL {row.get('delta_nll', float('nan')):+.4f}  top1 {row['top1_agreement']:.3f}  mean|dlp| {row.get('delta_logprob_mean', float('nan')):.4f}  max|dlp| {row.get('delta_logprob_max', float('nan')):.3f}")
PY
