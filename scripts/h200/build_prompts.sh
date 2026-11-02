#!/usr/bin/env bash
# Build the exact-length needle prompts (64K / 128K / 256K, two per length) used by every benchmark.
# Corpus: local technical-documentation markdown (KernelWiki by default); override with H200_CORPUS="dir1 dir2".
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
source "$HERE/env.sh"
CORPUS=${H200_CORPUS:-"/mnt/data1/$USER/KernelWiki/wiki /mnt/data1/$USER/KernelWiki/sources"}
OUT=${1:-$H200_WORK/prompts.json}
# shellcheck disable=SC2086
python "$REPO/scripts/build_prompts.py" --model "$QWEN38_MODEL" --corpus $CORPUS \
    --lengths "${LENGTHS:-65536,131072,260096}" --concurrency "${CONC:-2}" --output "$OUT"
