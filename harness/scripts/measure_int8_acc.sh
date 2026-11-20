#!/usr/bin/env bash
# Measure the int8 body chain against the fp16 model on one frame, and emit both halves as records.
#
#   harness/scripts/measure_int8_acc.sh <frame.raw> [width] [height]
#
# The frame must come from remastering/ (see DELEGATION.md): a casual clip upscaled to the target grid
# is a population this project never decided on. One GPU job at a time - the caller holds the lock:
#   flock -n /var/tmp/opencode/gpu.lock harness/scripts/measure_int8_acc.sh <frame>
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${RVE_TRT_PY:-$HOME/.cache/opencode/venvs/trt310/bin/python}"
MODEL="${RVE_MODEL:-/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2/custom_models/2x-Futsuu-Anime(1).pth}"
FRAME="$1"; W="${2:-1920}"; H="${3:-1080}"
OUT="${4:-/tmp/opencode/int8-acc}"
mkdir -p "$OUT"

echo "== int8 chain (emits certification + chain time)"
"$PY" harness/harness/scripts/int8_pipeline.py "$FRAME" --width "$W" --height "$H" --iters 10 --dump "$OUT/int8.raw" --emit "$OUT/pipeline.jsonl"

echo "== fp16 reference of the same frame"
"$PY" harness/harness/scripts/measure_residual.py "$FRAME" "$W" "$H" "$MODEL" "" "$OUT/ref.raw" | tail -2

echo "== the picture, one against the other"
"$PY" harness/harness/scripts/compare_raw.py "$OUT/ref.raw" "$OUT/int8.raw" --width "$W" --height "$H" --scale 2 --emit "$OUT/accuracy.jsonl"
echo "records: $OUT/pipeline.jsonl{,.time} $OUT/accuracy.jsonl"
