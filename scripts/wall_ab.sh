#!/usr/bin/env bash
# Interleaved wall A/B: the shipped fp16 engine against the int8 body, in ONE session.
#
# Why interleaved: the product's own phase log and the offline window disagree by more than run-to-run
# drift (1.35-1.42x vs 1.14x vs 0.95x), and every one of those numbers is a single draw taken while
# other work ran on the machine. Alternating the arms inside each round makes the drift common to both
# arms; the median of rounds 2+ is the statistic, rounds are not.
#
# What this does NOT do: it does not touch the gate (reuse is off on both arms, --no_temporal_reuse), it
# does not prove picture equality (that is int8-on-bd / gate-bd), and it says nothing about another card.
#
# Usage:
#   SINK=null  ROUNDS=3 scripts/wall_ab.sh          # render wall clock, nothing written
#   SINK=nvenc ROUNDS=3 scripts/wall_ab.sh          # the same with a real hevc_nvenc encode
#
# Environment: CLIPS (space separated, default the two BD cuts), ROUNDS, SINK, LOGDIR, RENDERS.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
APP=/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2
# The backend runs on the installed portable interpreter: the repo's own .venv has no cv2, and the
# installed packages are what every recorded wall number was taken with (REPORT-int8-lane-2026-11-16 10).
PY=${PY:-"$APP/python/python/bin/python3"}
CLIPS=${CLIPS:-"bd_violet20s.mkv bd_csm_same.mkv"}
ROUNDS=${ROUNDS:-3}
SINK=${SINK:-null}
CLIPROOT=${CLIPROOT:-/tmp/opencode}
LOGDIR=${LOGDIR:-"$ROOT/docs/wsp/verified/logs/wall-ab"}
RENDERS=${RENDERS:-/home/jorj/2Tb/wall-ab-2026-11-18}
LOCK=/var/tmp/opencode/gpu.lock

mkdir -p "$LOGDIR" "$RENDERS" /var/tmp/opencode

common=(
  --ffmpeg_path "$APP/bin/ffmpeg"
  --backend tensorrt
  --upscale_model "$APP/custom_models/2x-Futsuu-Anime(1).pth"
  --overwrite
  --no_temporal_reuse
)

run_arm() {  # clip arm round
  local clip=$1 arm=$2 round=$3 out preset
  local name="${clip%.mkv}_${arm}_r${round}_${SINK}"
  local log="$LOGDIR/$name.log"
  local args=("${common[@]}" --input "$CLIPROOT/$clip" --int8_fit "$arm")
  if [ "$SINK" = null ]; then
    out=null
    args+=(--output null --video_encoder_preset ffv1 --video_pixel_format yuv444p)
  else
    out="$RENDERS/$name.mkv"
    args+=(--output "$out" --video_encoder_preset x265_nvenc)
  fi
  {
    echo "# wall_ab $(date -Is) clip=$clip arm=$arm round=$round sink=$SINK"
    echo "# load_avg $(cut -d' ' -f1-3 /proc/loadavg)"
    echo "# clocks $(nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv,noheader 2>/dev/null || echo unavailable)"
    echo "# cmd rve-backend.py ${args[*]}"
  } > "$log"
  ( cd "$ROOT/backend" && flock -n "$LOCK" timeout 1800 "$PY" rve-backend.py "${args[@]}" ) >> "$log" 2>&1 \
    || { echo "# EXIT $?" >> "$log"; }
  grep -ao "render loop: [0-9.]* s for [0-9]* frames ([0-9.]* ms/frame)" "$log" | tail -1 \
    | sed "s|^|$name  |" || echo "$name  NO-LOOP-LINE"
}

for clip in $CLIPS; do
  for round in $(seq 1 "$ROUNDS"); do
    # alternate the order so a warm-up trend lands on both arms
    if [ $((round % 2)) -eq 1 ]; then arms=(fp16 int8); else arms=(int8 fp16); fi
    for arm in "${arms[@]}"; do
      run_arm "$clip" "$arm" "$round"
    done
  done
done
