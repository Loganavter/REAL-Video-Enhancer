#!/usr/bin/env bash
# live_matrix.sh — the PIXEL half of harness/tests/test_preview_state_matrix.py.
#
# The offscreen rows prove what the pane holds; these rows prove what reached the screen, on a leased display, by
# driving the app's own control channel. One line per row:
#
#     state | expected | observed | artifact
#
# and a non-zero exit when a row did not hold. Usage:
#
#     harness/scripts/live.sh up                     # boots a private Xvfb + the app (the ONLY X server involved)
#     harness/scripts/live_matrix.sh <clip.mkv>
#
# Prerequisites for the log-derived rows (`compare fill`, `pair_off`): the app must have been booted with
# `RVE_PREVIEW_LOG=1` (see `src/ui/video/telemetry.py`); the script reads `$RVE_PREVIEW_LOG_FILE` (default
# `preview-log.txt` at the checkout root, which is where the app's CWD puts it). Without that file the rows that need a
# number print UNPROVEN rather than a guess.
#
# The script NEVER starts an X server and never kills anything: it assumes `live.sh up` already ran and talks to that
# instance through its control dir (`$RVE_LIVE_STATE`, default /tmp/opencode/live). It also never presses a control on a
# hidden widget (`harness/docs/CODE_PATTERNS.md` §10: a click on a hidden control silently does nothing and reads as
# "the product").
#
# Which verb each row uses:
#   * `preview:<clip>` — start a preview job (it answers when the backend child exits, so the script runs it in the
#     background and drives the pane while it renders);
#   * `click:preview_mode_sbs` / `click:preview_play` — the two controls the rows about modes and transport need;
#   * `drag:preview_timeline@x1,y1:x2,y2` — the visible scrubber (a real press + four moves + a release, i.e. the
#     per-step `SEEK_TO` stream audit §6.3 measured). It is the window's ONLY seek: the pane's own painted strip, its
#     four gesture guards and its fraction-valued `seekRequested` are DELETED (audit §6.6, commit "one bar, one pane
#     protocol"), so `drag:previewLabel@...` is a pan/compare gesture and never a seek, and `state:previewLabel` no
#     longer carries `strip`/`strip_visible` (its `progress` and box still answer). Every seek row below drives
#     `preview_timeline`, and `state:preview_timeline` supplies that widget's own box for the coordinates;
#   * `state:previewLabel` — the pane's progress and box (a read; nothing is inferred from a screenshot where a number
#     exists);
#   * `shot` — the screen truth (`live.sh shot`, ImageMagick on the root window). `window.grab()` re-renders the tree
#     and hides exactly the stale-frame bugs these rows exist for, so it is not used.
#
# Exit: 0 = every row observed and held; 1 = at least one MISMATCH; 2 = no mismatch, but at least one row could not be
# observed (the table says which and why). The `playing, sbs, strip drag` row was the GATE for plan P1: it was expected to
# mismatch before `preview/pair.py` landed and must hold after it (79ea250c) - so a MISMATCH there is a regression now, not
# the expected state.
#
# Artifacts (shots, crops, the extracted log window per row) go to `$RVE_MATRIX_ARTIFACTS`
# (default /tmp/opencode/live-matrix-<stamp>); every row prints the files that back its verdict.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LIVE="${RVE_LIVE:-$ROOT/harness/scripts/live.sh}"
BASE_STATE="${RVE_LIVE_STATE:-/tmp/opencode/live}"
CLIP="${1:-${RVE_MATRIX_CLIP:-}}"
ART="${RVE_MATRIX_ARTIFACTS:-/tmp/opencode/live-matrix-$(date +%Y%m%d-%H%M%S)}"

die() { echo "live_matrix: $*" >&2; exit 2; }

[ -n "$CLIP" ] || die "usage: live_matrix.sh <clip.mkv>   (the app must already be up: live.sh up)"
[ -f "$CLIP" ] || die "no such clip: $CLIP"
[ -f "$LIVE" ] || die "no live.sh at $LIVE (RVE_LIVE overrides)"
if [ -f "$BASE_STATE/app.pid" ] && ! kill -0 "$(cat "$BASE_STATE/app.pid")" 2>/dev/null; then
    die "no live instance (looked at $BASE_STATE/app.pid) — run live.sh up first"
fi
[ -f "$BASE_STATE/app.pid" ] || die "no live instance (looked at $BASE_STATE/app.pid) — run live.sh up first"

STATE_DIR="$BASE_STATE"
[ -f "$BASE_STATE/state_dir" ] && STATE_DIR="$(cat "$BASE_STATE/state_dir")"
CONTROL="${RVE_UI_CAPTURE_SERVER:-${RVE_CONTROL_DIR:-}}"
if [ -z "$CONTROL" ] && [ -f "$STATE_DIR/control_dir" ]; then CONTROL="$(cat "$STATE_DIR/control_dir")"; fi
[ -n "$CONTROL" ] || CONTROL="$STATE_DIR/control"
[ -f "$CONTROL/ready" ] || die "the control channel is not ready at $CONTROL — is the app up?"

if command -v magick >/dev/null 2>&1; then IMAGEMAGICK="magick"; else IMAGEMAGICK="convert"; fi
command -v "$IMAGEMAGICK" >/dev/null 2>&1 || die "ImageMagick is missing: the pixel rows need it"

mkdir -p "$ART" || die "cannot create $ART"

# The pane's own telemetry file. An explicit `RVE_PREVIEW_LOG_FILE` wins (the app may not have written its header yet);
# otherwise the app's default (`preview-log.txt` at its CWD) and, failing that, the live state dir.
PREVIEW_LOG="${RVE_PREVIEW_LOG_FILE:-$ROOT/preview-log.txt}"
if [ ! -f "$PREVIEW_LOG" ] && [ -z "${RVE_PREVIEW_LOG_FILE:-}" ] && [ -f "$STATE_DIR/preview-log.txt" ]; then
    PREVIEW_LOG="$STATE_DIR/preview-log.txt"
fi

# --- the channel ----------------------------------------------------------------------------------------------------

ctrl() {                       # ctrl <verb> [seconds] -> the app's JSON answer
    local verb="$1" seconds="${2:-12}"
    python3 - "$verb" "$seconds" "$CONTROL" <<'PY'
import json, os, sys, time, uuid
verb, seconds, root = sys.argv[1], float(sys.argv[2]), sys.argv[3]
rid = uuid.uuid4().hex[:10]
requests = os.path.join(root, "requests")
answers = os.path.join(root, "answers")
os.makedirs(requests, exist_ok=True)
os.makedirs(answers, exist_ok=True)
target = os.path.join(answers, rid + ".json")
stage = os.path.join(requests, rid + ".tmp")
with open(stage, "w", encoding="utf-8") as handle:
    handle.write(verb)
os.replace(stage, os.path.join(requests, rid))     # the sanctioned publish form (drain_requests)
deadline = time.time() + seconds
while time.time() < deadline:
    if os.path.exists(target):
        try:
            with open(target, encoding="utf-8") as handle:
                print(json.dumps(json.load(handle), ensure_ascii=False))
            sys.exit(0)
        except ValueError:
            pass
    time.sleep(0.05)
print(json.dumps({"error": "no answer in %.0fs" % seconds, "verb": verb}))
sys.exit(1)
PY
}

json_get() {                   # json_get <json> <dotted.path> [default]
    python3 - "$@" <<'PY'
import json, sys
raw, path = sys.argv[1], sys.argv[2]
default = sys.argv[3] if len(sys.argv) > 3 else ""
try:
    value = json.loads(raw)
    for part in path.split("."):
        value = value[int(part)] if part.lstrip("-").isdigit() else value[part]
    print("" if value is None else value)
except Exception:
    print(default)
PY
}

log_mark() { if [ -f "$PREVIEW_LOG" ]; then wc -l < "$PREVIEW_LOG"; else echo 0; fi; }

log_since() {                  # the log window a row owns: everything written after its mark
    local mark="$1"
    if [ -f "$PREVIEW_LOG" ]; then tail -n "+$((mark + 1))" "$PREVIEW_LOG"; fi
}

shot() {                       # shot <name> -> the root-window capture's path
    local path="$ART/$1.png"
    "$LIVE" shot "$path" >/dev/null 2>&1 && echo "$path"
}

crop_stat() {                  # crop_stat <png> <x> <y> <w> <h> <fx>
    "$IMAGEMAGICK" "$1" -crop "${4}x${5}+${2}+${3}" +repage -colorspace Gray -format "%[fx:$6]" info: 2>/dev/null
}

picture_rect() {               # the newest draw line's rect="x,y,w,h" - the picture itself, with no bar or transport
    python3 - "$1" <<'PYEOF'
import re, sys
rect = None
for line in open(sys.argv[1], errors="replace"):
    m = re.search(r" rect=(\d+),(\d+),(\d+),(\d+)", line)
    if m:
        rect = m.groups()
print(" ".join(rect) if rect else "")
PYEOF
}

crop_sig() {                   # crop_sig <png> <x> <y> <w> <h> -> a hash of the region's pixels
    "$IMAGEMAGICK" "$1" -crop "${4}x${5}+${2}+${3}" +repage -colorspace Gray -format "%[signature]" info: 2>/dev/null
}

crop_out() {                   # crop_out <png> <name> <x> <y> <w> <h> -> the crop file kept as the row's artifact
    local path="$ART/$2.png"
    "$IMAGEMAGICK" "$1" -crop "${5}x${6}+${3}+${4}" +repage "$path" 2>/dev/null && echo "$path"
}

# --- small parsers over the app's own record ------------------------------------------------------------------------

pair_lines() {                 # half=pair draws: "<result_t> <original_t|->" per line
    python3 -c '
import sys
for line in sys.stdin:
    if "preview: draw" not in line or "mode=sbs" not in line or "half=pair" not in line:
        continue
    fields = dict(part.split("=", 1) for part in line.split() if "=" in part)
    print(fields.get("result_t", "-"), fields.get("original_t", "-"))
'
}

draw_generations() {           # the generation each sbs draw named, in order
    python3 -c '
import sys
for line in sys.stdin:
    if "preview: draw" in line and "mode=sbs" in line and "generation=" in line:
        print(line.split("generation=", 1)[1].split()[0])
'
}

pair_off() {                   # pair_off=<off>/<total> from the newest stats line -> "<off> <total>"
    python3 -c '
import sys
last = None
for line in sys.stdin:
    if "preview: stats" in line and "pair_off=" in line:
        last = line
if last is None:
    print("- -")
else:
    token = last.split("pair_off=", 1)[1].split()[0]
    off, _, total = token.partition("/")
    print(off or "-", total or "-")
'
}

fill_max() {                   # fill_max <field>: the HIGHEST `compare fill` value in the window (a peak, not an instant)
    python3 -c '
import sys
wanted = sys.argv[1]
best = None
for line in sys.stdin:
    if "preview: compare fill" not in line:
        continue
    fields = dict(part.split("=", 1) for part in line.split() if "=" in part)
    try:
        value = int(fields.get(wanted, ""))
    except ValueError:
        continue
    best = value if best is None else max(best, value)
print("-" if best is None else best)
' "$1"
}

fill_field() {                 # fill_field <field>: the newest `compare fill` value
    python3 -c '
import sys
wanted = sys.argv[1]
last = None
for line in sys.stdin:
    if "preview: compare fill" in line:
        last = line
if last is None:
    print("-")
else:
    fields = dict(part.split("=", 1) for part in last.split() if "=" in part)
    print(fields.get(wanted, "-"))
' "$1"
}

# --- the table ------------------------------------------------------------------------------------------------------

MISMATCHES=0
UNPROVEN=0

row() {                        # row <state> <expected> <observed> <artifact> <OK|MISMATCH|UNPROVEN>
    printf '%-28s | %-58s | %-58s | %s\n' "$1" "$2" "$3" "$4"
    case "$5" in
        OK) ;;
        MISMATCH) MISMATCHES=$((MISMATCHES + 1)) ;;
        *) UNPROVEN=$((UNPROVEN + 1)) ;;
    esac
}

wait_state() {                 # wait_state <python expression over `s`> <seconds> -> 0 when it held once
    local expr="$1" seconds="${2:-20}" waited=0 state=""
    while [ "$waited" -lt $((seconds * 5)) ]; do
        state="$(ctrl state:previewLabel 8)"
        if python3 - "$state" "$expr" <<'PY'
import json, sys
try:
    s = json.loads(sys.argv[1])
except Exception:
    sys.exit(1)
if s.get("error"):
    sys.exit(1)
try:
    sys.exit(0 if eval(sys.argv[2], {"s": s}) else 1)     # noqa: S307 - the expression is this script's own
except Exception:
    sys.exit(1)
PY
        then return 0; fi
        sleep 0.2
        waited=$((waited + 1))
    done
    return 1
}

pane_geometry() {              # pane_geometry -> "<x> <y> <w> <h>" of previewLabel on the screen
    local state; state="$(ctrl state:previewLabel 8)"
    printf '%s %s %s %s' \
        "$(json_get "$state" global.0 0)" "$(json_get "$state" global.1 0)" \
        "$(json_get "$state" size.0 0)" "$(json_get "$state" size.1 0)"
}

# =====================================================================================================================
# Stage A — a preview job, paused, then playing. Rows 1..5, 8 and 7 run here.
# =====================================================================================================================

echo "live_matrix: instance $STATE_DIR, control $CONTROL"
echo "live_matrix: clip $CLIP"
echo "live_matrix: artifacts $ART"
echo "live_matrix: preview log ${PREVIEW_LOG}"
if [ -f "$PREVIEW_LOG" ]; then
    LEVEL="$(grep 'preview telemetry session' "$PREVIEW_LOG" | tail -1 | sed -n 's/.*level=\([0-9]*\).*/\1/p')"
    [ "${LEVEL:-0}" -ge 1 ] || echo "live_matrix: WARNING the app was booted with RVE_PREVIEW_LOG=${LEVEL:-?}: the" \
        "\`compare fill\`/\`pair_off\` rows cannot be observed. Reboot live.sh with RVE_PREVIEW_LOG=1."
fi
printf '\n%-28s | %-58s | %-58s | %s\n' "state" "expected" "observed" "artifact"
printf '%s\n' "------------------------------------------------------------------------------------------------------------------------------------------"

ctrl "input:$CLIP" 20 > "$ART/input.json" || echo "live_matrix: WARNING input: did not answer; the job may use the loaded clip"
ctrl "click:preview_mode_sbs" 10 > "$ART/mode.json"

# The preview job answers when its backend child exits, so it runs in the background while the rows are observed.
ctrl "preview:$CLIP" 900 > "$ART/preview.json" &
PREVIEW_PID=$!

# A produced, painted frame in sbs is the precondition of every row below.
wait_state 's.get("progress") and s["progress"][1] and s["progress"][0] > 0' 120 \
    || echo "live_matrix: WARNING no painted frame within 120 s; the rows below will say what they saw"
read -r PX PY PW PH <<<"$(pane_geometry)"
[ "${PW:-0}" -gt 8 ] || die "previewLabel has no box (state: $PX,$PY ${PW}x${PH})"
HALF_W=$((PW / 2))
echo "live_matrix: pane at ${PX},${PY} ${PW}x${PH} (halves ${HALF_W} px)"

# The visible scrubber's own box: `drag:` takes WIDGET-local points, and the strip is a small widget in the bottom
# bar - reading the pane's box for it would put every "drag" outside the control and the row would read as "the
# product ignored the gesture" (code-patterns.md §10).
TIMELINE_STATE="$(ctrl state:preview_timeline 10)"
TLW="$(json_get "$TIMELINE_STATE" size.0 0)"
TLH="$(json_get "$TIMELINE_STATE" size.1 0)"
[ "${TLW:-0}" -gt 8 ] || die "preview_timeline has no box: ${TIMELINE_STATE:0:200}"
echo "live_matrix: timeline ${TLW}x${TLH}"
strip_drag() {                 # strip_drag <from-fraction> <to-fraction>
    python3 - "$1" "$2" "$TLW" "$TLH" <<'PY'
import sys
start, end, width, height = (float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
y = height // 2
x1 = int(max(1.0, min(width - 2.0, start * width)))
x2 = int(max(1.0, min(width - 2.0, end * width)))
print(f"{x1},{y}:{x2},{y}")
PY
}

# --- row 1: paused, sbs, pair ---------------------------------------------------------------------------------------

wait_state 's["progress"][0] > 0' 60 || true
sleep 1
RIGHT_BEFORE="$(shot row1-pair)"
if [ -n "$RIGHT_BEFORE" ]; then
    L1="$(crop_stat "$RIGHT_BEFORE" "$PX" "$PY" "$HALF_W" "$PH" standard_deviation)"
    R1="$(crop_stat "$RIGHT_BEFORE" "$PX" $((PX + HALF_W)) "$HALF_W" "$PH" standard_deviation)"
    crop_out "$RIGHT_BEFORE" row1-left "$PX" "$PY" "$HALF_W" "$PH" >/dev/null
    crop_out "$RIGHT_BEFORE" row1-right "$PX" $((PX + HALF_W)) "$HALF_W" "$PH" >/dev/null
fi
QUIET_MARK=$(log_mark)
sleep 2
WINDOW="$(log_since "$QUIET_MARK")"
ASKS="$(printf '%s\n' "$WINDOW" | fill_field asks)"
POOL="$(printf '%s\n' "$WINDOW" | fill_field pool)"
if [ -n "${L1:-}" ] && [ -n "${R1:-}" ]; then
    if python3 -c "import sys; sys.exit(0 if ($L1 > 0.02 and $R1 > 0.02) else 1)"; then
        if [ "$ASKS" = "0" ]; then
            row "paused, sbs, pair" "both halves painted; compare fill asks=0 over 2 s; no gap" \
                "left stddev=$L1 right stddev=$R1; asks=$ASKS pool=$POOL" "$RIGHT_BEFORE" OK
        elif [ "$ASKS" = "-" ]; then
            row "paused, sbs, pair" "both halves painted; compare fill asks=0 over 2 s; no gap" \
                "left stddev=$L1 right stddev=$R1 (both painted); no compare fill line to read" \
                "$RIGHT_BEFORE" UNPROVEN
        else
            row "paused, sbs, pair" "both halves painted; compare fill asks=0 over 2 s; no gap" \
                "left stddev=$L1 right stddev=$R1; asks=$ASKS pool=$POOL" "$RIGHT_BEFORE" MISMATCH
        fi
    else
        row "paused, sbs, pair" "both halves painted; compare fill asks=0 over 2 s; no gap" \
            "left stddev=$L1 right stddev=$R1 (a half is flat)" "$RIGHT_BEFORE" MISMATCH
    fi
else
    row "paused, sbs, pair" "both halves painted; compare fill asks=0 over 2 s; no gap" \
        "no screenshot (live.sh shot failed)" "$ART" UNPROVEN
fi
printf '%s\n' "$WINDOW" > "$ART/row1-fill-window.txt"

# --- row 2: paused, sbs, no source ----------------------------------------------------------------------------------
# The channel cannot orphan the source deliberately, but it CAN ask for a moment the source decoder has never decoded:
# a seek to 95 % of the clip while the tail is held. The renderer answers the result from its block; the compare's
# source pays a keyframe rewind. Through that window the pane must keep the LAST matched pair on the right half
# instead of painting the new result alone - measurable: while the left half is flat (a gap), the right half's pixels
# must not change.

ctrl "drag:preview_timeline@$(strip_drag 0.25 0.95)" 20 > "$ART/row2-drag.json"
sleep 0.4
SEEN=0
LONE=0
WORST="$ART"
SIG_BEFORE=""
[ -n "${RIGHT_BEFORE:-}" ] && SIG_BEFORE="$(crop_sig "$RIGHT_BEFORE" "$PX" $((PX + HALF_W)) "$HALF_W" "$PH")"
for step in $(seq 1 12); do
    SHOT="$(shot "row2-$step")"
    [ -n "$SHOT" ] || continue
    SEEN=$((SEEN + 1))
    L="$(crop_stat "$SHOT" "$PX" "$PY" "$HALF_W" "$PH" standard_deviation)"
    SIG="$(crop_sig "$SHOT" "$PX" $((PX + HALF_W)) "$HALF_W" "$PH")"
    if [ -n "${L:-}" ] && python3 -c "import sys; sys.exit(0 if $L <= 0.02 else 1)"; then
        if [ -n "$SIG_BEFORE" ] && [ -n "$SIG" ] && [ "$SIG" != "$SIG_BEFORE" ]; then
            LONE=$((LONE + 1))
            WORST="$SHOT"
        fi
    fi
    sleep 0.1
done
if [ "$SEEN" -eq 0 ]; then
    row "paused, sbs, no source" "left half empty; the result is never painted alone" "no screenshot" "$ART" UNPROVEN
elif [ "$LONE" -eq 0 ]; then
    row "paused, sbs, no source" "left half empty; the result is never painted alone" \
        "$SEEN shots across the source gap; 0 with a changed right half over a flat left one" "$WORST" OK
else
    row "paused, sbs, no source" "left half empty; the result is never painted alone" \
        "$LONE of $SEEN shots painted a NEW result beside a flat left half" "$WORST" MISMATCH
fi

# --- row 3: playing, sbs, steady ------------------------------------------------------------------------------------

ctrl "click:preview_play" 10 > "$ART/row3-play.json"
sleep 1
MARK3=$(log_mark)
sleep 3
WINDOW3="$(log_since "$MARK3")"
read -r OFF3 TOTAL3 <<<"$(printf '%s\n' "$WINDOW3" | pair_off)"
PAIRS3="$(printf '%s\n' "$WINDOW3" | pair_lines | wc -l)"
if [ "$OFF3" = "-" ]; then
    row "playing, sbs, steady" "every painted pair carries the source; delta_frames=0" \
        "$PAIRS3 pair draws, no stats line (RVE_PREVIEW_LOG<1?)" "$ART/row3-window.txt" UNPROVEN
elif [ "$OFF3" = "0" ] && [ "${TOTAL3:-0}" -gt 0 ]; then
    row "playing, sbs, steady" "every painted pair carries the source; delta_frames=0" \
        "pair_off=0/$TOTAL3 over 3 s, $PAIRS3 pair draws" "$ART/row3-window.txt" OK
else
    row "playing, sbs, steady" "every painted pair carries the source; delta_frames=0" \
        "pair_off=$OFF3/$TOTAL3 over 3 s, $PAIRS3 pair draws" "$ART/row3-window.txt" MISMATCH
fi
printf '%s\n' "$WINDOW3" > "$ART/row3-window.txt"

# --- row 4: playing, sbs, strip drag (EXPECTED RED until plan P1) ----------------------------------------------------

sleep 0.2
read -r OFF4A TOTAL4A <<<"$(log_since "$MARK3" | pair_off)"
MARK4=$(log_mark)
ctrl "drag:preview_timeline@$(strip_drag 0.10 0.90)" 25 > "$ART/row4-drag.json"
sleep 2
WINDOW4="$(log_since "$MARK4")"
read -r OFF4B TOTAL4B <<<"$(printf '%s\n' "$WINDOW4" | pair_off)"
SHOT4="$(shot row4-after-drag)"
if ! [[ "${OFF4B:-}" =~ ^[0-9]+$ && "${OFF4A:-}" =~ ^[0-9]+$ ]]; then
    # No `stats` line in one of the two windows: the level is too low (or the render never drew), so the row is
    # UNPROVEN rather than a shell arithmetic error. A row that cannot be observed must say so (exit 2), never crash.
    row "playing, sbs, strip drag" "the last matched pair holds; the gap is bounded; no lone half" \
        "no stats line in a window (RVE_PREVIEW_LOG<2?)" "$ART/row4-window.txt" UNPROVEN
else
    GAIN=$((OFF4B - OFF4A))
    if [ "$GAIN" -le 0 ]; then
        row "playing, sbs, strip drag" "the last matched pair holds; the gap is bounded; no lone half" \
            "pair_off $OFF4A/$TOTAL4A -> $OFF4B/$TOTAL4B: $(printf '%s\n' "$WINDOW4" | pair_lines | grep -c -- " -$") lone-half draws" \
            "${SHOT4:-$ART/row4-window.txt}" OK
    else
        row "playing, sbs, strip drag" "the last matched pair holds; the gap is bounded; no lone half" \
            "pair_off $OFF4A/$TOTAL4A -> $OFF4B/$TOTAL4B (+$GAIN mismatched paints); lone-half draws: $(printf '%s\n' "$WINDOW4" | pair_lines | grep -c -- " -$")" \
            "${SHOT4:-$ART/row4-window.txt}" MISMATCH
    fi
fi
printf '%s\n' "$WINDOW4" > "$ART/row4-window.txt"

# --- row 5: seek on a paused pane -----------------------------------------------------------------------------------

ctrl "click:preview_play" 10 > "$ART/row5-pause.json"        # the toggle: playing -> paused
sleep 0.5
MARK5=$(log_mark)
ctrl "drag:preview_timeline@$(strip_drag 0.80 0.40)" 25 > "$ART/row5-seek.json"
sleep 2
WINDOW5="$(log_since "$MARK5")"
PAIRS5="$(printf '%s\n' "$WINDOW5" | pair_lines)"
SAME5="$(printf '%s\n' "$PAIRS5" | python3 -c '
import sys
same = total = 0
for line in sys.stdin:
    parts = line.split()
    if len(parts) != 2:
        continue
    result, source = parts
    if result == "-" or source == "-":
        continue
    total += 1
    if abs(float(result) - float(source)) <= 0.02:
        same += 1
print(same, total)
')"
read -r SAME5_N TOTAL5_N <<<"$SAME5"
SHOT5="$(shot row5-after-seek)"
if [ "${TOTAL5_N:-0}" -eq 0 ]; then
    row "seek on a paused pane" "a gap, then the new moment; left and right are the same moment" \
        "no pair draw named both halves in the window" "$ART/row5-window.txt" UNPROVEN
elif [ "$SAME5_N" -eq "$TOTAL5_N" ]; then
    row "seek on a paused pane" "a gap, then the new moment; left and right are the same moment" \
        "$SAME5_N/$TOTAL5_N pair draws have result_t == original_t" "${SHOT5:-$ART/row5-window.txt}" OK
else
    row "seek on a paused pane" "a gap, then the new moment; left and right are the same moment" \
        "$((TOTAL5_N - SAME5_N))/$TOTAL5_N pair draws name two moments" "${SHOT5:-$ART/row5-window.txt}" MISMATCH
fi
printf '%s\n' "$WINDOW5" > "$ART/row5-window.txt"

# --- row 8: refused press -------------------------------------------------------------------------------------------
# A press whose route refuses it must leave the result generation and the register untouched (audit §6.3 probe B).
# The channel offers no verb that forces such a refusal on a pane that owns a register (every live refusal needs a
# pane with no block/owner, i.e. no register to preserve). The offscreen row + audit probe B carry this one; the live
# script refuses to fake it.
row "refused press" "generation and register unchanged" \
    "not arrangeable through the control channel: the live refusals need a pane with no register" \
    "$ROOT/harness/tests/test_preview_state_matrix.py" UNPROVEN

# --- row 6: render end -> finished file -----------------------------------------------------------------------------

echo "live_matrix: waiting for the preview job to finish ($ART/preview.json)"
WAITED=0
while kill -0 "$PREVIEW_PID" 2>/dev/null && [ "$WAITED" -lt 600 ]; do sleep 1; WAITED=$((WAITED + 1)); done
kill -0 "$PREVIEW_PID" 2>/dev/null && echo "live_matrix: WARNING the preview job outlived the wait; row 6 reads what it finds"
MARK6=$(log_mark)
sleep 2
WINDOW6="$(log_since "$MARK6")"
GEN6="$(printf '%s\n' "$WINDOW6" | draw_generations | tr '\n' ' ' | sed 's/ *$//')"
PAIRS6="$(printf '%s\n' "$WINDOW6" | pair_lines)"
FIRST6="$(printf '%s\n' "$PAIRS6" | head -1)"
FINISHED_PLAYHEAD="$(json_get "$(ctrl state:previewLabel 8)" progress.2 "")"
SHOT6="$(shot row6-finished-file)"
if [ -z "$GEN6" ] || [ -z "$FIRST6" ]; then
    row "render end -> finished file" "no frame of the previous file; the first pair is whole" \
        "no sbs draw in the handover window" "${SHOT6:-$ART}" UNPROVEN
else
    read -r R6 S6 <<<"$FIRST6"
    if [ "$S6" = "-" ]; then
        row "render end -> finished file" "no frame of the previous file; the first pair is whole" \
            "first handover draw at $R6 has NO source half (generations: $GEN6)" "${SHOT6:-$ART}" MISMATCH
    elif python3 -c "import sys; sys.exit(0 if abs(float('$R6') - float('$S6')) <= 0.02 else 1)"; then
        # The finished file stands where the render stood; a pair from the PREVIOUS file would name another moment.
        if [ -n "$FINISHED_PLAYHEAD" ] && python3 -c "import sys; sys.exit(0 if abs(float('$R6') - float('$FINISHED_PLAYHEAD')) <= 0.05 else 1)"; then
            row "render end -> finished file" "no frame of the previous file; the first pair is whole" \
                "first pair $R6/$S6, the finished file's playhead $FINISHED_PLAYHEAD; generations: $GEN6" \
                "${SHOT6:-$ART}" OK
        else
            row "render end -> finished file" "no frame of the previous file; the first pair is whole" \
                "first pair $R6/$S6 names a moment the finished file does not stand on ($FINISHED_PLAYHEAD)" \
                "${SHOT6:-$ART}" MISMATCH
        fi
    else
        row "render end -> finished file" "no frame of the previous file; the first pair is whole" \
            "first handover pair mixes moments $R6/$S6, generations: $GEN6" "${SHOT6:-$ART}" MISMATCH
    fi
fi
printf '%s\n' "$WINDOW6" > "$ART/row6-window.txt"

# --- row 7: cleared pane --------------------------------------------------------------------------------------------
# Entering the Render Queue tab with no remembered item clears the pane (`_enter_queue_pane` -> `_clear_pane`). The
# empty field must stay empty: 2 s and 10 s of shots, both flat.

ctrl "tab:processSettingsContainer@Render Queue" 10 > "$ART/row7-tab.json"
sleep 0.5
SHOT7A="$(shot row7-at-0.5s)"
sleep 1.5
SHOT7B="$(shot row7-at-2s)"
sleep 8
SHOT7C="$(shot row7-at-10s)"
CLEAN=0
TOTAL7=0
for pair in "0.5s:$SHOT7A" "2s:$SHOT7B" "10s:$SHOT7C"; do
    label="${pair%%:*}"; file="${pair#*:}"
    [ -n "$file" ] || continue
    TOTAL7=$((TOTAL7 + 1))
    L="$(crop_stat "$file" "$PX" "$PY" "$HALF_W" "$PH" standard_deviation)"
    R="$(crop_stat "$file" "$PX" $((PX + HALF_W)) "$HALF_W" "$PH" standard_deviation)"
    if python3 -c "import sys; sys.exit(0 if float('${L:-0}') <= 0.02 and float('${R:-0}') <= 0.02 else 1)"; then
        CLEAN=$((CLEAN + 1))
    else
        echo "live_matrix: cleared pane not empty at $label (left ${L:-?} right ${R:-?})"
    fi
done
if [ "$TOTAL7" -eq 0 ]; then
    row "cleared pane" "empty at 2 s and at 10 s" "no screenshot" "$ART" UNPROVEN
elif [ "$CLEAN" -eq "$TOTAL7" ]; then
    row "cleared pane" "empty at 2 s and at 10 s" "$CLEAN/$TOTAL7 shots flat on both halves (0.5/2/10 s)" "$SHOT7C" OK
else
    row "cleared pane" "empty at 2 s and at 10 s" "$((TOTAL7 - CLEAN))/$TOTAL7 shots still show a picture" "$SHOT7C" MISMATCH
fi
ctrl "tab:processSettingsContainer@General" 10 > "$ART/row7-tab-back.json"

# =====================================================================================================================
# Stage B — an export, sbs. Row 9: both halves from the block while the render is alive; held_block only while alive.
# =====================================================================================================================

ctrl "queue:$CLIP" 30 > "$ART/row9-queue.json"
ctrl "click:preview_mode_sbs" 10 > "$ART/row9-mode.json"
MARK9=$(log_mark)
ctrl "start:render" 30 > "$ART/row9-start.json"
# The block's own picture arrives on the render's first composited frame, so wait for it rather than guessing a delay.
LIVE_WAIT=0
WINDOW9=""
HELD_BLOCK_LIVE="-"
while [ "$LIVE_WAIT" -lt 45 ]; do
    WINDOW9="$(log_since "$MARK9")"
    HELD_BLOCK_LIVE="$(printf '%s\n' "$WINDOW9" | fill_max held_block)"
    if [ "$HELD_BLOCK_LIVE" != "-" ] && [ "${HELD_BLOCK_LIVE:-0}" -gt 0 ]; then break; fi
    sleep 1; LIVE_WAIT=$((LIVE_WAIT + 1))
done
HELD_DECODER_LIVE="$(printf '%s\n' "$WINDOW9" | fill_max held_decoder)"
PAIRS9="$(printf '%s\n' "$WINDOW9" | pair_lines | wc -l)"
SHOT9="$(shot row9-export-sbs)"
# Wait for the job to end through the transport glyph (`checked` is the running state), then read the pool again: the
# block's pictures belong to the render pass, so `_close_display_state` must have dropped them.
STARTED=0
WAITED=0
while [ "$WAITED" -lt 300 ]; do
    CHECKED="$(json_get "$(ctrl state:startRenderButton 8)" checked "")"
    if [ "$CHECKED" = "True" ]; then STARTED=1; fi
    if [ "$STARTED" = "0" ] && [ "$WAITED" -ge 20 ]; then break; fi      # it never started: stop guessing
    if [ "$STARTED" = "1" ] && [ "$CHECKED" = "False" ]; then break; fi
    sleep 2; WAITED=$((WAITED + 2))
done
sleep 2
MARK9B=$(log_mark)
sleep 3
WINDOW9B="$(log_since "$MARK9B")"
HELD_BLOCK_DEAD="$(printf '%s\n' "$WINDOW9B" | fill_field held_block)"
if [ "$STARTED" = "0" ]; then
    row "export, sbs" "both halves from one owner; held_block only while the render is alive" \
        "the export never started (startRenderButton stayed unchecked); queue: $(json_get "$(cat "$ART/row9-queue.json" 2>/dev/null)" selectedRows "-")" \
        "${SHOT9:-$ART/row9-window.txt}" UNPROVEN
elif [ "$HELD_BLOCK_LIVE" = "-" ] || [ "$HELD_BLOCK_DEAD" = "-" ]; then
    row "export, sbs" "both halves from one owner; held_block only while the render is alive" \
        "no compare fill line (RVE_PREVIEW_LOG<1?); pair draws: $PAIRS9" "${SHOT9:-$ART/row9-window.txt}" UNPROVEN
elif [ "${HELD_BLOCK_LIVE:-0}" -gt 0 ] && [ "${HELD_BLOCK_DEAD:-1}" -eq 0 ]; then
    row "export, sbs" "both halves from one owner; held_block only while the render is alive" \
        "held_block $HELD_BLOCK_LIVE (decoder $HELD_DECODER_LIVE) while alive, $HELD_BLOCK_DEAD after; $PAIRS9 pair draws" \
        "${SHOT9:-$ART/row9-window.txt}" OK
else
    row "export, sbs" "both halves from one owner; held_block only while the render is alive" \
        "held_block $HELD_BLOCK_LIVE while alive, $HELD_BLOCK_DEAD after; $PAIRS9 pair draws" \
        "${SHOT9:-$ART/row9-window.txt}" MISMATCH
fi
printf '%s\n' "$WINDOW9" > "$ART/row9-window.txt"

# --- the verdict ----------------------------------------------------------------------------------------------------

echo
echo "live_matrix: $MISMATCHES mismatch(es), $UNPROVEN unproven row(s); artifacts in $ART"
echo "live_matrix: the 'playing, sbs, strip drag' row was the gate for plan P1 (pair writer, 79ea250c)."
echo "             Since then a MISMATCH there is a regression, not the expected state."
if [ "$MISMATCHES" -gt 0 ]; then exit 1; fi
if [ "$UNPROVEN" -gt 0 ]; then exit 2; fi
exit 0
