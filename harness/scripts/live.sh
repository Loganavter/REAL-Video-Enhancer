#!/usr/bin/env bash
# live.sh — a private, self-serve live instance of RVE (the RVE analogue of BlinkTable's MCP control surface).
#
# Why this exists: the offscreen platform has no QRhi and re-renders on grab(), so it cannot answer "does this paint",
# "does this composite", "is the frame stale" — and the owner is not always at the machine to be the instrument. This
# script boots the real app on a PRIVATE X display nobody else uses, exposes the app's own control channel, and takes
# screen-truth screenshots, so a supervisor can test without a human.
#
#   live.sh up [display]        boot Xvfb (if that display is not already up) + the app; wait for the control channel
#   live.sh down                stop the app and the Xvfb this script started (kill by PID, never by pattern)
#   live.sh status              what is running, the control dir, the last log lines
#   live.sh shot [path]         root-window screenshot (the real framebuffer, not a QWidget::grab)
#   live.sh ask "<verb>" [sec]  one control-channel request (where:, rhi:, grab:, probe_at:, click:, queue:, ...)
#   live.sh click X Y           a real click through X (xdotool), i.e. with hit-testing
#   live.sh move X Y            move the pointer
#   live.sh key <key>           press a key (xdotool key), e.g. `key Return`, `key ctrl+s`
#   live.sh type "<text>"       type text
#
# Traps this script encodes (each cost a session once):
#   * never the owner's display (:0/:1 and their Wayland socket): the display is leased from :90 up, and a lease that is
#     already in use is skipped;
#   * kill by PID from the pid files — `pkill -f live_boot.py` matches the shell that runs it and SIGTERMs itself;
#   * `xcb` needs a real X server and software GL: `LIBGL_ALWAYS_SOFTWARE=1` is what lets QRhi/OpenGL initialise on
#     llvmpipe when there is no GPU-backed GL;
#   * the GTK appmenu module aborts the process under xcb unless the platform theme and GTK modules are emptied;
#   * `window.grab()` (the control channel's `grab:` verb) RE-RENDERS and hides stale-frame bugs; `import -window root`
#     is the screen truth. Use both.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"            # the RVE checkout
TOOLKIT="${RVE_TOOLKIT_PATH:-$(dirname "$ROOT")/sli-ui-toolkit/src}"
VENV="${RVE_VENV:-/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2/venv}"
BASE_STATE="${RVE_LIVE_STATE:-/tmp/opencode/live}"
STATE="$BASE_STATE"
# The app derives its drop box from the directory of argv[0], so a boot from harness/scripts would put it under
# harness/scripts and every waiter would watch the wrong path. State it explicitly and wait on that exact path.
CONTROL="${RVE_UI_CAPTURE_SERVER:-${RVE_CONTROL_DIR:-$STATE/control}}"
DISPLAY_NUMBER="${RVE_LIVE_DISPLAY:-90}"
# One display per lane: a lease from the machine tool when it is installed (`live-lease alloc`), the free-:90 picker
# below when it is not. A lease owns a per-instance runtime dir, so `up` moves the state dir there and leaves a pointer
# in BASE_STATE - every later command (ask/shot/down/status) must find the instance it talks to.
LANE="${RVE_LIVE_LANE:-rve}"
TASK="${RVE_LIVE_TASK:-live}"

mkdir -p "$STATE"

die() { echo "live: $*" >&2; exit 1; }

use_state() {                       # one state dir per instance: pid files, log, control dir, the lease id
    STATE="$1"
    mkdir -p "$STATE"
    if [ -z "${RVE_UI_CAPTURE_SERVER:-}${RVE_CONTROL_DIR:-}" ]; then
        CONTROL="$STATE/control"
    fi
    echo "$STATE" > "$BASE_STATE/state_dir"
}

if [ -f "$BASE_STATE/state_dir" ] && [ "${1:-}" != "up" ]; then
    use_state "$(cat "$BASE_STATE/state_dir")"
fi

pick_display() {
    local n="$DISPLAY_NUMBER"
    while [ -e "/tmp/.X11-unix/X$n" ] || [ -e "/tmp/.X$n-lock" ]; do
        n=$((n + 1))
        [ "$n" -gt 120 ] && die "no free display between :$DISPLAY_NUMBER and :120"
    done
    echo "$n"
}

app_env() {
    local display="$1"
    echo "DISPLAY=$display QT_QPA_PLATFORM=xcb QT_QPA_PLATFORMTHEME= GTK_MODULES= XDG_CURRENT_DESKTOP= LIBGL_ALWAYS_SOFTWARE=1"
}

wait_ready() {
    # `seconds` is seconds, and the tick is half of one: the counter is compared against twice the budget, not against
    # the budget itself (the old body counted half-seconds against a seconds bound, so a 40 s wait was really 20 s).
    local seconds="${1:-40}" waited=0
    # Second statement on purpose: `local seconds=... budget=$((seconds * 2))` expands the arithmetic BEFORE the
    # assignment runs, so under `set -u` it died with "seconds: unbound variable" on the first boot after it was
    # introduced (measured 2026-11-27, the live run that was meant to prove something else).
    local budget=$(( seconds * 2 ))
    while [ "$waited" -lt "$budget" ]; do
        [ -f "$CONTROL/ready" ] && return 0
        sleep 0.5
        # One tick is half a second, so the counter advances by ONE: `waited / 2 + 1` never leaves 1 (0 -> 1 -> 1 ->
        # ...), which made this loop a hang instead of a timeout for every caller that waited on a boot that never
        # came (control audit, 2026-11-27).
        waited=$((waited + 1))
        [ -f "$STATE/app.pid" ] && ! kill -0 "$(cat "$STATE/app.pid")" 2>/dev/null && {
            echo "live: the app exited while waiting; last log lines:" >&2
            tail -20 "$STATE/app.log" >&2
            return 1
        }
    done
    return 1
}

cmd_up() {
    local display_number="${1:-}" started_xvfb=0 display="" lease=""
    local prev; prev="$(cat "$BASE_STATE/state_dir" 2>/dev/null || true)"
    if [ -n "$prev" ] && [ -f "$prev/app.pid" ] && kill -0 "$(cat "$prev/app.pid")" 2>/dev/null; then
        die "an app is already running (pid $(cat "$prev/app.pid"), state $prev) — live.sh down first"
    fi
    # A lease coordinates with the other lanes on this machine (and owns its Xvfb); the picker below is the fallback
    # for a host without the tool. Never :0 — the owner's screen is not ours.
    if [ -z "$display_number" ] && command -v live-lease >/dev/null 2>&1; then
        lease="$(live-lease alloc --lane "$LANE" --task "$TASK" 2>/dev/null | tail -1)"
        if [ -n "$lease" ] && eval "$(live-lease env "$lease" 2>/dev/null)"; then
            display="${DISPLAY:-}"
            if [ -n "$display" ]; then
                echo "$lease" > "$STATE/lease"
                # the lease's runtime dir is per-instance: two lanes must not share pid files or a control dir
                [ -n "${RVE_LIVE_STATE:-}" ] || use_state "${XDG_RUNTIME_DIR:-$STATE}/rve-live"
                echo "$lease" > "$STATE/lease"
            fi
        fi
        if [ -z "$display" ] && [ -n "$lease" ]; then
            live-lease release "$lease" >/dev/null 2>&1 && lease=""
        fi
    fi
    if [ -z "$display" ]; then
        if [ -z "$display_number" ]; then
            display_number="$(pick_display)"
        fi
        display=":$display_number"
        if ! xdpyinfo -display "$display" >/dev/null 2>&1; then
            echo "live: starting Xvfb $display"
            Xvfb "$display" -screen 0 1400x1000x24 -nolisten tcp > "$STATE/xvfb.log" 2>&1 &
            echo $! > "$STATE/xvfb.pid"
            started_xvfb=1
            for _ in $(seq 1 20); do
                xdpyinfo -display "$display" >/dev/null 2>&1 && break
                sleep 0.25
            done
            xdpyinfo -display "$display" >/dev/null 2>&1 || { tail -5 "$STATE/xvfb.log" >&2; die "Xvfb $display did not come up"; }
        fi
    else
        for _ in $(seq 1 40); do
            xdpyinfo -display "$display" >/dev/null 2>&1 && break
            sleep 0.25
        done
        xdpyinfo -display "$display" >/dev/null 2>&1 || die "the leased display $display did not come up (lease ${lease:-?})"
    fi
    [ -f "$STATE/app.pid" ] && kill -0 "$(cat "$STATE/app.pid")" 2>/dev/null && die "an app is already running (pid $(cat "$STATE/app.pid")) — live.sh down first"
    rm -f "$CONTROL/ready"
    echo "$CONTROL" > "$STATE/control_dir"
    echo "live: booting the app on $display (control $CONTROL)"
    # shellcheck disable=SC2046
    # The walk is scheduled at RVE_UI_CAPTURE_DELAY_MS (40 s by default, so a person's capture run settles first) and
    # `ready` is written AFTER it. A harness boot must not wait 40 s for its own channel, and once `wait_ready` had a
    # real timeout the two budgets raced: the wait expired exactly when the walk began (measured 2026-11-27). Short
    # delay here, unchanged budget in `wait_ready`.
    env $(app_env "$display") RVE_UI_CAPTURE_SERVER="$CONTROL" RVE_UI_CAPTURE_DELAY_MS="${RVE_UI_CAPTURE_DELAY_MS:-500}" \
        PYTHONPATH="$ROOT:$ROOT/src:$TOOLKIT" RVE_TOOLKIT_PATH="$TOOLKIT" \
        PYTHONDONTWRITEBYTECODE=1 "$VENV/bin/python" -u "$ROOT/harness/scripts/live_boot.py" \
        > "$STATE/app.log" 2>&1 &
    echo $! > "$STATE/app.pid"
    echo "$display" > "$STATE/display"
    echo "$started_xvfb" > "$STATE/xvfb_owned"
    wait_ready "${2:-40}" || die "the control channel did not become ready; see $STATE/app.log"
    echo "live: ready on $display, control channel $CONTROL"
}

cmd_down() {
    if [ -f "$STATE/app.pid" ]; then
        local pid; pid="$(cat "$STATE/app.pid")"
        kill "$pid" 2>/dev/null && echo "live: stopped the app (pid $pid)"
        rm -f "$STATE/app.pid"
    fi
    if [ "$(cat "$STATE/xvfb_owned" 2>/dev/null || echo 0)" = "1" ] && [ -f "$STATE/xvfb.pid" ]; then
        kill "$(cat "$STATE/xvfb.pid")" 2>/dev/null && echo "live: stopped Xvfb (pid $(cat "$STATE/xvfb.pid"))"
        rm -f "$STATE/xvfb.pid"
    fi
    if [ -f "$STATE/lease" ]; then
        local lease; lease="$(cat "$STATE/lease")"
        live-lease release "$lease" >/dev/null 2>&1 && echo "live: released lease $lease"
        rm -f "$STATE/lease"
    fi
    rm -f "$CONTROL/ready"
    rm -f "$BASE_STATE/state_dir"
}

cmd_status() {
    echo "display: $(cat "$STATE/display" 2>/dev/null || echo none)"
    echo "control: $CONTROL"
    if [ -f "$STATE/lease" ]; then
        echo "lease:   $(cat "$STATE/lease") ($(live-lease status "$(cat "$STATE/lease")" 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin).get("state","?"))' 2>/dev/null || echo '?'))"
    fi
    if [ -f "$STATE/app.pid" ] && kill -0 "$(cat "$STATE/app.pid")" 2>/dev/null; then
        echo "app: running (pid $(cat "$STATE/app.pid"))"
    else
        echo "app: not running"
    fi
    [ -f "$STATE/app.log" ] && { echo "log tail:"; tail -8 "$STATE/app.log"; }
}

cmd_shot() {
    local path="${1:-$STATE/shot-$(date +%Y%m%d-%H%M%S).png}"
    local display; display="$(cat "$STATE/display" 2>/dev/null)" || die "no live instance — live.sh up"
    DISPLAY="$display" import -window root "$path" || die "screen capture failed (ImageMagick 'import' missing?)"
    echo "$path"
}

cmd_ask() {
    local verb="${1:?usage: live.sh ask \"<verb>\" [seconds]}"
    local seconds="${2:-15}"
    python3 - "$verb" "$seconds" "$CONTROL" <<'PY'
import json, os, sys, time, uuid
verb, seconds, root = sys.argv[1], float(sys.argv[2]), sys.argv[3]
request_id = uuid.uuid4().hex[:10]
answers = os.path.join(root, "answers")
os.makedirs(answers, exist_ok=True)
target = os.path.join(answers, f"{request_id}.json")
with open(os.path.join(root, "request"), "w", encoding="utf-8") as handle:
    handle.write(f"{verb}#{request_id}")
deadline = time.time() + seconds
while time.time() < deadline:
    if os.path.exists(target):
        try:
            with open(target, encoding="utf-8") as handle:
                print(json.dumps(json.load(handle), ensure_ascii=False, indent=1))
            sys.exit(0)
        except ValueError:
            pass
    time.sleep(0.05)
print(json.dumps({"error": f"no answer within {seconds:.0f}s", "verb": verb}, ensure_ascii=False))
sys.exit(1)
PY
}

cmd_xdo() {
    local display; display="$(cat "$STATE/display" 2>/dev/null)" || die "no live instance — live.sh up"
    DISPLAY="$display" xdotool "$@" || die "xdotool $* failed"
}

case "${1:-}" in
    up)     shift; cmd_up "$@" ;;
    down)   cmd_down ;;
    status) cmd_status ;;
    shot)   shift; cmd_shot "$@" ;;
    ask)    shift; cmd_ask "$@" ;;
    click)  shift; cmd_xdo mousemove "$1" "$2" click 1 ;;
    move)   shift; cmd_xdo mousemove "$1" "$2" ;;
    key)    shift; cmd_xdo key "$@" ;;
    type)   shift; cmd_xdo type "$@" ;;
    *)      sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
