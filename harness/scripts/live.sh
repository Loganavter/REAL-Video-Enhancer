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
STATE="${RVE_LIVE_STATE:-/tmp/opencode/live}"
# The app derives its drop box from the directory of argv[0], so a boot from harness/scripts would put it under
# harness/scripts and every waiter would watch the wrong path. State it explicitly and wait on that exact path.
CONTROL="${RVE_UI_CAPTURE_SERVER:-${RVE_CONTROL_DIR:-$STATE/control}}"
DISPLAY_NUMBER="${RVE_LIVE_DISPLAY:-90}"

mkdir -p "$STATE"

die() { echo "live: $*" >&2; exit 1; }

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
    local seconds="${1:-40}" waited=0
    while [ "$waited" -lt "$seconds" ]; do
        [ -f "$CONTROL/ready" ] && return 0
        sleep 0.5
        waited=$((waited / 2 + 1))
        [ -f "$STATE/app.pid" ] && ! kill -0 "$(cat "$STATE/app.pid")" 2>/dev/null && {
            echo "live: the app exited while waiting; last log lines:" >&2
            tail -20 "$STATE/app.log" >&2
            return 1
        }
    done
    return 1
}

cmd_up() {
    local display_number="${1:-}" started_xvfb=0
    if [ -z "$display_number" ]; then
        display_number="$(pick_display)"
    fi
    local display=":$display_number"
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
    [ -f "$STATE/app.pid" ] && kill -0 "$(cat "$STATE/app.pid")" 2>/dev/null && die "an app is already running (pid $(cat "$STATE/app.pid")) — live.sh down first"
    rm -f "$CONTROL/ready"
    echo "$CONTROL" > "$STATE/control_dir"
    echo "live: booting the app on $display (control $CONTROL)"
    # shellcheck disable=SC2046
    env $(app_env "$display") RVE_UI_CAPTURE_SERVER="$CONTROL" PYTHONPATH="$ROOT:$ROOT/src:$TOOLKIT" RVE_TOOLKIT_PATH="$TOOLKIT" \
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
    rm -f "$CONTROL/ready"
}

cmd_status() {
    echo "display: $(cat "$STATE/display" 2>/dev/null || echo none)"
    echo "control: $CONTROL"
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
