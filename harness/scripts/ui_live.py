"""Drive the app's capture server: boot it once, then take snapshots in seconds.

The server lives in `src/ui/capture_server.py` (env `RVE_UI_CAPTURE_SERVER=<dir>`); this script is the supervisor side
of the file protocol, and it owns the private display and the process lifetime so a session only pays the 40 s settle
once.

    python3 harness/scripts/ui_live.py boot [--display :99] [--wait 150]
    python3 harness/scripts/ui_live.py shot nav3-tab0,nav4        # -> prints the folder
    python3 harness/scripts/ui_live.py shot all
    python3 harness/scripts/ui_live.py stop
    python3 harness/scripts/ui_live.py status

State lives in `/tmp/opencode/live/<display>` (server dir, pid, Xvfb pid). Everything is plain files and subprocesses,
so a crash leaves a readable trace instead of a hung socket.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join("/tmp/opencode", "live")


def _state_dir(display: str) -> str:
    return os.path.join(BASE, display.replace(":", "d"))


def _paths(display: str) -> dict[str, str]:
    state = _state_dir(display)
    server = os.path.join(state, "server")
    return {
        "state": state,
        "server": server,
        "ready": os.path.join(server, "ready"),
        "request": os.path.join(server, "request"),
        "done": os.path.join(server, "done"),
        "quit": os.path.join(server, "quit"),
        "app_pid": os.path.join(state, "app.pid"),
        "xvfb_pid": os.path.join(state, "xvfb.pid"),
        "log": os.path.join(state, "app.log"),
    }


def _wait_for(path: str, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path):
            return True
        time.sleep(0.2)
    return False


def boot(display: str, wait: float, extra_env: dict[str, str]) -> int:
    paths = _paths(display)
    os.makedirs(paths["server"], exist_ok=True)
    for stale in (paths["ready"], paths["done"], paths["request"], paths["quit"]):
        try:
            os.unlink(stale)
        except FileNotFoundError:
            pass
    if os.path.exists(paths["app_pid"]):
        try:
            os.kill(int(open(paths["app_pid"]).read().strip()), 0)
            print(f"an app is already running for {display}; run `stop` first", file=sys.stderr)
            return 1
        except (OSError, ValueError):
            pass

    xvfb = subprocess.Popen(["Xvfb", display, "-screen", "0", "1600x1000x24"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(paths["xvfb_pid"], "w") as handle:
        handle.write(str(xvfb.pid))
    time.sleep(1.5)

    env = dict(os.environ)
    env.update({
        "DISPLAY": display,
        "QT_QPA_PLATFORM": "xcb",
        "PYTHONDONTWRITEBYTECODE": "1",
        "RVE_UI_CAPTURE_SERVER": paths["server"],
        "RVE_UI_STABLE_TEXT": env.get("RVE_UI_STABLE_TEXT", "systemInfoText"),
    })
    env.pop("RVE_UI_CAPTURE_DIR", None)
    env.pop("RVE_UI_LAYOUT_DUMP", None)
    env.update(extra_env)
    log = open(paths["log"], "w")
    app = subprocess.Popen([os.path.join(ROOT, "run.sh")], cwd=ROOT, env=env,
                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    with open(paths["app_pid"], "w") as handle:
        handle.write(str(app.pid))
    print(f"booting on {display} (pid {app.pid}); log {paths['log']}")
    if not _wait_for(paths["ready"], wait):
        print(f"the server did not become ready within {wait:.0f} s - see {paths['log']}", file=sys.stderr)
        return 2
    print(f"ready: {paths['server']}")
    return 0


def shot(display: str, pages: str, wait: float) -> int:
    paths = _paths(display)
    if not os.path.exists(paths["ready"]):
        print("no live server - run `boot` first", file=sys.stderr)
        return 2
    try:
        os.unlink(paths["done"])
    except FileNotFoundError:
        pass
    with open(paths["request"] + ".tmp", "w", encoding="utf-8") as handle:
        handle.write(pages)
    os.replace(paths["request"] + ".tmp", paths["request"])
    if not _wait_for(paths["done"], wait):
        print(f"no snapshot within {wait:.0f} s - see {paths['log']}", file=sys.stderr)
        return 2
    info = json.load(open(paths["done"], encoding="utf-8"))
    folder = os.path.join(paths["server"], info["stamp"])
    print(folder)
    print(f"{len(info['pages'])} page(s): {', '.join(info['pages'])}", file=sys.stderr)
    return 0


def stop(display: str, wait: float = 15.0) -> int:
    paths = _paths(display)
    if os.path.exists(paths["ready"]):
        open(paths["quit"], "w").close()
        _wait_for(paths["quit"] + ".gone", 0.1)   # the file may stay; the process exit is what matters
    for key in ("app_pid", "xvfb_pid"):
        try:
            pid = int(open(paths[key]).read().strip())
        except (OSError, ValueError):
            continue
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.kill(pid, sig)
                time.sleep(0.4)
            except ProcessLookupError:
                break
        try:
            os.unlink(paths[key])
        except FileNotFoundError:
            pass
    for stale in ("ready", "done", "request", "quit"):
        try:
            os.unlink(os.path.join(paths["server"], stale))
        except FileNotFoundError:
            pass
    print("stopped")
    return 0


def status(display: str) -> int:
    paths = _paths(display)
    alive = False
    try:
        pid = int(open(paths["app_pid"]).read().strip())
        os.kill(pid, 0)
        alive = True
    except (OSError, ValueError):
        pass
    print(json.dumps({"display": display, "app_alive": alive, "ready": os.path.exists(paths["ready"]),
                      "server": paths["server"], "log": paths["log"]}, indent=1))
    return 0 if alive else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["boot", "shot", "stop", "status", "clean"])
    ap.add_argument("pages", nargs="?", default="all")
    ap.add_argument("--display", default=":99")
    ap.add_argument("--wait", type=float, default=150.0)
    ap.add_argument("--env", action="append", default=[], metavar="K=V")
    args = ap.parse_args()
    extra = dict(item.split("=", 1) for item in args.env)
    if args.command == "boot":
        return boot(args.display, args.wait, extra)
    if args.command == "shot":
        return shot(args.display, args.pages, args.wait)
    if args.command == "stop":
        return stop(args.display)
    if args.command == "status":
        return status(args.display)
    shutil.rmtree(_state_dir(args.display), ignore_errors=True)
    print("cleaned")
    return 0


if __name__ == "__main__":
    sys.exit(main())
