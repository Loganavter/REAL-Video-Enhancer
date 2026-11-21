"""Queue and finish ONE render through the GUI: the W6 gate nothing has run yet.

It drives the live app's own controls (the capture server's `render:` request): set the input field, press
"Add to Render Queue", press "Start", wait for the app to re-enable Start. No pipeline call is made directly, so a page
that cannot queue a job - the failure mode a UI migration actually risks - is caught.

    python3 harness/scripts/ui_live.py boot --display :99
    python3 harness/scripts/ui_render_gate.py /tmp/opencode/tiny.mp4 [--display :99]

Exit codes: 0 render finished and the output file exists, 1 the app reported a failure, 2 no live server.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

BASE = "/tmp/opencode/live"


def _paths(display: str) -> dict[str, str]:
    state = os.path.join(BASE, display.replace(":", "d"))
    return {"ready": os.path.join(state, "server", "ready"), "request": os.path.join(state, "server", "request"),
            "done": os.path.join(state, "server", "done"), "log": os.path.join(state, "app.log")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--display", default=":99")
    ap.add_argument("--wait", type=float, default=240.0)
    args = ap.parse_args()
    paths = _paths(args.display)
    if not os.path.exists(paths["ready"]):
        print("no live server - run `python3 harness/scripts/ui_live.py boot` first", file=sys.stderr)
        return 2
    try:
        os.unlink(paths["done"])
    except FileNotFoundError:
        pass
    with open(paths["request"] + ".tmp", "w", encoding="utf-8") as handle:
        handle.write(f"render:{os.path.abspath(args.video)}")
    os.replace(paths["request"] + ".tmp", paths["request"])
    import time

    deadline = time.time() + args.wait
    while time.time() < deadline:
        if os.path.exists(paths["done"]):
            break
        time.sleep(0.3)
    else:
        print(f"no answer within {args.wait:.0f} s - see {paths['log']}", file=sys.stderr)
        return 1
    result = json.load(open(paths["done"], encoding="utf-8"))
    print(json.dumps(result, indent=1))
    if not result.get("finished"):
        print(f"render did not finish: {result.get('error')}", file=sys.stderr)
        return 1
    if not result.get("output") or not os.path.exists(result["output"]):
        print(f"finished but no output file: {result.get('output')!r}", file=sys.stderr)
        return 1
    print(f"render gate OK: {result['output']} ({result.get('output_bytes', 0)} bytes, {result['seconds']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
