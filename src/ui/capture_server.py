"""A long-lived capture server: boot the app once, then answer snapshot requests from disk.

Why: every measurement used to cost a full app start plus the backend settle (30-40 s) because the harness could only
talk to a process it had just launched. Iterating on one page meant paying that price per iteration, which is most of
the wall time of a migration session.

With ``RVE_UI_CAPTURE_SERVER=<dir>`` the app boots once, warms every page, writes ``<dir>/ready`` and then polls
``<dir>/request``. A request names the pages to capture (or ``all``); the server shows each, saves ``<page>.png`` and
``<page>.json`` into ``<dir>/<stamp>/`` and writes ``<dir>/done`` with that stamp. A single page then costs about a
second instead of a minute, and the process stays up for the next request.

Protocol (files, so any tool can drive it - see ``harness/scripts/ui_live.py``):

    request   "all" or "nav3-tab0,nav4"     written by the driver
    ready     ""                             written once, after the warm-up walk
    done      "<stamp>"                      written per request; the driver removes `request` first
    quit      ""                             asks the app to exit

Why not a socket: the app may block its event loop in a modal dialog or a long paint; a file poll keeps the protocol
trivial and crash-safe (a half-written request is simply not a valid list of pages).
"""

from __future__ import annotations

import json
import os
import time


def enabled() -> bool:
    return bool(os.environ.get("RVE_UI_CAPTURE_SERVER"))


def _stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"


def install(window) -> None:
    """Arm the server on ``window``. Never returns anything; the app keeps running."""
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from src.ui import layout_dump, page_capture

    root = os.environ["RVE_UI_CAPTURE_SERVER"]
    os.makedirs(root, exist_ok=True)
    delay = int(os.environ.get("RVE_UI_CAPTURE_DELAY_MS", "40000"))
    step = int(os.environ.get("RVE_UI_CAPTURE_STEP_MS", "120"))
    steps = page_capture.steps(window)
    if not steps:
        print("capture-server: the window has no pages to walk", flush=True)
        return
    by_name = {name: action for name, action in steps}
    ready_path = os.path.join(root, "ready")
    request_path = os.path.join(root, "request")
    done_path = os.path.join(root, "done")
    for path in (ready_path, done_path, request_path):
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

    state = {"warming": True, "index": 0}

    def warm(index: int) -> None:
        if index >= len(steps):
            state["warming"] = False
            with open(ready_path, "w", encoding="utf-8") as handle:
                handle.write(_stamp())
            print(f"capture-server: ready ({len(steps)} pages) in {root}", flush=True)
            QTimer.singleShot(0, poll)
            return
        _name, action = steps[index]
        action()
        QApplication.processEvents()
        QTimer.singleShot(step, lambda: warm(index + 1))

    def poll() -> None:
        if os.path.exists(os.path.join(root, "quit")):
            print("capture-server: quit requested", flush=True)
            QApplication.quit()
            return
        if state["warming"]:
            QTimer.singleShot(120, poll)
            return
        if os.path.exists(request_path):
            try:
                with open(request_path, encoding="utf-8") as handle:
                    wanted = handle.read().strip()
            except OSError:
                wanted = ""
            os.unlink(request_path)
            capture(wanted)
        QTimer.singleShot(120, poll)

    def capture(wanted: str) -> None:
        names = list(by_name) if wanted in {"", "all"} else [part.strip() for part in wanted.split(",") if part.strip()]
        unknown = [name for name in names if name not in by_name]
        for name in unknown:
            print(f"capture-server: no page named {name!r}", flush=True)
        names = [name for name in names if name in by_name]
        stamp = _stamp()
        out_dir = os.path.join(root, stamp)
        os.makedirs(out_dir, exist_ok=True)
        for name in names:
            by_name[name]()
            QApplication.processEvents()
            page_capture.stabilise(window)
            pixmap = window.grab()
            pixmap.save(os.path.join(out_dir, f"{name}.png"))
            layout_dump.write_page(window, name, out_dir=out_dir)
            print(f"capture-server: {name} -> {out_dir}", flush=True)
        tmp = done_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump({"stamp": stamp, "pages": names}, handle)
        os.replace(tmp, done_path)

    QTimer.singleShot(delay, lambda: warm(0))
