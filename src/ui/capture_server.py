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
            with open(ready_path, "w", encoding="utf-8") as ready_file:
                ready_file.write(_stamp())
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
                with open(request_path, encoding="utf-8") as request_file:
                    wanted = request_file.read().strip()
            except OSError:
                wanted = ""
            os.unlink(request_path)
            handle(wanted)
        QTimer.singleShot(120, poll)

    def handle(wanted: str) -> None:
        if wanted.startswith("render:"):
            render(wanted[len("render:"):].strip())
            return
        capture(wanted)

    def _finish(payload: dict) -> None:
        tmp = done_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as done_file:
            json.dump(payload, done_file)
        os.replace(tmp, done_path)

    def render(video: str) -> None:
        """Queue and START a render through the same controls the user presses.

        Staged polling, not sleeps: the input field's `textChanged` runs `loadVideo`, which probes the file and can take
        seconds; pressing "Add to Render Queue" before that finishes leaves the queue empty (measured - the first
        version used a fixed 1.2 s delay and answered "the queue stayed empty").
        """
        stamp = _stamp()
        result = {"stamp": stamp, "kind": "render", "video": video, "queued": 0, "started": False,
                  "finished": False, "seconds": 0.0, "output": None, "error": None}
        started_at = time.time()

        def output_of_last_item() -> str | None:
            queue = getattr(window.renderQueue, "queue", []) or []
            if not queue:
                return None
            item = queue[-1]
            for attr in ("output", "outputPath", "output_file", "out_path"):
                value = getattr(item, attr, None)
                if value:
                    return str(value)
            return None

        def finish(error: str | None = None) -> None:
            result["seconds"] = round(time.time() - started_at, 1)
            result["queued"] = len(getattr(window.renderQueue, "queue", []) or [])
            result["output"] = output_of_last_item()
            if result["output"] and os.path.exists(result["output"]):
                result["output_bytes"] = os.path.getsize(result["output"])
            if error:
                result["error"] = error
            _finish(result)
            print(f"capture-server: render {result}", flush=True)

        def wait_finish(ticks: int = 0) -> None:
            if window.startRenderButton.isEnabled() and result["started"]:
                result["finished"] = True
                finish()
                return
            if ticks > 3000:                     # 3000 * 100 ms = 5 min
                finish("timeout waiting for the render to finish")
                return
            QTimer.singleShot(100, lambda: wait_finish(ticks + 1))

        def press_start(ticks: int = 0) -> None:
            if len(getattr(window.renderQueue, "queue", []) or []) > 0:
                result["started"] = True
                window.startRenderButton.click()
                QTimer.singleShot(250, wait_finish)
                return
            if ticks > 100:                      # 100 * 100 ms = 10 s
                finish("the queue stayed empty after Add to Render Queue")
                return
            QTimer.singleShot(100, lambda: press_start(ticks + 1))

        def select_a_model() -> None:
            """The app refuses to queue without one: "Please select at least one model!" (REAL-Video-Enhancer.py:537).

            The gate enables upscale (the model list is populated from the models directory) and picks the first model,
            because a queue that cannot be filled would make the rest of the gate measure nothing.
            """
            boxes = ("interpolateCheckBox", "upscaleCheckBox", "deblurCheckBox", "denoiseCheckBox",
                     "decompressCheckBox")
            if any(getattr(window, name, None) is not None and getattr(window, name).isChecked() for name in boxes):
                return
            box = getattr(window, "upscaleCheckBox", None)
            if box is not None:
                box.setChecked(True)
            combo = getattr(window, "upscaleModelComboBox", None)
            if combo is not None and combo.currentIndex() < 0 and combo.count() > 0:
                combo.setCurrentIndex(0)

        def press_add() -> None:
            select_a_model()
            window.addToRenderQueueButton.click()
            QTimer.singleShot(200, press_start)

        def wait_loaded(ticks: int = 0) -> None:
            info = getattr(window, "inputVideoInfoTextEdit", None)
            loaded = bool(info is not None and info.toPlainText().strip()) or getattr(window, "videoLength", 1) != 1
            if loaded:
                press_add()
                return
            if ticks > 150:                      # 15 s
                press_add()                      # let the app refuse on its own terms; its error is the evidence
                return
            QTimer.singleShot(100, lambda: wait_loaded(ticks + 1))

        window.inputFileText.setText(video)
        QTimer.singleShot(300, wait_loaded)

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
        with open(tmp, "w", encoding="utf-8") as done_file:
            json.dump({"stamp": stamp, "pages": names}, done_file)
        os.replace(tmp, done_path)

    QTimer.singleShot(delay, lambda: warm(0))
