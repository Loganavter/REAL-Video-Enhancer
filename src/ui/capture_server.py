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


#: Where the always-on channel lives: `<app root>/.rve-control`. Relative to the running script, so a launch from any
#: directory still gives one predictable drop box next to `log.txt`.
CONTROL_DIR_NAME = ".rve-control"


def control_root() -> str:
    """The request/reply directory: `RVE_UI_CAPTURE_SERVER` when set, else `<app root>/.rve-control`."""
    explicit = os.environ.get("RVE_UI_CAPTURE_SERVER")
    if explicit:
        return explicit
    import sys

    script = os.path.abspath(sys.argv[0]) if sys.argv and sys.argv[0] else os.getcwd()
    return os.path.join(os.path.dirname(script) or os.getcwd(), CONTROL_DIR_NAME)


def enabled() -> bool:
    """Always on unless explicitly switched off (`RVE_UI_CAPTURE_SERVER=0`).

    Why always: automation through the keyboard was the wrong instrument on this host - `Ctrl+Shift+<letter>` is eaten by
    the input method (only the modifier's auto-repeats reached the app), so a probe the owner pressed never ran. The
    reference host answers this with an always-available in-app control server (`QLocalServer` + JSON lines,
    BlinkTable `src/core/control/server.py`); this is the same idea over a drop-box directory that needs no listener and
    no clipboard/grab. The owner can still switch it off with `RVE_UI_CAPTURE_SERVER=0`.
    """
    value = os.environ.get("RVE_UI_CAPTURE_SERVER")
    if value is None:
        return True
    return str(value).strip().lower() not in ("0", "false", "no", "off")


def _stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"


def install(window) -> None:
    """Arm the server on ``window``. Never returns anything; the app keeps running."""
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from src.ui import layout_dump, page_capture

    root = control_root()
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

    # The warm-up walk exists for measurement runs (every page touched once so the first grab is settled). The always-on
    # diagnosis channel must NOT walk: it would switch pages under the owner's hands. It polls immediately and only warms
    # a page when that page is asked for.
    always_on = os.environ.get("RVE_UI_CAPTURE_SERVER") is None
    state = {"warming": not always_on, "index": 0, "request_id": None}

    if always_on:
        with open(ready_path, "w", encoding="utf-8") as ready_file:
            ready_file.write(_stamp())
        print(f"control channel: ready (no warm-up walk) in {root}", flush=True)

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

    if always_on:
        QTimer.singleShot(0, poll)

    def probe(spec: str) -> None:
        """Name the painter at a point: `probe:<objectName>@<window_x>,<window_y>`.

        The point is in WINDOW coordinates and is mapped into the widget here, because a widget's live origin cannot be
        reproduced offscreen (measured: `generalSettings` is 480x539 and a window point that looks "inside" maps outside
        it in a differently sized build). The answer is `childAt`'s chain - the only form that names a painter, since a
        custom `paintEvent` widget carries no fill attribute to query - plus each step's own pixel.
        """
        from PySide6.QtCore import QPoint
        from PySide6.QtWidgets import QWidget

        target, _, coords = spec.partition("@")
        target = target.strip()
        root_widget = window.findChild(QWidget, target)
        report = {"stamp": _stamp(), "kind": "probe", "target": target, "point": coords.strip() or None,
                  "error": None, "chain": []}
        if root_widget is None:
            report["error"] = "widget not found"
            _finish(report)
            return
        if not coords.strip():
            report["error"] = "no point given (expected name@x,y in window coordinates)"
            _finish(report)
            return
        try:
            wx, wy = (int(v) for v in coords.split(","))
        except ValueError:
            report["error"] = "point is not x,y"
            _finish(report)
            return
        node = root_widget
        local = root_widget.mapFrom(window, QPoint(wx, wy))
        report["local_point"] = [local.x(), local.y()]
        report["target_size"] = [root_widget.width(), root_widget.height()]
        point = local
        while True:
            child = node.childAt(point.x(), point.y())
            if child is None:
                break
            step = child.mapFrom(node, point)
            image = child.grab().toImage()
            pixel = None
            if 0 <= step.x() < image.width() and 0 <= step.y() < image.height():
                pixel = list(image.pixelColor(step.x(), step.y()).getRgb())
            report["chain"].append({"class": type(child).__name__, "name": child.objectName(),
                                    "geometry": [child.x(), child.y(), child.width(), child.height()],
                                    "local": [step.x(), step.y()], "pixel": pixel})
            node = child
            point = step
        _finish(report)
        print(f"capture-server: probe {target}@({wx},{wy}) -> {len(report['chain'])} step(s)", flush=True)

    def click(spec: str) -> None:
        """Send a real left click: `click:<objectName>@x,y`.

        A screenshot cannot tell "the row was selected" from "the row was clicked and nothing happened", and the
        offscreen tests deliver their own events - this routes the click the way the platform does, through the widget
        the hit-test finds.
        """
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import QWidget

        target, _, coords = spec.partition("@")
        target = target.strip()
        widget = window.findChild(QWidget, target)
        report = {"stamp": _stamp(), "kind": "click", "target": target, "point": coords.strip() or None,
                  "error": None, "hit": None}
        if widget is None:
            report["error"] = "widget not found"
            _finish(report)
            print(f"capture-server: click {target!r} -> widget not found", flush=True)
            return
        try:
            x, y = (int(value) for value in coords.split(","))
        except ValueError:
            report["error"] = "point is not x,y"
            _finish(report)
            return
        point = QPoint(x, y)
        hit = widget.childAt(point)
        report["hit"] = {"class": type(hit).__name__, "name": hit.objectName()} if hit is not None else None
        report["target_size"] = [widget.width(), widget.height()]
        # The geometry chain: a painted list is three widgets deep, and a press that never reaches the rows widget looks
        # exactly like "the click was ignored".
        chain = []
        node = widget
        rows = getattr(widget, "_rows", None)
        if rows is not None:
            chain.append({"what": "rows", "geom": [rows.x(), rows.y(), rows.width(), rows.height()],
                          "parent": type(rows.parent()).__name__,
                          "in_list": [rows.mapTo(widget, QPoint(0, 0)).x(), rows.mapTo(widget, QPoint(0, 0)).y()],
                          # What makes Qt SKIP a widget for mouse delivery (childAt ignores both of these, so a
                          # "childAt says _Rows" answer is not proof that the press lands there):
                          "enabled": bool(rows.isEnabled()), "visible": bool(rows.isVisible()),
                          "mouse_transparent": bool(rows.testAttribute(
                              Qt.WidgetAttribute.WA_TransparentForMouseEvents)),
                          "blocked_by_modal": bool(qt_widgets.QApplication.activeModalWidget()) if False else None})
        scroll = getattr(widget, "_scroll", None)
        if scroll is not None:
            viewport = scroll.viewport()
            chain.append({
                "what": "scroll", "geom": [scroll.x(), scroll.y(), scroll.width(), scroll.height()],
                "viewport": [viewport.x(), viewport.y(), viewport.width(), viewport.height()],
                # The attribute that makes Qt SKIP a widget for mouse delivery lives on EVERY level of the chain:
                # a transparent scroll area keeps the press at the list, and no child ever sees it.
                "scroll_transparent": bool(scroll.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)),
                "viewport_transparent": bool(viewport.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)),
                "scroll_enabled": bool(scroll.isEnabled()), "viewport_enabled": bool(viewport.isEnabled()),
            })
        report["geometry"] = chain
        # WHERE the press actually lands: `childAt` is this script's guess, an event filter is the platform's answer.
        from PySide6.QtCore import QEvent, QObject

        delivered = []

        class _Tap(QObject):
            def eventFilter(self, obj, event):
                if event.type() in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease):
                    delivered.append((event.type().name, type(obj).__name__, obj.objectName()))
                return False

        tap = _Tap()
        QApplication.instance().installEventFilter(tap)
        try:
            # Through the WINDOW, not the widget: `QTest.mouseClick(some_widget, ...)` posts straight to that widget,
            # which is how an offscreen test can "prove" a click on a row that the platform would deliver elsewhere.
            # The window handle goes through Qt's own hit-testing - the same path a real mouse takes.
            target_window = window.windowHandle()
            window_point = widget.mapTo(window, point)
            QTest.mouseClick(target_window, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                             window_point)
            for _ in range(6):
                QApplication.processEvents()
        finally:
            QApplication.instance().removeEventFilter(tap)
        report["delivered"] = delivered[:6]
        selected = getattr(widget, "selectedRows", None)
        report["selectedRows"] = sorted(selected()) if callable(selected) else None
        report["currentRow"] = widget.currentRow() if hasattr(widget, "currentRow") else None
        _finish(report)
        print(f"capture-server: click {target}@({x},{y}) -> hit={report['hit']} selected={report['selectedRows']}", flush=True)

    def drag(spec: str) -> None:
        """A real drag through the window: `drag:<objectName>@x1,y1>dydx...` - `x1,y1:x2,y2`.

        The rubber band is the gesture the owner asked for, and it cannot be proven any other way: a synthetic move sent
        straight to a widget bypasses the application filter that makes the band keep tracking.
        """
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import QWidget

        target, _, coords = spec.partition("@")
        widget = window.findChild(QWidget, target.strip())
        report = {"stamp": _stamp(), "kind": "drag", "target": target.strip(), "error": None}
        if widget is None or ":" not in coords:
            report["error"] = "widget not found" if widget is None else "expected x1,y1:x2,y2"
            _finish(report)
            return
        try:
            start_text, end_text = coords.split(":", 1)
            x1, y1 = (int(v) for v in start_text.split(","))
            x2, y2 = (int(v) for v in end_text.split(","))
        except ValueError:
            report["error"] = "points are not x,y:x,y"
            _finish(report)
            return
        target_window = window.windowHandle()
        p1 = widget.mapTo(window, QPoint(x1, y1))
        p2 = widget.mapTo(window, QPoint(x2, y2))
        QTest.mousePress(target_window, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p1)
        for step in range(1, 5):
            QApplication.processEvents()
            mid = QPoint(p1.x() + (p2.x() - p1.x()) * step // 4, p1.y() + (p2.y() - p1.y()) * step // 4)
            QTest.mouseMove(target_window, mid, 10)
        QTest.mouseRelease(target_window, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p2)
        for _ in range(8):
            QApplication.processEvents()
        selected = getattr(widget, "selectedRows", None)
        report["selectedRows"] = sorted(selected()) if callable(selected) else None
        _finish(report)
        print(f"capture-server: drag {target}@({x1},{y1})->({x2},{y2}) selected={report['selectedRows']}", flush=True)

    def queue(spec: str) -> None:
        """Load a folder into the queue: `queue:<folder-or-file>` - the user's own clicks, without a file dialog.

        A repaint artifact cannot be reproduced through `window.grab()` (it re-renders the tree and never reads the
        backing store), so the live app has to be put into the state that shows it: several rows queued, some selected.
        """
        import os as _os

        path = spec.strip()
        report = {"stamp": _stamp(), "kind": "queue", "path": path, "error": None, "rows": 0}
        if not _os.path.exists(path):
            report["error"] = "path does not exist"
            _finish(report)
            return
        if _os.path.isdir(path):
            window.loadVideo(path, multi_file=True)
        else:
            window.loadVideo(path)
        out = "/tmp/opencode/live-queue-out"
        _os.makedirs(out, exist_ok=True)
        window.outputFileText.setText(out)

        def add(ticks: int = 0) -> None:
            probe = getattr(window, "_batchProbe", None)
            if probe is not None and probe.isRunning() and ticks < 200:
                QTimer.singleShot(100, lambda: add(ticks + 1))
                return
            window.addToRenderQueue()
            if window.renderQueueListWidget.count() == 0:
                # No op runs without a model selected; the probe is a measurement tool, so it states one instead of
                # reporting "0 rows" with no reason.
                report["error"] = "nothing added (no model selected?)"
                box = getattr(window, "upscaleCheckBox", None)
                combo = getattr(window, "upscaleModelComboBox", None)
                if box is not None and combo is not None:
                    box.setChecked(True)
                    if combo.count():
                        combo.setCurrentIndex(combo.count() - 1)
                    window.addToRenderQueue()
            report["rows"] = window.renderQueueListWidget.count()
            report["files"] = [row.inputFile for row in window.renderQueue.getQueue()]
            _finish(report)
            print(f"capture-server: queue {path} -> {report['rows']} rows", flush=True)

        QTimer.singleShot(100, add)

    def select(spec: str) -> None:
        """Select rows by index: `select:renderQueueListWidget@0,1,2`."""
        from PySide6.QtWidgets import QWidget

        target, _, rows = spec.partition("@")
        widget = window.findChild(QWidget, target.strip())
        try:
            indices = [int(part) for part in rows.split(",") if part.strip()]
        except ValueError:
            indices = []
        report = {"stamp": _stamp(), "kind": "select", "error": None}
        if widget is None or not hasattr(widget, "setSelectedRows"):
            report["error"] = "no such list"
        else:
            widget.setSelectedRows(indices)
            for _ in range(6):
                QApplication.processEvents()
            report["selectedRows"] = widget.selectedRows()
        _finish(report)
        print(f"capture-server: select {spec} -> {report.get('selectedRows')}", flush=True)

    def move(direction: str) -> None:
        """Run the queue's Move Up / Move Down on the current selection: `move:up`."""
        queue_model = window.renderQueue
        before = [row.inputFile for row in queue_model.getQueue()]
        moved = queue_model.moveSelected(direction.strip())
        for _ in range(8):
            QApplication.processEvents()
        widget = window.renderQueueListWidget
        _finish({"stamp": _stamp(), "kind": "move", "moved": bool(moved),
                 "rows": widget.count(), "selectedRows": widget.selectedRows(),
                 "order": [row.inputFile for row in queue_model.getQueue()],
                 "changed": before != [row.inputFile for row in queue_model.getQueue()]})
        print(f"capture-server: move {direction} -> moved={moved} selected={widget.selectedRows()}", flush=True)

    def where(_spec: str = "") -> None:
        """Which code is actually RUNNING: `where:` - paths, commits, and the revision of the toolkit in memory.

        "Am I on the right checkout?" cannot be answered by looking at a screenshot, and a process started before a fix
        keeps running the old modules: this answer names the file each module came from, the git commit of every repo
        involved, and whether the functions the fix added exist in the loaded copy.
        """
        import inspect as _inspect
        import subprocess as _subprocess
        import sys as _sys

        def repo_state(path: str) -> dict:
            root = os.path.dirname(os.path.dirname(os.path.abspath(path)))
            try:
                head = _subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                                       capture_output=True, text=True, timeout=5).stdout.strip()
                branch = _subprocess.run(["git", "-C", root, "rev-parse", "--abbrev-ref", "HEAD"],
                                         capture_output=True, text=True, timeout=5).stdout.strip()
                dirty = _subprocess.run(["git", "-C", root, "status", "--porcelain"],
                                        capture_output=True, text=True, timeout=5).stdout.strip()
            except Exception as error:
                return {"root": root, "error": str(error)}
            return {"root": root, "head": head, "branch": branch, "dirty_files": len(dirty.splitlines())}

        report = {"stamp": _stamp(), "kind": "where", "executable": _sys.executable,
                  "argv0": _sys.argv[0], "python": _sys.version.split()[0], "modules": {}}
        for name in ("sli_ui_toolkit", "sli_ui_toolkit.ui.widgets.composite.top_tab_bar.host",
                     "sli_ui_toolkit.ui.widgets.composite.text_list_widget"):
            module = _sys.modules.get(name)
            path = getattr(module, "__file__", None) if module is not None else None
            report["modules"][name] = {"file": path, "loaded": module is not None}
        host_module = _sys.modules.get("sli_ui_toolkit.ui.widgets.composite.top_tab_bar.host")
        report["host_has_fit_pages"] = bool(host_module and hasattr(host_module.TopTabHost, "_fit_pages"))
        widget_module = _sys.modules.get("sli_ui_toolkit.ui.widgets.composite.text_list_widget")
        report["rows_fill_present"] = bool(widget_module and hasattr(widget_module.TextListWidget, "fill_colour"))
        toolkit_file = report["modules"].get("sli_ui_toolkit", {}).get("file")
        report["toolkit_repo"] = repo_state(toolkit_file) if toolkit_file else None
        report["app_repo"] = repo_state(_sys.argv[0])
        _finish(report)
        print(f"capture-server: where -> app {report['app_repo'].get('head')} toolkit "
              f"{report['toolkit_repo'].get('head') if report['toolkit_repo'] else 'n/a'}", flush=True)

    def widget_state(name: str) -> None:
        """The interactive state of one named widget: `state:<objectName>`.

        Why: "not blocked during the render" is a claim about a WIDGET, not about a screenshot - a toolkit control paints
        the same pixels whether or not it is disabled, so a grab cannot tell them apart. This answer can: `enabled`,
        visibility, and the scroll range/value of whatever scrolls inside it.
        """
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QWidget

        target = window.findChild(QWidget, name.strip())
        if target is None and name.strip() == window.objectName():
            target = window          # the window itself is not its own child
        report = {"stamp": _stamp(), "kind": "state", "target": name.strip(), "error": None}
        if target is None:
            report["error"] = "widget not found"
            _finish(report)
            print(f"capture-server: state {name!r} -> widget not found", flush=True)
            return
        report.update({
            "class": type(target).__name__,
            "enabled": bool(target.isEnabled()),
            "visible": bool(target.isVisible()),
            "size": [target.width(), target.height()],
            # The window's own attributes: `opaque_paint` is the flag that used to make the shell skip repaints.
            "parent": type(target.parentWidget()).__name__ if target.parentWidget() else None,
            "parent_name": target.parentWidget().objectName() if target.parentWidget() else None,
            "parent_size": [target.parentWidget().width(), target.parentWidget().height()]
            if target.parentWidget() else None,
            "opaque_paint": bool(target.testAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)),
            "translucent": bool(target.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)),
            "autofill": bool(target.autoFillBackground()),
        })
        for attr in ("count", "currentRow", "selectedRows", "selectedCount"):
            getter = getattr(target, attr, None)
            if callable(getter):
                try:
                    value = getter()
                except Exception:
                    continue
                report[attr] = sorted(value) if isinstance(value, list) else value
        bar = getattr(target, "verticalScrollBar", None)
        if callable(bar):
            try:
                scroll = bar()
                report["scroll"] = {"value": int(scroll.value()), "max": int(scroll.maximum())}
            except Exception:
                pass
        _finish(report)
        print(f"capture-server: state {name!r} -> {report.get('enabled')} enabled", flush=True)

    def handle(wanted: str) -> None:
        request_id = None
        if "#" in wanted:
            wanted, _, request_id = wanted.rpartition("#")
            wanted = wanted.strip()
            request_id = request_id.strip() or None
        state["request_id"] = request_id
        if wanted.startswith("probe:"):
            probe(wanted[len("probe:"):])
            return
        if wanted.startswith("repaint:"):
            # A full repaint on demand: the same thing a window resize does, so "does this clear it" needs no keyboard.
            before = _stamp()
            window.repaint()
            for _ in range(4):
                QApplication.processEvents()
            _finish({"stamp": before, "kind": "repaint", "done": True})
            print("capture-server: repaint -> done", flush=True)
            return
        if wanted.startswith("grab:"):
            target = wanted[len("grab:"):].strip() or os.path.join(root, f"grab-{_stamp()}.png")
            os.makedirs(os.path.dirname(target) or root, exist_ok=True)
            pixmap = window.grab()
            saved = pixmap.save(target)
            _finish({"stamp": _stamp(), "kind": "grab", "path": target, "saved": bool(saved),
                     "size": [pixmap.width(), pixmap.height()]})
            print(f"capture-server: grab -> {target} ({pixmap.width()}x{pixmap.height()})", flush=True)
            return
        if wanted.startswith("probe_at:"):
            from PySide6.QtCore import QPoint

            from src.ui.devtools import layout_audit

            try:
                gx, gy = (int(value) for value in wanted[len("probe_at:"):].strip().split(","))
            except ValueError:
                _finish({"stamp": _stamp(), "kind": "probe_at", "error": "expected x,y in global coordinates"})
                return
            lines = layout_audit.probe_at(window, QPoint(gx, gy))
            path = layout_audit.write_probe(window, control_root())
            _finish({"stamp": _stamp(), "kind": "probe_at", "lines": lines, "path": path,
                     "point": [gx, gy]})
            print(f"capture-server: probe_at ({gx},{gy}) -> {len(lines)} line(s)", flush=True)
            return
        if wanted.startswith("audit:"):
            from src.ui.devtools import layout_audit

            lines = layout_audit.audit(window)
            path = layout_audit.write(window)
            _finish({"stamp": _stamp(), "kind": "audit", "lines": lines, "path": path})
            print(f"capture-server: audit -> {len(lines) - 3} finding(s), {path}", flush=True)
            return
        if wanted.startswith("where:"):
            where(wanted[len("where:"):])
            return
        if wanted.startswith("state:"):
            widget_state(wanted[len("state:"):])
            return
        if wanted.startswith("click:"):
            click(wanted[len("click:"):])
            return
        if wanted.startswith("drag:"):
            drag(wanted[len("drag:"):])
            return
        if wanted.startswith("queue:"):
            queue(wanted[len("queue:"):])
            return
        if wanted.startswith("select:"):
            select(wanted[len("select:"):])
            return
        if wanted.startswith("move:"):
            move(wanted[len("move:"):])
            return
        if wanted == "stack":
            # Every thread's Python stack, written next to the server's other answers. The render runs in a worker, and
            # a stuck first frame needs the call it is stuck IN, not an inference from a progress bar.
            import faulthandler

            path = os.path.join(root, "stacks.txt")
            with open(path, "w", encoding="utf-8") as handle:
                faulthandler.dump_traceback(file=handle, all_threads=True)
            _finish({"stamp": _stamp(), "kind": "stack", "path": path})
            return
        if wanted.startswith("render:"):
            render(wanted[len("render:"):].strip())
            return
        capture(wanted)

    def _finish(payload: dict) -> None:
        payload.setdefault("request_id", state.get("request_id"))
        """Publish an answer. `done` is the mailbox every older caller reads; `answers/<id>.json` is the race-free one.

        One shared `done` file loses answers when two requests overlap: the waiter sees a payload of the wrong kind,
        deletes it, and then waits for an answer that was already consumed - which looks exactly like the app hanging
        (measured 2026-11-22: the owner closed the app while a probe client sat on that wait). The request may carry an
        id as `verb#<id>`; the answer is then also written to `answers/<id>.json`, and a client that knows its id never
        reads anybody else's reply.
        """
        answers_dir = os.path.join(root, "answers")
        request_id = payload.get("request_id")
        if request_id:
            try:
                os.makedirs(answers_dir, exist_ok=True)
                target = os.path.join(answers_dir, f"{request_id}.json")
                with open(target + ".tmp", "w", encoding="utf-8") as answer_file:
                    json.dump(payload, answer_file)
                os.replace(target + ".tmp", target)
            except OSError:
                pass
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
        result = {"stamp": stamp, "kind": "render", "video": video, "backend": None, "queued": 0, "started": False,
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
            result["backend"] = backend_of()
            result["output"] = output_of_last_item()
            if result["output"] and os.path.exists(result["output"]):
                result["output_bytes"] = os.path.getsize(result["output"])
            if error:
                result["error"] = error
            _finish(result)
            print(f"capture-server: render {result}", flush=True)

        def backend_of() -> str | None:
            """Which backend the job actually runs on - the payload has to say, or a stall cannot be attributed."""
            for attr in ("backendComboBox", "backend_combo", "backend"):
                combo = getattr(window, attr, None)
                if combo is not None and hasattr(combo, "currentText"):
                    try:
                        text = combo.currentText()
                    except Exception:
                        continue
                    if text:
                        return str(text)
            return None

        def progress_sample() -> dict:
            """What the UI says about the job right now - a timeout without this cannot tell a slow model load from a
            frame loop that never advances, and the previous run's `finished: False` said only that."""
            bar = getattr(window, "progressBar", None)
            queue = getattr(window, "renderQueue", None)
            sample = {
                "bar": int(bar.value()) if bar is not None else None,
                "bar_max": int(bar.maximum()) if bar is not None else None,
                "queue": len(getattr(queue, "queue", []) or []) if queue is not None else None,
                "start_enabled": bool(window.startRenderButton.isEnabled()),
                "status": (getattr(window, "STATUS", None).text() if getattr(window, "STATUS", None) is not None else None),
                "backend": backend_of(),
            }
            return sample

        def wait_finish(ticks: int = 0) -> None:
            if ticks % 50 == 0:                  # one sample every 5 s, so 15 per minute of a long model load
                result.setdefault("trace", []).append({"t": round(time.time() - started_at, 1), **progress_sample()})
            if window.startRenderButton.isEnabled() and result["started"]:
                result["finished"] = True
                finish()
                return
            if ticks > 9000:                     # 9000 * 100 ms = 15 min
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

        def select_a_backend() -> None:
            """Prefer a backend that initialises in this runtime.

            The default is TensorRT, and its engine build blocks forever here (measured: the backend child sits in
            `futex_wait` at ~4 % CPU while the app waits on its stdout, ProcessTab.py:323). Torch/PyTorch runs.
            """
            combo = getattr(window, "backendComboBox", None)
            if combo is None or not hasattr(combo, "count"):
                return
            for index in range(combo.count()):
                text = combo.itemText(index).lower()
                if "torch" in text and "tensorrt" not in text:
                    if combo.currentIndex() != index:
                        combo.setCurrentIndex(index)
                    return

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
            select_a_backend()
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
            # Settle the page switch before the grab: one event pass is not enough right after the warm-up walk, and a
            # grab of the previous page under the requested file name is a silent wrong reading (measured twice).
            for _ in range(25):
                QApplication.processEvents()
                if page_capture.is_current(window, name):
                    break
                time.sleep(0.02)
            else:
                print(f"capture-server: {name} did not become current - the dump names a page it is not showing",
                      flush=True)
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
