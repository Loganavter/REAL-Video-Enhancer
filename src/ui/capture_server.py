"""A long-lived capture server: boot the app once, then answer snapshot requests from disk.

Why: every measurement used to cost a full app start plus the backend settle (30-40 s) because the harness could only
talk to a process it had just launched. Iterating on one page meant paying that price per iteration, which is most of
the wall time of a migration session.

With ``RVE_UI_CAPTURE_SERVER=<dir>`` the app boots once, warms every page, writes ``<dir>/ready`` and then polls
``<dir>/request``. A request names the pages to capture (or ``all``); the server shows each, saves ``<page>.png`` and
``<page>.json`` into ``<dir>/<stamp>/`` and writes ``<dir>/done`` with that stamp. A single page then costs about a
second instead of a minute, and the process stays up for the next request.

Protocol (files, so any tool can drive it - see ``harness/scripts/ui_live.py``):

    request   "all" or "nav3-tab0,nav4"     written by the driver; may carry its answer id as `<verb>#<id>`
    requests/<id>  "<verb>"                  one file per request, the file name IS the id (see below)
    ready     ""                             written once, after the warm-up walk
    done      "<stamp>"                      written per request; the driver removes `request` first
    quit      ""                             asks the app to exit

Two request sources, one contract: **a request that reaches the intake is dispatched once or answered with an
error** - an empty file (a client writing in place truncates first) is left for the next tick, a request that stays
empty is rejected, and an unreadable verb is no longer a batch capture. The single `request` slot cannot hold two
writers inside one 120 ms poll, so a client that needs that guarantee publishes `requests/<id>` with
``write tmp; os.replace(tmp, final)``; the file name is its answer id and its content is the verb verbatim. The
legacy slot keeps working for every existing client (`ui_live.py shot`, `live.sh ask`, the MCP backend).

Why not a socket: the app may block its event loop in a modal dialog or a long paint; a file poll keeps the protocol
trivial and crash-safe (an unparsed request is answered, not mistaken for a page list).
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


#: The legacy single request slot. Kept because every existing client writes it (`harness/scripts/ui_live.py`,
#: `harness/scripts/live.sh ask`, the MCP backend, the supervisor's probes); a second client can overwrite it
#: inside one 120 ms tick, which is why `requests/` exists.
LEGACY_REQUEST_NAME = "request"
#: One file per request: `requests/<id>` - the file NAME is the id, and two clients publishing with
#: `write tmp; os.replace(tmp, final)` can never clobber each other.
REQUESTS_DIR_NAME = "requests"
#: How long an empty request file is left alone before it becomes a visible error. A client writing in place
#: truncates first, so the empty window is usually milliseconds; the grace only separates "still writing" from
#: "a client left an empty file behind".
EMPTY_REQUEST_GRACE_S = 2.0


def _read_request(path: str) -> str:
    """Read one request file without ever raising on its bytes.

    `errors="replace"` is the point: a verb carrying a path in a non-UTF-8 encoding used to raise
    `UnicodeDecodeError` out of `poll`, the timer was never rescheduled, and every later request went unanswered
    - an app that looks hung (measured 2026-11-24: `0xb7` in a Cyrillic path). A replacement character is a value
    the handler can report; an exception costs every request after it.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as request_file:
        return request_file.read()


def _discard(path: str) -> None:
    """Remove a request that has been answered. A failure leaves it for the next tick, never eats it silently."""
    try:
        os.unlink(path)
    except OSError:
        pass


def _dispatch_safely(dispatch, verb: str, request_id: str | None, path: str) -> None:
    """Run one dispatch; a handler that raises must not abort the batch (this file is already off the disk)."""
    try:
        dispatch(verb, request_id)
    except Exception as exc:
        print(f"capture-server: {path} ({verb!r}) failed: {type(exc).__name__}: {exc}", flush=True)


def page_request_error(wanted: str, known_pages) -> str | None:
    """The sentence `handle()`'s fallthrough must answer with, or None when `wanted` IS a page request.

    The fallthrough used to be `capture(wanted)`, and `capture("")` means "every page": a request file caught in
    its truncate-then-write window was not merely lost, it ran a full batch capture (measured 2026-11-24). An
    empty or unparseable verb answers an error bound to its id instead. A partly known list stays a capture
    request: `capture()` answers with the pages it could name and its payload lists them, so the dropped name is
    visible to the client.
    """
    if not wanted.strip():
        return "empty request - nothing was dispatched (the file was probably still being written)"
    if wanted.strip() == "all":
        return None
    names = [part.strip() for part in wanted.split(",") if part.strip()]
    if any(name in known_pages for name in names):
        return None
    return (f"unknown request {wanted!r}: expected 'all', a page name ({', '.join(known_pages)}) "
            f"or a verb such as state:, click:, preview:, render:")


def drain_requests(root: str, dispatch, *, state: dict | None = None, now: float | None = None,
                   grace_s: float = EMPTY_REQUEST_GRACE_S) -> int:
    """Take every readable request out of the drop box exactly once and hand it to `dispatch(verb, id)`.

    Two sources, one contract - a request either reaches `dispatch` or is answered with an error:

    * `request` (legacy single slot): the verb may carry its id as `verb#<id>`, and the text is passed verbatim
      so `handle()` keeps parsing it the way it always has;
    * `requests/<id>`: the file NAME is the id, so two clients publishing with `os.replace` never overwrite each
      other, and the content is the verb verbatim (a `#` inside it, as in `preview:<clip>#model=futsuu`, is the
      verb's own, not an id).

    Files are read with replacement characters (see `_read_request`), and an EMPTY file is left in place for the
    next tick: a client writing in place truncates first, and consuming that window loses the request silently.
    Past `grace_s` the empty file is a request the loop must answer - it is dispatched as the empty verb (which
    `page_request_error` rejects), then removed so it cannot answer again every tick.

    Returns the number of requests dispatched, rejections included: they were answered, not dropped.
    """
    if state is None:
        state = {}
    if now is None:
        now = time.monotonic()
    empty_since = state.setdefault("empty_since", {})

    pending: list[tuple[str, str | None, bool]] = []
    legacy_path = os.path.join(root, LEGACY_REQUEST_NAME)
    if os.path.isfile(legacy_path):
        pending.append((legacy_path, None, True))
    directory = os.path.join(root, REQUESTS_DIR_NAME)
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        names = []
    for name in names:
        if name.startswith(".") or name.endswith(".tmp"):
            # `write tmp; os.replace(tmp, final)` is the sanctioned way to publish a request: the stage is not a
            # request and must not be consumed (the client may be between the two calls).
            continue
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            pending.append((path, name, False))

    batch: list[tuple[str, str | None, bool, str]] = []
    for path, request_id, legacy in pending:
        try:
            text = _read_request(path)
        except OSError:
            # Not consumed: an unreadable file stays where its owner can see it and is retried next tick.
            continue
        batch.append((path, request_id, legacy, text))

    live = {path for path, _request_id, _legacy, _text in batch}
    for stale in [path for path in empty_since if path not in live]:
        del empty_since[stale]

    served = 0
    for path, request_id, _legacy, text in batch:
        if not text.strip():
            first = empty_since.setdefault(path, now)
            if now - first < grace_s:
                continue
            empty_since.pop(path, None)
            _discard(path)
            served += 1
            _dispatch_safely(dispatch, "", request_id, path)
            continue
        empty_since.pop(path, None)
        # Unlink BEFORE dispatch: `capture()` can run for seconds, and the legacy slot must be free for the next
        # writer - otherwise this unlink would delete the request that writer just published.
        _discard(path)
        served += 1
        _dispatch_safely(dispatch, text, request_id, path)
    return served


def validate_input_path(path: str) -> tuple[str | None, str | None]:
    """``(error, resolved)`` for an `input:` request - pure, so a test needs no window.

    An empty path and a path that does not exist are the two ways a probe gets this wrong (a typo, a relative path run
    from another directory); both are answered, never silently ignored.
    """
    text = (path or "").strip()
    if not text:
        return "empty path", None
    resolved = os.path.abspath(os.path.expanduser(text))
    if not os.path.isfile(resolved):
        return f"no such file: {resolved}", None
    return None, resolved


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
    state = {"warming": not always_on, "index": 0, "request_id": None, "walk_before": None, "empty_since": {}}

    if always_on:
        with open(ready_path, "w", encoding="utf-8") as ready_file:
            ready_file.write(_stamp())
        print(f"control channel: ready (no warm-up walk) in {root}", flush=True)

    def warm(index: int) -> None:
        if index == 0:
            # The walk is an INSTRUMENT, not a user: remember where the interface is before step 0, so finishing the
            # walk does not leave the window on the last page's last tab. On a fresh boot that leftover was the empty
            # Logs tab of the settings-nested host, and every probe that waited for `ready` photographed it (measured
            # 2026-11-24: `state:processSettingsContainer -> currentIndex: 4`, `state:tabWidget -> currentIndex: 2`;
            # one click on any tab fixed the view).
            state["walk_before"] = page_capture.walk_state(window)
        if index >= len(steps):
            # Restore BEFORE `ready` is written: a client that waits for the file must never observe a half-walked
            # interface. A capture ON REQUEST keeps its own behaviour (it ends on the last requested stem).
            before = state.pop("walk_before", None)
            page_capture.restore_walk_state(window, before)
            QApplication.processEvents()
            state["warming"] = False
            with open(ready_path, "w", encoding="utf-8") as ready_file:
                ready_file.write(_stamp())
            print(f"capture-server: ready ({len(steps)} pages) in {root}", flush=True)
            print(f"capture-server: walk restored to page {before.get('page') if before else None}", flush=True)
            QTimer.singleShot(0, poll)
            return
        _name, action = steps[index]
        action()
        QApplication.processEvents()
        QTimer.singleShot(step, lambda: warm(index + 1))

    def dispatch(wanted: str, request_id: str | None = None) -> None:
        """Dispatch ONE drained request, with its id bound before any handler can answer from a timer."""
        state["request_id"] = request_id
        try:
            handle(wanted, request_id)
        except Exception as exc:
            # This loop IS the channel, and a handler that raises also skips the reschedule below: one bad verb
            # then silences every later request (measured 2026-11-23). Answer the error, keep polling. The id is
            # explicit because a handler may have raised before it bound its own answer.
            print(f"capture-server: {wanted!r} failed: {type(exc).__name__}: {exc}", flush=True)
            try:
                _finish({"stamp": _stamp(), "kind": "error", "request_id": request_id,
                         "error": f"{type(exc).__name__}: {exc}"})
            except Exception:
                pass

    def poll() -> None:
        if os.path.exists(os.path.join(root, "quit")):
            print("capture-server: quit requested", flush=True)
            QApplication.quit()
            return
        if state["warming"]:
            QTimer.singleShot(120, poll)
            return
        try:
            # Both sources: the legacy `request` slot and one file per request under `requests/<id>`. Reading
            # with replacement characters, keeping an empty file for the next tick and answering an unknown verb
            # all live in `drain_requests` - the reasons are there and in the tests.
            drain_requests(root, dispatch, state=state)
        except Exception as exc:
            # The reschedule BELOW is the channel, so it runs whatever the intake did: one request that surprises
            # the reader must cost that request, never every later one (measured 2026-11-24: a `UnicodeDecodeError`
            # inside the old read left the app answering nothing at all).
            print(f"capture-server: request intake failed: {type(exc).__name__}: {exc}", flush=True)
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

    def tab(spec: str) -> None:
        """Select an inner tab by name or index: `tab:<objectName>@<index|label>`.

        The process page's tabs (General / Encoder Options / Render Queue / ...) are a toolkit `TopTabHost` inside the
        page, not the outer nav stack - `page:` cannot reach them, and clicking a strip whose tab widths depend on the
        theme and the label text is a guess that reads as "the click did nothing" (the same failure the old
        `click:processSettingsContainer@25,25` workaround existed for). This answers with the index and the text so a
        probe can prove WHICH tab it got.
        """
        from PySide6.QtWidgets import QWidget

        target, _, which = spec.partition("@")
        host = window.findChild(QWidget, target.strip())
        report = {"stamp": _stamp(), "kind": "tab", "target": target.strip(), "requested": which.strip(),
                  "error": None, "index": None, "text": None}
        if host is None or not hasattr(host, "setCurrentIndex"):
            report["error"] = "widget not found" if host is None else "not a tab host"
            _finish(report)
            return
        index = None
        if which.strip().isdigit():
            index = int(which.strip())
        else:
            try:
                for candidate in range(host.count()):
                    if host.tabText(candidate).strip().lower() == which.strip().lower():
                        index = candidate
                        break
            except Exception as exc:
                report["error"] = f"{type(exc).__name__}: {exc}"
        if index is None or not 0 <= index < host.count():
            report["error"] = f"no tab {which.strip()!r} on {target.strip()} ({host.count()} tabs)"
            _finish(report)
            return
        host.setCurrentIndex(index)
        for _ in range(4):
            QApplication.processEvents()
        report["index"] = int(host.currentIndex())
        try:
            report["text"] = host.tabText(report["index"])
        except Exception:
            report["text"] = None
        _finish(report)
        print(f"capture-server: tab {target.strip()} -> {report['index']} ({report['text']!r})", flush=True)

    def scale(spec: str) -> None:
        """Set the toolkit's process-wide UI scale: `scale:1.5`.

        The scale is one factor for the whole window (`UiScale`), so judging it needs the SAME window before and
        after - which a re-run with a different environment cannot give. The answer echoes the factor actually in
        force (the toolkit clamps to its own range) and the resolved application-font pixel size, so a probe can
        tell "the number moved" from "the number moved and the text followed"; page overflow is judged by the
        existing `audit:` verb right after.
        """
        from PySide6.QtWidgets import QApplication

        from sli_ui_toolkit.ui.managers import UiScale

        report = {"stamp": _stamp(), "kind": "scale", "requested": spec.strip(), "error": None}
        manager = UiScale.get_instance()
        try:
            requested = float(spec.strip())
        except ValueError:
            report["error"] = f"not a number: {spec.strip()!r}"
            _finish(report)
            return
        report["before"] = float(manager.factor())
        manager.set_factor(requested)
        for _ in range(6):
            QApplication.processEvents()
        report["factor"] = float(manager.factor())
        # Qt reports exactly one of the two (-1 for the unused channel): the app face is a POINT size on this host.
        font = QApplication.font()
        report["applicationFontPx"] = int(font.pixelSize())
        report["applicationFontPt"] = round(float(font.pointSizeF()), 2)
        _finish(report)
        print(f"capture-server: scale {requested} -> {report['factor']}", flush=True)

    def load_input(path: str) -> None:
        """Set the process page's input file the way the file dialog does, without the dialog: `input:<path>`.

        The dialog's last step is `MainWindow.loadVideo(inputFile)`; the input field, the probe, the queue add and the
        Render Preview button's enablement all hang off that call. Without this verb a live probe cannot reach any of
        it: measured 2026-11-24, `render:` and `preview:` answer NOTHING at all - no payload, no error, no log line -
        while no input file is set, which reads exactly like a dead channel.

        The answer reports what the APP ended up with, not what was asked: `loadVideo` refuses a file it cannot probe
        and leaves the field empty, so the verb says so instead of pretending success.
        """
        report = {"stamp": _stamp(), "kind": "input", "requested": path, "error": None, "loaded": None}
        error, resolved = validate_input_path(path)
        if error:
            report["error"] = error
            _finish(report)
            print(f"capture-server: input {path!r} -> {error}", flush=True)
            return
        loader = getattr(window, "loadVideo", None)
        if loader is None:
            report["error"] = "the window has no loadVideo"
            _finish(report)
            return
        loader(resolved)
        for _ in range(8):
            QApplication.processEvents()
        field = getattr(getattr(window.page_handles, "general", None), "inputFileText", None)
        shown = field.text().strip() if field is not None and hasattr(field, "text") else None
        report["loaded"] = shown or None
        if not shown:
            report["error"] = "the app did not take the file (its probe rejected it); the input field is empty"
        _finish(report)
        print(f"capture-server: input {resolved} -> loaded={report['loaded']!r} error={report['error']!r}", flush=True)

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
        # A toolkit `Button` carries `_rows` too - a LIST of content rows, not the painted rows widget. Reading
        # `.x()` off it raised, and the exception took the whole poll loop (the channel) down with it (measured
        # 2026-11-23: `click:previewPauseButton@20,20` and then every later request timed out).
        if rows is not None and not isinstance(rows, QWidget):
            rows = None
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
        if scroll is not None and not isinstance(scroll, QWidget):
            scroll = None
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

    def hold(spec: str) -> None:
        """Press the left button and LEAVE IT DOWN: `hold:<objectName>@x,y` (pair it with `release:`).

        The LMB compare is a hold, not a click: the pane shows the source while the button is down and clears it on
        release (`VideoPreviewSurface.mousePressEvent`/`mouseReleaseEvent`), so a `click:` is unobservable - the flash is
        gone before a grab and the report would read "nothing happened". Measured 2026-11-24: the owner's compare report
        ("a couple of frames of desync, or the current upscale frame") had no instrument at all until this verb existed.
        """
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import QWidget

        target, _, coords = spec.partition("@")
        widget = window.findChild(QWidget, target.strip())
        report = {"stamp": _stamp(), "kind": "hold", "target": target.strip(), "error": None}
        if widget is None or "," not in coords:
            report["error"] = "widget not found" if widget is None else "expected x,y"
            _finish(report)
            return
        try:
            x, y = (int(value) for value in coords.split(",", 1))
        except ValueError:
            report["error"] = "point is not x,y"
            _finish(report)
            return
        point = widget.mapTo(window, QPoint(x, y))
        QTest.mousePress(window.windowHandle(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point)
        state["held_point"] = point
        for _ in range(4):
            QApplication.processEvents()
        report["point"] = [point.x(), point.y()]
        _finish(report)
        print(f"capture-server: hold {target.strip()}@({x},{y})", flush=True)

    def release(spec: str = "") -> None:
        """Release the button `hold:` left down. `release:` alone releases at the held point."""
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import QWidget

        report = {"stamp": _stamp(), "kind": "release", "error": None}
        point = state.pop("held_point", None)
        if point is None:
            report["error"] = "no button is held"
            _finish(report)
            return
        target, _, coords = spec.partition("@")
        widget = window.findChild(QWidget, target.strip()) if target.strip() else None
        if widget is not None and "," in coords:
            try:
                x, y = (int(value) for value in coords.split(",", 1))
                point = widget.mapTo(window, QPoint(x, y))
            except ValueError:
                pass
        QTest.mouseRelease(window.windowHandle(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point)
        for _ in range(4):
            QApplication.processEvents()
        _finish(report)
        print(f"capture-server: release at ({point.x()},{point.y()})", flush=True)

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
        window.page_handles.general.outputFileText.setText(out)

        def add(ticks: int = 0) -> None:
            probe = getattr(window, "_batchProbe", None)
            if probe is not None and probe.isRunning() and ticks < 200:
                QTimer.singleShot(100, lambda: add(ticks + 1))
                return
            window.addToRenderQueue()
            if window.page_handles.queue.renderQueueListWidget.count() == 0:
                # No op runs without a model selected; the probe is a measurement tool, so it states one instead of
                # reporting "0 rows" with no reason.
                report["error"] = "nothing added (no model selected?)"
                box = window.page_handles.general.upscaleCheckBox
                combo = window.page_handles.general.upscaleModelComboBox
                box.setChecked(True)
                if combo.count():
                    combo.setCurrentIndex(combo.count() - 1)
                window.addToRenderQueue()
            report["rows"] = window.page_handles.queue.renderQueueListWidget.count()
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
        widget = window.page_handles.queue.renderQueueListWidget
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

    def tail_state() -> None:
        """The live preview reader's own state: `tail:`.

        "The pane shows nothing" has four different causes - no reader, a file that will not open, a reader that
        opened and read nothing, a reader that is held by the transport - and they look identical from outside. This
        answer tells them apart without a screenshot.
        """
        tab = getattr(window, "processTab", None)
        tail = getattr(tab, "_live_tail", None) if tab is not None else None
        report = {"stamp": _stamp(), "kind": "tail", "path": getattr(tab, "_live_tail_path", None),
                  "wanted": bool(getattr(tab, "_live_tail_wanted", False)), "paused": None,
                  "published": None, "produced": None, "thread": None, "reader": None, "error": None}
        if tail is None:
            report["error"] = "no live tail"
            _finish(report)
            return
        reader = getattr(tail, "_reader", None)
        report.update({
            "paused": bool(getattr(tail, "paused", False)),
            "published": int(getattr(tail, "published_frames", 0) or 0),
            "produced": float(getattr(tail, "produced_seconds", 0.0) or 0.0),
            "thread": "running" if tail.isRunning() else "stopped",
            "reader": None if reader is None else {
                "path": getattr(reader, "_path", None),
                "size": getattr(reader, "_size", None),
                "emitted": getattr(reader, "_emitted", None),
                "last_pts": getattr(reader, "_last_pts", None),
                "exhausted": getattr(reader, "_exhausted", None),
                "open_failures": getattr(reader, "_open_failures", None),
                "decode_failures": getattr(reader, "_decode_failures", None),
                "container": type(getattr(reader, "_container", None)).__name__,
            },
        })
        _finish(report)
        print(f"capture-server: tail {report}", flush=True)

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
        from PySide6.QtCore import QPoint as _QPoint

        origin = target.mapToGlobal(_QPoint(0, 0))
        report.update({
            "class": type(target).__name__,
            "enabled": bool(target.isEnabled()),
            "visible": bool(target.isVisible()),
            "size": [target.width(), target.height()],
            # Where the widget actually is on the screen: a screenshot diff of "does this pane move" needs the
            # rectangle, and the window can be placed anywhere on the X root by the environment.
            "global": [origin.x(), origin.y()],
            # The window's own attributes: `opaque_paint` is the flag that used to make the shell skip repaints.
            "parent": type(target.parentWidget()).__name__ if target.parentWidget() else None,
            "parent_name": target.parentWidget().objectName() if target.parentWidget() else None,
            "parent_size": [target.parentWidget().width(), target.parentWidget().height()]
            if target.parentWidget() else None,
            "opaque_paint": bool(target.testAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)),
            "translucent": bool(target.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)),
            "autofill": bool(target.autoFillBackground()),
        })
        is_checked = getattr(target, "isChecked", None)
        if callable(is_checked):
            try:
                report["checked"] = bool(is_checked())
            except Exception:
                pass
        progress = getattr(target, "_progress", None)
        if progress:
            # A pane's two time bases: (rendered, total, playhead, duration). "A seek holds while the renderer
            # advances" is a claim about `playhead` versus `rendered`, so the answer must carry both.
            report["progress"] = list(progress)
            report["strip_visible"] = bool(getattr(target, "_strip_visible", False))
            strip_rect = getattr(target, "_strip_rect", None)
            if callable(strip_rect):
                try:
                    rect = strip_rect()
                    report["strip"] = [round(rect.x()), round(rect.y()), round(rect.width()), round(rect.height())]
                except Exception:
                    pass
        for attr in ("count", "currentRow", "selectedRows", "selectedCount", "value", "text",
                     "currentIndex", "currentText"):
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

    def handle(wanted: str, request_id: str | None = None) -> None:
        # The legacy single file carries its id inside the verb (`state:x#abc`); a file under `requests/<id>`
        # carries it in its NAME, and its content is a verb verbatim (so `preview:<clip>#model=futsuu` keeps its
        # own `#`). `drain_requests` passes the two forms differently and they must not be confused.
        if request_id is None and "#" in wanted:
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
        if wanted.startswith("rhi:"):
            # Live proof that a QRhiWidget renders inside THIS window (plan-rve-video-preview-widget: P0 item 4). The
            # probe is created in the running window, renders a few frames, writes a readback PNG and is removed again.
            # `rhi:` (no argument) is enough; the answer is written into the usual JSON answer file.
            try:
                from src.ui.devtools.rhi_probe import probe as rhi_probe

                argument = wanted[len("rhi:"):].strip()
                hold_ms = int(argument) if argument.isdigit() else 0
                # Optional placement: "rhi:hold@x,y" puts the probe where a screenshot can judge compositing.
                place = None
                if "@" in argument:
                    hold_text, _, coordinates = argument.partition("@")
                    hold_ms = int(hold_text) if hold_text.isdigit() else 0
                    try:
                        x_text, _, y_text = coordinates.partition(",")
                        place = (int(x_text), int(y_text))
                    except (TypeError, ValueError):
                        place = None
                report = rhi_probe(window, out_dir=root, hold_ms=hold_ms, place=place)
            except Exception as exc:  # the experiment failing is a result, not a broken control channel
                report = {"error": f"{type(exc).__name__}: {exc}"}
            _finish({"stamp": _stamp(), "kind": "rhi", **report})
            print(f"capture-server: rhi -> {report}", flush=True)
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
        if wanted.startswith("job:"):
            # Put one queue row into a given state so the row painter can be looked at without a render:
            # `job:<index>@<status>:<fraction>` e.g. `job:1@rendering:0.75`.
            spec = wanted[len("job:"):].strip()
            index_text, _, state_text = spec.partition("@")
            status, _, fraction_text = state_text.partition(":")
            jobs = window.renderQueue.getQueue()
            report = {"stamp": _stamp(), "kind": "job", "error": None, "index": index_text}
            try:
                index = int(index_text)
                job = jobs[index]
            except (ValueError, IndexError):
                report["error"] = "no such job"
                _finish(report)
                return
            window.renderQueue.mark_rendering(job)
            if status == "done":
                window.renderQueue.mark_finished(job)
            elif fraction_text:
                window.renderQueue.update_progress(job, float(fraction_text))
            window.page_handles.queue.renderQueueListWidget.update()
            for _ in range(4):
                QApplication.processEvents()
            report["state"] = (job.state.status, job.state.progress)
            _finish(report)
            print(f"capture-server: job {index} -> {report['state']}", flush=True)
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
        if wanted.startswith("page:"):
            # `page:proc|general|settings|download`: the page stack is switched by name, so a probe can reach a control
            # without guessing a menu button's coordinates.
            name = wanted[len("page:"):].strip().lower()
            call = {"home": "switchToHomePage", "proc": "switchToProcessingPage",
                    "settings": "switchToSettingsPage", "download": "switchToDownloadPage"}.get(name)
            if call is None or not hasattr(window, call):
                _finish({"stamp": _stamp(), "kind": "page", "page": name, "error": "unknown page"})
                return
            getattr(window, call)()
            QApplication.processEvents()
            _finish({"stamp": _stamp(), "kind": "page", "page": name, "shown": True})
            return
        if wanted.startswith("state:"):
            widget_state(wanted[len("state:"):])
            return
        if wanted.startswith("tail:"):
            tail_state()
            return
        if wanted.startswith("tab:"):
            tab(wanted[len("tab:"):])
            return
        if wanted.startswith("scale:"):
            scale(wanted[len("scale:"):])
            return
        if wanted.startswith("input:"):
            load_input(wanted[len("input:"):].strip())
            return
        if wanted.startswith("click:"):
            click(wanted[len("click:"):])
            return
        if wanted.startswith("drag:"):
            drag(wanted[len("drag:"):])
            return
        if wanted.startswith("hold:"):
            hold(wanted[len("hold:"):])
            return
        if wanted.startswith("release:"):
            release(wanted[len("release:"):])
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
        if wanted.startswith("preview:"):
            preview(wanted[len("preview:"):].strip())
            return
        error = page_request_error(wanted, by_name)
        if error is not None:
            # Never `capture("")`: an empty verb used to mean every page, so a request caught mid-write ran a
            # batch. A visible rejection bound to the request's id is the contract.
            _finish({"stamp": _stamp(), "kind": "error", "error": error})
            print(f"capture-server: rejected {wanted!r}: {error}", flush=True)
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

    def select_a_backend() -> None:
        """Prefer a backend that initialises in this runtime.

        The default is TensorRT, and its engine build blocks forever here (measured: the backend child sits in
        `futex_wait` at ~4 % CPU while the app waits on its stdout, ProcessTab.py:323). Torch/PyTorch runs.
        """
        combo = window.page_handles.general.backendComboBox
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

        The gate enables upscale and picks a model whose OWN scale is the one the GUI asks for (`upscaleScaleSpinBox`),
        not simply the first entry. The first entry here is a 4x SPAN, and a 2x request then ran it and downscaled its
        5760x4320 output: measured 668.7 ms/frame against 75.2 ms/frame for the installed 2x model on the same
        1440x1080 clip (8.9x, owner: "4x SPAN model использовал бы futsuu"). A model's scale is in its name (`2x...`).
        """
        import re as _re

        general = window.page_handles.general
        boxes = (general.interpolateCheckBox, general.upscaleCheckBox, general.deblurCheckBox,
                 general.denoiseCheckBox, general.decompressCheckBox)
        if any(box.isChecked() for box in boxes):
            return
        general.upscaleCheckBox.setChecked(True)
        combo = general.upscaleModelComboBox
        if combo.count() == 0:
            return
        if combo.currentIndex() >= 0:
            return
        wanted = None
        try:
            wanted = int(round(float(general.upscaleScaleSpinBox.value())))
        except Exception:
            pass
        best = 0
        if wanted and wanted >= 2:
            for index in range(combo.count()):
                match = _re.match(r"\s*(\d+)x", combo.itemText(index))
                if match and int(match.group(1)) == wanted:
                    best = index
                    break
        combo.setCurrentIndex(best)

    def render(video: str) -> None:
        """Queue and START a render through the same controls the user presses.

        Staged polling, not sleeps: the input field's `textChanged` runs `loadVideo`, which probes the file and can take
        seconds; pressing "Add to Render Queue" before that finishes leaves the queue empty (measured - the first
        version used a fixed 1.2 s delay and answered "the queue stayed empty").
        """
        stamp = _stamp()
        result = {"stamp": stamp, "kind": "render", "video": video, "backend": None, "queued": 0, "started": False,
                  "finished": False, "seconds": 0.0, "output": None, "error": None}
        # This verb answers from timers, seconds after the request: bind the answer's id now (see `preview`).
        result["request_id"] = state.get("request_id")
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
            combo = window.page_handles.general.backendComboBox
            if combo is not None and hasattr(combo, "currentText"):
                try:
                    text = combo.currentText()
                except Exception:
                    text = ""
                if text:
                    return str(text)
            return None

        def progress_sample() -> dict:
            """What the UI says about the job right now - a timeout without this cannot tell a slow model load from a
            frame loop that never advances, and the previous run's `finished: False` said only that."""
            bar = window.page_handles.proc.progressBar
            queue = getattr(window, "renderQueue", None)
            sample = {
                "bar": int(bar.value()),
                "bar_max": int(bar.maximum()),
                "queue": len(getattr(queue, "queue", []) or []) if queue is not None else None,
                "start_enabled": bool(window.page_handles.proc.startRenderButton.isEnabled()),
                "status": window.page_handles.proc.STATUS.text(),
                "backend": backend_of(),
            }
            return sample

        def wait_finish(ticks: int = 0) -> None:
            if ticks % 50 == 0:                  # one sample every 5 s, so 15 per minute of a long model load
                result.setdefault("trace", []).append({"t": round(time.time() - started_at, 1), **progress_sample()})
            if window.page_handles.proc.startRenderButton.isEnabled() and result["started"]:
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
                window.page_handles.proc.startRenderButton.click()
                QTimer.singleShot(250, wait_finish)
                return
            if ticks > 100:                      # 100 * 100 ms = 10 s
                finish("the queue stayed empty after Add to Render Queue")
                return
            QTimer.singleShot(100, lambda: press_start(ticks + 1))

        def press_add() -> None:
            select_a_backend()
            select_a_model()
            window.page_handles.general.addToRenderQueueButton.click()
            QTimer.singleShot(200, press_start)

        def wait_loaded(ticks: int = 0) -> None:
            info = window.page_handles.general.inputVideoInfoTextEdit
            loaded = bool(info.toPlainText().strip()) or getattr(window, "videoLength", 1) != 1
            if loaded:
                press_add()
                return
            if ticks > 150:                      # 15 s
                press_add()                      # let the app refuse on its own terms; its error is the evidence
                return
            QTimer.singleShot(100, lambda: wait_loaded(ticks + 1))

        window.page_handles.general.inputFileText.setText(video)
        QTimer.singleShot(300, wait_loaded)

    def preview(video: str) -> None:
        """START the preview render the Render Preview button starts (`RenderOptions.isPreview = True`).

        `render:` cannot reach the LMB compare: a plain render leaves the preview pane empty, so pressing LMB compares
        nothing (measured 2026-11-23: two root captures of the same tree while LMB was held were identical). The preview
        path lives only behind `renderPreviewBtn` -> `MainWindow.renderPreview`, and the control channel could not press
        a button by name - which left the whole preview surface outside the supervisor's instrument. This verb presses
        that button, then answers with the preview file and what the pane holds, so the LMB check has a loop.
        """
        from src.constants import TEMP_DOWNLOAD_PATH

        model_hint = None
        if "#model=" in video:
            video, _, model_hint = video.partition("#model=")
            model_hint = model_hint.strip() or None
        result = {"stamp": _stamp(), "kind": "preview", "video": video, "started": False, "finished": False,
                  "seconds": 0.0, "output": None, "pane": None, "output_bytes": None, "error": None}
        # Bind the answer to THIS request now: `_finish` publishes under the id it is given, and the verb answers
        # seconds later from a timer. Without this, a request that arrived meanwhile claims the preview's answer and
        # the caller sees a payload of the wrong kind (measured 2026-11-23: `preview:` answered "widget not found",
        # which belonged to a `state:` probe).
        result["request_id"] = state.get("request_id")
        started_at = time.time()
        expected = os.path.join(TEMP_DOWNLOAD_PATH, f"{os.path.basename(video)}_preview.mkv")

        def pane_state() -> dict | None:
            pane = window.page_handles.proc.previewLabel
            if pane is None:
                return None
            frame = getattr(pane, "_frame", None)
            return {"class": type(pane).__name__, "visible": bool(pane.isVisible()),
                    "has_frame": frame is not None,
                    "frame": [frame.width(), frame.height()] if frame is not None else None,
                    # (rendered, total, playhead, duration): the strip's two bases, so a live seek can be read
                    # from outside instead of guessed from a screenshot.
                    "progress": list(getattr(pane, "_progress", ()) or ())}

        def render_process():
            """The backend child lives on the ProcessTab, NOT on the window.

            The first refusal check read `window.renderProcess`, which does not exist - so `getattr(..., None)` was
            always None and a legitimate render was declared a refusal (measured 2026-11-23 22:05: a 12 s clip
            rendered 300 frames in ~14 s while the verb answered "the app refused the preview").
            """
            tab = getattr(window, "processTab", None)
            return getattr(tab, "renderProcess", None) if tab is not None else None

        def toast_texts() -> list:
            """Whatever the app is telling the user right now.

            A refusal ("Video is not loaded!", "End time must be greater than start time!", "Please select at least
            one model!") is delivered as a toolkit toast and reaches nothing else - the app log stays silent, which is
            why a refused preview looked like a job that started and vanished.
            """
            try:
                from .dialogs.toolkit_dialogs import _toast_manager_for

                manager = _toast_manager_for(window)
                texts = []
                for toast in getattr(manager, "_toasts", {}).values():
                    label = getattr(toast, "message_label", None)
                    text = label.text() if label is not None else ""
                    if text:
                        texts.append(str(text))
                return texts
            except Exception:
                return []

        def finish(error: str | None = None) -> None:
            result["seconds"] = round(time.time() - started_at, 1)
            for candidate in (expected, str(window.page_handles.general.outputFileText.text() or "")):
                if candidate and os.path.exists(candidate):
                    result["output"] = candidate
                    result["output_bytes"] = os.path.getsize(candidate)
                    break
            result["pane"] = pane_state()
            result["toasts"] = toast_texts()
            if error:
                result["error"] = error
            _finish(result)
            print(f"capture-server: preview {result}", flush=True)

        def confirm_refusal() -> None:
            """A click that leaves no backend and no disabled button did nothing - say why, in the app's own words."""
            if render_process() is None and window.page_handles.proc.startRenderButton.isEnabled():
                reason = "; ".join(toast_texts()) or "no toast was shown"
                finish(f"the app refused the preview: {reason}")
                return
            wait_finish(1)

        def wait_finish(ticks: int = 0) -> None:
            if ticks == 0 and render_process() is None \
                    and window.page_handles.proc.startRenderButton.isEnabled():
                # "started" is what the click asked for, not what happened: a refusal returns from the slot before
                # any backend exists and leaves the button enabled - indistinguishable from a render that finished
                # in under a second (measured: a preview answered "finished in 0.9 s, no output" for exactly this
                # reason while the real cause was a missing input file). Give the toast 600 ms, then read it.
                QTimer.singleShot(600, confirm_refusal)
                return
            if result["started"] and window.page_handles.proc.startRenderButton.isEnabled():
                result["finished"] = True
                finish()
                return
            if ticks > 3000:                     # 3000 * 100 ms = 5 min
                finish("timeout waiting for the preview render to finish")
                return
            QTimer.singleShot(100, lambda: wait_finish(ticks + 1))

        def press_preview() -> None:
            select_a_backend()
            select_a_model()
            if model_hint:
                # `preview:<clip>#model=futsuu`: the model a measurement runs with is part of the measurement. The
                # same clip and resolution gave 75.2 ms/frame on a 2x Anime model and 621.6 ms/frame on a 2x
                # animation model (measured 2026-11-23) - an unnamed model makes every number unrepeatable.
                combo = window.page_handles.general.upscaleModelComboBox
                for index in range(combo.count()):
                    if model_hint.lower() in combo.itemText(index).lower():
                        combo.setCurrentIndex(index)
                        result["model"] = combo.itemText(index)
                        break
                else:
                    result["model_hint_missed"] = model_hint
            elif getattr(window.page_handles.general, "upscaleModelComboBox", None) is not None:
                result["model"] = window.page_handles.general.upscaleModelComboBox.currentText()
            # The button lives on the processing page and so does the pane: pressing it from another page would start a
            # render nobody can see (and make the strip unclickable for a live seek proof).
            try:
                window.switchToProcessingPage()
            except Exception:
                pass
            button = window.page_handles.proc.renderPreviewBtn
            if button is None:
                finish("no renderPreviewBtn on the window")
                return
            result["started"] = True
            button.click()
            QTimer.singleShot(400, wait_finish)

        def wait_loaded(ticks: int = 0) -> None:
            info = window.page_handles.general.inputVideoInfoTextEdit
            loaded = bool(info is not None and info.toPlainText().strip())
            if loaded or ticks > 150:
                press_preview()
                return
            QTimer.singleShot(100, lambda: wait_loaded(ticks + 1))

        window.page_handles.general.inputFileText.setText(video)
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
