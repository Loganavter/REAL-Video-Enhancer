"""Two debug hotkeys for "there is a rectangle on my screen that should not be there".

A stray rectangle has exactly two possible origins, and they need opposite fixes:

* nothing painted that area - a stale frame from before a resize/page switch, or a widget that intentionally paints
  nothing (`SurfaceLabel` with no token, `ImageLabel` with a null pixmap are `WA_TranslucentBackground`) letting whatever
  was there before show through;
* somebody paints it - a real widget or a painted plate.

`RVE_UI_DEBUG_KEYS=1` installs two keys that tell the two apart in one press each:

* **Ctrl+Shift+R** repaints the whole window from scratch. If the rectangle disappears, it was a stale frame.
* **Ctrl+Shift+H** hides the widget under the cursor (`QApplication.widgetAt`) and logs its chain. If the rectangle
  disappears with that widget hidden, it is inside that widget; if the rectangle stays, the painter is behind it.
* **Ctrl+Shift+D** writes `layout-audit-<stamp>.txt` next to `log.txt`: every widget whose own rect clips the children it
  lays out (with the overflow in px) and every stacked page whose size is not the stack's. That is the one answer the two
  keys above cannot give - "the rectangle IS painted, by a widget that lost 118 px of width".

Both log to the app log, so the chain is copyable without reading it off the screen. Environment-gated like the
inspector: they are diagnosis tools, not a shipping feature.
"""

from __future__ import annotations

import os

from PySide6.QtCore import QObject, Qt
from PySide6.QtGui import QCursor, QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication, QWidget

#: Environment variable that DISARMS the debug keys (`RVE_UI_DEBUG_KEYS=0`). They used to be opt-in, and the owner
#: pressed Ctrl+Shift+D on a normal launch: nothing happened, because the flag was not set - a diagnosis tool nobody
#: remembers to arm is a diagnosis tool that does not exist. Armed by default now; the shortcuts are obscure enough
#: (Ctrl+Shift+R/H/P/D) that they cannot be hit by accident.
ENV_FLAG = "RVE_UI_DEBUG_KEYS"

#: Forcing the compositor to blend the window instead of scanning it out directly.
ENV_FORCE_COMPOSITE = "RVE_UI_FORCE_COMPOSITE"

REPAINT_SHORTCUT = "Ctrl+Shift+R"
HIDE_SHORTCUT = "Ctrl+Shift+H"
#: Layout audit: names the widget whose own rect is SMALLER than the children it lays out (the clipping signature) and
#: every page inside a QStackedWidget whose size is not the stack's (the stale-page signature). It writes a file, so the
#: answer is copyable instead of argued over a screenshot.
AUDIT_SHORTCUT = "Ctrl+Shift+D"
#: Cursor probe: names the painter at the cursor and compares what it RENDERS (`grab()`) with what the screen shows
#: (`grabWindow`) - the pair that separates our painting from the compositor's.
PROBE_SHORTCUT = "Ctrl+Shift+P"


def enabled() -> bool:
    """True unless the host explicitly disarmed the keys (`RVE_UI_DEBUG_KEYS=0`)."""
    return str(os.environ.get(ENV_FLAG, "1")).strip().lower() not in ("0", "false", "no", "off")


def widget_chain(widget: QWidget | None) -> str:
    """"Class#name < Class#name < …" up to the window (a readable stand-in for a dump)."""
    parts: list[str] = []
    current = widget
    while current is not None:
        name = current.objectName()
        parts.append(f"{type(current).__name__}#{name}" if name else type(current).__name__)
        current = current.parentWidget()
    return " < ".join(parts) if parts else "(nothing under the cursor)"


def _app_root() -> str:
    """The directory of the running script (`REAL-Video-Enhancer.py`), i.e. where `log.txt` lives.

    Not `os.getcwd()`: launched from a desktop entry or another terminal the reports would land somewhere nobody looks.
    """
    import sys

    script = os.path.abspath(sys.argv[0]) if sys.argv and sys.argv[0] else os.getcwd()
    return os.path.dirname(script) or os.getcwd()


def _log(window, message: str) -> None:
    """Write into the app's own `log.txt` - that is the file a report is read from.

    It used to print to stdout, i.e. the terminal the app was launched from: the owner pressed the keys, saw nothing in
    `log.txt`, and the whole diagnosis stalled on an output stream nobody was reading (2026-11-22).
    """
    line = f"[debug-keys] {message}"
    try:
        from src.Util import log as app_log

        app_log(line)
        return
    except Exception:
        pass
    try:
        import os as _os

        with open(_os.path.join(_app_root(), "log.txt"), "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except Exception:
        print(line, flush=True)


def install(window) -> QObject | None:
    """Install the debug keys; returns the holder object (keeps the filter alive).

    One application-wide event filter rather than four `QShortcut`s: a shortcut only fires while its window is the active
    one, and it says nothing when it does NOT fire - so "I pressed it and nothing happened" could not be told apart from
    "the key never arrived". Every press is logged before it is handled.
    """
    if getattr(window, "_debug_keys", None) is not None:
        return None

    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QKeySequence

    class _Keys(QObject):
        def eventFilter(self, watched, event):  # noqa: N802 - Qt API
            if event.type() != QEvent.Type.KeyPress:
                return False
            sequence = QKeySequence(event.keyCombination())
            text = sequence.toString()
            try:
                if text not in _ACTIONS:
                    return False
            except Exception:
                return False
            _log(window, f"key {text}")
            try:
                _ACTIONS[text]()
            except Exception as error:                 # a broken probe must say so, not die silently
                _log(window, f"key {text} failed: {type(error).__name__}: {error}")
            return True

    def _repaint() -> None:
        _log(window, "repaint: full window")
        window.repaint()

    def _hide_under_cursor() -> None:
        widget = QApplication.widgetAt(QCursor.pos())
        if widget is None:
            _log(window, "hide: nothing under the cursor")
            return
        _log(window, f"hide: {type(widget).__name__}#{widget.objectName()} "
                     f"(chain: {widget_chain(widget)})")
        widget.hide()
        window.repaint()

    def _audit() -> None:
        from . import layout_audit

        path = layout_audit.write(window, _app_root())
        with open(path, encoding="utf-8") as handle:      # copy the verdict into the log: one press, one answer
            for line in handle.read().splitlines()[:12]:
                _log(window, f"audit: {line}")
        _log(window, f"audit written: {path}")

    def _probe() -> None:
        from . import layout_audit

        path = layout_audit.write_probe(window, _app_root())
        with open(path, encoding="utf-8") as handle:
            for line in handle.read().splitlines():
                _log(window, f"probe: {line}")
        _log(window, f"probe written: {path}")

    global _ACTIONS
    _ACTIONS = {
        REPAINT_SHORTCUT: _repaint,
        HIDE_SHORTCUT: _hide_under_cursor,
        AUDIT_SHORTCUT: _audit,
        PROBE_SHORTCUT: _probe,
    }

    app = QApplication.instance()
    if app is None:
        return None
    holder = _Keys(app)
    app.installEventFilter(holder)
    window._debug_keys = holder
    _log(window, f"debug keys armed: {REPAINT_SHORTCUT} = repaint, {HIDE_SHORTCUT} = hide under cursor, "
                 f"{AUDIT_SHORTCUT} = layout audit, {PROBE_SHORTCUT} = pixel probe (every press is logged)")
    return holder


_ACTIONS: dict = {}


def force_composite(window) -> bool:
    """Kept as the documented switch for the scanout test; the work itself happens in `MainWindowShell.build`.

    Marking the window translucent here was the first attempt and it forced compositing for the wrong reason: Qt gives a
    translucent window no frame, so the title bar - the only place to grab and move the window - disappeared (owner,
    2026-11-22). The window now keeps its decorations and simply withholds the opaque DECLARATION, which is the same test
    with the picture unchanged.
    """
    if not os.environ.get(ENV_FORCE_COMPOSITE):
        return False
    _log(window, "composite test: opaque region withheld (declaration skipped in MainWindowShell.build)")
    return True
