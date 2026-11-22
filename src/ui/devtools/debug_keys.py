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

#: Environment variable that arms the debug keys.
ENV_FLAG = "RVE_UI_DEBUG_KEYS"

#: Forcing the compositor to blend the window instead of scanning it out directly.
ENV_FORCE_COMPOSITE = "RVE_UI_FORCE_COMPOSITE"

REPAINT_SHORTCUT = "Ctrl+Shift+R"
HIDE_SHORTCUT = "Ctrl+Shift+H"
#: Layout audit: names the widget whose own rect is SMALLER than the children it lays out (the clipping signature) and
#: every page inside a QStackedWidget whose size is not the stack's (the stale-page signature). It writes a file, so the
#: answer is copyable instead of argued over a screenshot.
AUDIT_SHORTCUT = "Ctrl+Shift+D"


def enabled() -> bool:
    """True when the host asked for the debug keys (`RVE_UI_DEBUG_KEYS=1`)."""
    return bool(os.environ.get(ENV_FLAG))


def widget_chain(widget: QWidget | None) -> str:
    """"Class#name < Class#name < …" up to the window (a readable stand-in for a dump)."""
    parts: list[str] = []
    current = widget
    while current is not None:
        name = current.objectName()
        parts.append(f"{type(current).__name__}#{name}" if name else type(current).__name__)
        current = current.parentWidget()
    return " < ".join(parts) if parts else "(nothing under the cursor)"


def _log(window, message: str) -> None:
    """Use the host's log when it has one, else stderr - the app's log is where a report is read from."""
    logger = getattr(window, "log", None) or getattr(window, "_log", None)
    if callable(logger):
        logger(message)
        return
    print(f"[debug-keys] {message}", flush=True)


def install(window) -> QObject | None:
    """Install the hotkeys on `window`; returns the holder object (keeps the shortcuts alive)."""
    if getattr(window, "_debug_keys", None) is not None:
        return None

    holder = QObject(window)

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

        path = layout_audit.write(window)   # next to log.txt, like the SIGUSR1 stack dump
        with open(path, encoding="utf-8") as handle:      # copy the verdict into the log: one press, one answer
            for line in handle.read().splitlines()[:12]:
                _log(window, f"audit: {line}")
        _log(window, f"audit written: {path}")

    audit_shortcut = QShortcut(QKeySequence(AUDIT_SHORTCUT), window)
    audit_shortcut.activated.connect(_audit)
    holder._audit_shortcut = audit_shortcut  # type: ignore[attr-defined]

    repaint_shortcut = QShortcut(QKeySequence(REPAINT_SHORTCUT), window)
    repaint_shortcut.activated.connect(_repaint)
    hide_shortcut = QShortcut(QKeySequence(HIDE_SHORTCUT), window)
    hide_shortcut.activated.connect(_hide_under_cursor)

    holder._repaint_shortcut = repaint_shortcut  # type: ignore[attr-defined]
    holder._hide_shortcut = hide_shortcut  # type: ignore[attr-defined]
    window._debug_keys = holder
    _log(window, f"debug keys: {REPAINT_SHORTCUT} = repaint the window, {HIDE_SHORTCUT} = hide the widget "
                 f"under the cursor, {AUDIT_SHORTCUT} = write a layout audit")
    return holder


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
