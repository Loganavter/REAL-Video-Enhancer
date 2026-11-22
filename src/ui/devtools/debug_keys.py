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

REPAINT_SHORTCUT = "Ctrl+Shift+R"
HIDE_SHORTCUT = "Ctrl+Shift+H"


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

    repaint_shortcut = QShortcut(QKeySequence(REPAINT_SHORTCUT), window)
    repaint_shortcut.activated.connect(_repaint)
    hide_shortcut = QShortcut(QKeySequence(HIDE_SHORTCUT), window)
    hide_shortcut.activated.connect(_hide_under_cursor)

    holder._repaint_shortcut = repaint_shortcut  # type: ignore[attr-defined]
    holder._hide_shortcut = hide_shortcut  # type: ignore[attr-defined]
    window._debug_keys = holder
    _log(window, f"debug keys: {REPAINT_SHORTCUT} = repaint the window, {HIDE_SHORTCUT} = hide the widget "
                 f"under the cursor")
    return holder
