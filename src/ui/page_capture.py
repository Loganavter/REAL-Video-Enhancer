"""Capture every page of the app to PNG, so a migration can be measured instead of eyeballed.

Why: the per-screen loop was manual - launch on Xvfb, click the gear, screenshot, diff, click back. That is one
screen per session and it cannot answer "how much of the interface is migrated" at all. This walks the widget tree
the app itself builds (`stackedWidget` for the nav, each page's `QTabWidget` for its sub-tabs), grabs every page with
`QWidget.grab()` and names the file by position, then quits - so one run produces the whole table.

`grab()` is a render of the widget tree rather than a screen capture: no window-manager decoration, no partial
compositing, and it works for pages the nav never showed.

Enabled by ``RVE_UI_CAPTURE_DIR=<dir>``; the delay exists because RVE spends 30-40 s probing the backend before the
window is usable (a capture before that catches a half-applied settings page - measured).
"""

from __future__ import annotations

import os
import re

_STEPS: list[tuple[str, object]] = []


def enabled() -> bool:
    return bool(os.environ.get("RVE_UI_CAPTURE_DIR"))


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "page").lower()).strip("-") or "page"


def steps(window) -> list[tuple[str, object]]:
    """Every page of the window as ``(name, action that shows it)``.

    One owner for page enumeration: the capture walker and the layout dumper must visit the same pages in the same
    order, or their tables stop describing the same interface.
    """
    from PySide6.QtWidgets import QTabWidget

    stack = getattr(window, "stackedWidget", None)
    if stack is None:
        return []
    found: list[tuple[str, object]] = []
    for index in range(stack.count()):
        page = stack.widget(index)
        found.append((f"nav{index}", lambda i=index: stack.setCurrentIndex(i)))
        tabs = page.findChild(QTabWidget) if page is not None else None
        if tabs is not None and tabs.count() > 1:
            for tab_index in range(tabs.count()):
                found.append((f"nav{index}-tab{tab_index}",
                              lambda t=tabs, j=tab_index: t.setCurrentIndex(j)))
    return found


def install(window) -> None:
    """Arm the walker: it runs after ``RVE_UI_CAPTURE_DELAY_MS`` (default 40000), page by page, then quits."""
    from PySide6.QtCore import QTimer

    out_dir = os.environ["RVE_UI_CAPTURE_DIR"]
    os.makedirs(out_dir, exist_ok=True)
    delay = int(os.environ.get("RVE_UI_CAPTURE_DELAY_MS", "40000"))
    per_page = int(os.environ.get("RVE_UI_CAPTURE_STEP_MS", "260"))

    if os.environ.get("RVE_UI_STABLE_TEXT"):
        print(f"capture: stabilising {os.environ['RVE_UI_STABLE_TEXT']} before every grab", flush=True)

    steps_list = steps(window)
    if not steps_list:
        return
    _STEPS[:] = steps_list

    def capture(n: int) -> None:
        from PySide6.QtWidgets import QApplication

        if n >= len(_STEPS):
            QApplication.quit()
            return
        name, action = _STEPS[n]
        action()
        QApplication.processEvents()
        QTimer.singleShot(per_page, lambda: snap(name, n))

    def snap(name: str, n: int) -> None:
        stabilise(window)
        pixmap = window.grab()
        path = os.path.join(out_dir, f"{name}.png")
        pixmap.save(path)
        print(f"capture: {path} {pixmap.width()}x{pixmap.height()}", flush=True)
        QTimer.singleShot(0, lambda: capture(n + 1))

    QTimer.singleShot(delay, lambda: capture(0))


def stabilise(window) -> None:
    """Neutralise live machine values in labels named by ``RVE_UI_STABLE_TEXT`` (comma-separated object names).

    The Home page's system-info label carries free disk space, so two runs of the SAME build differ by a digit - and
    when a digit changes the line wrap, the whole bottom band moves (measured: A/A over budget on nav0, 2.7% of the
    page, with nothing changed in the code). A measurement that cannot be repeated is not a measurement, so the
    harness zeroes the digits of the named labels before every grab. Both sides of a comparison get the same
    treatment, so the screenshots stay comparable; the value itself is not interface.
    """
    import re

    names = [part.strip() for part in os.environ.get("RVE_UI_STABLE_TEXT", "").split(",") if part.strip()]
    if not names:
        return
    from PySide6.QtWidgets import QWidget

    for name in names:
        for widget in window.findChildren(QWidget, name):
            text = ""
            for getter in ("text", "toPlainText"):
                fn = getattr(widget, getter, None)
                if callable(fn):
                    try:
                        text = fn()
                    except Exception:
                        text = ""
                    break
            if not text:
                continue
            fixed = re.sub(r"\d", "0", text)
            if fixed == text:
                continue
            setter = getattr(widget, "setText", None) or getattr(widget, "setPlainText", None)
            if callable(setter):
                setter(fixed)
