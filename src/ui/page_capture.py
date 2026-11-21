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


def install(window) -> None:
    """Arm the walker: it runs after ``RVE_UI_CAPTURE_DELAY_MS`` (default 40000), page by page, then quits."""
    from PySide6.QtCore import QTimer

    out_dir = os.environ["RVE_UI_CAPTURE_DIR"]
    os.makedirs(out_dir, exist_ok=True)
    delay = int(os.environ.get("RVE_UI_CAPTURE_DELAY_MS", "40000"))
    per_page = int(os.environ.get("RVE_UI_CAPTURE_STEP_MS", "260"))

    from PySide6.QtWidgets import QTabWidget

    stack = getattr(window, "stackedWidget", None)
    if stack is None:
        return
    steps: list[tuple[str, object]] = []
    for index in range(stack.count()):
        page = stack.widget(index)
        steps.append((f"nav{index}", lambda i=index: stack.setCurrentIndex(i)))
        tabs = page.findChild(QTabWidget) if page is not None else None
        if tabs is not None and tabs.count() > 1:
            for tab_index in range(tabs.count()):
                steps.append((f"nav{index}-tab{tab_index}",
                              lambda t=tabs, j=tab_index: t.setCurrentIndex(j)))
    _STEPS[:] = steps

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
        pixmap = window.grab()
        path = os.path.join(out_dir, f"{name}.png")
        pixmap.save(path)
        print(f"capture: {path} {pixmap.width()}x{pixmap.height()}", flush=True)
        QTimer.singleShot(0, lambda: capture(n + 1))

    QTimer.singleShot(delay, lambda: capture(0))
