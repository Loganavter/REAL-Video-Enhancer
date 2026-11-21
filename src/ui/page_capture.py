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
        tabs = _tab_host(page)
        if tabs is not None and tabs.count() > 1:
            for tab_index in range(tabs.count()):
                found.append((f"nav{index}-tab{tab_index}",
                              lambda t=tabs, j=tab_index: t.setCurrentIndex(j)))
    return found


def _tab_host(page):
    """The page's tab container: the stock `QTabWidget`, or the toolkit `TopTabHost` that replaced it.

    A migrated page has no `QTabWidget` left, and `findChild(QTabWidget)` then reports nothing - which silently
    dropped every `nav3-tabN` stem from the walk and made a tab-host conversion unmeasurable (measured on the settings
    page: the capture wrote `nav3` only, and the tab comparison ran against a page it never captured).
    """
    from PySide6.QtWidgets import QTabWidget, QWidget

    tabs = page.findChild(QTabWidget) if page is not None else None
    if tabs is not None:
        return tabs
    if page is None:
        return None
    # The toolkit host keeps the Designer's object name (`tabWidget`), so it is found by name, not by class.
    host = page.findChild(QWidget, "tabWidget")
    if host is not None and hasattr(host, "count") and hasattr(host, "setCurrentIndex"):
        return host
    return None


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

    def warm(index: int) -> None:
        """Make every page current once before anything is captured or dumped.

        A page's children keep their construction-time geometry until the page has been current at least once, so a
        single pass writes dumps where the *later* pages are 100x30 boxes - which the comparison then reports as
        "visible False -> True" and a whole-page geometry difference that does not exist (measured).
        """
        from PySide6.QtWidgets import QApplication

        if index >= len(steps_list):
            QTimer.singleShot(0, lambda: capture(0))
            return
        _name, action = steps_list[index]
        action()
        QApplication.processEvents()
        QTimer.singleShot(per_page, lambda: warm(index + 1))

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
        # One walk, both artefacts: a capture run that also asked for a layout dump writes the JSON for the same page
        # here instead of paying a second 40 s settle in a separate process (the two walks visit identical pages).
        try:
            from src.ui import layout_dump
            if layout_dump.enabled():
                layout_dump.write_page(window, name)
        except Exception as exc:
            print(f"capture: layout dump for {name} failed: {exc}", flush=True)
        QTimer.singleShot(0, lambda: capture(n + 1))

    QTimer.singleShot(delay, lambda: warm(0))


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
            # A RICH-TEXT pane's content is environment state, not interface: the Home page's changelog is fetched from
            # the network when it can be, so two runs of the same build show different notes and the page diff reads
            # 2.9% of "regression" that is a release note. Freeze such panes to one placeholder line.
            if hasattr(widget, "toPlainText") and os.environ.get("RVE_UI_STABLE_RICH", "1") == "1":
                widget.setPlainText("stable text")
                continue
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
