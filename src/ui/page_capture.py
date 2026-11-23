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


def _page_stack(window):
    """The nav stack: the typed handle when the window carries one, else the stack's object name.

    Until 2026-11-23 the stack was also an attribute of the window (a page-attribute copy); the handles
    (`src/ui/pages/handles.py`) are the declared edge now, and the object-name fallback keeps a dump harness's stub
    window working.
    """
    from PySide6.QtWidgets import QStackedWidget

    handles = getattr(window, "page_handles", None)
    if handles is not None:
        return handles.menu.stackedWidget
    return window.findChild(QStackedWidget, "stackedWidget")


def steps(window) -> list[tuple[str, object]]:
    """Every page of the window as ``(name, action that shows it)``.

    One owner for page enumeration: the capture walker and the layout dumper must visit the same pages in the same
    order, or their tables stop describing the same interface.
    """
    from PySide6.QtWidgets import QTabWidget

    stack = _page_stack(window)
    if stack is None:
        return []
    found: list[tuple[str, object]] = []
    for index in range(stack.count()):
        page = stack.widget(index)
        found.append((f"nav{index}", lambda i=index: stack.setCurrentIndex(i)))
        tabs = _tab_host(page)
        if tabs is not None and tabs.count() > 1:
            for tab_index in range(tabs.count()):
                # The action selects the PAGE too: the walk runs `nav2` before `nav2-tab0`, so in-order capture hid the
                # omission, but a single `shot nav2-tab2` switched a tab on a page that was not on screen and the grab
                # was the previous page (measured twice).
                found.append((f"nav{index}-tab{tab_index}",
                              lambda i=index, t=tabs, j=tab_index: _show(stack, i, t, j)))
    return found


def _show(stack, page_index: int, tabs, tab_index: int) -> None:
    """Make `page_index` current, then its tab `tab_index`."""
    stack.setCurrentIndex(page_index)
    tabs.setCurrentIndex(tab_index)


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
    # The toolkit host keeps whatever object name the Designer gave its tab container - `tabWidget` on the settings
    # page, `processSettingsContainer` on the process page - so it is found by CLASS, not by name.
    for child in page.findChildren(QWidget):
        if type(child).__name__ == "TopTabHost" and hasattr(child, "setCurrentIndex"):
            return child
    return None


def is_current(window, name: str) -> bool:
    """Is the interface actually showing the page `name` (a walker stem like `nav2` or `nav3-tab1`)?

    The capture server used to apply a page's action and grab the very next event pass; a request right after the
    warm-up walk then photographed the PREVIOUS page while naming the file after the requested one - measured twice:
    `shot nav2-tab1` wrote the download page, and the comparison charged that mismatch to the branch.
    """
    match = re.fullmatch(r"nav(\d+)(?:-tab(\d+))?", name)
    stack = _page_stack(window)
    if match is None or stack is None:
        return False
    index = int(match.group(1))
    if stack.currentIndex() != index:
        return False
    tab_text = match.group(2)
    if tab_text is None:
        return True
    host = _tab_host(stack.widget(index))
    return host is not None and host.currentIndex() == int(tab_text)


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

    def snap(name: str, n: int, tries: int = 0) -> None:
        from PySide6.QtWidgets import QApplication

        if not is_current(window, name) and tries < 20:
            # The walk applies a page's action and grabs one tick later; a tab step used to photograph the page that was
            # current before it (measured on a targeted capture). Re-apply and wait instead of writing a wrong pair.
            _STEPS[n][1]()
            QApplication.processEvents()
            QTimer.singleShot(max(60, per_page // 4), lambda: snap(name, n, tries + 1))
            return
        if tries >= 20:
            print(f"capture: {name} never became current - writing it anyway", flush=True)
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


def _stabilise_state(window) -> None:
    """Pin the app's own start-up choices, so two builds are compared in the same state.

    The Process page probes the machine on start: which video encoder is selected (`libx264` on one boot, `x265_nvenc`
    on another), whether the reuse boxes are enabled, and the command line those produce. Two boots of the SAME tree
    therefore differ by 2-6 % of that page - measured: nav2-tab0 read 3.58 and 5.93 with an identical tree and an
    identical nav1 (0.00, the environment marker). The value of a measurement is the interface, not the probe, so the
    harness states the state it compares in: `RVE_UI_STABLE_STATE=0` disables this.
    """
    if os.environ.get("RVE_UI_STABLE_STATE", "1") != "1":
        return
    # The combo is not necessarily a `QComboBox`: a migrated page builds the toolkit's `ComboBox` (a `Button` subclass),
    # and an `isinstance` check silently skipped it - the encoder stayed on whatever the probe picked while the stock
    # build was pinned, which is a state difference the page table then charged to the migration.
    def _is_combo(candidate) -> bool:
        return all(hasattr(candidate, name) for name in ("count", "itemText", "currentIndex", "setCurrentIndex"))

    # Pin by TEXT where the text is the interface: the encoder list is built from what the machine reports, so index 0
    # is a different encoder in two builds (stock `libx264`, ours `x265_nvenc`) and the page then differs in the command
    # line it shows - a state difference, not a UI one.
    wanted = {"encoder": "libx264", "upscaleModelComboBox": None}
    handles = window.page_handles
    pinned = {"encoder": handles.encoder.encoder, "upscaleModelComboBox": handles.general.upscaleModelComboBox}
    for name, preferred in wanted.items():
        combo = pinned[name]
        if not _is_combo(combo) or combo.count() == 0:
            continue
        target = 0
        if preferred:
            for index in range(combo.count()):
                if combo.itemText(index) == preferred:
                    target = index
                    break
        if os.environ.get("RVE_UI_STABLE_DEBUG"):
            print(f"stabilise: {name} items={[combo.itemText(i) for i in range(combo.count())][:12]} "
                  f"current={combo.currentText()!r} target={target}", flush=True)
        if combo.currentIndex() != target:
            combo.setCurrentIndex(target)
            if os.environ.get("RVE_UI_STABLE_DEBUG"):
                print(f"stabilise: {name} -> {combo.currentText()!r}", flush=True)


def stabilise(window) -> None:
    """Neutralise live machine values in labels named by ``RVE_UI_STABLE_TEXT`` (comma-separated object names).

    The Home page's system-info label carries free disk space, so two runs of the SAME build differ by a digit - and
    when a digit changes the line wrap, the whole bottom band moves (measured: A/A over budget on nav0, 2.7% of the
    page, with nothing changed in the code). A measurement that cannot be repeated is not a measurement, so the
    harness zeroes the digits of the named labels before every grab. Both sides of a comparison get the same
    treatment, so the screenshots stay comparable; the value itself is not interface.
    """
    import re

    _stabilise_state(window)
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
