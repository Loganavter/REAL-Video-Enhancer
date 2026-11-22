"""Find the widget that paints a rectangle where it should not be - a layout audit, not a guess.

`debug_keys` answers "is this rectangle painted by anyone?" (repaint vs hide). When a widget IS painted there but is
clipped - the owner's case: the right ~118 px of a page keeps the previous page's pixels and the current page's controls
are cut exactly at that edge - neither test names the culprit, because the culprit is a widget whose own rect is SMALLER
than the children it lays out.

The audit walks the window and reports, for every widget:

* children whose geometry extends beyond the parent's own rect (the clipping signature - a parent that lost the width
  its children were laid out for), with the overflow in px;
* pages inside a `QStackedWidget` whose size differs from the stack's (the stale-page signature).

It writes `layout-audit-<stamp>.txt` next to `log.txt`, so the answer to "which widget is it" is one file, not a
screenshot argument. Armed with `RVE_UI_DEBUG_KEYS=1` (Ctrl+Shift+D while the artifact is on screen) or called directly
from a probe.
"""

from __future__ import annotations

import os
import time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QStackedWidget, QWidget


def _rect(widget: QWidget) -> str:
    geometry = widget.geometry()
    return f"{geometry.x()},{geometry.y()} {geometry.width()}x{geometry.height()}"


def _chain(widget: QWidget | None) -> str:
    parts: list[str] = []
    current = widget
    while current is not None:
        name = current.objectName()
        parts.append(f"{type(current).__name__}#{name}" if name else type(current).__name__)
        current = current.parentWidget()
    return " < ".join(parts)


def audit(window: QWidget) -> list[str]:
    """Return the report as lines; empty when nothing looks clipped or stale."""
    lines: list[str] = ["layout audit " + time.strftime("%Y-%m-%d %H:%M:%S"),
                        f"window {_rect(window)}"]
    problems = 0

    for parent in window.findChildren(QWidget):
        if parent.isWindow() or not parent.isVisible() or parent.width() <= 0:
            continue
        inner = parent.contentsRect()
        for child in parent.children():
            if not isinstance(child, QWidget) or child.isWindow() or not child.isVisible():
                continue
            goal = child.geometry()
            if goal.isNull():
                continue
            over_right = goal.right() - inner.right()
            over_bottom = goal.bottom() - inner.bottom()
            # A scroll area's viewport is SUPPOSED to hold content taller/wider than itself - that is scrolling, not
            # clipping. Only geometry that is not a scroll view's content is a finding.
            if parent.objectName() == "qt_scrollarea_viewport":
                continue
            if over_right > 1 or over_bottom > 1:
                problems += 1
                lines.append(
                    f"CLIPPED {type(parent).__name__}#{parent.objectName()} {_rect(parent)} "
                    f"contents={inner.width()}x{inner.height()} clips "
                    f"{type(child).__name__}#{child.objectName()} {_rect(child)} "
                    f"(over right {over_right}px, bottom {over_bottom}px)"
                )

    for stack in window.findChildren(QStackedWidget):
        # A stack that is itself hidden or not laid out (a collapsed tab host reports 100x30) says nothing: its pages
        # are fitted when it grows, and reporting them is noise that buries the one real finding.
        if not stack.isVisible() or min(stack.width(), stack.height()) < 50:
            continue
        size = stack.size()
        for index in range(stack.count()):
            page = stack.widget(index)
            if page is None or page.size() == size:
                continue
            if stack.currentWidget() is page:
                # The stack may inset its current page by a few px on purpose (RVE moved the Designer's 5/10 px padding
                # into the layout). A gross difference is the bug; a small one is chrome.
                if abs(size.width() - page.width()) <= 24 and abs(size.height() - page.height()) <= 12:
                    continue
            problems += 1
            shown = "current" if stack.currentWidget() is page else "hidden"
            lines.append(
                f"STALE-PAGE {type(stack).__name__}#{stack.objectName()} {size.width()}x{size.height()} "
                f"page[{index}] {_page_name(page)} {page.width()}x{page.height()} ({shown})"
            )

    lines.insert(1, f"{problems} problem(s)")
    return lines


def probe_at(window: QWidget, global_pos=None) -> list[str]:
    """Name the painter at a screen point, and compare what it WOULD paint with what is on the screen.

    The audit answers "is something clipped" (geometry). When geometry is correct and the screen still shows a stale
    strip, the question changes: does the widget paint the right pixel and the compositor show another? The rendered
    answer comes from `widget.grab()` - a fresh re-render of that widget - and the screen answer from
    `QScreen.grabWindow`, so the two can be compared directly.
    """
    from PySide6.QtGui import QCursor

    pos = global_pos or QCursor.pos()
    lines = [f"probe {time.strftime('%H:%M:%S')} at global {pos.x()},{pos.y()}"]

    widget = QApplication.widgetAt(pos) if QApplication.instance() is not None else None
    if widget is None:
        lines.append("  nothing under the cursor (QApplication.widgetAt returned None)")
        return lines

    lines.append(f"  chain: {_chain(widget)}")
    node = widget
    depth = 0
    while node is not None and depth < 8:
        local = node.mapFromGlobal(pos)
        inside = node.rect().contains(local)
        native = bool(node.internalWinId())
        lines.append(
            f"  [{depth}] {type(node).__name__}#{node.objectName()} {_rect(node)} "
            f"content={node.contentsRect().width()}x{node.contentsRect().height()} "
            f"local={local.x()},{local.y()} inside={inside} native_win={native} "
            f"opaque={bool(node.testAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent))} "
            f"translucent={bool(node.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground))} "
            f"autofill={bool(node.autoFillBackground())}"
        )
        node = node.parentWidget()
        depth += 1

    # Rendered truth: re-render the widget under the cursor and read the pixel at the cursor.
    try:
        local = widget.mapFromGlobal(pos)
        image = widget.grab().toImage()
        if 0 <= local.x() < image.width() and 0 <= local.y() < image.height():
            rendered = image.pixelColor(local.x(), local.y()).name()
            lines.append(f"  RENDERED pixel at the cursor: {rendered}")
        else:
            lines.append(f"  RENDERED: cursor outside the grabbed image ({image.width()}x{image.height()})")
    except Exception as error:
        lines.append(f"  RENDERED: grab failed ({error})")

    # Screen truth: what the compositor is showing at that point. A native Wayland client cannot read the screen, and
    # then the human is the instrument - the verdict line says which two colours to compare.
    try:
        screen = window.screen()
        pixmap = screen.grabWindow(0, pos.x(), pos.y(), 1, 1)
        on_screen = pixmap.toImage().pixelColor(0, 0).name() if not pixmap.isNull() else "n/a"
        lines.append(f"  SCREEN   pixel at the cursor: {on_screen}")
    except Exception as error:
        on_screen = "n/a"
        lines.append(f"  SCREEN: grabWindow unavailable ({error}) - this session cannot read the screen from inside the app")
    rendered_colour = rendered if "rendered" in locals() else "n/a"
    if on_screen not in ("n/a",) and rendered_colour != "n/a":
        if on_screen != rendered_colour:
            lines.append("  VERDICT: MISMATCH - the widget renders one colour and the screen shows another "
                         "(compositor / backing store, not layout)")
        else:
            lines.append("  VERDICT: the screen agrees with the render - a widget really paints this pixel")
    else:
        lines.append(f"  VERDICT: compare by eye - the probe says this pixel RENDERS as {rendered_colour}. If the block "
                     f"you see is a different colour, the compositor is showing a stale frame; if it is this colour, a "
                     f"widget paints it and the chain above names it.")
    # Anything still mapped ON TOP of the window: a menu or tooltip that did not go away keeps the area under it
    # unrepainted (Qt treats a covered region as somebody else's), and the symptom is "every control stops updating
    # until I click somewhere" - the owner's words (2026-11-22).
    lines.append("  popups / top-level windows:")
    try:
        app = QApplication.instance()
        popups = []
        for top in app.topLevelWidgets() if app is not None else []:
            if not top.isVisible() or top is window:
                continue
            popups.append(
                f"    {type(top).__name__}#{top.objectName()} {_rect(top)} "
                f"opaque={bool(top.testAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent))} "
                f"translucent={bool(top.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground))}"
            )
        lines.extend(popups or ["    (none)"])
        if app is not None:
            active_popup = app.activePopupWidget()
            lines.append(f"  activePopupWidget: "
                         f"{type(active_popup).__name__ if active_popup is not None else 'none'}")
            modal = app.activeModalWidget()
            lines.append(f"  activeModalWidget: {type(modal).__name__ if modal is not None else 'none'}")
    except Exception as error:
        lines.append(f"  popups: unavailable ({error})")

    try:
        ratio = window.devicePixelRatioF()
        handle = window.windowHandle()
        lines.append(f"  surface: devicePixelRatio={ratio:g} platform={QApplication.platformName()} "
                     f"window_handle={'yes' if handle is not None else 'no'}")
    except Exception:
        pass
    return lines


def _page_name(page: QWidget) -> str:
    name = page.objectName()
    return f"{type(page).__name__}#{name}" if name else type(page).__name__


def write_probe(window: QWidget, directory: str | None = None) -> str:
    """Write the cursor probe (see `probe_at`) next to the app's log; returns the file path."""
    directory = directory or os.getcwd()
    path = os.path.join(directory, f"layout-probe-{time.strftime('%Y%m%d-%H%M%S')}.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(probe_at(window)) + "\n")
    return path


def write(window: QWidget, directory: str | None = None) -> str:
    """Run the audit and write it next to the app's log; returns the file path."""
    directory = directory or os.getcwd()
    path = os.path.join(directory, f"layout-audit-{time.strftime('%Y%m%d-%H%M%S')}.txt")
    report = audit(window)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report) + "\n")
    return path
