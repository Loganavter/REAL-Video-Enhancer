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

from PySide6.QtWidgets import QStackedWidget, QWidget


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


def _page_name(page: QWidget) -> str:
    name = page.objectName()
    return f"{type(page).__name__}#{name}" if name else type(page).__name__


def write(window: QWidget, directory: str | None = None) -> str:
    """Run the audit and write it next to the app's log; returns the file path."""
    directory = directory or os.getcwd()
    path = os.path.join(directory, f"layout-audit-{time.strftime('%Y%m%d-%H%M%S')}.txt")
    report = audit(window)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report) + "\n")
    return path
