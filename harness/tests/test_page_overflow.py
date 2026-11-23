"""A page may not contain a control that needs more width than the page can give it.

The defect this gate exists for (owner, 2026-11-24): the encoder-command field on the Encoder Options page was drawn
~900 px wide inside a ~480 px column and painted under the preview pane. Two independent decisions made it: the page
gave the field a 900 px minimum, and the glue re-set the field's width on every keystroke to the width of its text.
Neither is visible in any other gate - the page built, the suite was green, and the control simply left its box.

The class, stated once: **a widget's minimum width (an explicit `minimumSize` or the `minimumSizeHint` its content
produces) must fit the box it is laid out in.** The box is the nearest clipping ancestor - a scroll area's viewport,
because that is what the user can see at one time - and, for a page with no scroll area above the control, the page
itself. A control whose floor is larger than that box cannot be honoured by the layout: Qt gives it its minimum and the
widget is drawn outside the box it belongs to.

Measured at the window's own default size (``MainWindowShell`` resizes to 1163x700; every other layout gate measures
the larger 1163x900), with every page of every stack revealed in turn: a hidden `QStackedWidget` page is 80-100 px
wide because it was never laid out, and measuring it would report the placeholder instead of the page.

Run it against the tree at the commit before the cure and it names `EncoderCommand` and the content widget inside
`scrollArea_6`. That is the whole point: the next page that pins a control to its text width fails here, with the
object name, the number and the parent chain, not with a screenshot nobody diffs.
"""

from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("sli_ui_toolkit")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QAbstractScrollArea,
    QApplication,
    QMainWindow,
    QSizePolicy,
    QStackedWidget,
    QWidget,
)

#: The size the other layout gates (`harness/tests/test_page_fit_box.py`) measure the shell at.
DEFAULT_SIZE = (1163, 900)

#: The control the owner's report is about. It is pinned separately, below, because the generic gate names a row and
#: the row must stay recognisable.
ENCODER_COMMAND = "EncoderCommand"

#: How much of the parent chain a row carries: enough to see which container owns the offender, short enough to read.
CHAIN_DEPTH = 6


def _label(widget: QWidget) -> str:
    return widget.objectName() or type(widget).__name__


def _chain(widget: QWidget) -> str:
    parts: list[str] = []
    node: QWidget | None = widget
    while node is not None and len(parts) < CHAIN_DEPTH:
        parts.append(f"{_label(node)}[{node.width()}]")
        node = node.parentWidget()
    return " <- ".join(parts)


def _boot():
    """Build the shell at the default size, with every page laid out, and hand back the app and the window."""
    if "resources_rc" not in sys.modules:
        sys.modules["resources_rc"] = types.ModuleType("resources_rc")
    from src.ui import toolkit_boot, toolkit_theme
    from src.ui.shell.window import MainWindowShell

    app = QApplication.instance() or QApplication([])
    toolkit_boot.ensure()
    toolkit_theme.install(app)
    window = QMainWindow()
    MainWindowShell().build(window)
    window.resize(*DEFAULT_SIZE)
    window.show()
    for _ in range(8):
        app.processEvents()
    return app, window


def _settle(app, turns: int = 3) -> None:
    for _ in range(turns):
        app.processEvents()


def _page_overflows(app, window) -> tuple[list[str], list[str]]:
    """Walk every revealed page and return (too-wide-to-fit rows, hard-floor rows).

    A `QStackedWidget` page that is not current has never been laid out - it is 80-100 px wide and its children carry
    placeholder geometry - so the walk makes each page current before measuring it. Nested stacks (the settings tab
    host inside the proc page) are revealed the same way, because the outer page has to be current first.
    """
    too_wide: list[str] = []
    hard_floor: list[str] = []

    def box_for(widget: QWidget, page: QWidget) -> tuple[int, str]:
        node = widget.parentWidget()
        while node is not None and node is not page:
            if isinstance(node, QAbstractScrollArea):
                return node.viewport().width(), f"{_label(node)}.viewport"
            node = node.parentWidget()
        return page.width(), f"{_label(page)} (page)"

    def measure(widget: QWidget, page: QWidget) -> None:
        if widget is page or not widget.isVisible():
            return
        box, source = box_for(widget, page)
        if box <= 0:                       # a container that has not been sized yet says nothing about the page
            return
        minimum = widget.minimumSize().width()
        hint = max(widget.minimumSizeHint().width(), 0)   # an invalid hint is (-1, -1)
        needed = max(minimum, hint)
        if needed > box:
            too_wide.append(f"{_label(widget):<32} needs {needed:>4} px, box is {box:>4} px ({source}) :: "
                            f"{_chain(widget)}")
        parent = widget.parentWidget()
        layout = parent.layout() if parent is not None else None
        if layout is not None:
            margins = layout.contentsMargins()
            available = parent.width() - margins.left() - margins.right()
            if minimum > available > 0:
                hard_floor.append(f"{_label(widget):<32} minimumSize {minimum:>4} px, parent box is "
                                  f"{available:>4} px :: {_chain(widget)}")

    def walk(widget: QWidget, page: QWidget) -> None:
        measure(widget, page)
        children = widget.findChildren(QWidget, options=Qt.FindChildOption.FindDirectChildrenOnly)
        for child in children:
            if isinstance(child, QStackedWidget):
                for index in range(child.count()):
                    child.setCurrentIndex(index)
                    _settle(app)
                    walk(child.widget(index), child.widget(index))
            else:
                walk(child, page)

    stack = window.page_handles.menu.stackedWidget
    for index in range(stack.count()):
        stack.setCurrentIndex(index)
        _settle(app)
        walk(stack.widget(index), stack.widget(index))
    return too_wide, hard_floor


def _report(title: str, rows: list[str]) -> str:
    return f"{title} ({len(rows)}):\n" + "\n".join(f"  {row}" for row in rows)


def test_no_page_control_needs_more_width_than_its_box():
    """The gate: an explicit minimum or a content-driven hint that the page cannot honour is a control outside its page."""
    app, window = _boot()
    try:
        too_wide, hard_floor = _page_overflows(app, window)
    finally:
        window.close()
    assert not too_wide and not hard_floor, (
        "a control cannot fit the box its page gives it at "
        f"{DEFAULT_SIZE[0]}x{DEFAULT_SIZE[1]} - either it follows the layout, or the layout is wrong:\n"
        + _report("minimum width wider than the visible box", too_wide)
        + "\n"
        + _report("hard minimum wider than the parent's layout box", hard_floor)
    )


def _encoder_field(app, window) -> QWidget:
    handles = window.page_handles
    stack = handles.menu.stackedWidget
    stack.setCurrentIndex(2)                  # the proc page owns the Encoder Options tab
    _settle(app)
    encoders = [widget for widget in stack.widget(2).findChildren(QStackedWidget) if widget.count() == 5]
    assert encoders, "the Encoder Options tab host is gone - this pin is looking at nothing"
    page = None
    for host in encoders:
        for index in range(host.count()):
            if host.widget(index).findChild(QWidget, ENCODER_COMMAND) is not None:
                host.setCurrentIndex(index)
                _settle(app)
                page = host.widget(index)
    assert page is not None, f"no tab of the proc page holds a {ENCODER_COMMAND} field"
    return page.findChild(QWidget, ENCODER_COMMAND)


def test_the_gate_still_names_the_shape_it_was_written_for():
    """A gate that cannot fail is a comment. Re-state the defect on the live field and require the gate to name it.

    This is the record of the owner's report kept as a test: the field with a 900 px floor in a ~480 px column is
    `EncoderCommand needs 900 px, box is 404 px`, and the gate has to say so.
    """
    app, window = _boot()
    try:
        field = _encoder_field(app, window)
        field.setMinimumWidth(900)          # the floor `src/ui/pages/tab_6.py` used to state
        field.setFixedWidth(900)            # the text-driven width `src/ui/ProcessTab.py` used to set
        _settle(app)
        too_wide, _hard_floor = _page_overflows(app, window)
    finally:
        window.close()
    named = [row for row in too_wide if row.split()[0] == ENCODER_COMMAND]
    assert named, "the gate did not report the encoder field at a 900 px floor / 900 px width:\n" + _report(
        "minimum width wider than the visible box", too_wide)
    assert "900" in named[0] and f"{ENCODER_COMMAND}[900]" in named[0], (
        f"the row names neither the width nor the parent chain: {named[0]}")


def test_the_encoder_command_field_follows_its_column():
    """The control from the owner's report: at layout width, expanding, and its long text readable inside that width."""
    from PySide6.QtCore import Qt as QtCore_Qt

    app, window = _boot()
    try:
        field = _encoder_field(app, window)
        page = field.parentWidget()
        while page is not None and page.objectName() != "tab_6":
            page = page.parentWidget()
        assert page is not None, "the Encoder Options page root (tab_6) is gone"

        assert field.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding, (
            "the encoder-command field must follow its column; a text- or fixed-width policy is the defect this gate "
            "was written for")
        assert field.minimumWidth() <= page.width(), (
            f"the field's floor is {field.minimumWidth()} px in a {page.width()} px page - it can leave its column")
        assert field.width() <= page.width(), (
            f"the field is {field.width()} px wide in a {page.width()} px page - it is drawn outside its page")

        # A real encoder command is longer than the column. The field keeps it readable by scrolling its own text, so
        # the caret must stay inside the field's box when it sits at the end of the command (measured through Qt's IME
        # query - the toolkit line edit paints itself, it is not a QLineEdit).
        command = ("-c:v hevc_nvenc -preset p1 -tune hq -cq:v 18 -pix_fmt yuv420p10le "
                   "-c:a copy -c:s copy")
        field.setText(command)
        field.setCursorPosition(len(command))
        _settle(app)
        caret = field.inputMethodQuery(QtCore_Qt.InputMethodQuery.ImCursorRectangle)
        assert caret is not None and 0 <= caret.x() <= field.width(), (
            f"with a {len(command)}-character command the caret sits at x={caret.x() if caret else None} in a "
            f"{field.width()} px field - the text is clipped and the user cannot see the end of what they type")
        assert command in field.toolTip(), (
            "the field shows only the part of the command that fits, so the whole command has to be readable on hover "
            f"- the tooltip is {field.toolTip()!r}")
    finally:
        window.close()
