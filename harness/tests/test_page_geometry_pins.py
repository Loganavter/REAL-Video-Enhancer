"""The pins retired on 2026-11-28 follow the UI factor - or provably do not need to.

The static gate (`harness/scripts/ui_gates.py`, rule `raw-chrome-px`) reports an integer literal inside a geometry call,
so it can be satisfied by any edit that takes the literal out of the call - including one that moves the number into a
constant and never resolves it through `scaled_px`, or one that pins a rect nothing reads. The gate cannot tell those
apart, so this file is the other half: it builds the REAL shell at 1.0 and 2.0 and pins what each retired site does.

Three outcomes, and the reason each is the right one:

* **converted to a design length** - `startRenderButton`/`killRenderButton.maximumSize`. The Designer's 59x55 is host
  geometry, like the toolkit's own `size=(59, 55)` box, so it resolves at call time: 59x55 at 1.0, 118x110 at 2.0.
  The page used to pass the numbers inline (`scaled_px(59)`), which the gate still read as a raw literal; they are
  named design constants now, so the resolution point is visible in the source as well as on the widget.
* **a design value the toolkit owns** - `Button.iconSize`. `setIconSize` stores a design number and the style bridge
  multiplies it ONCE at paint time, so the page must pass it unscaled at every factor; multiplying here would apply
  the factor twice (the same contract `test_download_scale.py` measures for the download page's icon buttons).
* **not a pin at all** - the scroll areas' CONTENT rects (`setWidgetResizable(True)` sizes that widget to its viewport
  on the first layout, so the Designer's preview rect never reached the screen) and `interpolateModelComboBox`'s
  `setSizeIncrement(230, 0)` (`QWidget.sizeIncrement` only governs a user's interactive resize of a TOP-LEVEL window,
  and this combo is laid out by its row). Both are retired rather than scaled: pinning their "doubling" would pin a
  number nothing reads. The `settingsPage.scrollArea` maximum is that same shape - `112312` was the Designer's way of
  saying "no maximum", now the sentinel Qt defines for it (`16777215`, `QWIDGETSIZE_MAX`).

Everything is offscreen and builds the real `MainWindowShell`; the compiled resource blob is stubbed because a test tree
has no build step for it.
"""

from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("sli_ui_toolkit")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget, QWidget  # noqa: E402

from sli_ui_toolkit.ui.managers.ui_scale import UiScale, scaled_px  # noqa: E402

if "resources_rc" not in sys.modules:
    sys.modules["resources_rc"] = types.ModuleType("resources_rc")

from src.ui.pages.proc_page import (  # noqa: E402
    TRANSPORT_BUTTON_MAX_HEIGHT_PX,
    TRANSPORT_BUTTON_MAX_WIDTH_PX,
    TRANSPORT_GLYPH_HEIGHT_PX,
    TRANSPORT_GLYPH_WIDTH_PX,
)

#: The size every layout gate measures the shell at (`test_page_overflow.py`).
WINDOW_SIZE = (1163, 900)

#: Qt's `QWIDGETSIZE_MAX` - the value a maximum size states when the widget is free to grow.
QWIDGETSIZE_MAX = 16777215

#: The scroll contents whose `setGeometry(QRect(0, 0, w, h))` was retired, with the Designer's rect it stated.
RETIRED_CONTENT_RECTS = {
    "scrollAreaWidgetContents_2": (928, 302),
    "scrollAreaWidgetContents_4": (470, 1040),
    "scrollAreaWidgetContents": (1026, 652),
    "scrollAreaWidgetContents_3": (968, 528),
    "encoderSettings": (453, 581),
    "scrollAreaWidgetContents_6": (918, 48),
}

#: The two transport controls whose ceiling is a design length (`render_row_geometry` pins the row itself).
TRANSPORT_BUTTONS = ("startRenderButton", "killRenderButton")


def _settle(app: QApplication, turns: int = 3) -> None:
    for _ in range(turns):
        app.processEvents()


def _shell(app: QApplication, factor: float) -> QMainWindow:
    """The real window built AFTER the factor is set - every page resolves its design numbers at construction."""
    from src.ui import toolkit_boot, toolkit_theme
    from src.ui.shell.window import MainWindowShell

    UiScale.get_instance().set_factor(factor)
    window = QMainWindow()
    MainWindowShell().build(window)
    window.resize(*WINDOW_SIZE)
    window.show()
    _settle(app, 8)
    return window


def _measure(app: QApplication, window: QMainWindow) -> dict[str, dict]:
    """Walk every revealed page and read what each site states, while that page is the current one.

    A `QStackedWidget` page that is not current is 80-100 px wide and its children carry placeholder geometry, so each
    stack is revealed before it is walked (the same reason `test_page_overflow.py` does it). A measurement is taken only
    from the tree of the page that is current at that moment.
    """
    seen: dict[str, dict] = {}

    def note(widget: QWidget) -> None:
        name = widget.objectName()
        if name in RETIRED_CONTENT_RECTS:
            parent = widget.parentWidget()
            seen[name] = {
                "width": widget.width(),
                "height": widget.height(),
                "parent_class": type(parent).__name__ if parent is not None else None,
                "parent_name": parent.objectName() if parent is not None else None,
                "viewport_width": parent.width() if parent is not None else -1,
                "viewport_height": parent.height() if parent is not None else -1,
            }
        if name in TRANSPORT_BUTTONS:
            seen[name] = {
                "maximum": (widget.maximumWidth(), widget.maximumHeight()),
                "glyph": widget.property("iconSizePx"),
            }
        if name == "interpolateModelComboBox":
            increment = widget.sizeIncrement()
            seen[name] = {"increment": (increment.width(), increment.height())}
        if name == "scrollArea":
            parent = widget.parentWidget()
            if parent is not None and parent.objectName() == "settingsPage":
                seen["settingsPage.scrollArea"] = {
                    "maximum": (widget.maximumWidth(), widget.maximumHeight()),
                }

    def walk(widget: QWidget) -> None:
        note(widget)
        children = widget.findChildren(QWidget, options=Qt.FindChildOption.FindDirectChildrenOnly)
        for child in children:
            if isinstance(child, QStackedWidget):
                for index in range(child.count()):
                    child.setCurrentIndex(index)
                    _settle(app)
                    walk(child.widget(index))
            else:
                walk(child)

    stack = window.page_handles.menu.stackedWidget
    for index in range(stack.count()):
        stack.setCurrentIndex(index)
        _settle(app)
        walk(stack.widget(index))
    return seen


def test_every_retired_pin_follows_the_factor_or_says_why_it_does_not(qapp: QApplication):
    """1.0 is the number the page shipped; 2.0 is the same design value through `scaled_px` - or no pin at all."""
    from src.ui import toolkit_boot, toolkit_theme

    toolkit_boot.ensure()
    toolkit_theme.install(qapp)
    scale = UiScale.get_instance()
    factor_before = scale.factor()
    one_window = two_window = None
    expected_two = {}
    try:
        one_window = _shell(qapp, 1.0)
        one = _measure(qapp, one_window)
        two_window = _shell(qapp, 2.0)
        two = _measure(qapp, two_window)
        # Resolved while the factor is 2.0: the rule is `scaled_px` at call time, not "multiply by two" in the test.
        expected_two = {name: tuple(scaled_px(v) for v in one[name]["maximum"]) for name in TRANSPORT_BUTTONS}
    finally:
        for window in (one_window, two_window):
            if window is not None:
                window.close()
        scale.set_factor(factor_before)

    # --- a design length, resolved through `scaled_px` at call time -------------------------------------------------
    for name in TRANSPORT_BUTTONS:
        assert one[name]["maximum"] == (TRANSPORT_BUTTON_MAX_WIDTH_PX, TRANSPORT_BUTTON_MAX_HEIGHT_PX), (
            f"{name}.maximumSize at 1.0 is {one[name]['maximum']}, not the Designer's "
            f"{(TRANSPORT_BUTTON_MAX_WIDTH_PX, TRANSPORT_BUTTON_MAX_HEIGHT_PX)} - 1.0 must not move")
        assert one[name]["maximum"] == (59, 55), (
            f"{name}.maximumSize at 1.0 is {one[name]['maximum']}; the shipped number is 59x55")
        assert two[name]["maximum"] == expected_two[name], (
            f"{name}.maximumSize at 2.0 is {two[name]['maximum']}, not the 1.0 box through `scaled_px` "
            f"({expected_two[name]}): the ceiling stopped following the factor and the layout can centre the button "
            "in a cell it no longer matches")
        assert two[name]["maximum"] == (118, 110)

    # --- a design value the toolkit scales once, at paint time -------------------------------------------------------
    glyph = min(TRANSPORT_GLYPH_WIDTH_PX, TRANSPORT_GLYPH_HEIGHT_PX)     # `setIconSize` keeps the smaller side
    for name in TRANSPORT_BUTTONS:
        assert one[name]["glyph"] == two[name]["glyph"] == glyph, (
            f"{name} stores iconSizePx={two[name]['glyph']} at 2.0 and {one[name]['glyph']} at 1.0, but this is the "
            f"toolkit's design value ({glyph}) and the style bridge multiplies it once when it paints: scaling it here "
            "would apply the factor twice")

    # --- the "no maximum" sentinel: unbounded, and not what stops the content ----------------------------------------
    max_one = one["settingsPage.scrollArea"]["maximum"]
    max_two = two["settingsPage.scrollArea"]["maximum"]
    content = one["scrollAreaWidgetContents"]
    assert max_one == max_two == (QWIDGETSIZE_MAX, QWIDGETSIZE_MAX), (
        f"settingsPage.scrollArea maximumSize is {max_one} at 1.0 and {max_two} at 2.0; both dimensions are 'no "
        f"maximum' and must stay the sentinel, not become a design length")
    assert max_one[1] >= content["height"], (
        f"settingsPage.scrollArea caps its height at {max_one[1]} px around {content['height']} px of content")

    # --- not pins: the content's box is its viewport's, at every factor ---------------------------------------------
    for name, designer_rect in RETIRED_CONTENT_RECTS.items():
        for factor, measured in ((1.0, one), (2.0, two)):
            row = measured[name]
            assert row["parent_name"] == "qt_scrollarea_viewport", (
                f"{name} at {factor} is parented to {row['parent_name']!r}, not the scroll area's viewport - this "
                "site is only not-a-pin while the area resizes its content")
            assert row["width"] >= row["viewport_width"], (
                f"{name} is {row['width']} px wide in a {row['viewport_width']} px viewport at {factor}: the content "
                "is narrower than the box the user can see")
            assert (row["width"], row["height"]) != designer_rect, (
                f"{name} is exactly the retired Designer rect {designer_rect} at {factor} - the literal is being "
                "honoured again instead of the widget taking the size its scroll area gives it")
            assert row["height"] >= row["viewport_height"], (
                f"{name} is {row['height']} px tall in a {row['viewport_height']} px viewport at {factor}: the scroll "
                "area cannot reach its content")

    # The viewport follows the factor, so the content does too - this is what a pinned preview rect would freeze.
    assert two["scrollAreaWidgetContents_2"]["width"] != one["scrollAreaWidgetContents_2"]["width"], (
        "the download page's scroll content is the same width at 1.0 and 2.0 - it would be if the Designer's 928 px "
        "rect were honoured instead of the area's own sizing")
    assert two["encoderSettings"]["height"] > one["encoderSettings"]["height"], (
        "the Encoder Options content is the same height at 2.0 as at 1.0 while its rows doubled - the content is stuck "
        "at a pinned rect")

    # --- not a pin: `sizeIncrement` is a top-level-window property; a combo in a row never uses it -------------------
    increment_one = one["interpolateModelComboBox"]["increment"]
    increment_two = two["interpolateModelComboBox"]["increment"]
    assert increment_one == increment_two == (0, 0), (
        f"interpolateModelComboBox.sizeIncrement is {increment_one} at 1.0 and {increment_two} at 2.0; the Designer's "
        "(230, 0) was retired because a layout-managed combo never resizes by increment, and reintroducing it - scaled "
        "or not - would state a step nothing reads")
