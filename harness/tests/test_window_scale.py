"""The window's own geometry and the nav rail follow the UI scale - and 1.0 is exactly what it was.

This is the last wave of the UI-scale migration: typography already follows the factor (the toolkit's squared-size
`paint_font` bug is fixed upstream), the process column grows strictly 500 -> 750 -> 1000 and its page body scrolls
(wave 2). What was still frozen was the window itself: 1163x700, the 1050x700 minimum, the nav rail's 55 px cap, the
35 px icon boxes and the 10/5 page insets were literals, so at 2.0 the window stayed the 1.0 box and the roam rail's
buttons were CLIPPED (the toolkit's own hint is 72 px at 2.0 - the 36 px icon floor times the factor - against a 55 px
raw maximum).

The pins, factor by factor (measured offscreen on the real `MainWindowShell`, not derived):

* default size 1163x700 -> 1744x1050 -> 2326x1400 (`scaled_px` rounds 1163 * 1.5 = 1744.5 with Python's round);
* minimum 1050x700 -> 1366x700 -> 1366x700, and the cap is the policy: the honest content minimum would be the scaled
  2100x1400, wider than a 1920 laptop and taller than any of them, while the page body scrolls - so the minimum is a
  usability floor that stops at a 1366x768 laptop's client area;
* nav rail 54 -> 72 -> 90 px wide (the rail is its buttons plus the 18 px nav margins, which do not scale), the
  buttons' cap 55 -> 82 -> 110, and the icon box stays a DESIGN 35 at every factor because the toolkit's `sizeHint`
  and painter multiply it themselves (a scaled value here would paint a 140 px glyph at 2.0);
* page insets 10/5 -> 15/8 -> 20/10;
* the advanced settings page's gaps 11/9/6 -> 22/18/12, and its `setGeometry(460, 516)` measurement is gone.

Everything is offscreen and builds the real shell; the compiled resource blob is stubbed because a test tree has no
build step for it (`resources_rc` is the same stub `test_proc_page_scale.py` uses).
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("sli_ui_toolkit")

from PySide6.QtWidgets import QMainWindow  # noqa: E402

if "resources_rc" not in sys.modules:
    sys.modules["resources_rc"] = types.ModuleType("resources_rc")

from src.ui.shell import window as shell_window  # noqa: E402

#: Where the advanced settings page lives in the source, for the one static pin (a deleted statement has no runtime).
ADVANCED_SETTINGS = Path(__file__).resolve().parents[2] / "src" / "ui" / "pages" / "advanced_settings.py"

#: The window's own numbers at each factor, MEASURED on this tree. A change here is a geometry change and belongs in a
#: review, not in a tolerance.
DEFAULT_SIZE = {1.0: (1163, 700), 1.5: (1744, 1050), 2.0: (2326, 1400)}
MINIMUM_SIZE = {1.0: (1050, 700), 1.5: (1366, 700), 2.0: (1366, 700)}
PAGE_INSETS = {1.0: (10, 5), 1.5: (15, 8), 2.0: (20, 10)}
#: The nav buttons' own box (the toolkit's 36 px icon floor times the factor) and the rail that wraps them (that box
#: plus the nav's 2x9 px margins, which are not on the scale yet).
NAV_BUTTON_BOX = {1.0: (36, 36), 1.5: (54, 54), 2.0: (72, 72)}
NAV_BUTTON_CAP = {1.0: 55, 1.5: 82, 2.0: 110}
NAV_RAIL_WIDTH = {1.0: 54, 1.5: 72, 2.0: 90}

#: The advanced settings page's design gaps and their resolved values. 1.0 is the Designer's own statement
#: (11 body inset, 9 group gap, 6 row gap) - the identity pin for the page. `scaled_px` rounds halves to even
#: (9 * 1.5 = 13.5 -> 14, 11 * 1.5 = 16.5 -> 16).
ADVANCED_GAPS = {1.0: (11, 9, 6), 1.5: (16, 14, 9), 2.0: (22, 18, 12)}


@pytest.fixture()
def factor_state():
    """Leave the process-wide factor where it was: the singleton outlives a test and so does its window."""
    from src.ui import i18n
    from sli_ui_toolkit.ui.managers.ui_scale import UiScale

    scale = UiScale.get_instance()
    factor_before = scale.factor()
    language_before = i18n.current_language()
    yield scale
    scale.set_factor(factor_before)
    i18n.configure_i18n(language_before)


def _shell(qapp, factor: float):
    """The real window at `factor`, built while that factor is in force, shown and settled."""
    from src.ui import i18n, toolkit_boot, toolkit_theme
    from src.ui.shell.window import MainWindowShell
    from sli_ui_toolkit.ui.managers.ui_scale import UiScale

    i18n.configure_i18n("en")
    toolkit_boot.ensure()
    toolkit_theme.install(qapp)
    UiScale.get_instance().set_factor(factor)
    window = QMainWindow()
    builder = MainWindowShell()
    builder.build(window)
    window.show()
    _settle(qapp)
    return builder, window


def _settle(qapp, turns: int = 8) -> None:
    for _ in range(turns):
        qapp.processEvents()


def _size(widget) -> tuple[int, int]:
    return widget.width(), widget.height()


def _margins(widget) -> tuple[int, int]:
    margins = widget.contentsMargins()
    return margins.left(), margins.top()


def _nav_buttons(builder):
    return builder.homeBtn, builder.processBtn, builder.settingsBtn, builder.downloadBtn


# --- the 1.0 identity --------------------------------------------------------------------------------------------


def test_one_point_zero_is_the_window_it_was_before_this_wave(qapp, factor_state):
    """The gate: 1163x700, a 1050x700 minimum, the rail and the icon boxes exactly as shipped today."""
    builder, window = _shell(qapp, 1.0)
    try:
        assert _size(window) == DEFAULT_SIZE[1.0]
        assert (window.minimumSize().width(), window.minimumSize().height()) == MINIMUM_SIZE[1.0]
        assert _margins(builder.stackedWidget) == PAGE_INSETS[1.0]

        assert shell_window.NAV_RAIL_PX == 55, "the design constant must stay the Designer's number"
        assert shell_window.NAV_ICON_PX == 35
        for button in _nav_buttons(builder):
            assert button.maximumWidth() == NAV_BUTTON_CAP[1.0], (
                f"{button.objectName()} is capped at {button.maximumWidth()} px at 1.0, not 55")
            assert button.property("iconSizePx") == 35, (
                "the icon box is stated in DESIGN px - the toolkit multiplies it; a scaled value here would be "
                "scaled twice")
            assert _size(button) == NAV_BUTTON_BOX[1.0]
        assert builder.leftMenuSubContainer.width() == NAV_RAIL_WIDTH[1.0]
    finally:
        window.close()


def test_the_design_constants_are_resolved_at_call_time_not_at_import(factor_state):
    """A module-level `scaled_px(CONST)` would freeze the factor the process started with; these functions do not."""
    from sli_ui_toolkit.ui.managers.ui_scale import UiScale

    scale = UiScale.get_instance()
    scale.set_factor(1.0)
    default = shell_window.window_default_size()
    minimum = shell_window.window_minimum_size()
    assert (default.width(), default.height()) == (1163, 700)
    assert (minimum.width(), minimum.height()) == (1050, 700)

    scale.set_factor(1.5)
    assert shell_window.window_default_size().width() == 1744
    scale.set_factor(2.0)
    assert shell_window.window_default_size().width() == 2326
    assert shell_window.window_default_size().height() == 1400
    # `window` is imported once, at factor 1.0 in most runs - the numbers still moved, so nothing was frozen.
    assert shell_window.window_minimum_size().width() == 1366


# --- the raised factors ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("factor", (1.5, 2.0))
def test_the_window_opens_larger_and_the_rail_grows_with_it(qapp, factor_state, factor):
    builder, window = _shell(qapp, factor)
    try:
        assert _size(window) == DEFAULT_SIZE[factor], (
            f"at {factor} the window opened {_size(window)}, not {DEFAULT_SIZE[factor]}: the default size is still a "
            "raw literal")
        assert (window.minimumSize().width(), window.minimumSize().height()) == MINIMUM_SIZE[factor]
        assert _margins(builder.stackedWidget) == PAGE_INSETS[factor]

        for button in _nav_buttons(builder):
            assert button.property("iconSizePx") == 35, (
                f"{button.objectName()} stores a scaled icon box: the toolkit would scale it again and paint "
                f"{35 * factor:.0f} * {factor:.1f} px")
            assert button.maximumWidth() == NAV_BUTTON_CAP[factor]
            assert _size(button) == NAV_BUTTON_BOX[factor], (
                f"{button.objectName()} is {_size(button)} at {factor}: its box cannot grow past the raw 55 px cap")
        assert builder.leftMenuSubContainer.width() == NAV_RAIL_WIDTH[factor]
        assert builder.leftMenuSubContainer.width() > NAV_RAIL_WIDTH[1.0], "the rail must widen with the factor"
    finally:
        window.close()


def test_a_live_factor_change_restates_the_window_without_a_restart(qapp, factor_state):
    """The capture channel's `scale:<factor>` moves the real window - the follower must re-resolve, not freeze."""
    builder, window = _shell(qapp, 1.0)
    try:
        assert _size(window) == DEFAULT_SIZE[1.0]
        for step in (1.5, 2.0):
            factor_state.set_factor(step)
            _settle(qapp)
            assert _size(window) == DEFAULT_SIZE[step], (
                f"after a live switch to {step} the window is {_size(window)}, not {DEFAULT_SIZE[step]}")
            assert (window.minimumSize().width(), window.minimumSize().height()) == MINIMUM_SIZE[step]
            assert _margins(builder.stackedWidget) == PAGE_INSETS[step]
            assert builder.leftMenuSubContainer.width() == NAV_RAIL_WIDTH[step]

        factor_state.set_factor(1.0)
        _settle(qapp)
        assert _size(window) == DEFAULT_SIZE[1.0], "and back: 1.0 must be reachable again, exactly"
        assert _margins(builder.stackedWidget) == PAGE_INSETS[1.0]
    finally:
        window.close()


# --- the minimum-size policy ------------------------------------------------------------------------------------


def test_the_minimum_stops_at_a_laptop_and_the_column_still_fits_inside_it(qapp, factor_state):
    """The 2.0 minimum is 1366x700, not the scaled 2100x1400 - and the 1000 px column still fits at that floor.

    The policy: the minimum is a USABILITY floor, not a content floor. The page body scrolls (wave 2), so content that
    does not fit is reachable; a minimum larger than a laptop screen is a window a user cannot fit at all. What the cap
    may not do is squeeze the process column into a horizontal scroll of its own - hence the second half of this pin.
    """
    from sli_ui_toolkit.ui.managers.ui_scale import UiScale

    builder, window = _shell(qapp, 2.0)
    try:
        assert UiScale.get_instance().scaled_px(1050) == 2100, "the uncapped design minimum at 2.0"
        assert UiScale.get_instance().scaled_px(700) == 1400
        assert (window.minimumSize().width(), window.minimumSize().height()) == MINIMUM_SIZE[2.0]

        window.resize(window.minimumSize())
        _settle(qapp)
        assert window.width() == MINIMUM_SIZE[2.0][0]
        page = builder.procPage_page
        assert page.processContainer.width() == 1000
        assert page.pageScroll.viewport().width() >= 1000, (
            "at the minimum window the page area is narrower than the column: the cap would have traded a laptop-"
            "unfittable window for a horizontal scroll on the column itself")
    finally:
        window.close()


# --- the advanced settings page's borrowed geometry --------------------------------------------------------------


def test_the_advanced_page_no_longer_states_its_measurement_as_a_geometry():
    """`setGeometry(460, 516)` was the Designer canvas's measurement of a scroll-area child, not a design value.

    The scroll area is `setWidgetResizable(True)`, so it sets that widget's box from its viewport and the layout gives
    it its minimum; the statement was overwritten on the first resize. A static AST read is the honest pin here: the
    statement is deleted, so there is no runtime object to ask (a text search would also trip on the comment that
    records why it went).
    """
    tree = ast.parse(ADVANCED_SETTINGS.read_text(encoding="utf-8"))
    pinning = [node for node in ast.walk(tree)
               if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "setGeometry"]
    assert not pinning, (
        "the page states a geometry again: a scroll-area child's box comes from the area and the layout, not from a "
        "measurement")
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for alias in node.names}
    assert "QRect" not in imported, "the import went with the only statement that used it"


@pytest.mark.parametrize("factor", (1.0, 1.5, 2.0))
def test_the_advanced_page_gaps_follow_the_factor(qapp, factor_state, factor):
    """The page's insets and spacings are design gaps now, applied once at build and again on `scale_changed`."""
    inset, group, row = ADVANCED_GAPS[factor]
    builder, window = _shell(qapp, factor)
    try:
        page = builder.procPage_page.advancedSettings_page
        assert page.verticalLayout_33.spacing() == group
        assert page.verticalLayout_33.contentsMargins().left() == inset
        assert page.verticalLayout_33.contentsMargins().top() == inset
        assert page.verticalLayout_31.spacing() == row
        assert page.horizontalLayout_42.spacing() == row
        assert page.horizontalLayout_42.contentsMargins().left() == group
        assert page.horizontalLayout_16.spacing() == row
        assert page.horizontalLayout_16.contentsMargins().bottom() == group
        assert page.horizontalLayout_18.spacing() == row
        assert page.horizontalLayout_18.contentsMargins().right() == group
    finally:
        window.close()
