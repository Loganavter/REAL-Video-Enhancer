"""The nav rail and the page insets follow the UI scale; the WINDOW does not - measured against BlinkTable.

This is the last wave of the UI-scale migration: typography already follows the factor (the toolkit's squared-size
`paint_font` bug is fixed upstream), the process column's DESIRED width grows strictly 500 -> 750 -> 1000 and its page
body scrolls (wave 2); the layout policy now decides whether that desired column fits beside the preview or the page
stacks (wave 3 of the same owner decision), so the column does not simply overrun the window at a raised factor. What
was still frozen was the window itself: 1163x700, the 1050x700 minimum, the nav rail's 55 px cap, the
35 px icon boxes and the 10/5 page insets were literals, so at 2.0 the window stayed the 1.0 box and the roam rail's
buttons were CLIPPED (the toolkit's own hint is 72 px at 2.0 - the 36 px icon floor times the factor - against a 55 px
raw maximum).

The pins, factor by factor (measured offscreen on the real `MainWindowShell`, not derived):

* default size 1163x700 at EVERY factor, and the same 1050x700 minimum: the window is design space, the content inside
  it scales (BlinkTable, measured 2026-11-24: `src/ui/layout_geometry.py:16` `resize_when_hidden=False` +
  `shared_toolkit/ui/layout_sizing.py:445`, plus a contract test that forbids `scaled_px` in a window minimum). The
  first version of this wave DID scale the default - 2326x1400 at 2.0, wider than a 2000 px screen - and that is the
  defect this policy removes;
* `fit_size(size, available)` clamps the default into the screen, tested with a fake rect rather than the machine's;
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
from src.ui.layout_policy import BODY_INSET_PX, STACK  # noqa: E402

#: Where the advanced settings page lives in the source, for the one static pin (a deleted statement has no runtime).
ADVANCED_SETTINGS = Path(__file__).resolve().parents[2] / "src" / "ui" / "pages" / "advanced_settings.py"

#: The window's own numbers at each factor, MEASURED on this tree. A change here is a geometry change and belongs in a
#: review, not in a tolerance.
#: The window and its minimum are DESIGN space: the same numbers at every factor, by policy (see the module docstring).
DEFAULT_SIZE = {1.0: (1163, 700), 1.5: (1163, 700), 2.0: (1163, 700)}
MINIMUM_SIZE = {1.0: (1050, 700), 1.5: (1050, 700), 2.0: (1050, 700)}
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



def _expected_window_size(window):
    """What the window must be: the design default clamped into ITS screen, then raised to its minimum by Qt.

    The offscreen platform reports an 800x800 screen, so the built window here is the 1050x700 minimum while a real
    desktop shows the design 1163x700 - the live lease proved that at 1.0. Asserting the design value directly would
    make this suite machine-dependent; the pure value is pinned by `test_the_window_size_is_design_space...` instead.
    """
    from PySide6.QtCore import QSize

    screen = window.screen()
    available = None if screen is None else screen.availableGeometry()
    fitted = shell_window.fit_size(QSize(*DEFAULT_SIZE[1.0]), available)
    minimum = shell_window.window_minimum_size()
    return (max(fitted.width(), minimum.width()), max(fitted.height(), minimum.height()))

def _nav_buttons(builder):
    return builder.homeBtn, builder.processBtn, builder.settingsBtn, builder.downloadBtn


# --- the 1.0 identity --------------------------------------------------------------------------------------------


def test_one_point_zero_is_the_window_it_was_before_this_wave(qapp, factor_state):
    """The gate: 1163x700, a 1050x700 minimum, the rail and the icon boxes exactly as shipped today."""
    builder, window = _shell(qapp, 1.0)
    try:
        assert _size(window) == _expected_window_size(window), (
            f"the 1.0 window is {_size(window)}; the design default clamped into this screen is {_expected_window_size(window)}")
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


def test_the_window_size_is_design_space_at_every_factor_and_the_rail_is_not(factor_state):
    """The window must NOT move with the factor; the rail and the insets must. One test, both halves.

    A module-level `scaled_px(CONST)` would freeze the factor the process started with - that is why the rail resolves
    at call time. The window is the opposite case on purpose: it is design space, so a factor change must leave it alone.
    """
    from sli_ui_toolkit.ui.managers.ui_scale import UiScale

    scale = UiScale.get_instance()
    for factor in (1.0, 1.5, 2.0):
        scale.set_factor(factor)
        default = shell_window.window_default_size()
        minimum = shell_window.window_minimum_size()
        assert (default.width(), default.height()) == (1163, 700), (
            f"at {factor} the window default became {default.width()}x{default.height()}: the window is design space")
        assert (minimum.width(), minimum.height()) == (1050, 700), (
            f"at {factor} the minimum became {minimum.width()}x{minimum.height()}: a scaled floor can exceed the screen")


def test_no_scale_call_reaches_the_window_size_functions():
    """The contract, mechanically: `window_default_size`/`window_minimum_size` must not resolve the factor.

    BlinkTable guards the same rule with `tests/contracts/test_window_minimum_size_design_space.py`; a scaled floor can
    exceed a user's screen height and a scaled default is how a 2x window ends up wider than the display.
    """
    tree = ast.parse(Path(shell_window.__file__).read_text(encoding="utf-8"))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name in ("window_default_size", "window_minimum_size"):
        node = functions[name]
        calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)]
        names = {getattr(c.func, "id", getattr(c.func, "attr", "")) for c in calls}
        assert "scaled_px" not in names, f"{name} resolves the scale: {sorted(names)}"
        assert "UiScale" not in {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}, (
            f"{name} touches UiScale directly")


def test_fit_size_clamps_an_oversized_window_into_the_screen():
    """The clamp is pure, so it is proven with a fake screen instead of whatever display runs the suite."""
    from PySide6.QtCore import QRect, QSize

    screen = QRect(0, 0, 1280, 800)
    assert shell_window.fit_size(QSize(2326, 1400), screen) == QSize(1280, 800), (
        "a stored or scaled rect larger than the screen must be clamped, not opened off-screen")
    assert shell_window.fit_size(QSize(1163, 700), screen) == QSize(1163, 700), "a fitting size is untouched"
    assert shell_window.fit_size(QSize(1163, 700), None) == QSize(1163, 700), (
        "no screen information (offscreen/headless) must not invent a bound")


# --- the raised factors ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("factor", (1.5, 2.0))
def test_the_window_keeps_its_size_and_the_rail_grows_inside_it(qapp, factor_state, factor):
    builder, window = _shell(qapp, factor)
    try:
        assert _size(window) == _expected_window_size(window), (
            f"at {factor} the window opened {_size(window)}, not the design default clamped into this screen "
            f"({_expected_window_size(window)}): a scaled window default is how an app ends up wider than the screen")
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


def test_a_live_factor_change_moves_the_rail_and_leaves_the_window_alone(qapp, factor_state):
    """The channel's `scale:<factor>` must re-resolve the rail, and must NOT resize the window.

    Resizing on a scale change would undo a size the user chose by hand; BlinkTable's main-window policy returns before
    `resize()` for exactly that reason.
    """
    builder, window = _shell(qapp, 1.0)
    try:
        assert _size(window) == _expected_window_size(window)
        fixed_size = _size(window)
        for step in (1.5, 2.0):
            factor_state.set_factor(step)
            _settle(qapp)
            assert _size(window) == fixed_size, (
                f"after a live switch to {step} the window resized to {_size(window)}: the window is design space")
            assert (window.minimumSize().width(), window.minimumSize().height()) == MINIMUM_SIZE[step]
            assert _margins(builder.stackedWidget) == PAGE_INSETS[step]
            assert builder.leftMenuSubContainer.width() == NAV_RAIL_WIDTH[step]

        factor_state.set_factor(1.0)
        _settle(qapp)
        assert _size(window) == fixed_size
        assert _margins(builder.stackedWidget) == PAGE_INSETS[1.0]
    finally:
        window.close()


# --- the minimum-size policy ------------------------------------------------------------------------------------


def test_the_minimum_is_design_space_and_the_page_stacks_at_two_point_zero(qapp, factor_state):
    """The minimum stays 1050x700 at 2.0, and the page does NOT scroll horizontally: it stacks.

    The policy is BlinkTable's: a minimum is a usability floor, never a content floor. At 2.0 the honest content
    minimum would be 2100x1400 - larger than any laptop - so the window keeps a floor a user can fit and the overflow
    lives in the page's own scroll surface. The layout policy is the other half: a 1000 px column plus a 338 px
    preview needs 1362 px of content in a 1009 px viewport, and the capped column (647) is 3 px under the General
    tab's 650 px content floor, so the page STACKS instead of asking for a 353 px horizontal scroll. This pin used to
    assert the opposite (`horizontalScrollBar().maximum() > 0`); it is rewritten with the owner's 2026-11-24 decision.
    """
    from sli_ui_toolkit.ui.managers.ui_scale import UiScale

    builder, window = _shell(qapp, 2.0)
    try:
        assert UiScale.get_instance().scaled_px(1050) == 2100, "the scaled design minimum at 2.0"
        assert UiScale.get_instance().scaled_px(700) == 1400
        assert (window.minimumSize().width(), window.minimumSize().height()) == MINIMUM_SIZE[2.0]

        page = builder.procPage_page
        viewport = page.pageScroll.viewport().width()
        assert page.layout_mode == STACK, (
            "at 2.0 in the design window the column and the preview cannot both fit - the page stacks")
        assert page.pageScroll.horizontalScrollBar().maximum() == 0, (
            "a 1000 px column plus the preview does not fit the 1163 px window at 2.0; the answer is a stacked "
            "page, not a 353 px horizontal scroll")
        assert page.processContainer.width() == viewport - BODY_INSET_PX, (
            "a stacked column takes the body's whole content width")
        assert page.processInfoContainer.isVisible()
        assert page.processInfoContainer.width() == page.processContainer.width(), (
            "the stacked preview band is as wide as the column under it")
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
