"""The process page applies the layout policy on the REAL shell: side while both panes fit, stack when they cannot.

The 15 cells are the research cell's measured sweep (5 factors x the app's minimum/design/widest windows). Every
expectation here is a number from that sweep, not a tolerance:

* page horizontal scroll is **0 in every cell** (it was 353 px at 2.0 in the design window before the policy);
* each pane keeps at least **320 px** of width (the preview's usability floor; 338 at 2.0 is its own hint);
* in `side` the column never goes under its own content floor (500 / 575 / 650 at 1.0 / 1.75 / 2.0), so the General
  tab's own inner horizontal scroll stays 0;
* in `stack` both panes take the body's full content width (viewport - 18 px of body margins) and the preview sits
  above the column.

The 1.0 identity is pinned separately: 500 px column, 541 px preview, a 1065x872 body, no page scroll of either axis,
the 450 px strip and its five tabs. That is the page the owner ships at 1.0, and this wave must not move it.

Everything is offscreen and builds the real `MainWindowShell`; the compiled resource blob is stubbed because a test
tree has no build step for it.
"""

from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("sli_ui_toolkit")

from PySide6.QtWidgets import QMainWindow  # noqa: E402

if "resources_rc" not in sys.modules:
    sys.modules["resources_rc"] = types.ModuleType("resources_rc")

from src.ui.layout_policy import BODY_INSET_PX, PANE_FLOOR_PX, SIDE, STACK  # noqa: E402

#: The window the sweep measures (the research cell's 900 px viewport; the design default is 700, too short to judge
#: the vertical end of the column).
WINDOW_HEIGHT = 900

#: The three widths: the app's minimum, the design default and the widest supported panel.
WINDOWS = (1050, 1163, 1366)

#: `(factor, window) -> (mode, expected column)`. The side columns are the policy's answer; the stack column is the
#: body's full content width, stated through the viewport in the test so the arithmetic is visible.
EXPECTED = {
    (1.0, 1050): (SIDE, 500),
    (1.0, 1163): (SIDE, 500),
    (1.0, 1366): (SIDE, 500),
    (1.25, 1050): (SIDE, 595),
    (1.25, 1163): (SIDE, 625),
    (1.25, 1366): (SIDE, 625),
    (1.5, 1050): (SIDE, 580),
    (1.5, 1163): (SIDE, 693),
    (1.5, 1366): (SIDE, 750),
    (1.75, 1050): (STACK, None),
    (1.75, 1163): (SIDE, 678),
    (1.75, 1366): (SIDE, 875),
    (2.0, 1050): (STACK, None),
    (2.0, 1163): (STACK, None),
    (2.0, 1366): (SIDE, 850),
}

#: The column's content floor per factor: the General tab starts scrolling inside itself below this.
COLUMN_FLOORS = {1.0: 500, 1.25: 500, 1.5: 500, 1.75: 575, 2.0: 650}


@pytest.fixture()
def process_state():
    """Leave the process-wide factor and language where they were: the singleton outlives a test."""
    from src.ui import i18n

    scale = _scale()
    factor_before = scale.factor()
    language_before = i18n.current_language()
    yield scale
    scale.set_factor(factor_before)
    i18n.configure_i18n(language_before)


def _scale():
    from sli_ui_toolkit.ui.managers.ui_scale import UiScale

    return UiScale.get_instance()


def _shell(qapp, factor: float, width: int):
    """The real window at `factor` and `width`, shown and settled; the page has applied its policy by then."""
    from src.ui import i18n, toolkit_boot, toolkit_theme
    from src.ui.shell.window import MainWindowShell

    _scale().set_factor(factor)
    i18n.configure_i18n("en")
    toolkit_boot.ensure()
    toolkit_theme.install(qapp)
    window = QMainWindow()
    shell = MainWindowShell()
    shell.build(window)
    window.resize(width, WINDOW_HEIGHT)
    window.show()
    _settle(qapp)
    # A show can land its own geometry first; state the measured box once more and let the page's resize hook run.
    window.resize(width, WINDOW_HEIGHT)
    _settle(qapp)
    return shell, window


def _settle(qapp, turns: int = 10) -> None:
    for _ in range(turns):
        qapp.processEvents()


def _viewport(page):
    return page.pageScroll.viewport()


@pytest.mark.parametrize("factor,window", sorted(EXPECTED))
def test_every_measured_cell_has_no_page_scroll_and_both_panes_usable(qapp, process_state, factor, window):
    """The 15-cell sweep: the policy's mode, the column, zero page horizontal scroll, panes >= 320, no inner scroll."""
    expected_mode, expected_column = EXPECTED[(factor, window)]
    shell, win = _shell(qapp, factor, window)
    try:
        page = shell.procPage_page
        area = page.pageScroll
        viewport = _viewport(page).width()

        assert page.layout_mode == expected_mode, (
            f"at factor {factor} in a {window} px window (viewport {viewport}) the page is {page.layout_mode!r}, "
            f"not {expected_mode!r}")
        if expected_mode == STACK:
            expected_column = viewport - BODY_INSET_PX
            assert page.policy_column == expected_column
            assert page.processContainer.width() == expected_column, (
                "a stacked column takes the body's whole content width")
            assert page.processInfoContainer.width() == expected_column, (
                "a stacked preview takes the same width as the column")
        else:
            assert page.policy_column == expected_column
            assert page.processContainer.width() == expected_column, (
                f"the side column is {page.processContainer.width()} px, not the policy's {expected_column}")
            assert page.processContainer.width() >= COLUMN_FLOORS[factor], (
                f"a side column below its {COLUMN_FLOORS[factor]} px content floor would make General scroll inside")
            assert page.processInfoContainer.isVisible()

        assert area.horizontalScrollBar().maximum() == 0, (
            f"the page scrolls horizontally by {area.horizontalScrollBar().maximum()} px at factor {factor}, "
            f"window {window}: the policy's whole purpose is that it never needs to")
        assert area.widget().width() <= viewport

        for name, pane in (("column", page.processContainer), ("preview", page.processInfoContainer)):
            assert pane.width() >= PANE_FLOOR_PX, (
                f"the {name} is {pane.width()} px at factor {factor}, window {window}; the floor is {PANE_FLOOR_PX}")
        assert page.generalSettings_page.scrollArea_4.horizontalScrollBar().maximum() == 0, (
            "the General tab scrolls inside itself: the column went under its content floor")
        from src.ui.devtools.layout_audit import audit

        start_clips = [line for line in audit(win) if "startRenderButton" in line]
        assert not start_clips, (
            "the Start button still hangs below its row at this factor:\n" + "\n".join(start_clips))
    finally:
        win.close()


def test_one_point_zero_is_unchanged_at_the_design_window(qapp, process_state):
    """The identity gate: the shipped 1.0 page - 500 / 541, a 1065x872 body, no scroll, a 450 px strip, five tabs."""
    shell, win = _shell(qapp, 1.0, 1163)
    try:
        page = shell.procPage_page
        area = page.pageScroll
        assert page.layout_mode == SIDE
        assert page.processContainer.width() == 500
        assert page.processInfoContainer.width() == 541, (
            "the preview at 1.0 is the remainder of the 1065 px body: 1065 - 18 (margins) - 6 (gap) - 500 = 541")
        assert (area.widget().width(), area.widget().height()) == (1065, 872), (
            f"the 1.0 body is {area.widget().width()}x{area.widget().height()}, not the shipped 1065x872")
        assert area.horizontalScrollBar().maximum() == 0
        assert area.verticalScrollBar().maximum() == 0, (
            "at 1.0 in a 900 px window the page body fits; a vertical range here would move the page")
        assert page._tab_strip.strip_fit().needed == 450
        assert page._tab_strip.strip_fit().fits
        assert page.processSettingsContainer.count() == 5
    finally:
        win.close()


def test_a_live_factor_change_reapplies_the_policy_without_a_restart(qapp, process_state):
    """`scale:<factor>` on a running window: 500 side -> 693 side -> 991 stacked, at the design width."""
    shell, win = _shell(qapp, 1.0, 1163)
    try:
        page = shell.procPage_page
        assert (page.layout_mode, page.processContainer.width()) == (SIDE, 500)

        process_state.set_factor(1.5)
        _settle(qapp)
        assert (page.layout_mode, page.processContainer.width()) == (SIDE, 693), (
            "at 1.5 the desired 750 + a 320 preview does not fit the 1037 px viewport, so the column yields to 693")

        process_state.set_factor(2.0)
        _settle(qapp)
        assert page.layout_mode == STACK
        assert page.processContainer.width() == _viewport(page).width() - BODY_INSET_PX
        assert page.pageScroll.horizontalScrollBar().maximum() == 0
    finally:
        win.close()


def test_a_window_drag_reapplies_the_policy_without_a_factor_change(qapp, process_state):
    """2.0 stacks at 1163 and fits side by side at 1366; only the window width moved."""
    shell, win = _shell(qapp, 2.0, 1163)
    try:
        page = shell.procPage_page
        assert page.layout_mode == STACK

        win.resize(1366, WINDOW_HEIGHT)
        _settle(qapp)
        assert page.layout_mode == SIDE, "at 1366 the capped 850 px column still covers its 650 px content floor"
        assert page.processContainer.width() == 850
        assert page.pageScroll.horizontalScrollBar().maximum() == 0

        win.resize(1163, WINDOW_HEIGHT)
        _settle(qapp)
        assert page.layout_mode == STACK
        assert page.pageScroll.horizontalScrollBar().maximum() == 0
    finally:
        win.close()
