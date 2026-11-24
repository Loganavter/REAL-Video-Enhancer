"""The process page's layout policy, pinned in the 15 cells the research cell measured.

The measured table lives here, not in prose: 5 factors x 3 window widths. The viewports are the real page's
(`pageScroll.viewport().width()` on the offscreen shell at 1163x900-equivalent windows, measured 2026-11-24); the pane
floors are `max(320, pane.minimumSizeHint().width())` (320 at 1.0-1.75, 338 at 2.0); the column floors are the General
tab's content floor (500 at 1.0-1.5, 575 at 1.75, 650 at 2.0 - 618 px content plus 32 of insets).

Expected columns, measured from the policy the owner chose:

* 1.0 - side at the design 500 everywhere;
* 1.25 - 595 at 1050 (the desired 625 + a 320 preview does not fit), 625 at 1163/1366;
* 1.5 - 580 at 1050, 693 at 1163 (the column yields so the preview keeps 320), 750 at 1366;
* 1.75 - stack at 1050 (891 full width), side at 1163 (678) and 1366 (875);
* 2.0 - stack at 1050 (878) and 1163 (991: the capped 647 is 3 px under the 650 content floor), side at 1366 (850).

A change to any cell is a product decision, not a tolerance: edit this table only with a measurement.
"""

from __future__ import annotations

import pytest

from src.ui.layout_policy import (
    BODY_INSET_PX,
    COLUMN_DESIGN_PX,
    PANE_FLOOR_PX,
    SIDE,
    STACK,
    column_content_floor,
    layout_decision,
    scaled_design_px,
)

#: The 3 window widths the research swept (the app's minimum, the design default and the widest supported panel).
WINDOW_WIDTHS = (1050, 1163, 1366)

#: The page viewport each (factor, window) produces on the real shell, measured offscreen. The window's chrome
#: (rail + page insets) grows with the factor while the window does not, which is why this is a table rather than a
#: subtraction of one constant.
VIEWPORTS = {
    1.0: {1050: 952, 1163: 1065, 1366: 1268},
    1.25: {1050: 939, 1163: 1052, 1366: 1255},
    1.5: {1050: 924, 1163: 1037, 1366: 1240},
    1.75: {1050: 909, 1163: 1022, 1366: 1225},
    2.0: {1050: 896, 1163: 1009, 1366: 1212},
}

#: The preview pane's floor per factor: its own minimum hint, never below the 320 px usability floor.
PANE_FLOORS = {1.0: 320, 1.25: 320, 1.5: 320, 1.75: 320, 2.0: 338}

#: The column's content floor per factor: where the General tab stops scrolling inside itself.
COLUMN_FLOORS = {1.0: 500, 1.25: 500, 1.5: 500, 1.75: 575, 2.0: 650}

#: `(factor, window) -> (mode, column_px)`. One row per measured cell, 15 in total.
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
    (1.75, 1050): (STACK, 891),
    (1.75, 1163): (SIDE, 678),
    (1.75, 1366): (SIDE, 875),
    (2.0, 1050): (STACK, 878),
    (2.0, 1163): (STACK, 991),
    (2.0, 1366): (SIDE, 850),
}


def _decide(factor: float, window: int) -> tuple[str, int]:
    return layout_decision(factor, VIEWPORTS[factor][window], PANE_FLOORS[factor], COLUMN_FLOORS[factor])


def test_the_table_covers_five_factors_by_three_windows():
    """The sweep is the evidence: a missing cell is a decision nobody measured."""
    assert len(EXPECTED) == 15
    assert {factor for factor, _window in EXPECTED} == set(VIEWPORTS) == set(PANE_FLOORS) == set(COLUMN_FLOORS)
    assert {window for _factor, window in EXPECTED} == set(WINDOW_WIDTHS)


@pytest.mark.parametrize("factor,window", sorted(EXPECTED))
def test_every_measured_cell_decides_as_measured(factor, window):
    """The policy's answer cell by cell, including the two cells the owner's brief names by hand."""
    mode, column = _decide(factor, window)
    assert (mode, column) == EXPECTED[(factor, window)], (
        f"at factor {factor} in a {window} px window (viewport {VIEWPORTS[factor][window]}) the policy answered "
        f"{(mode, column)}; measured {(EXPECTED[(factor, window)])}")


def test_one_point_zero_is_side_by_side_at_the_design_column_everywhere():
    """The identity half of the owner's decision: 1.0 must not move."""
    assert scaled_design_px(COLUMN_DESIGN_PX, 1.0) == 500
    for window in WINDOW_WIDTHS:
        assert _decide(1.0, window) == (SIDE, 500)


def test_two_point_zero_in_the_design_window_stacks_and_gives_both_panes_the_full_width():
    """The headline: 1163 at 2.0 is a stacked page, not a 353 px horizontal scroll."""
    assert _decide(2.0, 1163) == (STACK, 1009 - BODY_INSET_PX)
    # The side-by-side alternative would have kept only 647: three pixels under the General tab's 650 content floor.
    assert VIEWPORTS[2.0][1163] - BODY_INSET_PX - 6 - PANE_FLOORS[2.0] == 647 < COLUMN_FLOORS[2.0]


def test_one_point_seven_five_at_the_minimum_window_is_the_first_stack():
    """The cascade's edge cell: 565 px is under the 575 content floor, so the column cannot yield that far."""
    mode, column = _decide(1.75, 1050)
    assert (mode, column) == (STACK, 909 - BODY_INSET_PX)
    assert VIEWPORTS[1.75][1050] - BODY_INSET_PX - 6 - PANE_FLOORS[1.75] == 565 < COLUMN_FLOORS[1.75]


def test_two_point_zero_at_the_widest_window_still_fits_side_by_side():
    """Stacking is a last resort, not a permanent 2.0 mode: at 1366 the capped column does cover its floor."""
    mode, column = _decide(2.0, 1366)
    assert (mode, column) == (SIDE, 850)
    assert column >= COLUMN_FLOORS[2.0]
    assert VIEWPORTS[2.0][1366] - column - BODY_INSET_PX - 6 == PANE_FLOORS[2.0]


def test_the_capped_column_must_cover_its_content_floor_exactly():
    """The boundary is a floor, not a fudge: 650 fits, 649 stacks (the 2.0 decision is 3 px from the edge)."""
    assert layout_decision(2.0, 1012, 338, 650) == (SIDE, 650)
    assert layout_decision(2.0, 1011, 338, 650) == (STACK, 1011 - BODY_INSET_PX)


def test_the_desired_column_still_yields_to_the_preview_floor_before_stacking():
    """At 1.0's desired 500 the first branch needs 500+320+24 = 844 px of viewport; below that it caps, not stacks."""
    assert layout_decision(1.0, 844, 320, 500) == (SIDE, 500)
    assert layout_decision(1.0, 843, 320, 500) == (STACK, 843 - BODY_INSET_PX), (
        "843 - 24 - 320 = 499 is under the 500 floor, so this is already a stack; 844 is the last side cell")
    # A wider pane floor makes the same window yield sooner - the preview's floor is what the column yields to.
    assert layout_decision(1.0, 900, 400, 450) == (SIDE, 476)


def test_the_design_column_is_scaled_at_the_factor_in_force():
    """The policy resolves 500 through the factor, like the toolkit does (half to even)."""
    assert [scaled_design_px(COLUMN_DESIGN_PX, factor) for factor in (1.0, 1.25, 1.5, 1.75, 2.0)] == [
        500, 625, 750, 875, 1000]


def test_the_column_content_floor_is_the_general_tabs_measured_width_plus_its_insets():
    """618 px of General content at 2.0 plus 32 is the 650 the stack decision turns on."""
    assert column_content_floor(618) == 650
    assert column_content_floor(545) == 577
    # Before the first layout the measurement can read 0; the design floor is the answer, never 32.
    assert column_content_floor(0) == 500
