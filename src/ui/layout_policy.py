"""The process page's layout policy: the column yields first, the layout stacks last.

Owner decisions (2026-11-24), stated as arithmetic so the page cannot invent a different answer per resize:

* **cascade, then stack** - while the column plus the preview's own floor fits the viewport, they sit side by side;
  when it does not, the column shrinks to whatever the preview's floor leaves it; only when even that would put the
  column under its own content floor does the layout stack (the preview band above the column);
* the desired column is the DESIGN 500 px resolved through the scale factor (500 / 750 / 1000), never an
  import-time snapshot;
* the preview's floor is its own minimum hint, raised to a 320 px usability floor (320 at 1.0-1.75, 338 at 2.0);
* the column's floor is its content floor, measured off the General tab's scroll content (618 px at 2.0) plus the
  column's insets (32): 500 at 1.0-1.5, ~575 at 1.75, 650 at 2.0. There is a dedicated helper for that sum so the
  page and the tests do not re-invent the 32.

Why a pure function and not page code: the decision is the thing to review, and it has to be pinned in 15 cells
(5 factors x 3 window widths) without a display. The Qt half only applies the returned pair.

The measured table (research cell, 2026-11-24; windows 1050/1163/1366, viewports 952/1065/1268 at 1.0 ... 896/1009/
1212 at 2.0) is pinned in `harness/tests/test_layout_policy.py`. The interesting cells are 1.5 at 1163 (the column
yields to 693 so the preview keeps 320), 1.75 at 1050 (the first stack) and 2.0 at 1163 (the design window stacks:
647 px is what the column could keep, 3 px under its 650 content floor).
"""

from __future__ import annotations

#: The process column's width in DESIGN px. The research and the owner both name 500 -> 750 -> 1000.
COLUMN_DESIGN_PX = 500

#: The page body's own horizontal frame: `gridLayout_3`'s 9 px margins plus the 6 px gap between the two columns.
#: The margin half is a constant of the page; the gap half is only paid in `side` mode.
BODY_INSET_PX = 18
PANE_GAP_PX = 6

#: The preview's usability floor. Below 320 px the picture is a strip, which is the squeeze the owner reported at 1.5.
PANE_FLOOR_PX = 320

#: The column's content floor is its measured content width (the General tab's scroll content) plus this pad - the
#: tab pane's insets and the scrollbar the content itself would grow.
COLUMN_FLOOR_PAD_PX = 32

#: The floor a narrow column never goes under, regardless of the measurement: the 1.0 content needs no more than 500.
COLUMN_FLOOR_MIN_PX = 500

#: The column content's DESIGN width: 325 design px -> 650 at 2.0, where the General tab's scroll content measures
#: 618 px and the column's insets add 32. This is the width-independent form of the measured floor. The measurement
#: alone is not safe as a policy input: it moves with the width it is measuring (at a 647 px column the same content
#: reflows to a 603 px hint), which would let the 2.0 cell decide `side` on one resize and `stack` on the next.
COLUMN_FLOOR_DESIGN_PX = 325

#: `mode` values the page and the tests share, as plain strings (this module has no Qt, not even an enum dependency).
SIDE = "side"
STACK = "stack"


def scaled_design_px(value: int | float, factor: float) -> int:
    """``value`` at ``factor``, rounded the way the toolkit's ``scaled_px`` rounds (half to even, minimum 1).

    Kept here rather than imported from the toolkit so the policy is testable on an interpreter without PySide6;
    ``round`` is the same rounding the toolkit uses, and the minimum-1 clamp is the same one.
    """
    return max(1, int(round(float(value) * float(factor))))


def column_content_floor(general_content_px: int | float) -> int:
    """The column width at which its own content stops scrolling inside itself.

    ``general_content_px`` is the General tab's scroll content's minimum width (618 px at 2.0 on the real page). The
    pad is the column's insets around it; the 500 px floor is the 1.0 content's own need, which the measurement can
    under-report before the page has been laid out once.

    Use this for a MEASURED floor (a test, a probe). The page's policy uses :func:`column_floor_for_factor`, because a
    live measurement moves with the width being tested.
    """
    return max(COLUMN_FLOOR_MIN_PX, int(general_content_px) + COLUMN_FLOOR_PAD_PX)


def column_floor_for_factor(factor: float) -> int:
    """The column's content floor for a factor: the 325 design px content resolved the same way as the column.

    650 at 2.0 (the measured 618 + 32), 569 at 1.75 (the research's ~575, measured at that factor's font metrics),
    500 at 1.0-1.5 (the measured content is narrower than the design floor). Width-independent on purpose: the
    content hint reflows with the column, so feeding the live hint back into the decision lets one resize flip the
    answer (measured: 618 px at a 1000 px column, 603 px at the 647 px column the same content would get).
    """
    return max(COLUMN_FLOOR_MIN_PX, scaled_design_px(COLUMN_FLOOR_DESIGN_PX, factor))


def layout_decision(factor: float, viewport_px: int, pane_floor_px: int, column_floor_px: int) -> tuple[str, int]:
    """``(mode, column_px)`` for this factor, viewport and the two pane floors.

    The three cases, in the owner's order:

    1. **side at the desired column** while ``desired + pane_floor + 24`` fits the viewport;
    2. **side with the column capped** at ``viewport - 24 - pane_floor`` while that still covers the column's content
       floor;
    3. **stack**, where both panes take the body's full content width (the viewport less the 18 px body frame; the
       6 px inter-column gap is not paid when there is one column).

    A non-positive viewport (a page built before its first show) answers the desired column in ``side``: the first
    real resize re-runs the policy, and guessing a stack from a zero-width box would flash one.
    """
    viewport = max(0, int(viewport_px))
    pane_floor = max(1, int(pane_floor_px))
    column_floor = max(1, int(column_floor_px))
    desired = scaled_design_px(COLUMN_DESIGN_PX, factor)
    side_cost = BODY_INSET_PX + PANE_GAP_PX

    if viewport <= 0:
        return SIDE, desired
    if desired + pane_floor + side_cost <= viewport:
        return SIDE, desired
    capped = viewport - side_cost - pane_floor
    if capped >= column_floor:
        return SIDE, capped
    return STACK, max(0, viewport - BODY_INSET_PX)


__all__ = [
    "BODY_INSET_PX",
    "COLUMN_DESIGN_PX",
    "COLUMN_FLOOR_DESIGN_PX",
    "COLUMN_FLOOR_MIN_PX",
    "COLUMN_FLOOR_PAD_PX",
    "PANE_FLOOR_PX",
    "PANE_GAP_PX",
    "SIDE",
    "STACK",
    "column_content_floor",
    "column_floor_for_factor",
    "layout_decision",
    "scaled_design_px",
]
