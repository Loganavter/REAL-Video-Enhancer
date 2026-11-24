"""The stacked preview band and the sticky render row (owner, 2026-11-24).

The owner's three decisions, pinned on the real shell:

* when the policy stacks, the preview is a **band above the column** whose height is **16:9 of its width** (973x547 in
  the design window at 2.0), never a share of the window;
* the band has a **collapse toggle**; collapsed, the preview is gone and the column gets the body's whole width. In side
  mode that is a visible widening (500 -> 1047 at 1.0); in stacked mode the column already had the full width, and
  collapsing reclaims the vertical space. The toggle itself floats on the page (not inside the band) so a collapsed
  band can always be expanded again;
* the **render row** (Start / pause / kill / progress bar + detail) moves into a **sticky footer** in stacked mode: the
  footer is a child of the page viewport, so scrolling the body does not move it. In side mode the row stays in the
  column's flow and the footer is hidden.

The pane's ownership (`ProcessTab._queue_owns_pane`) is not touched: the preview widget is never re-parented, only its
container's grid cell changes, so the queue tab keeps painting into the same pane.

Everything is offscreen and builds the real `MainWindowShell`; the compiled resource blob is stubbed because a test
tree has no build step for it.
"""

from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("sli_ui_toolkit")

from PySide6.QtCore import QPoint  # noqa: E402
from PySide6.QtWidgets import QMainWindow  # noqa: E402

if "resources_rc" not in sys.modules:
    sys.modules["resources_rc"] = types.ModuleType("resources_rc")

from src.ui.layout_policy import BODY_INSET_PX, SIDE, STACK  # noqa: E402

WINDOW_SIZE = (1163, 900)


@pytest.fixture()
def factor_state():
    """Leave the process-wide factor and language where they were."""
    from src.ui import i18n
    from sli_ui_toolkit.ui.managers.ui_scale import UiScale

    scale = UiScale.get_instance()
    factor_before = scale.factor()
    language_before = i18n.current_language()
    yield scale
    scale.set_factor(factor_before)
    i18n.configure_i18n(language_before)


def _shell(qapp, factor: float, size=WINDOW_SIZE):
    from src.ui import i18n, toolkit_boot, toolkit_theme
    from src.ui.shell.window import MainWindowShell
    from sli_ui_toolkit.ui.managers.ui_scale import UiScale

    UiScale.get_instance().set_factor(factor)
    i18n.configure_i18n("en")
    toolkit_boot.ensure()
    toolkit_theme.install(qapp)
    window = QMainWindow()
    shell = MainWindowShell()
    shell.build(window)
    window.resize(*size)
    window.show()
    _settle(qapp)
    window.resize(*size)
    _settle(qapp)
    return shell, window


def _settle(qapp, turns: int = 10) -> None:
    for _ in range(turns):
        qapp.processEvents()


def _page(shell):
    return shell.procPage_page


def _viewport(page):
    return page.pageScroll.viewport()


def _row_bottom_in_page(page) -> int:
    """The render row's bottom edge in the page's coordinates (where a sticky footer must sit)."""
    row = page.renderRow
    return row.mapTo(page.procPage, QPoint(0, 0)).y() + row.height()


def _viewport_bottom_in_page(page) -> int:
    viewport = _viewport(page)
    return viewport.mapTo(page.procPage, QPoint(0, 0)).y() + viewport.height()


# --- the band's 16:9 height ----------------------------------------------------------------------------------------


def test_the_stacked_band_is_sixteen_by_nine_of_its_width(qapp, factor_state):
    """At 2.0 in the design window the band is 973x547 - 16:9 of its own width, not a window share."""
    shell, window = _shell(qapp, 2.0)
    try:
        page = _page(shell)
        assert page.layout_mode == STACK, "this pin needs the stacked layout at 2.0"
        band = page.previewBand
        assert band.isVisible()
        assert (band.width(), band.height()) == (973, 547), (
            f"the band is {band.width()}x{band.height()}, not the measured 973x547")
        assert band.height() == round(band.width() * 9 / 16), (
            "the band's height must be 16:9 of its own width at every stacked cell")
        # The band is above the column: its bottom edge is at or above the column's top edge.
        band_bottom = band.mapTo(page.procPage, QPoint(0, band.height())).y()
        column_top = page.processContainer.mapTo(page.procPage, QPoint(0, 0)).y()
        assert band_bottom <= column_top, "the preview band must be ABOVE the column, not beside it"
    finally:
        window.close()


# --- the collapse toggle --------------------------------------------------------------------------------------------


def test_the_toggle_collapses_the_band_and_the_column_takes_the_whole_width(qapp, factor_state):
    """Stacked: collapse hides the preview; the toggle stays reachable and expands it again."""
    shell, window = _shell(qapp, 2.0)
    try:
        page = _page(shell)
        toggle = page.previewBandToggle
        assert toggle.isVisible(), "the collapse toggle is the band's control and must be on screen"
        full = _viewport(page).width() - BODY_INSET_PX

        toggle.clicked.emit()
        _settle(qapp)
        assert page.preview_collapsed
        assert not page.processInfoContainer.isVisible(), "the collapsed band is hidden"
        assert page.processContainer.width() == full, (
            "a hidden band must leave the column the body's whole width")
        assert toggle.isVisible(), (
            "the toggle floats on the page, so a collapsed band can always be expanded again")

        toggle.clicked.emit()
        _settle(qapp)
        assert not page.preview_collapsed
        assert page.processInfoContainer.isVisible()
        assert page.layout_mode == STACK
        assert page.previewBand.height() == round(page.previewBand.width() * 9 / 16)
    finally:
        window.close()


def test_collapsing_in_side_mode_widens_the_column(qapp, factor_state):
    """The side half of the same rule: the column grows from its 500 px design width to the full body width."""
    shell, window = _shell(qapp, 1.0)
    try:
        page = _page(shell)
        assert page.layout_mode == SIDE
        assert page.processContainer.width() == 500
        assert not page.previewBandToggle.isVisible(), "there is no band to collapse in side mode"

        page.set_preview_collapsed(True)
        _settle(qapp)
        full = _viewport(page).width() - BODY_INSET_PX
        assert full == 1047
        assert page.processContainer.width() == full, (
            f"the column stayed at {page.processContainer.width()} px; a hidden band gives it all {full}")
        assert not page.processInfoContainer.isVisible()
        assert page.pageScroll.horizontalScrollBar().maximum() == 0

        page.set_preview_collapsed(False)
        _settle(qapp)
        assert page.processContainer.width() == 500, "expanding restores the shipped side geometry"
        assert page.processInfoContainer.isVisible()
    finally:
        window.close()


# --- the sticky render row ------------------------------------------------------------------------------------------


def test_the_render_row_is_sticky_in_stacked_mode_and_in_flow_in_side_mode(qapp, factor_state):
    """Stacked: the footer is on the viewport's bottom and scrolling does not move it. Side: the row is in the column."""
    shell, window = _shell(qapp, 2.0)
    try:
        page = _page(shell)
        footer = page.renderRowFooter
        assert page.layout_mode == STACK
        assert footer.isVisible(), "the stacked page's render row lives in the sticky footer"
        assert page.renderRow.parentWidget() is footer
        bottom = _viewport_bottom_in_page(page)
        assert abs(_row_bottom_in_page(page) - bottom) <= 1, (
            f"the sticky row's bottom is {_row_bottom_in_page(page)}; the viewport's bottom is {bottom}")

        # Scrolling the body must not move the footer: it is a child of the page, not of the scrolling body.
        bar = page.pageScroll.verticalScrollBar()
        assert bar.maximum() > 0, "the stacked page must be scrollable for this pin to mean anything"
        bar.setValue(bar.maximum())
        _settle(qapp)
        assert abs(_row_bottom_in_page(page) - bottom) <= 1, "the sticky footer scrolled with the body"
    finally:
        window.close()

    shell, window = _shell(qapp, 1.0)
    try:
        page = _page(shell)
        assert page.layout_mode == SIDE
        assert not page.renderRowFooter.isVisible(), "side mode keeps the render row in the column's flow"
        assert page.renderRow.parentWidget() is page.bottomMenuSubContainer, (
            "the side-mode render row belongs to the process column, not the overlay footer")
    finally:
        window.close()
