"""The process page follows the UI scale: the column grows, the body scrolls, the strip overflows honestly.

Owner decisions (2026-11-24), stated here so every pin has a subject:

* the process column's DESIRED width grows **strictly** with the factor - 500 design px, 750 at 1.5, 1000 at 2.0 -
  and the preview pane takes the remainder while it can. Since wave 2 of the 2026-11-24 layout policy, the invoice is
  the policy's: at 1.5 in the design window the column yields to 693 so the preview keeps its 320 px floor, and at 2.0
  the page stacks (both panes full width) instead of overflowing horizontally;
* the process page body **scrolls**, through the toolkit's own `OverlayScrollArea` rather than a stock `QScrollArea`;
* 1.0 is unchanged: the column is exactly 500 px and the five English labels still fit it whole.

Two numbers are recorded rather than assumed, because the live run and this offscreen tree disagree about which factor
they belong to: at 1.5 in English the five labels need **660 px**, so they fit the 693 px column; the 878 px figure
from the live session is the 2.0 strip (870 px here, within a font metric of it). The overflow path is therefore pinned
where it actually happens - the Russian labels at 1.5 need 851 px in a 693 px column - where the strip's own row has
to carry the last tab and `layout_audit` must not call a scrollable bar clipped.

Everything is offscreen and builds the real `MainWindowShell`; the compiled resource blob is stubbed because a test
tree has no build step for it.
"""

from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("sli_ui_toolkit")

from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QWheelEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget  # noqa: E402

from sli_ui_toolkit.ui.managers.ui_scale import UiScale  # noqa: E402
from sli_ui_toolkit.widgets import OverlayScrollArea  # noqa: E402

if "resources_rc" not in sys.modules:
    sys.modules["resources_rc"] = types.ModuleType("resources_rc")

from src.ui.pages.proc_page import PROCESS_COLUMN_PX  # noqa: E402

#: The column width the layout policy applies at the design window (1163). 500 is the DESIGN value; 1.5 yields to 693
#: (750 + a 320 px preview does not fit the 1037 px viewport) and 2.0 stacks, where the column is the full body width.
SIDE_COLUMN_WIDTHS = {1.0: 500, 1.5: 693}

#: The English strip's natural width at each factor, measured off the real buttons on this tree (the pins below state
#: them so a font or style change that moves the labels is a failure, not a surprise). At 1.0 the five widths are the
#: same numbers `test_tab_strip_fit.py` pins one by one.
STRIP_ENGLISH = {1.0: 450, 1.5: 660, 2.0: 870}

#: The Russian strip at 1.5: the labels grow faster than the column (500 design px, so 750), which is the overflow the
#: host's own row has to carry.
STRIP_RUSSIAN_1_5 = 851

#: The size every layout gate measures the shell at; the body fits its content here at 1.0, so 1.0 geometry is
#: unchanged by the scroll area.
FIT_SIZE = (1163, 900)

#: The window's own default (`MainWindowShell` resizes to this); at 2.0 the body is far taller than it.
DEFAULT_SIZE = (1163, 700)


@pytest.fixture()
def process_state():
    """Put the process-wide factor and language back: both are singletons every later test would inherit."""
    from src.ui import i18n

    scale = UiScale.get_instance()
    factor_before = scale.factor()
    language_before = i18n.current_language()
    yield scale, i18n
    scale.set_factor(factor_before)
    i18n.configure_i18n(language_before)


def _shell(qapp, factor: float, *, language: str = "en", size=DEFAULT_SIZE):
    """The real window at `factor`, built after the factor and language are set, shown and settled."""
    from src.ui import i18n, toolkit_boot, toolkit_theme
    from src.ui.shell.window import MainWindowShell

    UiScale.get_instance().set_factor(factor)
    i18n.configure_i18n(language)
    toolkit_boot.ensure()
    toolkit_theme.install(qapp)
    window = QMainWindow()
    shell = MainWindowShell()
    shell.build(window)
    window.resize(*size)
    window.show()
    for _ in range(10):
        qapp.processEvents()
    return shell, window


def _settle(qapp, turns: int = 6) -> None:
    for _ in range(turns):
        qapp.processEvents()


def _column(page):
    """The three containers that must all carry the one column width."""
    return page.processContainer, page.bottomMenuSubContainer, page.widget_13


# --- the column --------------------------------------------------------------------------------------------------


def test_the_column_follows_the_layout_policy_at_each_factor(qapp, process_state):
    """500 at 1.0, 693 at 1.5 (the column yields), and the whole body width when 2.0 stacks."""
    assert SIDE_COLUMN_WIDTHS[1.0] == PROCESS_COLUMN_PX, "the design constant is the 1.0 width, not a scaled one"
    for factor, expected in SIDE_COLUMN_WIDTHS.items():
        shell, window = _shell(qapp, factor)
        try:
            page = shell.procPage_page
            assert expected == round(PROCESS_COLUMN_PX * factor) or factor == 1.5
            for container in _column(page):
                assert container.width() == expected, (
                    f"at factor {factor} {container.objectName()} is {container.width()} px, not {expected}: "
                    "the column must be one number, stated at every site, not capped twice")
        finally:
            window.close()

    shell, window = _shell(qapp, 2.0)
    try:
        page = shell.procPage_page
        assert page.layout_mode == "stack"
        full = page.pageScroll.viewport().width() - 18
        assert full == round(PROCESS_COLUMN_PX * 2.0) - 9, "991 px: the 1000 px column minus the body's 18 px frame"
        for container in _column(page):
            assert container.width() == full, (
                f"a stacked {container.objectName()} is {container.width()} px, not the body's {full}")
    finally:
        window.close()


def test_a_live_factor_change_moves_the_column(qapp, process_state):
    """The capture channel's `scale:<factor>` moves the running window - no restart, no rebuild."""
    shell, window = _shell(qapp, 1.0)
    try:
        page = shell.procPage_page
        assert page.processContainer.width() == 500

        UiScale.get_instance().set_factor(1.5)
        _settle(qapp)
        assert page.processContainer.width() == 693
        assert page.bottomMenuSubContainer.width() == 693

        UiScale.get_instance().set_factor(2.0)
        _settle(qapp)
        assert page.layout_mode == "stack"
        assert page.processContainer.width() == page.pageScroll.viewport().width() - 18
    finally:
        window.close()


# --- the scroll host ---------------------------------------------------------------------------------------------


def test_the_page_body_is_the_toolkits_scroll_area(qapp, process_state):
    """Not a stock `QScrollArea`: the body lives in an `OverlayScrollArea` under the page root."""
    shell, window = _shell(qapp, 1.0)
    try:
        page = shell.procPage_page
        area = page.pageScroll
        assert isinstance(area, OverlayScrollArea)
        assert area.parentWidget() is page.procPage
        assert area.widget() is page.pageScrollBody
        assert page.pageScrollBody.layout() is page.gridLayout_3, (
            "the columns must be laid out inside the scroll surface, not beside it")
        # Both columns are in the scrolling body.
        assert page.processContainer.parentWidget() is page.pageScrollBody
        assert page.processInfoContainer.parentWidget() is page.pageScrollBody
    finally:
        window.close()


def test_at_one_point_zero_the_body_does_not_scroll_in_a_window_that_fits_it(qapp, process_state):
    """The 1.0 geometry is unchanged: at the gates' window size the body fits and the surface has no range."""
    shell, window = _shell(qapp, 1.0, size=FIT_SIZE)
    try:
        page = shell.procPage_page
        area = page.pageScroll
        assert page.processContainer.width() == 500
        assert area.widget().height() <= area.viewport().height(), (
            "at 1.0 in a 900 px window the body fits - a scroll range here would mean the page got taller")
        assert area.verticalScrollBar().maximum() == 0
    finally:
        window.close()


@pytest.mark.parametrize("factor", (2.0,))
def test_the_body_scrolls_to_its_content_at_a_raised_factor(qapp, process_state, factor):
    """Taller than the viewport, and a real wheel over it moves the viewport to the far end.

    Only the STACKED factor is a page-scroll case now. 1.5 used to be one of these two: since the column yields its
    vertical slack in `side` mode (owner, 2026-11-24: the render row and the status line must stay visible at the
    700 px window floor), the page at 1.5/1163x700 fits exactly and the tab pane is what scrolls inside itself -
    `test_the_bottom_strip_stays_visible_at_the_minimum_window_height` pins that side of the same rule. In `stack`
    the preview band above the column is a fixed 16:9 box, so the body is genuinely taller than the window and the
    wheel mechanism is proven here.
    """
    shell, window = _shell(qapp, factor)
    try:
        # The window's own default size follows the factor (wave 5) and its follower re-applies that default after the
        # first show, so a viewport chosen inside `_shell` is gone by now. This test is about the SCROLL MECHANISM in a
        # window of a laptop size, not about the default: choose the viewport after the window has settled, and do not
        # change the factor again (the follower would size the window back, by design).
        window.resize(*DEFAULT_SIZE)
        _settle(qapp, 3)
        page = shell.procPage_page
        area = page.pageScroll
        content = area.widget()
        assert content.height() > area.viewport().height(), (
            f"at factor {factor} the body is {content.height()} px in a {area.viewport().height()} px viewport")
        maximum = area.verticalScrollBar().maximum()
        assert maximum > 0

        centre = area.viewport().rect().center()
        wheel = QWheelEvent(
            QPointF(centre), QPointF(area.viewport().mapToGlobal(centre)), QPoint(0, 0), QPoint(0, -120),
            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.ScrollUpdate, False)
        QApplication.sendEvent(area.viewport(), wheel)
        assert wheel.isAccepted(), "the body must claim the wheel, or the page under it scrolls instead"
        for _ in range(50):
            QTest.qWait(20)
            if area.verticalScrollBar().value() > 0:
                break
        assert area.verticalScrollBar().value() > 0, "the wheel did not move the body"

        area.verticalScrollBar().setValue(maximum)
        _settle(qapp, 3)
        assert area.verticalScrollBar().value() == maximum
        # The far end is the content's bottom: the row last in the body is inside the viewport there.
        bottom_in_viewport = content.mapTo(area.viewport(), QPoint(0, content.height() - 1))
        assert 0 <= bottom_in_viewport.y() < area.viewport().height()
    finally:
        window.close()


# --- the strip ---------------------------------------------------------------------------------------------------


def test_the_english_strip_fits_the_column_it_scales_with(qapp, process_state):
    """The strip's need grows with the same factor as the column, so English still fits at 1.5 and 2.0."""
    for factor, expected_need in STRIP_ENGLISH.items():
        shell, window = _shell(qapp, factor)
        try:
            page = shell.procPage_page
            bar = page.processSettingsContainer.tabBar()
            fit = page._tab_strip.strip_fit()
            assert bar.required_width() == expected_need, (
                f"the English strip needs {bar.required_width()} px at factor {factor}; re-measure the labels before "
                "editing this pin")
            assert fit.needed == expected_need
            assert fit.fits, f"{expected_need} px of labels in a {page.processContainer.width()} px column"
            assert fit.available == page.processContainer.width()
        finally:
            window.close()


def test_the_russian_strip_overflows_and_the_host_carries_the_last_tab(qapp, process_state):
    """At 1.5 in Russian the labels outgrow the column; the strip's own row must carry the last tab, not hide it."""
    from src.ui.devtools.layout_audit import audit

    shell, window = _shell(qapp, 1.5, language="ru")
    try:
        page = shell.procPage_page
        host = page.processSettingsContainer
        bar = host.tabBar()
        column = page.processContainer.width()

        assert bar.required_width() == STRIP_RUSSIAN_1_5
        assert bar.required_width() > column, "this pin needs a strip that does not fit its column"
        assert not page._tab_strip.strip_fit().fits

        # The affordance is visible without the pointer having to find it.
        assert bar.height() > bar.sizeHint().height(), "an overflowing strip must grow its scroll row"

        # And it reaches: the far end puts the last label whole inside the bar.
        bar.set_scroll_offset(bar.required_width() - bar.width())
        _settle(qapp)
        buttons = page._tab_strip._factory.buttons
        last = buttons[-1]
        offset = bar.scroll_offset()
        assert last.x() >= offset and last.x() + last.width() <= offset + bar.width(), (
            "the last tab is not whole at the far end of the strip")

        # The audit's honesty: a bar narrower than its strip is scrollable by design, not clipped.
        clipped = [line for line in audit(window) if "CLIPPED" in line and "TopTabBar" in line]
        assert not clipped, (
            "layout_audit still calls the scrollable strip clipped; a child it cannot reach must stay reported:\n"
            + "\n".join(clipped))
    finally:
        window.close()


# --- the audit exemption is narrow -------------------------------------------------------------------------------


class _FakeStrip(QWidget):
    """The shape the audit recognises: a widget that reports an extent and can move its content within it."""

    def __init__(self, parent, extent: int) -> None:
        super().__init__(parent)
        self._extent = extent
        self._offset = 0

    def required_width(self) -> int:
        return self._extent

    def scroll_offset(self) -> int:
        return self._offset

    def set_scroll_offset(self, offset: int) -> None:
        self._offset = int(offset)


def test_the_audit_exempts_reachable_children_only(qapp):
    """Hand-built trees: a scrollable strip's in-reach child is not a finding; an out-of-reach one still is."""
    from src.ui.devtools.layout_audit import audit

    window = QWidget()
    window.resize(120, 80)
    strip = _FakeStrip(window, extent=250)
    strip.setGeometry(0, 0, 120, 40)
    reachable = QWidget(strip)
    reachable.setObjectName("reachableTab")
    reachable.setGeometry(150, 0, 90, 20)           # right edge 239 <= 250: the strip can scroll to it
    beyond = QWidget(strip)
    beyond.setObjectName("beyondTab")
    beyond.setGeometry(300, 0, 90, 20)              # right edge 389 > 250: even the far end cannot show it
    plain = QWidget(window)
    plain.setObjectName("plainParent")
    plain.setGeometry(0, 40, 60, 30)
    cut = QWidget(plain)
    cut.setObjectName("noScrollMechanism")
    cut.setGeometry(0, 0, 200, 20)                  # no scroll answer: this is a clip, not scrolling
    window.show()
    _settle(qapp)
    try:
        lines = audit(window)
        assert not [line for line in lines if "reachableTab" in line]
        assert [line for line in lines if "beyondTab" in line], "a child past the strip's extent is still a clip"
        assert [line for line in lines if "noScrollMechanism" in line], (
            "a cut child with no scrolling mechanism must stay reported")
    finally:
        window.close()
