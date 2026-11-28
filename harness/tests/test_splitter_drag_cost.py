"""What one drag of the preview splitter costs (owner, 2026-11-28).

*"оптимизировать обновление контроллов при движении сплиттера, а то сейчас это прям очень медленный процесс пока он
для этих двух десятков геометрии её каждый тик пересчитает паровозиком"* - make the controls cheap to update while the
splitter moves: today every `splitterMoved` (one per pixel) re-states the column's width on its two strips, and each
write reflows the whole General tab - rows, combos, spin boxes, the ribbon grid - to a width the user never rests on.

**Measured on this tree, offscreen, the real `MainWindowShell` at 1.0/1163x900 with the General tab current and the
upscale section open, the splitter dragged over its whole range in 60 steps (`moveSplitter` + `processEvents` per
step), medians of 3 rounds, this host:**

* one reflow per pixel (the old slot): **13.56 ms/step, 804 ms per gesture**;
* of that, one `setFixedWidth` of the column's settings strip is **2.80 ms** by itself (the render row's own write is
  0.02 ms), and the deferred paint of the reflowed controls is the rest;
* the same gesture as a real one - press on the handle, 60 moves, release - with the contents frozen for its duration
  and stated once on release: **3.07 ms/step, 186 ms per gesture** (median of 5 rounds), i.e. 4.2x cheaper, on exactly
  the geometry a single `moveSplitter` to the end position produces.

So the budget this file pins is **under 2x the measured gesture and under half the per-pixel path on the same run**;
the ratio is the primary criterion (it travels between machines), the absolute ceiling is the floor under it.

What the frozen frame looks like: the splitter's panes (and so the preview's edge, the handle) follow the pointer, while
the column's two strips keep the width the press landed on until the release. Both the card's fill and the page behind
it are the `surface.page` token (`widget_2` and `procPage` in `proc_page.py`), so the frozen frame reads as extra
padding on the column's right, not as a hole - and the release states the one width the gesture ended on. `stack` drags
are untouched: they move heights, so no width is written per pixel there, and nothing is frozen.

What each pin proves:

* **a gesture costs less than a reflow per pixel** - the same 60 moves with the handle pressed are at least twice as
  cheap as the same 60 moves without it (which is the old, immediate path), and the strips do not chase the handle
  while the mouse is down (`bottomMenuSubContainer` keeps the width it had when the press landed);
* **the gesture lands on the geometry a single move produces** - the classic deferred-reflow bug is a page left at an
  intermediate width, so the split, the strips, the tab host and the General tab's scroll content are compared against
  a fresh single `moveSplitter` to the same position;
* **a move that changes nothing writes nothing** - a move to the width already in force posts no layout request at
  all, which is what makes a drag held at a floor free.

Everything is offscreen and builds the real shell; the compiled resource blob is stubbed because a test tree has no
build step for it.
"""

from __future__ import annotations

import statistics
import sys
import time
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("sli_ui_toolkit")

from PySide6.QtCore import QEvent, QObject, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QMainWindow  # noqa: E402

if "resources_rc" not in sys.modules:
    sys.modules["resources_rc"] = types.ModuleType("resources_rc")

from src.ui.layout_policy import BODY_INSET_PX, PANE_FLOOR_PX  # noqa: E402

#: The size every other layout gate measures the shell at (`test_preview_splitter.py`, `test_page_fit_box.py`).
WINDOW_SIZE = (1163, 900)

#: One gesture, in steps: the whole range of the 1.0 splitter (500 px column floor -> preview floor) in 60 moves.
STEPS = 60

#: Rounds per path; the comparison uses the medians, so one scheduling hiccup cannot fail the pin.
ROUNDS = 3

#: Measured after the change: 186 ms per gesture (3.07 ms/step, median of 5 rounds). ~2.7x headroom; the per-pixel path
#: it replaces costs 804 ms on the same host, so the ceiling stays under that even on a machine twice as fast.
GESTURE_BUDGET_MS = 500.0

#: The gesture must be at least this much cheaper than one reflow per pixel, measured on the SAME run.
MIN_SPEEDUP = 2.0


@pytest.fixture()
def process_state():
    """Leave the process-wide factor and language where they were: the singleton outlives a test."""
    from src.ui import i18n
    from sli_ui_toolkit.ui.managers.ui_scale import UiScale

    scale = UiScale.get_instance()
    factor_before = scale.factor()
    language_before = i18n.current_language()
    yield scale
    scale.set_factor(factor_before)
    i18n.configure_i18n(language_before)


def _settle(qapp, turns: int = 10) -> None:
    for _ in range(turns):
        qapp.processEvents()


def _shell(qapp, factor: float, *, size=WINDOW_SIZE):
    """The real window at `factor`, on the process page, with the General tab current and upscale OPEN.

    Upscale open is the state with the most rows in the tab (`test_general_tab_fit.py` opens it the same way): the
    cost being pinned is the settings reflow, so the measurement asks for the heaviest one.
    """
    from PySide6.QtWidgets import QStackedWidget

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
    stack = window.findChild(QStackedWidget, "stackedWidget")
    assert stack is not None, "the nav page stack is missing"
    stack.setCurrentIndex(("home", "more", "proc", "settings", "download").index("proc"))
    page = shell.procPage_page
    page.generalSettings_page.generalUpscaleContainer.setVisible(True)
    page.processSettingsContainer.setCurrentIndex(0)
    _settle(qapp)
    return shell, window, page


def _range(page) -> tuple[int, int]:
    """The side drag's own range, from the page's floors: the column's content floor to the preview's 320 px floor."""
    body = page.pageScroll.viewport().width() - BODY_INSET_PX
    return page._column_floor_px(), body - page.preview_splitter.handleWidth() - PANE_FLOOR_PX


def _drag(page, qapp, *, press: bool, steps: int = STEPS) -> tuple[list[float], int]:
    """Move the boundary across its range in `steps`, `processEvents()` after each move - what a drag does.

    `press` is the only difference between the two paths: with it the handle is under the mouse, so the page freezes
    its contents for the gesture (what a user does); without it every move is an immediate reflow (what the old slot
    did, and what a programmatic `setSizes` still does). Returns the per-step milliseconds and the width the last move
    landed on.
    """
    splitter = page.preview_splitter
    handle = splitter.handle(1)
    lo, hi = _range(page)
    if press:
        QTest.mousePress(handle, Qt.MouseButton.LeftButton, pos=handle.rect().center())
        qapp.processEvents()
    times: list[float] = []
    target = lo
    try:
        for step in range(steps):
            target = int(round(lo + (hi - lo) * (step + 1) / steps))
            started = time.perf_counter()
            splitter.moveSplitter(target, 1)
            qapp.processEvents()
            times.append((time.perf_counter() - started) * 1000.0)
    finally:
        if press:
            QTest.mouseRelease(handle, Qt.MouseButton.LeftButton, pos=handle.rect().center())
            qapp.processEvents()
    return times, target


def _geometry(page) -> tuple:
    """Everything a deferred reflow could leave behind at an intermediate value."""
    splitter = page.preview_splitter
    content = page.generalSettings.findChild(type(page.pageScrollBody), "scrollAreaWidgetContents_4")
    return (
        tuple(splitter.sizes()),
        page.processContainer.width(),
        page.bottomMenuSubContainer.width(),
        page.renderRow.width(),
        page.processSettingsContainer.width(),
        page.generalSettings.width(),
        None if content is None else content.width(),
    )


def _reset(page, qapp) -> None:
    """Back to the range's start without a press (so the strips are applied immediately), then settle."""
    page.preview_splitter.moveSplitter(_range(page)[0], 1)
    _settle(qapp)


# --- the cost ----------------------------------------------------------------------------------------------------


def test_a_gesture_costs_less_than_a_reflow_per_pixel(qapp, process_state):
    """A whole drag under the mouse is at least `MIN_SPEEDUP`x cheaper than the same drag reflowing every pixel.

    Before: `_on_split_moved` called `_apply_column_widths` unconditionally, so each of the 60 moves re-stated the
    column's width on its two strips and reflowed the General tab (13.56 ms/step, 804 ms per gesture, measured - the
    docstring at the top of this file carries the split of that cost). After: a move only records the width, the
    settings host is frozen while the handle is under the mouse, and the width is stated once on release.

    The per-pixel path is still reachable (a programmatic `setSizes` does not press the handle), and it is measured on
    the SAME run, so the ratio is not a claim about this host's absolute speed. The strips must also not chase the
    handle mid-gesture - that is what the optimisation IS, and asserting it stops the test from passing when the
    freeze silently stops happening.
    """
    shell, window, page = _shell(qapp, 1.0)
    try:
        frozen_start = page.bottomMenuSubContainer.width()
        per_pixel, gesture = [], []

        # One throwaway round each: the first gesture pays for whatever is still cold (fonts, the tab's first pass).
        _drag(page, qapp, press=False)
        _reset(page, qapp)
        _drag(page, qapp, press=True)
        _reset(page, qapp)

        for _ in range(ROUNDS):
            per_pixel.append(sum(_drag(page, qapp, press=False)[0]))
            _reset(page, qapp)
            gesture.append(sum(_drag(page, qapp, press=True)[0]))
            _reset(page, qapp)

        baseline = statistics.median(per_pixel)
        after = statistics.median(gesture)
        assert after < GESTURE_BUDGET_MS, (
            f"a 60-step gesture took {after:.0f} ms (budget {GESTURE_BUDGET_MS:.0f} ms); "
            f"per pixel it is {baseline:.0f} ms")
        assert after * MIN_SPEEDUP <= baseline, (
            f"the gesture is {after:.0f} ms against {baseline:.0f} ms for one reflow per pixel "
            f"({baseline / max(after, 1e-9):.2f}x, needs {MIN_SPEEDUP:.1f}x): the contents are not being frozen for it")

        # The freeze itself: mid-gesture the column's settings strip keeps the width the press landed on, and the
        # page's own splitter is the only thing following the handle.
        handle = page.preview_splitter.handle(1)
        QTest.mousePress(handle, Qt.MouseButton.LeftButton, pos=handle.rect().center())
        _settle(qapp)
        lo, hi = _range(page)
        for step in range(STEPS):
            page.preview_splitter.moveSplitter(int(round(lo + (hi - lo) * (step + 1) / STEPS)), 1)
            qapp.processEvents()
        mid = (page.processContainer.width(), page.bottomMenuSubContainer.width())
        assert mid == (hi, frozen_start), (
            f"mid-gesture the column is {mid[0]} px and its strip {mid[1]} px; expected the pane at {hi} and the strip "
            f"frozen at {frozen_start} - the contents are reflowing per pixel again")
        QTest.mouseRelease(handle, Qt.MouseButton.LeftButton, pos=handle.rect().center())
        _settle(qapp)
    finally:
        window.close()


# --- the deferred reflow lands where a single move lands ----------------------------------------------------------


def test_the_gesture_lands_on_the_geometry_a_single_move_produces(qapp, process_state):
    """A frozen drag must not leave the page at an intermediate width - the classic deferred-reflow bug.

    The gesture ends with the mouse released; the width is stated then, once. The same position reached with one
    programmatic `moveSplitter` must give the identical page: the split, both strips, the tab host, the General tab and
    its scroll content (the last one is the settings reflow's own answer - the value that would stay stale if the
    release forgot to flush).
    """
    shell, window, page = _shell(qapp, 1.0)
    try:
        _drag(page, qapp, press=True)
        _settle(qapp)
        landed = _geometry(page)
        lo, hi = _range(page)
        assert page.processContainer.width() == hi, "the gesture did not reach the end of its own range"
        assert page.bottomMenuSubContainer.width() == hi, "the release did not state the final width on the strip"

        # Drive away and back with ONE move: the deferred path's answer has to match the immediate one exactly.
        page.preview_splitter.moveSplitter(lo, 1)
        _settle(qapp)
        page.preview_splitter.moveSplitter(hi, 1)
        _settle(qapp)
        assert _geometry(page) == landed, (
            f"the gesture landed on {landed}, a single move on {_geometry(page)} - a deferred reflow left the page at "
            "an intermediate width")
    finally:
        window.close()


# --- a repeat is not a reflow ------------------------------------------------------------------------------------


class _LayoutRequests(QObject):
    """Count `LayoutRequest` events on the column's settings strip."""

    def __init__(self, parent: QObject) -> None:
        super().__init__(parent)
        self.count = 0

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API
        if event.type() == QEvent.Type.LayoutRequest:
            self.count += 1
        return False


def test_a_move_that_changes_nothing_writes_nothing(qapp, process_state):
    """A move onto the width already in force posts no layout request at all.

    The old slot re-stated both widths on every `splitterMoved`, and a drag held against a floor keeps emitting moves
    with the same position - a write whose value did not change is still a layout request, so the strip is watched
    directly: the settled width is re-stated through `_apply_column_widths`, and the strip must stay silent.
    """
    shell, window, page = _shell(qapp, 1.0)
    try:
        page.preview_splitter.moveSplitter(600, 1)
        _settle(qapp)
        column = page.processContainer.width()
        assert page.bottomMenuSubContainer.width() == column

        counter = _LayoutRequests(page.bottomMenuSubContainer)
        page.bottomMenuSubContainer.installEventFilter(counter)
        try:
            # Drain what the layout pass above already queued: the count below is about the repeats only.
            _settle(qapp)
            counter.count = 0
            for _ in range(5):
                page._apply_column_widths(column)
                page.preview_splitter.moveSplitter(column, 1)
                _settle(qapp)
            assert counter.count == 0, (
                f"{counter.count} layout request(s) for a width already in force ({column} px)")
        finally:
            page.bottomMenuSubContainer.removeEventFilter(counter)
    finally:
        window.close()
