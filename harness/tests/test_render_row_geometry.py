"""The render-buttons row may not re-flow when a job starts.

The defect this gate exists for (owner, 2026-11-24): "the buttons row re-flows when a job starts". At idle the row is
`[start] [pause] [kill] [progress bar]`, with pause and kill hidden through `hide_layout_widgets`, so their width leaves
the layout entirely. `ProcessTab.run` then shows the container again - and the row lays itself out a second time: the
progress bar gives up ~two button widths and everything after the buttons moves at the exact moment the user presses
start. Nothing in the suite could see it: every state was laid out correctly, and the defect is the *step between* two
correct states.

Qt's answer is `QSizePolicy.setRetainSizeWhenHidden(True)`: a hidden widget keeps its box, so hiding is a paint decision
rather than a layout one. The row keeps the same geometry in both states and the bar never jumps.

The file has two halves, on purpose:

* the **mechanism** half is toolkit-free: two button-sized widgets in a nested `QHBoxLayout` next to an expanding
  stretch widget, exactly the shape of `onRenderButtonsContiainer`. It first *reproduces* the defect (hide without
  retention, the stretch widget grows by two button widths), then requires the cure (hide with retention, geometry
  unchanged). A gate that cannot fail is a comment, and this one demonstrates it can.
* the **app** half builds the real shell the way `harness/tests/test_page_overflow.py` does and asserts the same on
  `startRenderButton`, `pauseRenderButton`, `killRenderButton` and `progressBar` between idle and job start. It skips
  cleanly where `sli_ui_toolkit` is not importable (a worker cell); the supervisor runs it on the shared tree.

Run against the tree before the cure and the app half names the shrinking progress bar and the moved pause button. That
is the record of the owner's report kept as a test.
"""

from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QRect  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QHBoxLayout,
    QSizePolicy,
    QWidget,
)

#: The size every other layout gate measures the shell at (`test_page_overflow.py`).
WINDOW_SIZE = (1163, 900)

#: The Designer's button box: `startRenderButton`, `pauseRenderButton` and `killRenderButton` are ceilinged at 59x55.
BUTTON_SIZE = (59, 55)

#: A row wide enough to give the bar room on both sides of the two buttons (the shell's own row is capped at 500 px).
STAND_IN_ROW_WIDTH = 420


def _settle(app: QApplication, turns: int = 3) -> None:
    for _ in range(turns):
        app.processEvents()


# --------------------------------------------------------------------------------------------------------------------
# The mechanism: a hidden widget keeps its box only when its policy says so.
# --------------------------------------------------------------------------------------------------------------------


def _stand_in_row(app: QApplication, *, retain: bool) -> tuple[QWidget, list[QWidget], QWidget]:
    """Build `[lead] [pause|kill] [bar]` the shape of the render row, and lay it out.

    The two buttons sit in a nested `QHBoxLayout` of their own, as `pauseRenderButton`/`killRenderButton` do inside
    `onRenderButtonsContiainer`, because a nested layout's hint is what its parent layout reads - the mechanism has to
    hold across that seam, not only between siblings of one row.
    """
    row = QWidget()
    row.setObjectName("standInRow")
    row.setFixedHeight(BUTTON_SIZE[1])
    layout = QHBoxLayout(row)
    layout.setContentsMargins(9, 9, 9, 9)
    layout.setSpacing(6)

    lead = QWidget(row)
    lead.setObjectName("standInLead")
    lead.setMinimumSize(*BUTTON_SIZE)
    lead.setMaximumSize(*BUTTON_SIZE)
    layout.addWidget(lead)

    group = QHBoxLayout()
    group.setSpacing(0)
    buttons: list[QWidget] = []
    for name in ("standInPause", "standInKill"):
        button = QWidget(row)
        button.setObjectName(name)
        button.setMinimumSize(*BUTTON_SIZE)
        button.setMaximumSize(*BUTTON_SIZE)
        button.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        if retain:
            policy = button.sizePolicy()
            policy.setRetainSizeWhenHidden(True)
            button.setSizePolicy(policy)
        buttons.append(button)
        group.addWidget(button)
    layout.addLayout(group)

    bar = QWidget(row)
    bar.setObjectName("standInBar")
    bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    layout.addWidget(bar)

    row.resize(STAND_IN_ROW_WIDTH, BUTTON_SIZE[1])
    row.show()
    _settle(app)
    return row, buttons, bar


def test_hiding_the_buttons_without_retention_moves_the_bar(qapp: QApplication):
    """The defect, reproduced: a hidden widget that does not retain its size takes its width out of the layout.

    This is the state the shell was in: `hide_layout_widgets` calls `setVisible(False)` and the policy is the toolkit's
    plain `Minimum, Fixed`, so the row is laid out as if the buttons did not exist.
    """
    row, buttons, bar = _stand_in_row(qapp, retain=False)
    try:
        before = bar.geometry()
        for button in buttons:
            button.setVisible(False)
        _settle(qapp)
        after = bar.geometry()

        assert not buttons[0].sizePolicy().retainSizeWhenHidden(), (
            "the control row must not retain its buttons here - this half is the pre-fix state")
        assert after.width() > before.width() + 2 * BUTTON_SIZE[0] - 1, (
            f"hiding two {(BUTTON_SIZE[0])} px buttons without retention must hand their width to the stretch "
            f"widget: it went from {before.width()} px to {after.width()} px. If it did not, this mechanism no longer "
            "describes the defect and the cure below proves nothing.")
    finally:
        row.close()


def test_hiding_the_buttons_with_retention_keeps_the_bar(qapp: QApplication):
    """The cure: with `retainSizeWhenHidden`, hiding the same buttons changes nothing the user can see."""
    row, buttons, bar = _stand_in_row(qapp, retain=True)
    try:
        before = bar.geometry()
        for button in buttons:
            button.setVisible(False)
        _settle(qapp)
        assert bar.geometry() == before, (
            f"the stretch widget moved although the buttons retained their size: {before} -> {bar.geometry()}")
        assert buttons[0].sizePolicy().retainSizeWhenHidden()
    finally:
        row.close()


# --------------------------------------------------------------------------------------------------------------------
# The app: the real row, idle vs the moment a job starts.
# --------------------------------------------------------------------------------------------------------------------


def _boot():
    """Build the shell at the default size, the way `test_page_overflow.py` does."""
    if "resources_rc" not in sys.modules:
        sys.modules["resources_rc"] = types.ModuleType("resources_rc")
    from src.ui import toolkit_boot, toolkit_theme
    from src.ui.shell.window import MainWindowShell

    app = QApplication.instance() or QApplication([])
    toolkit_boot.ensure()
    toolkit_theme.install(app)
    from PySide6.QtWidgets import QMainWindow

    window = QMainWindow()
    MainWindowShell().build(window)
    window.resize(*WINDOW_SIZE)
    window.show()
    _settle(app, 8)
    return app, window


def _reveal_proc_page(app: QApplication, window) -> None:
    """Make the proc page current - an unlaid-out `QStackedWidget` page measures a placeholder, not the page."""
    handles = window.page_handles
    stack = handles.menu.stackedWidget
    for index in range(stack.count()):
        page = stack.widget(index)
        if page.findChild(QWidget, "startRenderButton") is not None:
            stack.setCurrentIndex(index)
            _settle(app)
            return
    pytest.fail("no page of the main stack holds a startRenderButton - this gate is looking at nothing")


def _row_rects(proc, row: QWidget) -> dict[str, QRect]:
    """The four controls from the owner's report, in the row's own coordinate space.

    `startRenderButton` lives inside `widget_24` while pause/kill belong to `widget_13`, so their local geometries are
    not comparable: a moved `widget_24` would leave the button's own geometry unchanged and the user still sees it move.
    Mapping through the row is the only honest measurement.
    """
    widgets = {
        "startRenderButton": proc.startRenderButton,
        "pauseRenderButton": proc.pauseRenderButton,
        "killRenderButton": proc.killRenderButton,
        "progressBar": proc.progressBar,
    }
    return {name: QRect(widget.mapTo(row, QPoint(0, 0)), widget.size()) for name, widget in widgets.items()}


def test_the_render_row_keeps_its_boxes_from_idle_to_job_start(qapp: QApplication):
    """The gate: pressing start may not move the row's controls. Only which of them is visible may change."""
    pytest.importorskip("sli_ui_toolkit")

    app, window = _boot()
    try:
        _reveal_proc_page(app, window)
        proc = window.page_handles.proc
        row = proc.progressBar.parentWidget()
        assert row is not None, "the progress bar has no parent - the row this gate measures is gone"

        # Idle, exactly as `ProcessTab.__init__` leaves it: `hide_layout_widgets(onRenderButtonsContiainer)` (a
        # `setVisible(False)` per widget) with the start button shown. The glue is not built by `MainWindowShell`, so
        # the test states the idle side of the transition the same way the glue does.
        proc.pauseRenderButton.setVisible(False)
        proc.killRenderButton.setVisible(False)
        proc.startRenderButton.setVisible(True)
        _settle(app)

        assert not proc.pauseRenderButton.isVisible() and not proc.killRenderButton.isVisible(), (
            "idle must hide the pause/kill pair - otherwise this gate is measuring a row that never changes")
        assert proc.startRenderButton.isVisible(), "idle must show the start button"
        idle = _row_rects(proc, row)

        # Exactly what `ProcessTab.run` performs when a job starts: `show_layout_widgets(onRenderButtonsContiainer)`
        # (a `setVisible(True)` per widget) and the start button hidden.
        proc.pauseRenderButton.setVisible(True)
        proc.killRenderButton.setVisible(True)
        proc.startRenderButton.setVisible(False)
        _settle(app)

        assert proc.pauseRenderButton.isVisible() and proc.killRenderButton.isVisible(), (
            "a running job must show the pause/kill pair")
        assert not proc.startRenderButton.isVisible(), "a running job must hide the start button"
        running = _row_rects(proc, row)

        moved = [f"{name}: {idle[name]} -> {running[name]}" for name in idle if idle[name] != running[name]]
        assert not moved, (
            "the render row re-flowed when the job started - a hidden button must keep its box "
            "(`QSizePolicy.setRetainSizeWhenHidden(True)`), or the progress bar jumps under the user's pointer:\n"
            + "\n".join(f"  {line}" for line in moved))

        for name in ("startRenderButton", "pauseRenderButton", "killRenderButton"):
            widget = getattr(proc, name)
            assert widget.sizePolicy().retainSizeWhenHidden(), (
                f"{name} does not retain its size when hidden: {widget.sizePolicy().horizontalPolicy()} without "
                "retention is the defect this gate was written for")
    finally:
        window.close()
