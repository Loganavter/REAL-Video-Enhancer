"""The render-buttons row: every button is always in the row, and only its enabled state changes.

This file changed sides twice on the owner's calls, and the three states are the history worth keeping:

1. **Morning, 2026-11-24** - "the buttons row re-flows when a job starts": pause/kill were hidden at idle and shown
   when a render began, so the progress bar gave up their width at the press. The cure was
   `setRetainSizeWhenHidden(True)`.
2. **Evening, 2026-11-24** - the cost of that cure: "why is there a left indent - the bar got shorter and the button
   stepped after it". Retention keeps the row stable by reserving the hidden pair's 84 px, which reads as an empty
   gap. The owner chose the longer idle bar over the stable row.
3. **Same evening, final** - "better show the cancel button always, just inactive while nothing runs". That is the
   contract below: no button is ever hidden, so the row has ONE geometry in every state, and nothing is reserved
   because nothing is missing. `ProcessTab._set_render_controls` is the single place a state is applied, and the
   mechanism half of this file keeps the retention demonstration as the reason the third option was chosen.
4. **Then** - "join the pause and play buttons into one, just change the icon". The transport is now a single toggle:
   unchecked = play glyph (idle / paused), checked = pause glyph (a job is running), and the press is dispatched by
   `_render_state` in the glue rather than by the glyph.

States (from `ProcessTab.RENDER_CONTROLS`): idle -> transport usable (play), cancel visible but disabled; running ->
transport usable (pause), cancel usable; paused -> transport usable (play = resume), cancel usable.
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
        "killRenderButton": proc.killRenderButton,
        "progressBar": proc.progressBar,
    }
    return {name: QRect(widget.mapTo(row, QPoint(0, 0)), widget.size()) for name, widget in widgets.items()}


#: The three states the glue applies, as `ProcessTab.RENDER_CONTROLS` states them (`src/ui/ProcessTab.py`).
STATES = {
    "idle": {"transport": True, "kill": False},
    "running": {"transport": True, "kill": True},
    "paused": {"transport": True, "kill": True},
}


def _apply_state(proc, state: str) -> None:
    """The same two lines `ProcessTab._set_render_controls` performs: show, then enable or disable.

    The test mirrors the glue because importing `src/ui/ProcessTab.py` needs the app's full dependency set (`cpuinfo`,
    the backend) that this offscreen suite deliberately does not have; `test_the_glue_owns_the_row_state` below pins
    that the glue is the one place these calls live.
    """
    wanted = STATES[state]
    for name, widget in (("transport", proc.startRenderButton), ("kill", proc.killRenderButton)):
        widget.setVisible(True)
        widget.setEnabled(bool(wanted[name]))
    proc.startRenderButton.setChecked(state == "running")


def test_every_render_button_stays_in_the_row_and_only_its_state_changes(qapp: QApplication):
    """The final contract: the row's boxes may not move, and no button may disappear."""
    pytest.importorskip("sli_ui_toolkit")

    app, window = _boot()
    try:
        _reveal_proc_page(app, window)
        proc = window.page_handles.proc
        row = proc.progressBar.parentWidget()
        assert row is not None, "the progress bar has no parent - the row this gate measures is gone"

        rects = {}
        for state in STATES:
            _apply_state(proc, state)
            _settle(app)
            for name in ("startRenderButton", "killRenderButton"):
                assert getattr(proc, name).isVisible(), f"{name} is hidden in the {state} state"
            rects[state] = _row_rects(proc, row)
            assert proc.startRenderButton.isEnabled() == STATES[state]["transport"]
            assert proc.killRenderButton.isEnabled() == STATES[state]["kill"]
            # the glyph IS the state: checked = a job is running = the pause icon
            assert proc.startRenderButton.isChecked() == (state == "running"), (
                f"the transport's glyph does not follow the {state} state")

        idle = rects["idle"]
        for state, measured in rects.items():
            moved = [f"{name}: {idle[name]} -> {measured[name]}" for name in idle if idle[name] != measured[name]]
            assert not moved, (
                f"the render row moved between idle and {state} - a disabled button keeps its box, a hidden one does "
                "not:\n" + "\n".join(f"  {line}" for line in moved))

        # No reservation is needed any more: nothing is hidden, so no policy may claim a box for a hidden widget.
        for name in ("startRenderButton", "killRenderButton"):
            assert not getattr(proc, name).sizePolicy().retainSizeWhenHidden(), (
                f"{name} retains its size when hidden - with nothing hidden that can only hide a real regression")
    finally:
        window.close()


def test_the_glue_owns_the_row_state():
    """One place applies it: `ProcessTab` enables/disables, and nothing in the file hides a render button."""
    import ast
    from pathlib import Path

    source = Path(__file__).resolve().parents[2] / "src" / "ui" / "ProcessTab.py"
    text = source.read_text(encoding="utf-8")
    assert "_set_render_controls(" in text, "the glue lost its single state writer"
    assert "RENDER_CONTROLS" in text, "the states are no longer stated as data"
    for name in ("startRenderButton", "killRenderButton"):
        for call in (f"{name}.setVisible(False)", f"{name}.hide()"):
            assert call not in text, f"{call} is back - the row's buttons stay in the row, only disabled"
    assert "pauseRenderButton" not in text, (
        "the pause button came back - play/pause/resume are one transport whose glyph follows the state")

    tree = ast.parse(text)
    states = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "RENDER_CONTROLS" for target in node.targets):
            states = ast.literal_eval(node.value)
    assert states == STATES, f"the glue's states drifted from the contract: {states}"
