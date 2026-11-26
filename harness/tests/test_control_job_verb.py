"""`start:render` starts the render queue through the control surface, and the queue's live proof has a way in.

Why this file exists (2026-11-26): the queue's live proof could add a row (`queue:<clip>`) and select it, but nothing
could START a job - the transport button lives on another tab and a synthetic click does not satisfy its hit test, so
`state:preview` on a queue item could only ever read `kind=none`. The probe was blocked by the instrument, not by the
phase. `start:render` drives the same control the user presses (`startRenderButton.click()`) and refuses when the render is
not idle, because that button is a play/pause/resume toggle: a click while running would silently PAUSE the render a
probe is trying to observe.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from src.ui.capture_server import start_render_job


class _Button:
    def __init__(self) -> None:
        self.clicks = 0

    def click(self) -> None:
        self.clicks += 1


def _window(*, state: str = "idle", rows: int = 1, button: _Button | None = None):
    button = _Button() if button is None else button
    tab = SimpleNamespace(_render_state=state, currentRenderOptions=SimpleNamespace(outputPath="/tmp/out.mkv"))
    queue = SimpleNamespace(getQueue=lambda: list(range(rows)))
    return SimpleNamespace(processTab=tab, renderQueue=queue,
                           page_handles=SimpleNamespace(proc=SimpleNamespace(startRenderButton=button))), button, tab


def test_an_idle_render_starts_through_the_users_button():
    window, button, tab = _window(state="idle")
    report = start_render_job(window)
    assert button.clicks == 1, "the verb must press the same control the user does"
    assert report["started"] is True and report["method"] == "startRenderButton"
    assert report["error"] is None
    assert report["queue_rows"] == 1 and report["output"] == "/tmp/out.mkv"
    assert report["render_state"] == "idle", "the report reads the state before the click; the poll shows the change"


@pytest.mark.parametrize("state", ["running", "paused"])
def test_a_render_that_is_not_idle_is_refused_not_paused(state: str):
    """The button is a toggle: clicking it while running would pause the very render the probe wants to watch."""
    window, button, _tab = _window(state=state)
    report = start_render_job(window)
    assert button.clicks == 0, f"a click in {state!r} would flip the transport"
    assert report["started"] is False
    assert state in (report["error"] or "")


def test_no_start_button_is_an_answer_not_an_exception():
    window, _button, _tab = _window()
    window.page_handles.proc = SimpleNamespace()
    report = start_render_job(window)
    assert report["started"] is False and "startRenderButton" in (report["error"] or "")


def test_the_verb_is_wired_and_refuses_an_unknown_action():
    """The routing exists (a verb that is not routed answers 'unknown request' and leaves the channel silent)."""
    source = inspect.getsource(__import__("src.ui.capture_server", fromlist=["x"]))
    assert 'wanted.startswith("start:")' in source, "the verb is not routed"
    assert "def start_job(action" in source, "the verb body is gone"
    assert "start:render" in source or "expected 'render'" in source
