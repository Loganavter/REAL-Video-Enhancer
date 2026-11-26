"""The `preview:` verb may answer `finished` only after the BACKEND CHILD exits and its file holds bytes.

Why this file exists (measured three times on 2026-11-25: 18:19, 18:43, 18:45): `preview:<clip>` answered
`started=true, finished=true` in ~1.2 s with `output=null` while the render actually ran 18-20 s and wrote 493
frames. The rule read `startRenderButton.isEnabled()` as "the job is over", and `ProcessTab.RENDER_CONTROLS` enables
that button in `idle`, `running` AND `paused` (one button carries play/pause/resume) - so the check was always true
and the verb concluded "finished" before the backend child existed. Every supervisor's live preview proof then read
as a refusal.

The rule under test is the one the verb runs: `capture_server.PreviewWatch.tick()` reads the SAME attribute paths
the live verb reads (`processTab.renderProcess`, `processTab._render_state`, `page_handles.proc.startRenderButton`)
and the real preview file on disk. A fake window/process/button with a fake clock reproduces the race with no render,
no Qt event loop and no display.

Red-before: `PreviewWatch` did not exist (the rule lived as two nested closures inside `install()`), so this file
failed at import/attribute lookup on the old tree. The first test is the exact regression: an enabled button while
the app is still `running` and no child has spawned yet must be PENDING, never FINISHED.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from src.ui import capture_server


class _Button:
    """`startRenderButton` - enabled in every render state; that is the whole point."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def isEnabled(self) -> bool:
        return self.enabled


class _Process:
    """`subprocess.Popen` as the rule sees it: `poll()` is the exit code, or None while the child is alive."""

    def __init__(self, code: int | None = None) -> None:
        self.code = code
        self.identity = object()

    def poll(self) -> int | None:
        return self.code


class _Window:
    """Only the attribute paths the live rule reads - the rest of the window is irrelevant to a verdict."""

    def __init__(self) -> None:
        self.processTab = SimpleNamespace(renderProcess=None, _render_state="idle")
        self.page_handles = SimpleNamespace(proc=SimpleNamespace(startRenderButton=_Button()))


def _watch(tmp_path, started_at: float = 0.0):
    output = tmp_path / "clip.mkv_preview.mkv"
    return capture_server.PreviewWatch(str(output), started_at), _Window(), output


def _verdict(watch, window, now: float) -> str:
    return watch.tick(window, now=now)[0]


# -- the regression: the button looks enabled the whole time -------------------------------------------------------


def test_an_enabled_button_without_a_child_is_pending_never_finished(tmp_path):
    """The exact 2026-11-25 race: the app is `running`, the transport button is enabled, no child yet."""
    watch, window, _output = _watch(tmp_path)
    window.processTab._render_state = "running"      # the click reached the app (startRender ran synchronously)
    window.page_handles.proc.startRenderButton.enabled = True

    for now in (0.4, 1.0, 2.0, 4.0):
        assert _verdict(watch, window, now) == capture_server.PREVIEW_PENDING, (
            "an enabled button is not evidence that a render finished - it answers PENDING until the child exits")


def test_a_live_backend_child_keeps_the_verb_pending_for_the_whole_render(tmp_path):
    """18-20 s of real work: alive child + a growing (partial) file stays pending until the child exits."""
    watch, window, output = _watch(tmp_path)
    window.processTab._render_state = "running"
    watch.previous_child = window.processTab.renderProcess    # nothing was running before the press

    child = _Process(None)
    window.processTab.renderProcess = child
    for now in (1.0, 5.0, 9.0, 13.0, 17.0):
        assert _verdict(watch, window, now) == capture_server.PREVIEW_PENDING

    output.write_bytes(b"partial preview - the writer is still going")
    assert _verdict(watch, window, 18.0) == capture_server.PREVIEW_PENDING, (
        "a non-empty file while the child is ALIVE is a render in progress, not a finished job")


# -- finished needs the exit AND a real file ----------------------------------------------------------------------


def test_finished_requires_the_child_exit_and_a_non_empty_file(tmp_path):
    watch, window, output = _watch(tmp_path)
    window.processTab._render_state = "running"
    child = _Process(None)
    window.processTab.renderProcess = child

    output.write_bytes(b"")                                  # removed/opened, not written yet
    child.code = 0                                           # exited cleanly, file still empty
    assert _verdict(watch, window, 19.0) != capture_server.PREVIEW_FINISHED

    output.write_bytes(b"x" * 4096)
    window.processTab._render_state = "idle"                 # the app's completion path has run too
    assert _verdict(watch, window, 19.1) == capture_server.PREVIEW_FINISHED


def test_the_apps_own_completion_path_can_stand_in_for_the_file_check(tmp_path):
    """`guiChangesOnRenderCompletion` returning the glue to `idle` is the app saying "this job is over"."""
    watch, window, _output = _watch(tmp_path)
    window.processTab._render_state = "running"
    child = _Process(None)
    window.processTab.renderProcess = child

    child.code = 0
    window.processTab._render_state = "idle"
    assert _verdict(watch, window, 20.0) == capture_server.PREVIEW_FINISHED


def test_a_nonzero_exit_is_an_error_even_with_a_file(tmp_path):
    watch, window, output = _watch(tmp_path)
    window.processTab._render_state = "running"
    child = _Process(None)
    window.processTab.renderProcess = child
    output.write_bytes(b"x" * 10)

    child.code = 1
    verdict, reason = watch.tick(window, now=21.0)
    assert verdict == capture_server.PREVIEW_FAILED
    assert reason and "code 1" in reason


def test_an_exited_child_with_no_file_and_no_completion_is_not_a_finish(tmp_path):
    watch, window, _output = _watch(tmp_path)
    window.processTab._render_state = "running"              # the app never reported completion
    child = _Process(None)
    window.processTab.renderProcess = child

    child.code = 0
    assert _verdict(watch, window, 22.0) == capture_server.PREVIEW_PENDING, (
        "the last bytes and the hand-over run after wait(); the file check gets a bounded settle window")
    verdict, reason = watch.tick(window, now=22.0 + capture_server.PREVIEW_EXIT_SETTLE_S)
    assert verdict == capture_server.PREVIEW_FAILED
    assert reason and "no preview file" in reason


# -- a refusal is the app's answer, not a finish ------------------------------------------------------------------


def test_no_child_ever_is_a_refusal_not_a_finish(tmp_path):
    """A refused click (`getCurrentRenderOptions()` -> 1, an end time before the start) leaves idle + enabled."""
    watch, window, _output = _watch(tmp_path)
    window.processTab._render_state = "idle"
    window.page_handles.proc.startRenderButton.enabled = True

    assert _verdict(watch, window, 1.0) == capture_server.PREVIEW_PENDING
    assert _verdict(watch, window, capture_server.PREVIEW_SPAWN_GRACE_S) == capture_server.PREVIEW_PENDING
    refusals = watch.tick(window, now=capture_server.PREVIEW_SPAWN_GRACE_S
                         + capture_server.PREVIEW_REFUSAL_GRACE_S + 0.1)
    assert refusals[0] == capture_server.PREVIEW_REFUSED, (
        "the app refused the preview: answer with its own toast, never with finished")


def test_a_disabled_button_with_no_child_is_not_a_finish_either(tmp_path):
    """A start the app is still preparing can leave the button disabled; that is not a verdict."""
    watch, window, _output = _watch(tmp_path)
    window.page_handles.proc.startRenderButton.enabled = False
    assert _verdict(watch, window, 10.0) == capture_server.PREVIEW_PENDING


# -- a stale child from an earlier job is not this job -------------------------------------------------------------


def test_a_stale_child_and_an_existing_file_are_not_this_jobs_finish(tmp_path):
    """`renderProcess` is not cleared between jobs; an earlier job's exit must not answer a new press."""
    watch, window, output = _watch(tmp_path)
    stale = _Process(0)                                      # an earlier render, long finished
    window.processTab.renderProcess = stale
    watch.previous_child = stale                             # recorded before the press
    output.write_bytes(b"x" * 4096)                          # an earlier preview's file, still on disk

    assert _verdict(watch, window, 1.0) == capture_server.PREVIEW_PENDING

    fresh = _Process(None)
    window.processTab.renderProcess = fresh
    window.processTab._render_state = "running"
    assert _verdict(watch, window, 2.0) == capture_server.PREVIEW_PENDING
    output.write_bytes(b"y" * 8192)
    fresh.code = 0
    assert _verdict(watch, window, 19.0) == capture_server.PREVIEW_FINISHED


def test_a_zero_byte_preview_file_is_not_output(tmp_path):
    empty = tmp_path / "empty_preview.mkv"
    empty.write_bytes(b"")
    assert capture_server.preview_output_bytes(str(empty)) is None
    empty.write_bytes(b"x")
    assert capture_server.preview_output_bytes(str(empty)) == 1
    assert capture_server.preview_output_bytes(str(tmp_path / "missing.mkv")) is None


# -- the verb applies this rule -----------------------------------------------------------------------------------


def test_the_preview_verb_applies_the_tested_rule():
    """A rule proved only in a test file proves nothing about the verb: pin the wiring to the same object."""
    source = inspect.getsource(capture_server.install)
    assert "PreviewWatch(" in source
    assert "watch.tick(window)" in source, (
        "the verdict that answers the request must come from PreviewWatch, not from a second copy of the rule")


# --- P5: the preview pane's whole state in ONE answer ------------------------------------------------

from src.ui.preview.session import PreviewSession  # noqa: E402  (after the module under test, as the file's style)


class _Clock:
    """The session's clock as it needs it: `seek_to(seconds) -> bool`."""

    def __init__(self, accepts: bool = True) -> None:
        self.calls: list[float] = []
        self.accepts = bool(accepts)

    def seek_to(self, seconds: float) -> bool:
        self.calls.append(float(seconds))
        return self.accepts


def _preview_window(session=None, *, with_controller: bool = True):
    """Only the attribute paths the preview verb reads: a controller's `session()` and the tab it lives on."""
    tab = SimpleNamespace(_preview_controller=None)
    if with_controller:
        tab._preview_controller = SimpleNamespace(session=lambda: session)
    elif session is not None:
        tab._preview_session_local = session
    return SimpleNamespace(processTab=tab)


def _state_source(**overrides):
    facts = {
        "kind": "queue_file",
        "duration": 12.0,
        "produced": 12.0,
        "played": 3.0,
        "playhead": 3.0,
        "finished": True,
        "paint": {"width_px": 400.0, "one_bar": True, "rects": []},
        "owner": {"kind": "queue_file", "may_paint": False, "may_want": False, "may_seek_any": True,
                  "has_clock": True, "has_surface": True},
    }
    facts.update(overrides)
    return facts


def _bound_session(**facts) -> PreviewSession:
    session = PreviewSession()
    session.bind(clock=_Clock(), duration=20.0, reach=20.0, frame_interval=1.0 / 24.0)
    session.set_dump_source(lambda: _state_source(**facts))
    return session


_DUMP_KEYS = ("snapshot", "ticket", "epoch", "paint", "owner")


def test_the_preview_verb_answers_the_five_keys_for_a_tab_with_a_controller():
    session = _bound_session()
    answer = capture_server.preview_state_answer(_preview_window(session))

    assert answer["kind"] == "preview" and answer["error"] is None, answer
    assert all(key in answer for key in _DUMP_KEYS), answer
    assert answer["snapshot"]["kind"] == "queue_file"
    assert answer["snapshot"]["duration"] == pytest.approx(12.0)
    assert answer["snapshot"]["produced"] == pytest.approx(12.0)
    assert answer["snapshot"]["played"] == pytest.approx(3.0)
    assert answer["snapshot"]["playhead"] == pytest.approx(3.0)
    assert answer["snapshot"]["finished"] is True
    assert answer["paint"]["width_px"] == pytest.approx(400.0)
    assert answer["owner"]["kind"] == "queue_file" and answer["owner"]["may_seek_any"] is True
    assert answer["epoch"] == 0 and answer["ticket"] is None, "no press yet: nothing to settle"


def test_the_preview_verb_without_a_session_is_a_documented_error_not_an_exception():
    """A tab that never got a controller still answers in the verb's shape: five keys, one reason."""
    answer = capture_server.preview_state_answer(_preview_window(None, with_controller=False))
    assert all(key in answer for key in _DUMP_KEYS), answer
    assert answer["error"] == "no preview session on this page", answer
    assert all(answer[key] is None for key in _DUMP_KEYS), answer

    bare = capture_server.preview_state_answer(SimpleNamespace(processTab=None))
    assert bare["error"] == "no process tab"
    assert all(key in bare for key in _DUMP_KEYS)


def test_a_dump_taken_mid_press_reports_the_open_ticket_and_the_press_epoch():
    """The verb's whole point: one read that says a press is still open, on the epoch the press opened."""
    session = _bound_session()
    assert session.seek_to(6.0) is True
    assert session.epoch == 1

    answer = capture_server.preview_state_answer(_preview_window(session))

    assert answer["ticket"]["open"] is True
    assert answer["ticket"]["target_s"] == pytest.approx(6.0)
    assert answer["epoch"] == 1
    assert answer["snapshot"]["epoch"] == 1, "the snapshot and the top-level epoch are the same generation"
    assert answer["snapshot"]["open"] is True


def test_the_widget_state_answer_still_serves_the_scrub_block():
    """The per-widget answer is not removed by the pane-wide one: its `scrub` block stays where it was."""
    source = inspect.getsource(capture_server.install)
    assert "scrub_report" in source and 'report["scrub"] = scrub_report()' in source


def test_the_preview_verb_is_its_own_branch_before_the_widget_lookup():
    """A widget lookup would answer geometry for a widget named `preview`; the pane's state needs the session."""
    source = inspect.getsource(capture_server.install)
    branch = source.index('wanted.strip() == "state:preview"')
    widget = source.index('wanted.startswith("state:")')
    assert branch < widget, "the pane's own verb must be answered before the per-widget branch"
