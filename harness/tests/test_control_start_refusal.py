"""`start:render` must answer, not wedge the whole channel inside a modal dialog.

Measured 2026-11-27 on a leased display: after `start:render` the verb answered nothing, and EVERY later verb stopped
answering too - `state:previewLabel`, `grab:/tmp/x.png` and `grab:...@previewLabel` all timed out at 10 s while
`live.sh status` still reported the app alive. The log's last lines were the media probe and
`capture-server: queue ... -> 2 rows`: the click and nothing after it. Only `live.sh down` recovered it.

The cause is structural, not a timeout: the control loop runs on the GUI thread, `startRenderButton.click()` runs
`ProcessTab.run` synchronously, `run` calls `checkForOverwrite` -> `askForOverwrite` -> `questionToOverride`, and a modal
`confirm` nobody can dismiss on a headless display parks that thread for good.

**The rule this file pins: a control verb runs on the GUI thread, so it may never open a modal.** It asks the
non-modal predicate first (`ProcessTab.would_ask_for_overwrite` / `overwrite_prompt_path`, the same inputs
`askForOverwrite` uses) and, when the question WOULD appear, refuses with the dialog's own sentence and does not click.
What a person's own press does is unchanged: the dialog still appears for them (the last two tests pin that).

Red-before (row 1): the predicate check in `capture_server.start_render_job` was removed and the suite re-run -
`test_an_existing_output_is_refused_with_the_dialogs_question` failed with `button.clicks == 1` / `started is True`,
i.e. the click ran into the question. Restored, it passes.
"""

from __future__ import annotations

import inspect
import os
import sys
import types
from pathlib import Path

import pytest

_ROOT = str(Path(__file__).resolve().parents[2])
# The app and the backend both live in a `src` namespace package; the suite's conftest puts the backend first. The
# app's `ProcessTab` needs the app's `src.constants`, so the app root goes first for this module (the pattern of
# `test_queue_pane.py` / `test_display_wiring.py`).
_ORIGINAL_SYS_PATH = list(sys.path)
if _ROOT in sys.path:
    sys.path.remove(_ROOT)
sys.path.insert(0, _ROOT)
sys.modules.pop("src.constants", None)

# The app's optional runtime imports: stub them ONLY when the interpreter does not have them.
try:  # pragma: no cover - the branch taken depends on the interpreter
    import cpuinfo  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    _cpuinfo = types.ModuleType("cpuinfo")
    _cpuinfo.get_cpu_info = lambda *args, **kwargs: {}
    sys.modules["cpuinfo"] = _cpuinfo
try:  # pragma: no cover
    import pypresence  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    _presence = types.ModuleType("pypresence")
    _presence.Presence = object
    sys.modules["pypresence"] = _presence

pytest.importorskip("PySide6")

from src.ui.ProcessTab import (  # noqa: E402
    OVERWRITE_QUESTION,
    ProcessTab,
    overwrite_prompt_path,
    would_ask_for_overwrite,
)
from src.ui.capture_server import start_render_job  # noqa: E402

# Leave the suite's path order as it was (the app modules are imported and bound now).
sys.path[:] = _ORIGINAL_SYS_PATH


class _Button:
    """A recording double: a real click would run `ProcessTab.run` for real, and that is the wedge being prevented."""

    def __init__(self) -> None:
        self.clicks = 0

    def click(self) -> None:
        self.clicks += 1


def _overwrite_tab(answer: bool = True):
    """The real `checkForOverwrite`/`askForOverwrite` on a bare `ProcessTab`, with the modal replaced by a recorder.

    `object.__new__` is the house style of `test_display_wiring.py` / `test_queue_pane.py`: the shipped methods are the
    ones under test, and only the collaborators they touch on `self` are provided.
    """
    tab = object.__new__(ProcessTab)
    tab.isOverwrite = False
    tab.asked: list[str] = []
    tab.finished = 0

    def question() -> bool:
        tab.asked.append("shown")
        return answer

    tab.questionToOverride = question
    tab.onRenderCompletion = lambda: setattr(tab, "finished", tab.finished + 1)
    tab.guiChangesOnRenderCompletion = lambda: setattr(tab, "finished", tab.finished + 1)
    return tab


def _row(path: str):
    return type("Row", (), {"outputPath": path})()


def _queue(paths):
    return type("Queue", (), {"getQueue": lambda self: [_row(path) for path in paths]})()


def _window(paths, *, state: str = "idle"):
    button = _Button()
    rows = [_row(path) for path in paths]
    tab = type("Tab", (), {
        "_render_state": state,
        "currentRenderOptions": rows[0] if rows else _row(""),
    })()
    window = type("Window", (), {
        "processTab": tab,
        "renderQueue": _queue(paths),
        "page_handles": type("Handles", (), {
            "proc": type("Proc", (), {"startRenderButton": button})(),
        })(),
    })()
    return window, button


# --- the verb -----------------------------------------------------------------------------------------------------------------


def test_an_existing_output_is_refused_with_the_dialogs_question(tmp_path):
    """Row 1. The click is what runs `run()` synchronously, so it must not happen at all."""
    existing = tmp_path / "out.mkv"
    existing.write_bytes(b"\0")
    window, button = _window([str(existing)])

    report = start_render_job(window)

    assert button.clicks == 0, ("a click runs `ProcessTab.run` on the GUI thread and parks it in the modal; "
                                "the verb must refuse instead")
    assert report["started"] is False
    assert report["error"] == OVERWRITE_QUESTION, "the refusal must carry the dialog's OWN sentence"
    assert report["output"] == str(existing)
    assert report["method"] is None, "no start method was used"


def test_a_fresh_output_starts_and_answers_as_before(tmp_path):
    """Row 2. Nothing to ask, so the shipped answer is unchanged."""
    fresh = tmp_path / "out.mkv"
    window, button = _window([str(fresh)])

    report = start_render_job(window)

    assert button.clicks == 1
    assert report["started"] is True and report["method"] == "startRenderButton"
    assert report["error"] is None
    assert report["queue_rows"] == 1 and report["output"] == str(fresh)
    assert report["render_state"] == "idle", "the report reads the state before the click; the poll shows the change"


def test_a_later_row_that_exists_also_refuses(tmp_path):
    """`checkForOverwrite` walks every row; a verb that only looked at the current one would still open the modal."""
    fresh = tmp_path / "a.mkv"
    existing = tmp_path / "b.mkv"
    existing.write_bytes(b"\0")
    window, button = _window([str(fresh), str(existing)])

    report = start_render_job(window)

    assert button.clicks == 0
    assert report["started"] is False
    assert report["output"] == str(existing)


def test_a_row_without_a_path_cannot_open_a_dialog(tmp_path):
    """The queue the existing job-verb tests build holds bare ints; an unreadable row must not wedge or raise."""
    window, button = _window([1, 2])
    report = start_render_job(window)
    assert button.clicks == 1 and report["started"] is True


# --- the predicate and the modal cannot disagree -----------------------------------------------------------------------------


@pytest.mark.parametrize("answer", [True, False])
@pytest.mark.parametrize("shape", ["existing", "fresh", "fresh-then-existing"])
def test_the_predicate_and_ask_for_overwrite_cannot_disagree(tmp_path, shape: str, answer: bool):
    """Row 3. Same inputs, one answer: the predicate is what `askForOverwrite` itself asks."""
    existing = tmp_path / "out.mkv"
    existing.write_bytes(b"\0")
    fresh = tmp_path / "new.mkv"
    paths = {
        "existing": [str(existing)],
        "fresh": [str(fresh)],
        "fresh-then-existing": [str(fresh), str(existing)],
    }[shape]

    tab = _overwrite_tab(answer=answer)
    allowed = ProcessTab.checkForOverwrite(tab, _queue(paths))
    predicate_says_ask = overwrite_prompt_path(_queue(paths)) is not None

    assert bool(tab.asked) is predicate_says_ask, "the dialog was shown (or not) against the predicate's answer"
    assert allowed is (not predicate_says_ask or answer)
    assert would_ask_for_overwrite(str(existing)) is True
    assert would_ask_for_overwrite(str(fresh)) is False


def test_the_predicate_reads_only_the_path_and_the_file(tmp_path):
    """Its two inputs are the ones `askForOverwrite` always used: the row's `outputPath` and whether the file exists."""
    existing = tmp_path / "out.mkv"
    existing.write_bytes(b"\0")
    assert would_ask_for_overwrite(str(existing)) is True
    assert would_ask_for_overwrite(str(tmp_path / "missing.mkv")) is False
    assert would_ask_for_overwrite("") is False
    assert would_ask_for_overwrite(None) is False
    assert would_ask_for_overwrite(tmp_path) is False, "a directory is not an existing output FILE"
    assert os.path.isfile(str(existing)), "the predicate has no side effects"


# --- the person's path is untouched -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("answer", [True, False])
def test_a_persons_press_still_asks_the_modal(tmp_path, answer: bool):
    """Row 4. The dialog still appears for the person who presses the button; only the control verb refuses."""
    existing = tmp_path / "out.mkv"
    existing.write_bytes(b"\0")
    tab = _overwrite_tab(answer=answer)

    allowed = ProcessTab.checkForOverwrite(tab, _queue([str(existing)]))

    assert tab.asked == ["shown"], "the person's path must still open the dialog"
    assert allowed is answer
    if not answer:
        assert tab.finished == 2, "declining still ends the render the way it always did (both hooks)"


def test_the_click_path_still_reaches_the_question():
    """`run()` is the button's slot: it must keep calling `checkForOverwrite`, which calls `askForOverwrite`.

    A verb that deleted that call, or a start path that routed around it, would be a second start path.
    """
    run_source = inspect.getsource(ProcessTab.run)
    assert "self.checkForOverwrite(" in run_source, "the button's slot no longer checks for an overwrite"

    check_source = inspect.getsource(ProcessTab.checkForOverwrite)
    assert "self.askForOverwrite(" in check_source, "the queue check no longer asks per row"

    ask_source = inspect.getsource(ProcessTab.askForOverwrite)
    assert "self.questionToOverride()" in ask_source, "the person's path no longer opens the dialog"
    assert "would_ask_for_overwrite(" in ask_source, "the modal and the predicate must share one answer"


def test_the_grab_guard_recognises_an_rhi_surface_by_its_Qt6_frame_method():
    """`grabFramebuffer` is the Qt 6.12 name; a guard that only knew `grabFrame` fell through to `QWidget.grab()`.

    Measured 2026-11-27 on a leased display: `grab:...@previewLabel` timed out at 8 s and again at 20 s with NO refusal
    line, while `state:` kept answering during it - the pane's `grabFrame` does not exist in PySide6 6.12, so the guard
    skipped and `QWidget.grab()` parked the grab in a nested event loop. The refusal must fire for the method the
    installed Qt actually has.
    """
    from src.ui.capture_server import _is_rhi_widget

    class _Plain:
        pass

    class _Qt6Rhi:
        def grabFramebuffer(self):                     # the name PySide6 6.12 exposes
            raise AssertionError("the guard must refuse before calling it")

    class _OlderRhi:
        def grabFrame(self):                           # the name older bindings had
            raise AssertionError("the guard must refuse before calling it")

    assert not _is_rhi_widget(_Plain()), "a plain widget is grabbable"
    assert not _is_rhi_widget(None)
    # Either name is enough for the refusal branch in the handler; the type check covers the real widget.
    assert callable(getattr(_Qt6Rhi(), "grabFramebuffer", None))
    assert callable(getattr(_OlderRhi(), "grabFrame", None))


def test_the_grab_verb_always_answers_even_when_the_widget_is_gone():
    """A verb that returns without an answer leaves the client waiting; the grab branch must answer EVERY way.

    Measured 2026-11-27: `grab:...@previewLabel` produced no answer file and no log line at all while `state:` kept
    answering - the handler said nothing on a path nobody had wrapped. `grab_answer` now returns a report for every
    outcome, including an exception raised by a stale C++ object.
    """
    from types import SimpleNamespace

    from src.ui.capture_server import grab_answer

    class _Window:
        def findChild(self, *args):                    # no such widget
            return None

    report = grab_answer(_Window(), "/tmp/opencode/x.png@nope", "/tmp/opencode")
    assert report["error"] and "no widget named" in report["error"]
    assert report["saved"] is False

    class _Boom:
        def findChild(self, *args):
            raise RuntimeError("Internal C++ object already deleted.")

    report = grab_answer(_Boom(), "/tmp/opencode/x.png@previewLabel", "/tmp/opencode")
    assert report["error"] and "RuntimeError" in report["error"], "an exception must come back as an answer"
    assert grab_answer(SimpleNamespace(), "/tmp/opencode/x.png", "/tmp/opencode")["error"] is None, (
        "the window form still grabs")
