"""The overwrite question is asked BEFORE the window changes anything.

Owner, 2026-11-28: the overwrite notice "жестко ебашит интерфейс что тот подвисает". The cause was an ordering one:
`MainWindow.startRender` disabled the page, hid the QRhi preview and told the transport "running" and only THEN called
`ProcessTab.run`, whose first interesting act was the modal. So the person watched the UI switch to a render state and
stall (hiding a `QRhiWidget` rebuilds its surface - on Wayland that is where `eglSwapBuffers failed` comes from) and was
asked afterwards whether to render at all.

The contract pinned here is structural, because the fault WAS structural: in `startRender`, the confirmation comes
before the first mutation, and the subsequent `run(...)` is told not to ask again (a second modal would be the old bug
in the other direction).
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "REAL-Video-Enhancer.py"

#: The mutations that used to happen before the question: the start button's enabled state, the preview's and the
#: label's visibility, and the page-wide lock.
MUTATIONS = ("disableProcessPage", "setEnabled", "setVisible")


def _start_render_body() -> ast.FunctionDef:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "startRender":
            return node
    raise AssertionError("MainWindow.startRender not found - did it get renamed?")


def test_the_confirmation_precedes_every_state_change():
    body = _start_render_body()
    asked = [
        node.lineno
        for node in ast.walk(body)
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "checkForOverwrite"
    ]
    assert asked, "startRender must ask the overwrite predicate itself, before it mutates anything"

    mutated = [
        node.lineno
        for node in ast.walk(body)
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") in MUTATIONS
    ]
    assert mutated, "the mutations should still be here - this test is about their ORDER, not their removal"
    assert min(asked) < min(mutated), (
        f"the overwrite question (line {min(asked)}) must come BEFORE the first state change "
        f"(line {min(mutated)}): a question about whether to start must not change what it asks about")


def test_the_queue_run_is_told_the_question_was_answered():
    body = _start_render_body()
    runs = [
        node
        for node in ast.walk(body)
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "run"
    ]
    assert len(runs) == 1, f"expected one `run(...)` call in startRender, found {len(runs)}"
    keywords = {kw.arg: kw.value for kw in runs[0].keywords}
    assert "confirm_overwrite" in keywords, (
        "startRender must pass confirm_overwrite=False: the question was already answered above, and a second modal "
        "would be the old defect reversed")
    value = keywords["confirm_overwrite"]
    assert isinstance(value, ast.Constant) and value.value is False, (
        f"confirm_overwrite must be the literal False, got {ast.dump(value)}")


def _function(name: str, source: Path = Path("src/ui/ProcessTab.py")) -> ast.FunctionDef:
    tree = ast.parse((ROOT / source).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {source}")


def test_a_no_changes_nothing_so_there_is_nothing_to_restore():
    """"No" must not call the completion/restore pair: with the question first, nothing was changed to undo.

    Owner, 2026-11-28: "ещё лучше сам путь изменить - убрать лишние вызовы на случай No или же Yes". The pair existed
    because the old order had already locked the page and switched the transport to "running" before asking; keeping it
    would be a restore path with no state to restore.
    """
    body = _function("askForOverwrite")
    calls = {
        # A plain `would_ask_for_overwrite(...)` is a Name, a method is an Attribute - take either spelling.
        getattr(node.func, "attr", "") or getattr(node.func, "id", "")
        for node in ast.walk(body)
        if isinstance(node, ast.Call)
    }
    assert "would_ask_for_overwrite" in calls, "the predicate is still the subject"
    assert "questionToOverride" in calls, "this method is the person's path - it must still ask"
    for forbidden in ("onRenderCompletion", "guiChangesOnRenderCompletion"):
        assert forbidden not in calls, (
            f"askForOverwrite must not call {forbidden}(): the question now comes before any state change")


def test_run_asks_before_it_touches_the_pane():
    """A direct `run()` caller gets the same guarantee: "No" leaves the transport and the pane untouched."""
    body = _function("run")
    asked = [
        node.lineno
        for node in ast.walk(body)
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "checkForOverwrite"
    ]
    assert asked, "run() must ask the predicate itself so a direct caller cannot start without the answer"
    touched = [
        node.lineno
        for node in ast.walk(body)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", "") in ("_stop_preview_playback", "_stop_live_tail",
                                               "_sync_preview_transport", "setStatus", "_set_render_controls")
    ]
    assert touched, "the pane's preparation should still exist - this is about its ORDER"
    assert min(asked) < min(touched), (
        f"the question (line {min(asked)}) must precede the first pane change (line {min(touched)})")
