"""The window's one exit path joins the QThreads it owns.

Owner, 2026-11-28: `QThread: Destroyed while thread is still running`. The notes fetch is a 10 s network request in a
`QThread` that nothing waited for, so closing the window while the request was in flight let the Python object die under
its own thread. `ProcessTab`'s 100 ms clock, live tail and compare decoder had the same shape on their own paths.

Pinned here: the exit path calls both stoppers, the notes stopper actually waits, and the process tab's stopper joins a
running clock thread instead of dropping it.
"""

from __future__ import annotations

import ast
import pathlib

from PySide6.QtWidgets import QApplication
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _function(path: pathlib.Path, name: str) -> ast.FunctionDef:
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {path}")


def _called_names(node: ast.AST) -> set[str]:
    return {
        getattr(call.func, "attr", "") or getattr(call.func, "id", "")
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
    }


def test_the_exit_path_joins_both_owners():
    body = _function(pathlib.Path("REAL-Video-Enhancer.py"), "shutdown")
    names = _called_names(body)
    assert "stop_notes_fetch" in names, "the window must join the notes fetch on the way out"
    assert "stop_all_threads" in names, "the process tab's threads must be joined on the way out"


def test_the_notes_stopper_waits():
    body = _function(pathlib.Path("src/ui/HomeTab.py"), "stop_notes_fetch")
    names = _called_names(body)
    assert "wait" in names, "a stoppable QThread is joined with `wait`, not dropped"


def test_the_process_stopper_joins_the_clock_tail_and_decoder():
    body = _function(pathlib.Path("src/ui/ProcessTab.py"), "stop_all_threads")
    names = _called_names(body)
    for expected in ("stop", "quit", "wait", "_stop_live_tail", "_stop_source_decoder"):
        assert expected in names, f"stop_all_threads must {expected} its threads"


class _FakeFetcher:
    def __init__(self, running: bool = True) -> None:
        self.running = running
        self.interrupted = False
        self.waited: int | None = None

    def isRunning(self) -> bool:  # noqa: N802 - Qt spelling
        return self.running

    def requestInterruption(self) -> None:  # noqa: N802 - Qt spelling
        self.interrupted = True

    def wait(self, ms: int) -> bool:  # noqa: N802 - Qt spelling
        self.waited = ms
        self.running = False
        return True


@pytest.fixture()
def qapp():
    return QApplication.instance() or QApplication([])


def test_stop_notes_fetch_interrupts_and_waits_then_forgets_the_thread(qapp):
    from src.ui.HomeTab import HomeTab

    page = object.__new__(HomeTab)          # only the fetcher slot matters here; no window, no network
    fetcher = _FakeFetcher(running=True)
    page._fetcher = fetcher
    page.stop_notes_fetch(1234)
    assert fetcher.interrupted is True
    assert fetcher.waited == 1234, "the join must be bounded, not infinite"
    assert page._fetcher is None, "the thread object is not reused after the join"


def test_stop_notes_fetch_is_silent_when_the_fetch_already_finished(qapp):
    from src.ui.HomeTab import HomeTab

    page = object.__new__(HomeTab)
    fetcher = _FakeFetcher(running=False)
    page._fetcher = fetcher
    page.stop_notes_fetch()
    assert fetcher.interrupted is False, "nothing to interrupt when the thread is done"
    assert fetcher.waited is None
    assert page._fetcher is None
