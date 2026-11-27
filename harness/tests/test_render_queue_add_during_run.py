"""The queue accepts new rows while a pass runs (owner, 2026-11-26).

*"добавить возможность пополнять очередь во время работы"* - a row added mid-run is rendered in the same pass, in
order, without restarting.

What was blocking it (measured on `98898708`, not guessed):

* `ProcessTab.renderToPipeThread` iterated `renderQueue.pending_jobs()` - **a snapshot taken once** - so a row added
  while the pass ran was never seen by it. The failure is reproduced by
  `test_a_snapshot_pass_misses_a_row_added_mid_run` below, which drives the old shape on purpose.
* `RenderQueue.add` silently returned while the queue was locked, and `startRender` locks it, so the row did not even
  reach the list. The lock's job is to stop rows MOVING under the consumer (remove/reorder); an append moves nothing.

The model now takes one row at a time from the live list (`take_next_pending`), with `queue`/`inputNameList` behind one
`threading.RLock` for every reader and writer. The tests below pin: order and marking, a row added from another thread,
the pass ending, a removed row not coming back, two consumers never sharing a row, and that the shipped render loop
calls `take_next_pending` rather than taking a snapshot. `test_the_add_row_stays_usable_while_a_render_runs` pins the
start path that makes the gesture reachable (the render lock used to grey the Add button out through its disabled
parent).

Red-before: with `take_next_pending` deleted, every model test here fails with `AttributeError` (the method did not
exist); with `ProcessTab.renderToPipeThread` put back on `pending_jobs()`, the loop gate fails and the snapshot test
shows the row that is never taken.
"""

from __future__ import annotations

import ast
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.ui.RenderQueue import RenderQueue

ROOT = Path(__file__).resolve().parents[2]


class _Rows:
    """The `QListWidget`-shaped surface `RenderQueue` uses, without any Qt."""

    def __init__(self) -> None:
        self.items: list[str] = []
        self.selected: set[int] = set()
        self.current = -1

    def addItem(self, text: str) -> None:
        self.items.append(text)

    def insertItem(self, index: int, text: str) -> None:
        self.items.insert(index, text)

    def clear(self) -> None:
        self.items.clear()
        self.selected.clear()
        self.current = -1

    def setItems(self, items) -> None:
        self.items = list(items)
        self.selected.clear()
        self.current = -1

    def selectedRows(self) -> list[int]:
        return sorted(self.selected)

    def setSelectedRows(self, rows) -> None:
        self.selected = {int(row) for row in rows}

    def setCurrentRow(self, index: int) -> None:
        self.current = int(index)

    def currentRow(self) -> int:
        return self.current


def _option(name: str):
    return SimpleNamespace(inputFile=name)


def _queue(*names: str) -> tuple[RenderQueue, _Rows]:
    rows = _Rows()
    queue = RenderQueue(rows)
    for name in names:
        queue.add(_option(name))
    return queue, rows


# -- 1. order, and the row is marked `rendering` as it is taken -------------------------------------------------------

def test_take_next_pending_returns_rows_in_order_and_marks_each_rendering():
    queue, _rows = _queue("a", "b", "c")

    first = queue.take_next_pending()

    assert first.inputFile == "a"
    assert first.state.status == "rendering", "a taken row is no longer `pending`, or a second take returns it again"
    assert first.state.progress == 0.0
    assert first.state.started_mono is not None, "the row's own clock starts when the pass takes it"

    second = queue.take_next_pending()
    assert second.inputFile == "b"
    assert second is not first, "the same row must not be handed out twice"
    assert [job.inputFile for job in queue.pending_jobs()] == ["c"]


# -- 2. a row added from ANOTHER THREAD while a job renders is taken by the same pass ---------------------------------

def test_a_row_added_from_another_thread_is_taken_by_the_same_pass():
    queue, _rows = _queue("first")
    produced: list[SimpleNamespace] = []
    producer_done = threading.Event()

    def producer() -> None:
        # The GUI thread's side: the Add button's path, one row at a time while the job runs.
        for index in range(40):
            job = _option(f"late-{index}")
            queue.add(job)
            produced.append(job)
            time.sleep(0.001)
        producer_done.set()

    worker = threading.Thread(target=producer, name="queue-add-producer")
    worker.start()

    taken = [queue.take_next_pending()]          # the row the pass started with
    while True:
        job = queue.take_next_pending()
        if job is None:
            if producer_done.is_set():
                break
            # The real pass is inside a backend process here; the test only has to wait for the next row.
            time.sleep(0.0005)
            continue
        taken.append(job)
        time.sleep(0.0002)                        # "render" it

    worker.join(timeout=30)
    assert not worker.is_alive()

    names = [job.inputFile for job in taken]
    expected = ["first"] + [job.inputFile for job in produced]
    assert len(names) == len(expected), "a row added mid-run was skipped, or one was taken twice"
    assert len(set(names)) == len(names), "the pass took the same row twice"
    assert names == expected, "the pass must keep queue order"


# -- 3. the pass ends when nothing is pending -------------------------------------------------------------------------

def test_the_pass_ends_with_none_after_the_last_pending_row():
    queue, _rows = _queue("a", "b")

    assert [queue.take_next_pending().inputFile, queue.take_next_pending().inputFile] == ["a", "b"]
    assert queue.take_next_pending() is None

    # A finished row stays in the list as history and is not taken again.
    queue.mark_finished(queue.getQueue()[0])
    assert queue.take_next_pending() is None


def test_the_old_snapshot_pass_misses_a_row_added_mid_run():
    """The red-before shape: one `pending_jobs()` call, then a row arrives.

    This is not a test of the product - it is the old contract, written down so the difference the change makes is a
    runnable fact rather than a claim in a commit message. The current loop must NOT behave like this
    (`test_the_render_loop_takes_from_the_live_queue_not_a_snapshot`).
    """
    queue, _rows = _queue("first")
    snapshot = queue.pending_jobs()               # what `renderToPipeThread` used to iterate

    queue.add(_option("added-while-rendering"))

    assert [job.inputFile for job in snapshot] == ["first"], "the snapshot never learns about the mid-run row"
    assert [job.inputFile for job in queue.pending_jobs()] == ["first", "added-while-rendering"], (
        "the row is in the queue - the snapshot is what lost it")
    assert [job.inputFile for job in [queue.take_next_pending(), queue.take_next_pending()]] == [
        "first", "added-while-rendering"], "the live take finds both"


# -- 4. a removed row is not taken, and removing one does not crash the pass ------------------------------------------

def test_a_row_removed_before_its_turn_is_not_taken_and_does_not_crash():
    queue, rows = _queue("a", "b", "c")
    assert queue.take_next_pending().inputFile == "a"

    rows.setSelectedRows([1])                     # "b", still pending - removed between two takes
    assert queue.removeSelected() == 1

    assert queue.take_next_pending().inputFile == "c", "a removed row must not be resurrected"
    assert queue.take_next_pending() is None
    assert [job.inputFile for job in queue.getQueue()] == ["a", "c"]


def test_removing_the_row_being_rendered_leaves_the_pass_alive():
    """The row the pass holds is detached, not destroyed: the render finishes, nothing looks it up again."""
    queue, rows = _queue("a", "b")
    running = queue.take_next_pending()

    rows.setSelectedRows([0])
    assert queue.removeSelected() == 1

    assert queue.take_next_pending().inputFile == "b"
    queue.mark_finished(running)                  # the detached job's own accounting still works
    assert queue.take_next_pending() is None
    assert [job.inputFile for job in queue.getQueue()] == ["b"]


def test_the_shipped_ui_locks_remove_but_not_add_while_a_pass_runs():
    """Recorded behaviour, because the brief's "the UI allows removing a pending row" is not what the code does.

    `startRender` -> `disableProcessPage` locks the queue; `removeSelected`/`moveSelected` refuse while locked, so a
    pending row cannot be removed from the window mid-run. The model is still written to survive it (the two tests
    above), and the next owner to unlock removal will find that guarantee already pinned.
    """
    queue, rows = _queue("a", "b")
    queue.set_locked(True)
    rows.setSelectedRows([1])

    assert queue.removeSelected() == 0
    queue.add(_option("c"))
    assert [job.inputFile for job in queue.getQueue()] == ["a", "b", "c"]


# -- 5. two consumers never take the same row (the lock's actual job) -------------------------------------------------

def test_two_consumers_never_take_the_same_row():
    queue, _rows = _queue(*[f"job-{index}" for index in range(120)])
    taken: list[str] = []
    guard = threading.Lock()

    def consume() -> None:
        while True:
            job = queue.take_next_pending()
            if job is None:
                return
            with guard:
                taken.append(job.inputFile)

    consumers = [threading.Thread(target=consume, name=f"consumer-{index}") for index in range(4)]
    for thread in consumers:
        thread.start()
    for thread in consumers:
        thread.join(timeout=30)
        assert not thread.is_alive()

    assert len(taken) == 120, f"taken {len(taken)} of 120 - the scan and the mark are not one critical section"
    assert sorted(taken) == sorted(f"job-{index}" for index in range(120))
    assert queue.take_next_pending() is None


# -- 6. the shipped loop, mechanically: it takes from the live queue --------------------------------------------------

def _calls(node: ast.AST, name: str) -> bool:
    return any(isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute) and item.func.attr == name
               for item in ast.walk(node))


def _render_loop() -> ast.FunctionDef:
    tree = ast.parse((ROOT / "src" / "ui" / "ProcessTab.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ProcessTab":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "renderToPipeThread":
                    return item
    raise AssertionError("ProcessTab.renderToPipeThread is gone")


def test_the_render_loop_takes_from_the_live_queue_not_a_snapshot():
    loop = _render_loop()

    taking = [node for node in ast.walk(loop)
              if isinstance(node, ast.While)
              and any(isinstance(sub, ast.NamedExpr) and _calls(sub.value, "take_next_pending")
                      for sub in ast.walk(node.test))]
    assert taking, ("renderToPipeThread no longer iterates `take_next_pending()` - a row added while the pass runs is "
                    "invisible to it again")
    assert not _calls(loop, "pending_jobs"), (
        "the pass is back on a one-shot `pending_jobs()` snapshot: rows added mid-run would never be rendered")


# -- 7. the start path: the Add row is usable while a render runs -----------------------------------------------------

def _window_methods(*names: str) -> dict:
    """Compile `MainWindow`'s named methods on their own.

    The entry module cannot be imported here (it pulls `cpuinfo` and the whole application), so the method bodies are
    extracted from the shipped file by AST and bound to a stand-in window by the test. What runs is the real source.
    """
    source = (ROOT / "REAL-Video-Enhancer.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    window = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MainWindow")
    found = [node for node in window.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in found} == set(names), f"missing methods: {set(names) - {n.name for n in found}}"
    namespace: dict = {}
    exec(compile(ast.Module(body=found, type_ignores=[]), "REAL-Video-Enhancer.py", "exec"), namespace)
    return namespace


def test_the_add_row_stays_usable_while_a_render_runs(qapp):
    pytest.importorskip("sli_ui_toolkit")
    from PySide6.QtWidgets import QWidget

    if "resources_rc" not in sys.modules:          # the page imports the compiled resource module
        sys.modules["resources_rc"] = types.ModuleType("resources_rc")
    from src.ui.pages.general_settings import GeneralSettingsPage

    page = GeneralSettingsPage()
    page.build()

    class _Queue:
        locked = False

        def set_locked(self, locked: bool) -> None:
            self.locked = bool(locked)

    window = SimpleNamespace(
        page_handles=SimpleNamespace(
            general=page,
            proc=SimpleNamespace(RenderedPreviewControlsContainer=QWidget()),
            advanced=SimpleNamespace(advancedSettings=QWidget()),
            encoder=SimpleNamespace(encoderSettings=QWidget()),
        ),
        renderQueue=_Queue(),
    )
    disable = types.MethodType(_window_methods("disableProcessPage")["disableProcessPage"], window)
    enable = types.MethodType(_window_methods("enableProcessPage")["enableProcessPage"], window)

    disable()

    assert window.renderQueue.locked is True
    assert page.addToRenderQueueButton.isEnabled(), (
        "the Add button is greyed during a render - the queue cannot be topped up from the window")
    assert page.inputFileSelectButton.isEnabled(), "the next file cannot be chosen while a render runs"
    assert page.batchSelectButton.isEnabled()
    assert not page.inputFileText.isEnabled(), (
        "the path field stays read-only: only a probed file (the Select button) may become a job")
    assert not page.upscaleModelComboBox.isEnabled(), (
        "the settings form must stay locked: a running job's options are not editable")
    assert not page.qualityPresetComboBox.isEnabled()

    enable()

    assert window.renderQueue.locked is False
    assert page.upscaleModelComboBox.isEnabled(), "enableProcessPage must undo every lock disableProcessPage set"
    assert page.qualityPresetComboBox.isEnabled()
    assert page.addToRenderQueueButton.isEnabled()
