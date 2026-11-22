"""`RenderQueue` behaviour: selection, block moves, and the render lock.

The queue page lost its three buttons (owner, 2026-11-22): actions come from the right-click menu and apply to the whole
selection, so the model has to remove and reorder several rows at once and keep the three parallel lists (options,
names, visible rows) in step. It also must refuse those mutations while a render walks the queue - but stay usable for
scrolling, which is why the lock is here and not `setEnabled(False)` on the page.

The widget is stubbed at exactly the compat surface the manager calls, so this runs on the portable interpreter without
PySide6; the real selection gestures are covered by the toolkit's own list tests.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.ui.RenderQueue import RenderQueue


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


def test_removing_a_selection_removes_every_row_in_all_three_lists():
    queue, rows = _queue("a", "b", "c", "d")
    rows.setSelectedRows([1, 3])

    assert queue.removeSelected() == 2

    assert [option.inputFile for option in queue.getQueue()] == ["a", "c"]
    assert queue.inputNameList == ["a", "c"]
    assert rows.items == ["a", "c"]


def test_removing_nothing_is_a_no_op():
    queue, rows = _queue("a")
    assert queue.removeSelected() == 0
    assert rows.items == ["a"]


def test_a_selected_block_moves_together():
    queue, rows = _queue("a", "b", "c", "d")
    rows.setSelectedRows([1, 2])

    assert queue.moveSelected("up") is True

    assert [option.inputFile for option in queue.getQueue()] == ["b", "c", "a", "d"]
    assert rows.items == ["b", "c", "a", "d"]
    assert rows.selected == {0, 1}, "the block stays selected after the move"


def test_a_block_at_the_edge_moves_as_far_as_it_can_and_stops():
    queue, rows = _queue("a", "b", "c")
    rows.setSelectedRows([0, 1])
    assert queue.moveSelected("up") is False
    assert rows.items == ["a", "b", "c"]

    rows.setSelectedRows([1, 2])
    assert queue.moveSelected("down") is False


def test_moving_down_keeps_the_selection_on_the_same_items():
    queue, rows = _queue("a", "b", "c", "d")
    rows.setSelectedRows([0])

    assert queue.moveSelected("down") is True

    assert rows.items == ["b", "a", "c", "d"]
    assert [rows.items[row] for row in rows.selected] == ["a"]


def test_a_locked_queue_refuses_mutations_but_still_reports_its_rows():
    queue, rows = _queue("a", "b")
    rows.setSelectedRows([0])
    queue.set_locked(True)
    assert queue.is_locked() is True

    assert queue.removeSelected() == 0
    assert queue.moveSelected("down") is False
    queue.add(_option("c"))
    assert rows.items == ["a", "b"], "the running render's list must not change underneath it"

    queue.set_locked(False)
    assert queue.moveSelected("down") is True


def test_clear_resets_all_three_lists():
    queue, rows = _queue("a", "b")
    rows.setSelectedRows([0])
    queue.clear()
    assert queue.getQueue() == []
    assert queue.inputNameList == []
    assert rows.items == []
