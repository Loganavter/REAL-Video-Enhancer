"""The tail's seek slot is taken atomically (audit 2026-11-27 §6.9).

`ask, self._pending_seek = self._pending_seek, None` evaluates the right-hand side, then stores both names: a `seek()`
from the GUI thread landing between the read and the `None` store was overwritten by that store, so the press vanished -
the tail kept its old position and the ticket eventually gave up. The window is microseconds, which is why it was found by
reading and never by a run, and why this file pins the SHAPE of the code: the setter and the take must share one lock, and
the take must read and clear inside one held section.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

ROOT = Path(__file__).resolve().parents[2]
TAIL = ROOT / "src" / "ui" / "video" / "tail.py"


def _method(cls_name: str, name: str) -> ast.FunctionDef:
    """The `name` method of `cls_name` - both `TailReader` and `LiveTailThread` define a `seek`, and only the second one
    owns the slot."""
    tree = ast.parse(TAIL.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
    raise AssertionError(f"{cls_name}.{name} is not defined in {TAIL}")


def _lock_blocks(method: ast.FunctionDef, attribute: str = "_seek_lock") -> list[ast.With]:
    """Every `with self.<attribute>:` in the method."""
    return [node for node in ast.walk(method)
            if isinstance(node, ast.With) and any(
                isinstance(item.context_expr, ast.Attribute) and item.context_expr.attr == attribute
                for item in node.items)]


def test_the_setter_holds_the_lock():
    """`LiveTailThread.seek` is the GUI thread's writer; without the lock the loop can clear under it."""
    assert _lock_blocks(_method("LiveTailThread", "seek")), (
        "seek() writes the seek slot without `_seek_lock`: a take between the write and the next poll can clear it")


def test_the_take_reads_and_clears_inside_one_section():
    """The loop must read `_pending_seek` and store None under the SAME held lock - that is what makes take+clear atomic.

    A `with` that only wraps one of the two operations still has the window.
    """
    blocks = _lock_blocks(_method("LiveTailThread", "run"))
    assert blocks, "the loop takes the seek slot without `_seek_lock`"

    def reads_slot(node: ast.AST) -> bool:
        return any(isinstance(n, ast.Attribute) and n.attr == "_pending_seek" for n in ast.walk(node))

    def clears_slot(node: ast.AST) -> bool:
        for n in ast.walk(node):
            if isinstance(n, ast.Assign):
                for target in n.targets:
                    if isinstance(target, ast.Attribute) and target.attr == "_pending_seek" and isinstance(n.value, ast.Constant) and n.value.value is None:
                        return True
            if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Attribute) and n.target.attr == "_pending_seek":
                if isinstance(n.value, ast.Constant) and n.value.value is None:
                    return True
        return False

    assert any(reads_slot(block) and clears_slot(block) for block in blocks), (
        "the read and the clear of `_pending_seek` are not in one locked section: a seek() between them is lost")
