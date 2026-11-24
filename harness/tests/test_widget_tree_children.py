"""A page's direct children are not all widgets, and the pass that disables a page must know it.

Measured live 2026-11-24: `startRender` -> `disableProcessPage` iterated `parent.children()` and called
`child.setEnabled(False)` on everything. The i18n lane had just parented a `_PageRelay` (a plain `QObject`) under each
page root, so that call raised

    AttributeError: '_PageRelay' object has no attribute 'setEnabled'

inside `startRender` - no render could start, and `render:`/`preview:` answered nothing at all while the offscreen suite
stayed green (nothing in it starts a render). `src/ui/widget_tree.enabled_widget_children` is the one filter, and these
tests keep both halves honest: it must drop a non-widget child, and it must still hand back every widget child (a filter
so aggressive that the page is never disabled would be the opposite bug).
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject  # noqa: E402
from PySide6.QtWidgets import QWidget  # noqa: E402

from src.ui.widget_tree import enabled_widget_children  # noqa: E402


def test_a_plain_qobject_child_is_not_handed_back(qapp):
    """The exact shape that crashed: a widget root with a relay (QObject) and a widget child."""
    parent = QWidget()
    QWidget(parent)                       # a widget child
    QObject(parent)                       # the relay stand-in
    returned = enabled_widget_children(parent)
    assert len(returned) == 1
    assert all(isinstance(child, QWidget) for child in returned)
    assert not any(type(child) is QObject for child in returned), (
        "a plain QObject child reached the caller, and `setEnabled` on it is the AttributeError that killed startRender")


def test_every_widget_child_is_kept(qapp):
    """The pass must still disable the page: a filter that skips widgets is the opposite failure."""
    parent = QWidget()
    first, second = QWidget(parent), QWidget(parent)
    QObject(parent)
    assert set(enabled_widget_children(parent)) == {first, second}


def test_no_parent_is_an_empty_list_not_a_crash(qapp):
    assert enabled_widget_children(None) == []
