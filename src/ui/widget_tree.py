"""One rule about widget trees: a page's children are not all widgets.

A page root carries plain `QObject`s too - the i18n relay (`src/ui/i18n._PageRelay`) is one, parented under every page
root so the toolkit's retranslate walk can reach it. Code that iterates `parent.children()` and calls a widget method on
each child therefore crashes on a relay: measured live 2026-11-24, `startRender` -> `disableProcessPage` raised
`AttributeError: '_PageRelay' object has no attribute 'setEnabled'`, so no render could start at all while the offscreen
suite stayed green (nothing in it starts a render).

The fix is this filter, in one place, so the next pass that walks a page's children does not reinvent the collapse.
"""

from __future__ import annotations

from PySide6.QtWidgets import QWidget


def enabled_widget_children(parent) -> list:
    """The parent's DIRECT children that are widgets - the ones ``setEnabled`` means anything for.

    Direct children, not descendants: the callers state "the page's top-level pieces", and a descendant walk would
    reach into control internals and disable parts of a toolkit control.
    """
    if parent is None:
        return []
    return [child for child in parent.children() if isinstance(child, QWidget)]


__all__ = ["enabled_widget_children"]
