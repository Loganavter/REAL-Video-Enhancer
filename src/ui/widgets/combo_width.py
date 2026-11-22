"""Keep a combo as wide as its content, but never wider than the row it lives in.

The field's width is its longest item (a field that elides its own selected text is a bug), and a long item - a model
name, an encoder string - would happily push the row past the pane. The cap is the space the row actually has for the
field: the parent's width minus its layout margins, the other items' hints and the spacing between them.

The cap follows the parent's width, so a window resize re-computes it; a `Maximum` size policy then keeps the field at
`min(content, available)` and lets it shrink, never grow.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import QSizePolicy

#: A field narrower than this is unusable even in a cramped row; below it the row scrolls instead of the text hiding.
MINIMUM_FIELD_WIDTH = 96


class _RowWidthCap(QObject):
    def __init__(self, combo) -> None:
        super().__init__(combo)
        self._combo = combo
        parent = combo.parentWidget()
        if parent is not None:
            parent.installEventFilter(self)
        # The combo's own hint changes with its items; a `Maximum` policy alone let the row hand it spare space
        # (measured 231 px for a 139 px hint), so the cap is computed here, not delegated to the policy.
        combo.installEventFilter(self)
        self.apply()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API
        if event.type() in (QEvent.Type.Resize, QEvent.Type.Show, QEvent.Type.LayoutRequest):
            self.apply()
        return False

    def available_width(self) -> int:
        parent = self._combo.parentWidget()
        layout = parent.layout() if parent is not None else None
        if parent is None or layout is None:
            return 0
        margins = layout.contentsMargins()
        occupied = 0
        others = 0
        for index in range(layout.count()):
            item = layout.itemAt(index)
            widget = item.widget() if item is not None else None
            if widget is self._combo or widget is None:
                continue
            others += 1
            occupied += item.sizeHint().width()
        spacing = layout.spacing() * max(0, others)
        return parent.width() - margins.left() - margins.right() - occupied - spacing

    def apply(self) -> None:
        available = self.available_width()
        if available <= 0:
            return
        wanted = self._combo.sizeHint().width()
        self._combo.setMaximumWidth(max(MINIMUM_FIELD_WIDTH, min(available, wanted)))


def cap_combo_to_row(combo) -> None:
    """Cap `combo` at the width its row can give it; idempotent."""
    if combo is None or getattr(combo, "_row_width_cap", None) is not None:
        return
    combo.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
    combo._row_width_cap = _RowWidthCap(combo)
