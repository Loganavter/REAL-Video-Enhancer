"""Keep a combo as wide as its content, but never wider than the pane it lives in.

The field's width is its longest item (a field that elides its own selected text is a bug), and a long item - a model
name, an encoder string - would happily push the row past the pane. The cap is the space the row can actually give the
field: the nearer of the pane's content width and the row's own width, less the layout margins, the other items' hints
and the spacing between them.

Why the PANE is an input and not only the row (measured 2026-11-24, the owner's "Settings and back breaks the middle
of the layout"): a `Maximum` combo is clamped into the row's minimum hint, so the row's width is partly the cap's own
output. Reading it back is circular, and the loop had two visible ends in the process column at 1.0:

* enabling upscale makes the vertical scrollbar appear, the viewport drops 478 -> 470, and the row - laid out at the
  wider pre-scrollbar pass - kept its 438 px: the content settled at 478 against a 470 viewport, so every field in the
  General tab was clipped at the column's edge;
* a return from Settings re-laid the row at a transiently wider width, the cap took it as a new floor (+8), and the
  content grew to 486. It never came back, because nothing recomputed the cap once the row stopped resizing.

The pane's viewport is not derived from the cap, so it is the stable input: the cap follows it, and a viewport resize
(the scrollbar appearing is one) re-applies. Where a row has no enclosing scroll area (a dialog, a test stand) the
row's own width stays the only statement available.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import QAbstractScrollArea, QSizePolicy, QWidget

#: A field narrower than this is unusable even in a cramped row; below it the row scrolls instead of the text hiding.
MINIMUM_FIELD_WIDTH = 96


def _pane_anchor(row: QWidget) -> QWidget | None:
    """The scroll viewport `row` is laid out inside, or None when the row lives outside a scroll area."""
    widget = row.parentWidget()
    while widget is not None:
        ancestor = widget.parentWidget()
        if isinstance(ancestor, QAbstractScrollArea):
            return ancestor.viewport()
        if ancestor is None:
            return None
        widget = ancestor
    return None


def _pane_content_width(row: QWidget) -> int | None:
    """The width the pane gives `row`'s row: the enclosing viewport less every layout margin above the row.

    The row's own layout margins are NOT included - the caller subtracts them once, so the two answers (row width and
    pane width) are compared on the same terms. None means "no scroll area above this row".
    """
    insets = 0
    widget = row.parentWidget()
    while widget is not None:
        layout = widget.layout()
        if layout is not None:
            margins = layout.contentsMargins()
            insets += margins.left() + margins.right()
        ancestor = widget.parentWidget()
        if isinstance(ancestor, QAbstractScrollArea):
            return max(0, ancestor.viewport().width() - insets)
        if ancestor is None:
            return None
        widget = ancestor
    return None


class _RowWidthCap(QObject):
    def __init__(self, combo) -> None:
        super().__init__(combo)
        self._combo = combo
        parent = combo.parentWidget()
        if parent is not None:
            parent.installEventFilter(self)
        # The pane's viewport changes when a scrollbar appears or the window resizes without the row changing width;
        # without this subscription the cap would keep the pre-scrollbar answer (the 478 -> 470 case above).
        anchor = _pane_anchor(parent) if parent is not None else None
        if anchor is not None:
            anchor.installEventFilter(self)
        # The combo's own hint changes with its items; a `Maximum` policy alone let the row hand it spare space
        # (measured 231 px for a 139 px hint), so the cap is computed here, not delegated to the policy.
        combo.installEventFilter(self)
        self.apply()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API
        if event.type() in (QEvent.Type.Resize, QEvent.Type.Show, QEvent.Type.LayoutRequest):
            self.apply()
        return False

    def available_width(self) -> int:
        """The width the combo may take: the tighter of its row and the pane the row sits in."""
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
        frame = margins.left() + margins.right() + occupied + spacing
        row_room = parent.width() - frame
        pane = _pane_content_width(parent)
        pane_room = pane - frame if pane is not None else row_room
        # `min`, never the pane alone: a combo can share its row with widgets the loop above does not count (a sibling
        # OF the row, not an item in it), and the pane knows nothing about them. Tightening only, never widening.
        if row_room > 0 and pane_room > 0:
            return min(row_room, pane_room)
        return max(row_room, pane_room)

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
