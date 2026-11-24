"""One follower for the pages' design-px geometry: apply now, and again whenever ``UiScale`` changes.

The pattern is wave 2's (`src/ui/pages/proc_page.py`): a page states its boxes in **design px**, resolves them through
``scaled_px()`` at the moment the widgets exist, and re-resolves them on every `UiScale.scale_changed` so a live
`scale:<factor>` moves a built window without a restart. It is a small object **parented to the page** because a plain
callable connected to the process-wide `UiScale` singleton outlives the window and would fire into deleted C++ objects
after a close - and because the page builder itself is not a `QObject`, so it cannot parent anything directly.

Two rules this helper enforces by its shape:

* the callback runs ONCE in the constructor, so a page cannot forget the construction-time resolve (the tree's own
  history is a page that set its box from a literal and only scaled on a later change);
* the callback takes no arguments, so a page re-states the whole box rather than patching one edge - the same call is
  the 1.0 geometry and the 2.0 geometry.

It is deliberately public (``ScaleFollower``) while `proc_page.py` keeps its private copy: wave 3/4 pages share this
one, and importing a sibling page's underscore helper would couple them to a detail of that page.
"""

from __future__ import annotations

from PySide6.QtCore import QObject

from sli_ui_toolkit.ui.managers.ui_scale import UiScale


class ScaleFollower(QObject):
    """Calls ``apply`` once now and again on every `UiScale.scale_changed`.

    ``parent`` is normally the page root widget: Qt drops the connection when it is destroyed, so the follower cannot
    outlive the tree it resizes.
    """

    def __init__(self, parent: QObject, apply) -> None:
        super().__init__(parent)
        self._apply = apply
        UiScale.get_instance().scale_changed.connect(self._refresh)
        self._refresh()

    def _refresh(self, *_args) -> None:
        self._apply()
