"""One object that re-applies a widget's design px every time the UI scale changes.

Why this is a shared module and not a private class per page: the migration's last wave gives two owners the same
hook - the window shell (its default size, its minimum, the nav rail's width and icon boxes, the page insets) and the
advanced settings page (its gaps) - and a private copy per module is how two implementations drift apart. `src/ui/pages/`
carried two more copies for a few hours (`scale_follower.py` and `ui_scale_follow.py`) while three waves ran in parallel;
they were folded into this module on 2026-11-24, and `proc_page.py`'s private `_UiScaleFollower` went with them.

The contract is the toolkit's (`sli_ui_toolkit.ui.managers.ui_scale`): a consumer reads the factor through
`scaled_px()` and subscribes to `scale_changed` to re-resolve its geometry. The important half is **never at import** -
a module-level `scaled_px(CONST)` freezes whatever factor the process happened to start with, which is why the constants
these owners resolve are plain design numbers and the call happens in `apply`.

Parenting matters: the follower is a `QObject` parented to the widget it serves, so Qt drops the connection with the
widget. The toolkit's manager is a process-wide singleton and a plain callable connected to it outlives the window,
firing into deleted C++ objects after a close.
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QObject

from sli_ui_toolkit.ui.managers.ui_metrics import AXIS_SCALE, UiMetrics
from sli_ui_toolkit.ui.managers.ui_scale import UiScale


class ScaleFollower(QObject):
    """Calls ``apply`` once now, on every `UiScale.scale_changed`, and once more on the coalesced bus.

    The second subscription is not redundant: a toolkit control subscribes to `UiMetrics.changed` at construction, so
    an owner that sized it from `scale_changed` alone would be overwritten by that control's own deferred pass (the
    toolkit's `CustomLineEdit` re-pins its height on a scale change). Owners create the follower AFTER their widgets, so
    its slot runs last in the coalesced emission and their design numbers have the final word.

    Both subscriptions are why every page and the window import THIS class instead of keeping a private copy: three
    parallel waves grew three copies of the same eight lines (2026-11-24), and the strongest contract is the one worth
    keeping - a page that only followed `scale_changed` lost its row heights to the toolkits re-pin.
    """

    def __init__(self, parent: QObject, apply: Callable[[], None]) -> None:
        super().__init__(parent)
        self._apply = apply
        UiScale.get_instance().scale_changed.connect(self._refresh)
        UiMetrics.get_instance().changed.connect(self._on_metrics)
        self._refresh()

    def _refresh(self, *_args) -> None:
        self._apply()

    def _on_metrics(self, axes) -> None:
        if AXIS_SCALE in axes:
            self._apply()


__all__ = ["ScaleFollower"]
