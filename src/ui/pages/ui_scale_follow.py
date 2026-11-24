"""The one follower the settings pages share for a live ``UiScale`` change.

`src/ui/pages/proc_page.py` keeps a private copy of this idea for its process column. The three page builders in this
wave (``general_settings``/``tab_2``/``tab_3``) need exactly the same eight lines, and copying them three times would
turn one bug into three; this module is the single place.

Why a ``QObject`` parent: the connection is to the process-wide ``UiScale`` singleton, which outlives a window. Parenting
the follower to the page root drops the connection when Qt deletes the page, so a closed window cannot be told about a
factor change and fire into deleted C++ objects.

The contract is deliberately "run this whole apply once now, then once per change": every size in this wave is a
DESIGN value resolved through ``scaled_px()`` at apply time, never a number captured at import. A page therefore passes
one method that states all of its geometry, and construction is just the first run of it.
"""

from __future__ import annotations

from PySide6.QtCore import QObject

from sli_ui_toolkit.ui.managers.ui_metrics import AXIS_SCALE, UiMetrics
from sli_ui_toolkit.ui.managers.ui_scale import UiScale


class UiScaleFollower(QObject):
    """Calls ``apply`` once now, on every `UiScale.scale_changed`, and once more on the coalesced bus.

    The second subscription is not redundant: a toolkit control subscribes to `UiMetrics.changed` at construction, so
    a page that sized it from `scale_changed` alone would be overwritten by that control's own deferred pass (the
    toolkit's `CustomLineEdit` re-pins its height on a scale change). The follower is created after the page's widgets,
    so its slot runs LAST in the coalesced emission and the page's design numbers have the final word.
    """

    def __init__(self, parent: QObject, apply) -> None:
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


__all__ = ["UiScaleFollower"]
