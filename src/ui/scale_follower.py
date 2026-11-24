"""One object that re-applies a widget's design px every time the UI scale changes.

Why this is a shared module and not a private class per page: the migration's last wave gives two owners the same
hook - the window shell (its default size, its minimum, the nav rail's width and icon boxes, the page insets) and the
advanced settings page (its gaps) - and a private copy per module is how two implementations drift apart. `proc_page.py`
still carries its own `_UiScaleFollower` from the wave that introduced the idiom; it is deliberately left alone here
(that wave's file is not this one's to edit), and the next change to it can import this class instead of keeping a
second copy.

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

from sli_ui_toolkit.ui.managers.ui_scale import UiScale


class ScaleFollower(QObject):
    """Calls ``apply`` once now and again on every `UiScale.scale_changed`.

    ``apply`` takes no arguments (the signal's factor is ignored because the callers re-read it through `scaled_px()`;
    taking it as a parameter would invite a stale copy of the number).
    """

    def __init__(self, parent: QObject, apply: Callable[[], None]) -> None:
        super().__init__(parent)
        self._apply = apply
        UiScale.get_instance().scale_changed.connect(self._refresh)
        self._refresh()

    def _refresh(self, *_args) -> None:
        self._apply()


__all__ = ["ScaleFollower"]
