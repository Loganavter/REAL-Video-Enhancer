"""The wheel scrolls the PAGE - a control under the cursor must not take it.

Every toolkit control that steps a value from the wheel inherits `WheelScrollPolicyMixin`, and its flag is off by
default: the first tick over a combo (or a spin box) focuses it and changes the value. RVE's pages are long columns
with a scroll area, so the pointer crosses combos on the way down and scrolling silently rewrote settings - the owner
called it "tons of misclicks" (2026-11-22).

`WheelScrollPolicyMixin` already carries the fix (`wheel_requires_focus`): with it set, `shouldHandleWheelEvent` ignores
the event unless the control has keyboard focus, and an ignored wheel event propagates to the enclosing scroll area,
which is exactly the wanted behaviour. It is applied here as one host decision instead of a constructor argument at
every one of the ~40 call sites - the same shape as `typography.apply_control_fonts`.

Controls that SCROLL instead of stepping a value are deliberately untouched: `MinimalistScrollBar` is a `QScrollBar`
and does not carry the mixin, so the video scrub bar keeps stepping on the wheel.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import QApplication


class _LateControlsWheelPolicy(QObject):
    """Apply the policy to controls that appear AFTER the startup walk.

    The walk covers everything that exists when the window is built, but the app also creates controls later - dialogs,
    the inspector, a combo rebuilt with new items - and those were left on the toolkit default (the wheel steps them).
    An application-wide filter closes that gap at the moment a widget is polished/shown, which is the earliest point a
    control can receive the wheel.
    """

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt API
        if event.type() in (QEvent.Type.Polish, QEvent.Type.Show):
            setter = getattr(obj, "setWheelRequiresFocus", None)
            if setter is not None and not getattr(obj, "wheelRequiresFocus", lambda: False)():
                setter(True)
        return False


def _install_late_policy(root) -> None:
    app = QApplication.instance()
    if app is None or getattr(app, "_wheel_requires_focus_guard", None) is not None:
        return
    guard = _LateControlsWheelPolicy(app)
    app.installEventFilter(guard)
    app._wheel_requires_focus_guard = guard


def apply_wheel_requires_focus(root) -> int:
    """Make every value-stepping control under ``root`` require focus for the wheel; returns the count changed.

    Two layers: the walk over what exists now, and an application-wide filter for controls created later (dialogs,
    rebuilt combos) - otherwise a control made after this call kept the toolkit default and stepped on the wheel while
    the page scrolled (owner, 2026-11-22: "not every combo obeys the no-scroll policy").

    Idempotent: a control that already states the policy is skipped, so a second walk costs nothing.
    """
    _install_late_policy(root)
    changed = 0
    for widget in root.findChildren(object):
        setter = getattr(widget, "setWheelRequiresFocus", None)
        if setter is None:
            continue
        if getattr(widget, "wheelRequiresFocus", lambda: False)():
            continue
        setter(True)
        changed += 1
    return changed
