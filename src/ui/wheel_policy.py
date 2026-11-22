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


def apply_wheel_requires_focus(root) -> int:
    """Make every value-stepping control under ``root`` require focus for the wheel; returns the count changed.

    Idempotent: a control that already states the policy is skipped, so a second walk (a dialog built later, another
    page) costs nothing.
    """
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
