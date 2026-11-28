"""One design height for every combobox in the app: the toolkit's own.

Owner, 2026-11-28: *"make every combobox's open button the same height in the app"*. Measured on the built shell at
1.0, twenty comboboxes carried TWO heights: fourteen at 33 px (their own hint, untouched) and six at 30 px - the six
the General page capped with its settings-row literal, which is BELOW the toolkit's own ``minimumSizeHint().height()``
of 33. A field shorter than its own minimum is the squashed chevron the owner saw.

The number is stated once, here, as the toolkit's design height: ``ComboBox.BASE_HEIGHT``. It is the control's
contract (the plate stroke, the vertical padding and the 12 pt line all live inside it), so the app does not invent a
second one and does not lower the toolkit's from a page. A page that sizes a combo resolves this through
``scaled_px`` at apply time, never at import.

``ScrollableComboBox`` is the toolkit's second combo family and states the same 33 in its constructor; the app does not
build one today (measured 2026-11-28: zero instances in the shell), so this module only carries the design number.
"""

from __future__ import annotations

from sli_ui_toolkit.ui.managers.ui_scale import scaled_px
from sli_ui_toolkit.widgets import ComboBox

#: The one combobox height, in DESIGN px: the toolkit's own minimum, so no host row can squeeze a field below it.
COMBO_HEIGHT_PX = ComboBox.BASE_HEIGHT


def apply_combo_height(combo) -> None:
    """Give ``combo`` exactly the one design height at the factor in force now.

    Both bounds, not a hint: a combo lays out with ``Fixed`` vertical policy and a layout would otherwise hand it the
    row's spare height, or - the 2026-11-28 defect - a shorter row's maximum and clip it. This is re-run on every
    scale change by the page's ``ScaleFollower``, so the height follows the factor like the rest of the row.
    """
    height = scaled_px(COMBO_HEIGHT_PX)
    combo.setMinimumHeight(height)
    combo.setMaximumHeight(height)


__all__ = ["COMBO_HEIGHT_PX", "apply_combo_height"]
