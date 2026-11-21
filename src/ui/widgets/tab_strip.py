"""RVE's tab strips: one factory and one style, shared by every page that has tabs.

The pages used to build the host differently - the process page with a `TabStyle`, the settings page with the toolkit's
defaults - so the two strips drifted into two looks (a flat button strip on one, a clipped folder tab on the other).
Both now state the same thing here, and both hand the button to the toolkit through `tab_factory` (the host's half of
`TopTabHost`): a strip-level knob was the wrong place for a per-button look.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSizePolicy

from sli_ui_toolkit.widgets import Button, TabStyle

#: Height of a strip's row. The button is `TAB_INSET` shorter, which is the air under the plate.
TAB_HEIGHT = 34
TAB_INSET = 4


def tab_style() -> TabStyle:
    """The one strip style: flat buttons, a plate on the selected tab, a gap under the strip."""
    return TabStyle(
        flat=True,
        border=False,
        gap=8,
        min_width=88,
        padding=12,
        spacing=4,
        radius=8,
        scroll_policy=Qt.ScrollBarPolicy.ScrollBarAsNeeded,
        reserve_scroll_row=True,
        selected_token="button.dialog.default.background.pressed",
    )


def tab_button(text: str, tab_height: int, style) -> Button:
    """A tab as a plain toolkit button: rounded plate, its own checked state, its own height."""
    tab = Button(
        text=text or " ",
        toggle=True,
        variant="surface",
        size=(0, max(1, tab_height - TAB_INSET)),
        corner_radius=8,
        content_padding=(12, 4, 12, 4),
    )
    tab.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
    return tab
