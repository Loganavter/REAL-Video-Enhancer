"""Cut `ImageLabel#previewLabel` down one property at a time - the owner's bisect, not a look-alike window.

`RVE_UI_PREVIEW_LABEL=<mode>` (boot-time; one run per mode, same trigger as the artifact):

    default   what ships: an `ImageLabel` with `WA_TranslucentBackground` and a null pixmap, so it paints NOTHING and the
              parent's plate shows through
    opaque    stop being translucent and erase with the palette (`WA_TranslucentBackground` off + autoFill), i.e. the
              label becomes an opaque rectangle of the window palette colour
    fill      keep it translucent but paint an opaque plate of the pane's own token (`dialog.background`) - the label
              then covers exactly its rect with the pane colour
    hide      hide the label completely: the parent's plate is all that is left in that area
    small     keep it as shipped but cap it to 200x150, so the plate no longer fills the pane

Each mode logs which one is active, so a screenshot or a log tail says what was running.

Why this order: the label is the only widget in the pane that paints nothing, and `WA_TranslucentBackground` is the flag
that lets whatever is behind it show through. If `opaque` or `fill` changes the artifact and `hide` does not, the label's
own paint path owns it; if `hide` changes it too, the parent plate or the surface does.
"""

from __future__ import annotations

import os

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QWidget

ENV_FLAG = "RVE_UI_PREVIEW_LABEL"


def mode() -> str:
    return os.environ.get(ENV_FLAG, "").strip() or "default"


def _log(label: QWidget, message: str) -> None:
    print(f"[preview-label] {message}", flush=True)


def cut(label: QWidget) -> str:
    """Apply the requested mode to the preview label; returns the mode actually applied."""
    active = mode()
    if active == "default":
        return active
    if active == "opaque":
        label.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        label.setAutoFillBackground(True)
    elif active == "fill":
        set_background = getattr(label, "set_background_token", None)
        if set_background is None:
            _log(label, "fill: this label has no set_background_token")
            return "default"
        set_background("dialog.background")
    elif active == "hide":
        label.hide()
    elif active == "small":
        label.setMaximumSize(QSize(200, 150))
    else:
        _log(label, f"{active}: unknown mode (default | opaque | fill | hide | small)")
        return "default"
    _log(label, f"mode={active} (translucent={label.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)}, "
                f"autoFill={label.autoFillBackground()}, "
                f"max={label.maximumSize().width()}x{label.maximumSize().height()})")
    return active
