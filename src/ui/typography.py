"""One owner for RVE's font sizes.

Today's pages carry the Designer's font temporaries (``font``, ``font1`` .. ``font8``) and apply them with 61
``setFont`` calls. Those temporaries are gone; every widget asks this module for its font instead, so a size is named
once and the rendered result stays byte-identical.

The values below were measured from the Designer's own compiled output,
``harness/fixtures/designer_mainwindow.py`` (the nine ``QFont()`` temporaries and their ``setPointSize`` /
``setBold`` / ``setUnderline`` / ``setFamilies`` calls):

    temporary  resolved spec                        widgets
    ---------  -----------------------------------  ------------------------------------------------------
    font       pointSize 25, families ["Fira Mono"] home ``label``
    font1      pointSize 15                         ``systemInfoText``
    font2      no point size (application default), ``renderPreviewBtn``
               bold off
    font3      pointSize 10                         ``ETA``, ``FPS``, ``STATUS``
    font4      pointSize 12                         the body (48 labels, buttons, the scroll area)
    font5      pointSize 15, underline              ``label_74``, ``label_41``, ``label_40``, ``label_42``
    font6      pointSize 13                         ``renderOutput``
    font7      pointSize 25, underline              download ``label_4``
    font8      no point size (application default), ``low_storage_label``
               bold on

So the explicit point sizes that actually appear are 25, 15, 13, 12 and 10 (plus the two widgets the Designer never
sized - they keep the application default, which is why ``pt=None`` is supported). ``harness/tests/
test_typography_parity.py`` pins each page's use back to this fixture; a drifted size fails the suite.

``GROUP_TITLE_PT`` is 15 and ``CAPTION_PT`` is 10 because that is what the fixture says - the group names are the
contract, the numbers are the measurement.
"""

from __future__ import annotations

from PySide6.QtGui import QFont, QFontInfo

#: Body text: the 48 widgets the Designer sized at 12 pt.
BODY_PT = 12
#: Section titles: the 15 pt headings (``font1``/``font5``); the underlined ones pass ``underline=True``.
GROUP_TITLE_PT = 15
#: Small readouts: the 10 pt process labels (``ETA``/``FPS``/``STATUS``).
CAPTION_PT = 10
#: The render log's 13 pt text (``renderOutput``).
SMALL_PT = 13
#: Display text: the 25 pt heading (Home ``label``, Download ``label_4``).
DISPLAY_PT = 25


def font(
    pt: float | None = BODY_PT,
    *,
    bold: bool = False,
    italic: bool = False,
    underline: bool = False,
    family: str | None = None,
) -> QFont:
    """Return the font for a size group.

    ``pt=None`` means "no explicit point size" and keeps the application default - the two widgets the Designer
    handed a bare ``QFont()`` (``renderPreviewBtn``, ``low_storage_label``) rely on that.
    """
    resolved = QFont()
    if pt is not None:
        # setPointSize keeps the exact integer the Designer used instead of a float that only compares equal.
        if float(pt).is_integer():
            resolved.setPointSize(int(pt))
        else:
            resolved.setPointSizeF(float(pt))
    if family is not None:
        resolved.setFamilies([family])
    resolved.setBold(bold)
    resolved.setItalic(italic)
    resolved.setUnderline(underline)
    return resolved


def apply(
    widget,
    pt: float | None = BODY_PT,
    *,
    bold: bool = False,
    italic: bool = False,
    underline: bool = False,
    family: str | None = None,
) -> None:
    """Give ``widget`` the font of a size group - the only way page code sets a font."""
    widget.setFont(font(pt, bold=bold, italic=italic, underline=underline, family=family))


#: RVE's controls are 12 PX while its labels are 12 pt (= 16 px at 96 dpi): the deleted compatibility layer forced that
#: pixel size in each adapter's constructor. Without it a control inherits the 12 pt font of the scroll area it sits in -
#: measured as the single largest parity delta after the layer went (every control +33 % text, every row taller).
CONTROL_PX = 12


def apply_control_fonts(root) -> None:
    """Pin the toolkit CONTROLS under ``root`` to RVE's control size, once per window.

    A page's `apply(container, BODY_PT)` cascades a 16 px font into every toolkit control inside it; the toolkit controls
    do not set a font of their own, so the cascade wins. One walk after the window is built states the rule in one place
    instead of 147 call sites, and it is idempotent.
    """
    from sli_ui_toolkit.widgets import (Button, CheckBox, ComboBox, CustomLineEdit, DoubleSpinBox, SpinBox,
                                        Stepper)

    for cls in (Button, CheckBox, ComboBox, CustomLineEdit, DoubleSpinBox, SpinBox, Stepper):
        for widget in root.findChildren(cls):
            current = widget.font()
            if current.pixelSize() == CONTROL_PX:
                continue
            current.setPixelSize(CONTROL_PX)
            widget.setFont(current)


def point_size_of(widget) -> float:
    """The widget's resolved point size (``QFontInfo``, so an unset size shows the application default)."""
    return QFontInfo(widget.font()).pointSizeF()
