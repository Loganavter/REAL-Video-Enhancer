"""One owner for RVE's font sizes.

Today's pages carry the Designer's font temporaries (``font``, ``font1`` .. ``font8``) and apply them with 61
``setFont`` calls. Those temporaries are gone; every widget asks this module for its font instead, so a size is named
once and the rendered result stays byte-identical.

The values below were measured, once, from the Designer's compiled output (the nine ``QFont()`` temporaries and their
``setPointSize`` / ``setBold`` / ``setUnderline`` / ``setFamilies`` calls). That fixture is retired as of 2026-11-23 -
the Designer is no longer a reference in this tree - so the table below is the record; the last commit that carried
the fixture is `8d607cad`.

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

Scale (2026-11-24): every number below is a **design** value. ``font``/``apply`` resolve it through the toolkit's
``UiFont``, which multiplies it by the live ``UiScale`` factor exactly once, and ``apply`` keeps the widget subscribed
to ``font_changed``/``scale_changed`` so a later family or scale change right here reaches it. Before this, the module
built a plain ``QFont`` once: ``UiScale.set_factor(2.0)`` reached the toolkit's own widgets and nothing in RVE
(``harness/tests/test_typography_scale.py`` is the pin).

The one place the two scales meet is ``label_px``/``CONTROL_RATIO``: Qt resolves point sizes at 96 dpi, so a 12 pt
label is 16 px, while a control is pinned to 12 px - the label is 16/12 = 1.33x the control. That 33 % gap is
deliberate (see ``CONTROL_PX``) and now scales with the same factor instead of being replaced by it.
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

#: Qt resolves a point size at 96 dpi; this is the only place the pt and px scales are compared.
DESIGN_DPI = 96.0


def label_px(pt: float) -> int:
    """The pixel size a design point size resolves to at ``DESIGN_DPI`` (12 pt -> 16 px)."""
    return int(round(float(pt) * DESIGN_DPI / 72.0))


def font(
    pt: float | None = BODY_PT,
    *,
    bold: bool = False,
    italic: bool = False,
    underline: bool = False,
    family: str | None = None,
) -> QFont:
    """Return the font for a size group, scaled by the live ``UiScale`` factor.

    ``pt`` is a **design** value: ``UiFont.resolve`` multiplies it by ``UiScale.factor()`` exactly once. ``pt=None``
    means "no explicit point size" and keeps the application default - the two widgets the Designer handed a bare
    ``QFont()`` (``renderPreviewBtn``, ``low_storage_label``) rely on that; the default itself is scaled through the
    same resolver, so those two follow the scale as well.

    A returned ``QFont`` is a snapshot: use :func:`apply` for a widget that must follow a later change.
    """
    from sli_ui_toolkit.ui.managers import UiFont

    return UiFont.get_instance().resolve(
        point_size=pt, bold=bold, italic=italic, underline=underline, family=family
    )


def _is_toolkit_text(widget) -> bool:
    """True for the toolkit ``Text`` painted label - the widget that resolves its own face.

    Read from the already-imported module, never imported here: this module is imported by the shell before the pages,
    and a lazy import of ``sli_ui_toolkit.widgets`` at that point would run the toolkit's scrollbar-policy retrofit over
    half a window (see ``toolkit_theme``'s note on import order).
    """
    import sys

    module = sys.modules.get("sli_ui_toolkit.widgets")
    text_cls = None if module is None else getattr(module, "Text", None)
    return text_cls is not None and isinstance(widget, text_cls)


def apply(
    widget,
    pt: float | None = BODY_PT,
    *,
    bold: bool = False,
    italic: bool = False,
    underline: bool = False,
    family: str | None = None,
) -> None:
    """Give ``widget`` the font of a size group - the only way page code sets a font.

    A toolkit ``Text`` (the painted label every page uses) resolves its OWN face on each scale/font change, so a widget
    font set from here would be overwritten by the next pass. It is pinned instead through ``setPixelSize`` with the
    DESIGN pixel size (``label_px(pt)``), which the toolkit multiplies by ``UiScale`` exactly once and re-resolves on
    ``scale_changed``/``font_changed``. At factor 1.0 this is the same pixel size Qt already resolved the point size to
    (12 pt = 16 px, 25 pt = 33 px), so the look does not move.

    Anything else (a stock widget) gets the resolved font from ``UiFont.apply``, which re-applies it on every later
    ``font_changed``/``scale_changed`` with these same overrides.

    ``RichTextView`` paints from an engine sized in its own design px (``setTextPixelSize``); ``apply`` leaves that
    alone - see ``DONE.md``'s finding - the pane still follows the SCALE through the toolkit's own metrics hook.
    """
    kind = _is_toolkit_text(widget)
    if kind:
        if family is not None:
            widget.setFamily(family)
        widget.setBold(bool(bold))
        widget.setItalic(bool(italic))
        widget.setUnderline(bool(underline))
        widget.setPixelSize(None if pt is None else label_px(pt))
        return
    from sli_ui_toolkit.ui.managers import UiFont

    UiFont.get_instance().apply(
        widget, point_size=pt, bold=bold, italic=italic, underline=underline, family=family
    )


def application_font(base: QFont | None = None) -> QFont:
    """The ``QApplication`` default face at the design control size (``CONTROL_PX``, NOT scaled).

    ``UiFont.base_font`` treats the application font as design space and multiplies it by ``UiScale`` at resolution
    time, so scaling it here as well would apply the factor twice. ``toolkit_theme.install`` calls this instead of
    stating a pixel size of its own.
    """
    resolved = QFont(base) if base is not None else QFont()
    resolved.setPixelSize(CONTROL_PX)
    return resolved


#: RVE's controls are 12 PX while its labels are 12 pt (= 16 px at 96 dpi): the deleted compatibility layer forced that
#: pixel size in each adapter's constructor. Without it a control inherits the 12 pt font of the scroll area it sits in -
#: measured as the single largest parity delta after the layer went (every control +33 % text, every row taller).
#: The 16 px label is 1.33x the 12 px control - a deliberate gap, stated once as ``CONTROL_RATIO``.
CONTROL_PX = 12

#: ``CONTROL_PX / label_px(BODY_PT)`` = 0.75: the deliberate control-vs-label ratio, in code as well as in prose above.
CONTROL_RATIO = CONTROL_PX / label_px(BODY_PT)


def apply_control_font(widget) -> None:
    """Pin one toolkit control to RVE's control scale (``CONTROL_PX`` design px, scaled once by ``UiScale``)."""
    from sli_ui_toolkit.ui.managers import UiFont

    UiFont.get_instance().apply(widget, pixel_size=CONTROL_PX)


def apply_control_fonts(root) -> None:
    """Pin the toolkit CONTROLS under ``root`` to RVE's control size, once per window.

    A page's `apply(container, BODY_PT)` cascades a 16 px font into every toolkit control inside it; the toolkit controls
    do not set a font of their own, so the cascade wins. One walk after the window is built states the rule in one place
    instead of 147 call sites, and it is idempotent: re-applying replaces the widget's previous live wiring.
    """
    from sli_ui_toolkit.widgets import (Button, CheckBox, ComboBox, CustomLineEdit, DoubleSpinBox, SpinBox,
                                        Stepper)

    for cls in (Button, CheckBox, ComboBox, CustomLineEdit, DoubleSpinBox, SpinBox, Stepper):
        for widget in root.findChildren(cls):
            apply_control_font(widget)


def point_size_of(widget) -> float:
    """The widget's resolved point size (``QFontInfo``, so an unset size shows the application default)."""
    return QFontInfo(widget.font()).pointSizeF()
