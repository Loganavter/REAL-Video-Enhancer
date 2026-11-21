"""Qt-shaped faces over the toolkit widgets, so a page can be migrated one class at a time.

Why adapters instead of rewriting every call site: the pages are the Designer's own statement sequence, and that
sequence is the identity gate - keeping `QLabel(parent)` + `setText(...)` + `setFont(...)` in place and changing *which
class* answers to the name means a migration step cannot silently drop a property. The adapters are thin on purpose:
constructor shape (Qt passes the parent first, the toolkit sometimes does not) and font translation.

Font translation is the one piece with a measurement behind it: the Designer's labels carry POINT sizes
(`pointSizeF 12`, `pixelSize -1`), and `Text` paints from its own family/pixel-size overrides, not from `QFont`. A
`setFont(QFont)` therefore becomes `setFamily` + `setPixelSize(QFontInfo(font).pixelSize())` + weight/slant/underline,
which is the resolved size `Text` needs. Passing the point size straight through is what made every caption render at
the variant's 12 px in the v2 spike.

Two classes are deliberately NOT here, and each is a toolkit row rather than a local workaround:

* `QListWidget` - the toolkit has no compact text list. `IconListWidget` is a nav list: one `Button` per row (hover
  and pressed plates), `row_height` 44 by default, 8 px between rows and 8 px of host margin (measured on
  `IconListWidget.__init__`), and it has no `insertItem`/`takeItem` - `RenderQueue` reorders the queue with exactly
  those two calls (`src/ui/RenderQueue.py`). Pinning a 20 px text row with a nav row would not reproduce the look.
* `QTextEdit` - `Text(mode="document")` renders markdown but never scrolls (the host composes the scroll area), and
  `ConsoleView` is a console (radius 6, padding 6, a floating scrollbar) with no markdown. RVE's four text edits are a
  scrollable markdown changelog, two static info panels and the render log; none has a one-widget counterpart.

Pixmap labels stay `QLabel` too, because `Text` paints text only.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtCore import QSize
from PySide6.QtWidgets import QSizePolicy
from PySide6.QtGui import QColor, QFont, QFontInfo, QFontMetrics

from sli_ui_toolkit.widgets import (
    Button,
    CheckBox,
    ComboBox,
    CustomLineEdit,
    DoubleSpinBox,
    Stepper,
    Text,
    ToastProgressBar,
)
# Not exported from `sli_ui_toolkit.widgets`: the step-button gutter is documented as a host mixin
# ("A host widget mixes this in and implements two hooks", `_step_buttons.py`), and `Stepper` composes it the same
# way. A toolkit row is to export it; importing the module it lives in is the honest use of the documented hook.
from sli_ui_toolkit.ui.widgets.atomic._step_buttons import StepButtonsMixin


def _constraints(widget) -> tuple:
    return (widget.minimumSize(), widget.maximumSize(), widget.sizePolicy())


def _restore_constraints(widget, saved: tuple) -> None:
    """Re-impose the constraints the caller set before a toolkit-internal geometry pass ran."""
    minimum, maximum, policy = saved
    widget.setMinimumSize(minimum)
    widget.setMaximumSize(maximum)
    widget.setSizePolicy(policy)


class QtLabel(Text):
    """`QLabel`'s shape over `Text`.

    Font handling is the whole trick. `Text` paints from its own family/pixel-size overrides, and `Text._apply_style`
    calls `self.setFont(...)` internally - so a plain override recurses into itself and the last style pass wins with
    the variant's default (measured: every label came out 12 px instead of the Designer's 16). The override therefore
    has two modes: an external `setFont(QFont)` translates the Qt font into `Text`'s overrides (resolving points to
    pixels), and the internal call is let through. A label with no explicit font adopts the font it inherits at
    construction, which is what `QLabel` does.
    """

    def __init__(self, parent=None, text: str = "") -> None:
        super().__init__(text, parent=parent)
        self._translating_font = False
        self._adopt_qt_font(self.font())

    def _adopt_qt_font(self, font: QFont) -> None:
        self._translating_font = True
        try:
            Text.setFamily(self, font.family() or None)
            Text.setPixelSize(self, QFontInfo(font).pixelSize())
            Text.setBold(self, bool(font.bold()))
            Text.setItalic(self, bool(font.italic()))
            Text.setUnderline(self, bool(font.underline()))
            Text.setStrikeOut(self, bool(font.strikeOut()))
        finally:
            self._translating_font = False

    def setFont(self, font: QFont) -> None:      # noqa: N802 - Qt spelling, and the generated code calls it
        if getattr(self, "_translating_font", False):
            Text.setFont(self, font)
            return
        saved = _constraints(self)
        self._adopt_qt_font(font)
        _restore_constraints(self, saved)

    def setText(self, text: str) -> None:        # noqa: N802
        saved = _constraints(self)
        Text.setText(self, text)
        _restore_constraints(self, saved)

    def setSizePolicy(self, horizontal, vertical=None) -> None:   # noqa: N802
        # The Designer sets policies on some labels (`label` on the home page is Preferred/Expanding). `Text`'s style
        # pass re-derives the policy on every font/text change and would drop it - so a policy set from outside is
        # remembered, and the style pass (marked by the toolkit's own `_applying_style`) cannot overwrite the memory.
        if getattr(self, "_applying_style", False) and getattr(self, "_rve_policy", None):
            # The style pass just re-derived a policy; the Designer's one is what the layout must see, so it is what
            # the widget is actually given (the memory alone was not enough - the widget kept the style's policy).
            remembered = self._rve_policy
            Text.setSizePolicy(self, QSizePolicy(remembered[0], remembered[1]))
            return
        if not getattr(self, "_applying_style", False):
            if isinstance(horizontal, QSizePolicy):
                # Qt's single-argument form (`setSizePolicy(sizePolicy2)` in the Designer's code) carries both
                # policies inside the object; storing the object instead of the pair made the memory read as
                # "vertical unknown" and the remembered policy was silently ignored.
                self._rve_policy = (horizontal.horizontalPolicy(), horizontal.verticalPolicy())
            else:
                self._rve_policy = (horizontal, vertical)
        if vertical is None:
            Text.setSizePolicy(self, horizontal)
        else:
            Text.setSizePolicy(self, horizontal, vertical)

    def _vertical_policy(self) -> QSizePolicy.Policy:
        # A QLabel's vertical policy is Preferred, which lets a layout give it the row's height; `Text` defaults to
        # Fixed unless it word-wraps, so every label in a tall row stayed 17 px high and the row's contents collapsed
        # to the top (measured: `encoderLabel_3` 98 px tall in the Designer build, 22 px after migration).
        return QSizePolicy.Policy.Preferred

    def setStyleSheet(self, css: str) -> None:   # noqa: N802
        # A QSS rule on a toolkit widget is a no-op, and RVE's pages still carry a few per-widget ones from the
        # Designer (`low_storage_label` is `QLabel { color: red; }`). Rather than let the colour vanish silently, the
        # two properties a label actually declares are translated to the painter's own setters; anything else is
        # reported once so the gate's finding is not the only trace.
        if not css or not css.strip():
            return
        import re

        colour = re.search(r"\bcolor\s*:\s*([^;}\s]+)", css)
        if colour:
            self.setColor(QColor(colour.group(1)))
        size = re.search(r"\bfont-size\s*:\s*(\d+)\s*(px|pt)?", css)
        if size:
            value = int(size.group(1))
            self.setPixelSize(value * 4 // 3 if size.group(2) == "pt" else value)

    def setPixmap(self, pixmap) -> None:         # noqa: N802
        raise NotImplementedError(
            "Text paints text only - a pixmap label stays QLabel (the splitter keeps such labels stock)")

    def setScaledContents(self, on: bool) -> None:   # noqa: N802
        raise NotImplementedError("Text has no scaled-contents mode - pixmap labels stay QLabel")


class QtButton(Button):
    """`QPushButton`'s shape over `Button`.

    RVE's QSS gives its buttons `text-align: left`, `padding: 5px 10px`, `border-radius: 10px`, so those are the
    adapter's defaults - the look is reproduced by the widget, not by a stylesheet that the painter would ignore.
    """

    def __init__(self, parent=None, text: str = "") -> None:
        # Those numbers are RVE's QSS, expressed as the widget's own properties: `QPushButton{padding:5px 10px;
        # border-radius:10px; text-align:left; background-color:#2c313c}`. A QSS rule would be ignored by a
        # toolkit widget (a custom QWidget never gets WA_StyledBackground), so the look has to live here.
        # RVE's button has no border (`QPushButton` rule sets only background, radius and padding), so the variant's
        # default border layer is switched off explicitly instead of being painted a hair lighter than the plate.
        # RVE's `padding: 5px 10px` is the QSS's box around the TEXT, and QPushButton draws a 16 px line inside a
        # 30 px button - 5 px above and below still leaves room. The toolkit reserves the font's full line spacing plus
        # the padding, and at RVE's 12 pt font that is 22 + 10 = 32 px against a Designer-pinned 30: the glyphs were
        # clipped through the middle (measured on every 30 px button). Two pixels vertically is what the stock look
        # actually uses.
        # `content_padding` is (left, top, right, bottom) - RVE's QSS is `padding: 5px 10px`, i.e. 10 px horizontally
        # and as little vertically as the Designer's pinned heights allow (the toolkit reserves a full line, so 5 px
        # above and below a 30 px button clips the glyphs).
        super().__init__(None, text=text, parent=parent, variant="surface", content_align=_LEFT,
                         content_padding=(10, 5, 10, 5), corner_radius=10,
                         border_color=QColor(0, 0, 0, 0))
        self._has_icon = False
        self._icon_box = QSize(0, 0)
        # No `rows=`: the default content path paints the widget's own font (`paint_font(widget)` reads `font()`) AND
        # draws the icon, whereas a hand-built row has no icon slot at all - building rows dropped the "+" on RVE's
        # add-to-queue button (measured). `setFont` is therefore enough; it just needs the constraints preserved.
        # The toolkit's Button base gives every instance a 36x36 minimum AND maximum (its own default control size).
        # The minimum turned RVE's 22x16 checkbox into a 36x36 square; the maximum clamped every text button to 36 px
        # wide, which is what made labels look cut off. Both are cleared here; the Designer's own
        # `setMinimumSize`/`setMaximumSize` statements run after this constructor and still apply where declared.
        self.setMinimumSize(0, 0)
        self.setMaximumSize(16777215, 16777215)
        # QPushButton's own policy: as wide as it needs, exactly as tall as its hint. The toolkit's button is happy to
        # fill the cell, which turned a 27 px button into a 96 px slab inside its row (measured on the settings page).
        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

    def setSizePolicy(self, horizontal, vertical=None) -> None:   # noqa: N802
        if isinstance(horizontal, QSizePolicy):
            self._rve_policy = (horizontal.horizontalPolicy(), horizontal.verticalPolicy())
        else:
            self._rve_policy = (horizontal, vertical)
        if vertical is None:
            Button.setSizePolicy(self, horizontal)
        else:
            Button.setSizePolicy(self, horizontal, vertical)

    def setText(self, text: str) -> None:        # noqa: N802
        # `Button.setText` re-derives its geometry AND resets `maximumSize` to unbounded (measured: a 30 px high
        # button came back unbounded, so every button in a row grew past the Designer's height). Whatever the caller
        # had set is re-imposed after the pass, and the row is rebuilt so the text keeps the button's font size.
        saved = _constraints(self)
        Button.setText(self, text)
        _restore_constraints(self, saved)

    def setFont(self, font: QFont) -> None:      # noqa: N802
        saved = _constraints(self)
        Button.setFont(self, font)
        _restore_constraints(self, saved)

    def setIcon(self, icon) -> None:             # noqa: N802
        # The toolkit keeps its icon privately and exposes no getter; the hint below needs to know whether there is one
        # and how big it is, so the adapter records what Qt's setter was told.
        self._has_icon = not icon.isNull()
        super().setIcon(icon)

    def setIconSize(self, size) -> None:         # noqa: N802
        self._icon_box = QSize(size)
        super().setIconSize(size)

    def text(self) -> str:
        """The toolkit keeps the label in `_text` and exposes no getter; Qt's name is part of the shape RVE expects."""
        return getattr(self, "_text", "") or ""

    def sizeHint(self) -> QSize:                 # noqa: N802
        # Same QSS semantics as the combo: `padding: 5px 10px` around the content, and QPushButton's 27 px row for a
        # 12 px font (measured). An icon-only button is the icon plus the padding - RVE's nav buttons are 55x45 for a
        # 35 px icon, which is `icon + 20` by `icon + 10` (measured), not the toolkit's fixed 36x36.
        has_icon = self._has_icon
        icon_size = self._icon_box
        text = self.text()
        fm = QFontMetrics(self.font())
        # QSS `padding: 5px 10px` around the content. The content is a line of text and/or an icon, so the height is the
        # taller of the two plus five - RVE's add-to-queue button carries a 28 px "+" and is 35 px tall, where a text
        # row alone would give 27 (measured against the Designer build).
        # QSS `padding: 5px 10px` is 10 px of vertical box around the content: RVE's stock button is 27 px tall for a
        # 17 px line, i.e. line + 10 - not line + 5, which is what a mis-calibrated hint produced (22 px) and why the
        # label was clipped through the bottom.
        content_h = max(fm.lineSpacing(), icon_size.height() if has_icon else 0)
        if not text:
            if has_icon:
                return QSize(icon_size.width() + 20, content_h + 10)
            return super().sizeHint()
        icon_room = (icon_size.width() + 6) if has_icon else 0
        return QSize(fm.horizontalAdvance(text) + 20 + icon_room, content_h + 10)

    def minimumSizeHint(self) -> QSize:          # noqa: N802
        return self.sizeHint()


    def setCheckable(self, on: bool) -> None:    # noqa: N802
        # A toolkit `Button`'s toggle mode is a constructor flag; turning it on later is not supported, so the
        # splitter keeps checkable buttons stock (nav rail, mode buttons) instead of pretending this worked.
        if on:
            raise NotImplementedError("Button toggle mode is constructor-only - checkable buttons stay QPushButton")

    def setAutoExclusive(self, on: bool) -> None:    # noqa: N802
        if on:
            raise NotImplementedError("Button has no auto-exclusive group - such buttons stay QPushButton")

    def setAutoRepeat(self, on: bool) -> None:       # noqa: N802
        if on:
            raise NotImplementedError("Button has no auto-repeat - such buttons stay QPushButton")


class QtCheckBox(CheckBox):
    # RVE's checkbox is Qt's: a 13 px box with a 1 px outline, no vertical padding inside a 16 px row. The toolkit's
    # default (20 px box, 5 px padding) is a 30 px row - measured against the Designer's 16 px, and it moved every
    # row it sat in. These are the toolkit's own class constants, tuned to the class being replaced.
    INDICATOR_SIZE = 16
    INDICATOR_RADIUS = 2
    SPACING = 6
    PADDING_H = 3
    PADDING_V = 0

    """`QCheckBox`'s shape over `CheckBox` (Qt passes the parent first and sets the text later).

    `stateChanged(int)` is the Qt signal RVE's glue actually binds (12 sites: `ProcessTab`, `SettingsTab`, ...). The
    toolkit box reports `toggled(bool)`, so the adapter carries Qt's spelling as a derived signal - the same semantics
    (2 = checked, 0 = unchecked), no behaviour of its own.
    """

    stateChanged = Signal(int)

    def __init__(self, parent=None, text: str | None = None) -> None:
        super().__init__(text or "", parent)
        self.toggled.connect(lambda checked: self.stateChanged.emit(2 if checked else 0))
        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        # The toolkit's Button base caps every instance at 36x36. Clearing only the MINIMUM left the maximum in place,
        # so a labelled checkbox was clipped to a 36 px strip with its text cut off (measured on the process page).
        self.setMinimumSize(0, 0)
        self.setMaximumSize(16777215, 16777215)

    def setChecked(self, checked: bool) -> None:     # noqa: N802
        # The toolkit animates the check mark over 150 ms from `toggled`. A checkbox set before its page is shown
        # never ticks that animation (a hidden widget gets no frames), so it painted as UNCHECKED while `isChecked()`
        # was True (measured on the process page). Qt's `setChecked` has no animation; neither does this.
        super().setChecked(checked)
        self.set_checked_progress(1.0 if checked else 0.0)

    def setText(self, text: str) -> None:            # noqa: N802
        saved = _constraints(self)
        super().setText(text)
        _restore_constraints(self, saved)

    def sizeHint(self) -> QSize:                 # noqa: N802
        # RVE's box is the QSS's `QCheckBox::indicator`: 14 px, `left: 2px`, `border-radius: 2px`, and QCheckBox's
        # own row height of 16 px for a 12 px font. The toolkit's constants are computed from ITS font, not the
        # widget's (measured: two boxes in one page hinted 15 and 24 px tall), so the hint is stated here instead.
        fm = QFontMetrics(self.font())
        text = self.text()
        width = 22 + ((6 + fm.horizontalAdvance(text)) if text else 0)
        return QSize(width, 16)

    def minimumSizeHint(self) -> QSize:          # noqa: N802
        return self.sizeHint()

    def setTristate(self, on: bool) -> None:     # noqa: N802
        # The toolkit box has two states; RVE has one tristate checkbox. Recorded rather than faked, and the splitter
        # keeps tristate boxes stock so no behaviour changes silently.
        raise NotImplementedError("CheckBox has no tristate - tristate boxes stay QCheckBox")


class QtComboBox(ComboBox):
    """`QComboBox`'s shape over `ComboBox`.

    The toolkit field is 33 px tall and never narrower than 100 px; the Designer's combos are 29 px and as narrow as
    their longest item (78 px on the settings page - measured). Its `sizeHint` floors the width to make a field of
    `"M"*n` legible, which RVE's layouts do not ask for, so the adapter asks the class it replaces instead: a stock
    `QComboBox` with the same font and items, cached until the items or the font change. That is exact by construction
    rather than a formula guessed from two samples.
    """

    BASE_HEIGHT = 29
    RADIUS = 10                      # RVE's `QComboBox{border-radius:10px}` (the toolkit default is 6)
    TEXT_HORIZONTAL_PADDING = 10     # RVE's `padding:5px 10px` (the toolkit default is 12)

    def __init__(self, parent=None, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        self._stock_hint = None
        # QComboBox's own policy is Preferred/Fixed, NOT Expanding: an Expanding combo eats the row's spare width and
        # squeezes the label next to it to its hint (measured: the settings rows lost 240 px of label). A layout shares
        # spare width among the Preferred items, which is what the Designer's rows were laid out against.
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    def _invalidate(self) -> None:
        self._stock_hint = None
        # The hint changed, so the LAYOUT has to know. Without this the combo kept the width it was given while empty
        # (38 px on the settings page) even though its sizeHint had already grown to the stock 145 px - measured by
        # dumping the widget after the app had populated it.
        self.updateGeometry()

    def addItem(self, text, userData=None) -> None:      # noqa: N802
        super().addItem(text, userData)
        self._invalidate()

    def addItems(self, texts) -> None:                   # noqa: N802
        super().addItems(texts)
        self._invalidate()

    def clear(self) -> None:                             # noqa: N802
        super().clear()
        self._invalidate()

    def setItemText(self, index: int, text: str) -> None:  # noqa: N802
        # The Designer adds EMPTY items and fills the text from `retranslateUi` (`setItemText`), so invalidating only
        # on add/clear cached the width of an empty combo: 38 px against the stock 145 (measured).
        super().setItemText(index, text)
        self._invalidate()

    def insertItem(self, index: int, text: str, userData=None) -> None:  # noqa: N802
        super().insertItem(index, text, userData)
        self._invalidate()

    def removeItem(self, index: int) -> None:             # noqa: N802
        super().removeItem(index)
        self._invalidate()

    def setFont(self, font) -> None:                     # noqa: N802
        super().setFont(font)
        self._invalidate()

    #: RVE's combo rule, verbatim - the oracle below must be measured under the same QSS the stock widget had,
    #: otherwise its padding and arrow are missing from the hint (measured: 70 vs the Designer's 78 px).
    RVE_QSS = "QComboBox{background-color:#2c313c; text-align:left; padding:5px 10px; border-radius:10px; color:white;}"

    def _stock_metrics(self):
        if self._stock_hint is None:
            from PySide6.QtWidgets import QComboBox

            oracle = QComboBox()
            oracle.setStyleSheet(self.RVE_QSS)
            oracle.setFont(self.font())
            for index in range(self.count()):
                oracle.addItem(self.itemText(index))
            self._stock_hint = oracle.sizeHint()
        return self._stock_hint

    def sizeHint(self) -> QSize:                          # noqa: N802
        # RVE's combo: `QComboBox{padding:5px 10px}`, so a field is its longest item plus 20 px, never under 80, and
        # 29 px tall (5+5 padding + 17 px of 12 px text + 2 border - measured on the Designer build).
        return self._stock_metrics()

    def minimumSizeHint(self) -> QSize:                   # noqa: N802
        return self.sizeHint()


class QtLineEdit(CustomLineEdit):
    """`QLineEdit`'s shape over `CustomLineEdit`.

    RVE's line edit is 25 px tall (the Designer's `setMinimumSize(0, 25)`), and `CustomLineEdit` pins its height in the
    constructor (`setFixedHeight(scaled_px(HEIGHT))`, `HEIGHT = 32`), which no `sizeHint` override can undo - the field
    stayed 32 px against the stock 25 (measured). The subclass states the height the class reads, so the fixed size and
    every internal rect agree.
    """

    HEIGHT = 25

    def __init__(self, parent=None, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)


def _token(name: str) -> QColor | None:
    """A toolkit token as a `QColor`, or `None` when the palette does not define it."""
    from sli_ui_toolkit.theme import ThemeManager

    try:
        value = ThemeManager.get_instance().try_get_color(name)
    except Exception:
        return None
    return QColor(value) if value is not None and value.isValid() else None


#: RVE's spin-box rules, verbatim (`settings_page.py`, `proc_page.py`): the oracle below must be measured under the
#: same QSS the stock widget had, otherwise its padding and its 20 px arrow column are missing from the hint (measured
#: on the combo: 70 vs the Designer's 78 px).
_ARROW_RULES = (
    "{klass}::up-arrow{{image:url(:/icons/icons/arrow-up.svg);}}"
    "{klass}::down-arrow{{image:url(:/icons/icons/arrow-down.svg);}}"
    "{klass}::up-button{{width:20px;height:15px;}}"
    "{klass}::down-button{{width:20px;height:15px;}}"
)
_SPIN_QSS = ("QSpinBox{background-color:#2c313c;padding:5px 10px;border-radius:10px;color:white;}"
             + _ARROW_RULES.format(klass="QSpinBox"))
_DOUBLE_SPIN_QSS = ("QDoubleSpinBox{background-color:#2c313c;padding:5px 10px;border-radius:10px;color:white;}"
                    + _ARROW_RULES.format(klass="QDoubleSpinBox"))


#: `QAbstractSpinBox`'s own text alignment (`QSpinBox().alignment()` = AlignLeft|AlignVCenter, measured). The
#: toolkit's field defaults to AlignCenter, which would move every value to the middle of its field.
_QT_ALIGNMENT = None


def _qt_alignment():
    """Qt's spin-box alignment, resolved lazily so this module still imports without a QApplication."""
    global _QT_ALIGNMENT
    if _QT_ALIGNMENT is None:
        from PySide6.QtCore import Qt

        _QT_ALIGNMENT = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
    return _QT_ALIGNMENT


class _QtSpinShape:
    """The `QAbstractSpinBox` shape both spin-box adapters share.

    Three measured facts decide it:

    * **The step buttons are part of the control.** RVE's page QSS styles `::up-button`/`::down-button`, so the
      counterpart is the toolkit's `Stepper` (a `SpinBox` with the ▲/▼ gutter), not a bare `SpinBox` - migrating to
      the bare field would silently drop the affordance. The gutter is stated at RVE's own width (20 px, no gap, a
      10 px glyph) instead of the toolkit's 44 px default, because two of RVE's boxes are pinned to 75 px wide.
    * **Qt owns the geometry, the toolkit pins its own.** `SpinBox.__init__`/`Stepper.__init__` call
      `setMinimumWidth(content_width)` and `SpinBox.on_scale_changed` re-applies `setFixedHeight(32)`. An explicit
      minimum above the Designer's `setMaximumSize(75, 30)` makes Qt **ignore the maximum** (`qSmartMinSize`
      bounds the hint by the maximum, then an explicit minimum overrides it) - the same class of defect that cost the
      combo 107 px. So the toolkit's pins are released, the Designer's own statements then apply, and the hint is
      the stock class's.
    * **The width is an oracle, the height a stated constant.** A stock `QSpinBox` under RVE's QSS hints 32 px tall;
      its width depends on the range and the decimals, so it is measured from a live stock widget with the same font
      and range (`QtComboBox` is the precedent), cached, and invalidated whenever the hint can move.
    """

    #: RVE's `QSpinBox{padding:5px 10px; border-radius:10px}` on the toolkit's own constants.
    RADIUS = 10
    H_PADDING = 10
    #: Measured: a stock spin box under RVE's QSS hints 32 px tall (a 20 px field with 5 px padding above and below).
    #: The toolkit pins the same 32 px, so the number is stated rather than inherited.
    HEIGHT = 32
    #: RVE's stepper column: `::up-button`/`::down-button{width:20px; height:15px}`.
    STEP_BUTTON_WIDTH = 20
    STEP_BUTTON_GAP = 0
    STEP_ARROW_SIZE = 10

    RVE_QSS = ""
    _stock_hint: QSize | None = None
    _designer_max_height = 16777215
    _designer_min_width = 0

    # -- construction shared by both adapters ------------------------------------------------------------

    def _init_shape(self, minimum, maximum) -> None:
        self._stock_hint = None
        self.setRange(minimum, maximum)
        # The toolkit's content-width pin is not the Designer's width; the Designer's own `setMinimumSize` /
        # `setMaximumSize` statements run after this constructor and are the authority.
        self._release_width_pin()
        self.setMinimumHeight(0)
        self.setMaximumHeight(16777215)
        # QAbstractSpinBox's own policy (measured on the stock class): as wide as its hint, exactly as tall.
        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

    def _release_width_pin(self) -> None:
        # `setMinimumWidth(0)` restores the toolkit's floor, except where the Designer stated one of its own.
        self.setMinimumWidth(self._designer_min_width)

    def setMinimumSize(self, *args) -> None:                 # noqa: N802
        # The Designer's `setMinimumSize(QSize(75, 30))` runs before `setMinimum`/`setMaximum` on two boxes; a range
        # change releases the toolkit's content-width pin, so the Designer's width has to be remembered here.
        size = args[0] if len(args) == 1 else QSize(args[0], args[1])
        self._designer_min_width = size.width()
        super().setMinimumSize(*args)

    def _stock_oracle(self):           # pragma: no cover - each adapter names the stock class it replaces
        raise NotImplementedError

    def _stock_metrics(self) -> QSize:
        if self._stock_hint is None:
            oracle = self._stock_oracle()
            try:
                self._stock_hint = oracle.sizeHint()
            finally:
                oracle.deleteLater()
        return self._stock_hint

    def _invalidate(self) -> None:
        self._stock_hint = None
        # A cached hint the layout never hears about is the bug that cost the combo 107 px.
        self.updateGeometry()

    # -- Qt's range API over the toolkit's ------------------------------------------------------------------

    def setRange(self, minimum, maximum) -> None:            # noqa: N802
        super().setRange(minimum, maximum)
        # `SpinBox.setRange` calls `setMinimumWidth(minimumSizeHint().width())`; that minimum is the toolkit's
        # content width (text + a 40 px gutter), not the Designer's. See the class docstring.
        self._release_width_pin()
        self._invalidate()

    def setMinimum(self, value) -> None:                     # noqa: N802
        self.setRange(value, self._maximum)

    def setMaximum(self, value) -> None:                     # noqa: N802
        self.setRange(self._minimum, value)

    def setButtonSymbols(self, symbols) -> None:             # noqa: N802
        """`QAbstractSpinBox::ButtonSymbols` over the stepper's visibility.

        RVE's boxes ask for `UpDownArrows` (the Qt default). `NoButtons` is the toolkit's own
        `setStepButtonsVisible(False)`; `PlusMinus` has no counterpart and is a splitter blocker.
        """
        from PySide6.QtWidgets import QAbstractSpinBox

        kind = QAbstractSpinBox.ButtonSymbols
        if symbols == kind.UpDownArrows:
            self.setStepButtonsVisible(True)
        elif symbols == kind.NoButtons:
            self.setStepButtonsVisible(False)
        else:
            raise NotImplementedError("Stepper draws ▲/▼ only - PlusMinus spin boxes stay QSpinBox")

    # -- geometry -------------------------------------------------------------------------------------------

    def setMaximumSize(self, *args) -> None:                 # noqa: N802
        # Remember the Designer's height ceiling: a later scale pass re-pins `setFixedHeight(32)` and would make
        # Qt ignore it (see the class docstring).
        size = args[0] if len(args) == 1 else QSize(args[0], args[1])
        self._designer_max_height = size.height()
        super().setMaximumSize(*args)

    def on_scale_changed(self, *_args) -> None:
        super().on_scale_changed(*_args)
        # `SpinBox.on_scale_changed` re-pins `setFixedHeight(scaled_px(32))`; RVE's Designer states the height.
        self.setMinimumHeight(0)
        self.setMaximumHeight(self._designer_max_height)
        if hasattr(self, "_sync_text_margins"):
            self._sync_text_margins()
        if hasattr(self, "_sync_step_buttons"):
            self._sync_step_buttons()
        self.updateGeometry()

    def setFont(self, font: QFont) -> None:                  # noqa: N802
        super().setFont(font)
        self._invalidate()

    def sizeHint(self) -> QSize:                             # noqa: N802
        return QSize(self._stock_metrics().width(), self.HEIGHT)

    def minimumSizeHint(self) -> QSize:                      # noqa: N802
        return self.sizeHint()


class QtSpinBox(_QtSpinShape, Stepper):
    """`QSpinBox`'s shape over `Stepper` (`SpinBox` plus the ▲/▼ gutter).

    The value surface RVE binds is Qt's: `setRange`/`setMinimum`/`setMaximum`/`setValue`/`value`, plus
    `text()`/`textChanged` (which `SettingsTab` binds for the GPU ids - `CustomLineEdit` is the field those come from).
    """

    RVE_QSS = _SPIN_QSS
    #: Qt's default range, not `SpinBox`'s 0..100: the Designer sets a minimum and leaves the maximum to the class.
    QT_DEFAULT_RANGE = (0, 99)

    def __init__(self, parent=None) -> None:
        super().__init__(parent, alignment=_qt_alignment())
        self._init_shape(*self.QT_DEFAULT_RANGE)

    def _stock_oracle(self):
        from PySide6.QtWidgets import QSpinBox

        oracle = QSpinBox()
        oracle.setStyleSheet(self.RVE_QSS)
        oracle.setFont(self.font())
        oracle.setRange(int(self._minimum), int(self._maximum))
        oracle.setValue(int(self._value))
        return oracle


class QtDoubleSpinBox(_QtSpinShape, StepButtonsMixin, DoubleSpinBox):
    """`QDoubleSpinBox`'s shape over the toolkit's `DoubleSpinBox` plus the same step gutter.

    `DoubleSpinBox` has no stepper class of its own; `StepButtonsMixin` is the toolkit's documented host hook for
    exactly this composition (the one `Stepper` uses), so the double gets the same ▲/▼ without a second control.

    One semantic difference is closed here, not inherited: `DoubleSpinBox.setValue` **snaps** to `singleStep`
    (`round((value - minimum) / step)`), so RVE's stored `scene_change_detection_threshold,3.5` came back as `3.0`
    at the default step of 1 - the exact "field shows the wrong value" defect this migration was told not to
    reintroduce. Qt's `QDoubleSpinBox.setValue` clamps and rounds to `decimals`; so does this.
    """

    RVE_QSS = _DOUBLE_SPIN_QSS
    #: Qt's default range for `QDoubleSpinBox` (the toolkit defaults to 0..100).
    QT_DEFAULT_RANGE = (0.0, 99.99)

    def __init__(self, parent=None) -> None:
        super().__init__(parent, alignment=_qt_alignment())
        self._init_step_gutter(True)
        self._init_shape(*self.QT_DEFAULT_RANGE)

    # -- the step gutter (both hooks `StepButtonsMixin` asks a host for) --------------------------------------

    def _step(self, delta: int) -> None:
        self.setValue(self.value() + float(delta) * (self.singleStep() or 1.0))

    def _sync_text_margins(self) -> None:
        gutter = self._step_gutter_width()
        self.setTextMargins(self.H_PADDING, 0, gutter + 6 if gutter else self.H_PADDING, 0)

    def _update_content_width(self) -> None:
        """The toolkit pins `setMinimumWidth(content_width)` here; the Designer's width is the authority."""

    # -- Qt's value semantics (no snapping) ------------------------------------------------------------------

    def setValue(self, value) -> None:                       # noqa: N802
        self._apply_value(float(value))

    def _apply_value(self, value: float) -> None:
        clamped = min(max(float(value), self._minimum), self._maximum)
        clamped = round(clamped, self._decimals)
        if abs(clamped - self._value) > 1e-12:
            self._value = clamped
            self.valueChanged.emit(self._value)
        text = self._format(clamped)
        if self.text() != text:
            self.setText(text)
        self.updateGeometry()

    def setDecimals(self, decimals: int) -> None:            # noqa: N802
        super().setDecimals(decimals)
        # Qt rounds the value to the new display precision; `DoubleSpinBox.setDecimals` only reformats the text.
        self._apply_value(self._value)
        self._invalidate()

    def _stock_oracle(self):
        from PySide6.QtWidgets import QDoubleSpinBox

        oracle = QDoubleSpinBox()
        oracle.setStyleSheet(self.RVE_QSS)
        oracle.setFont(self.font())
        oracle.setDecimals(self._decimals)
        oracle.setRange(self._minimum, self._maximum)
        oracle.setValue(self._value)
        return oracle


class QtProgressBar(ToastProgressBar):
    """`QProgressBar`'s shape over `ToastProgressBar`.

    RVE's bar is a `#2c313c` track, a white chunk, radius 10 and no text, in an Expanding/Fixed row. The toolkit's
    only progress control paints a rounded track and fill, so the adapter points its two colour hooks at the tokens
    RVE's `QProgressBar` rules were already mapped to (`toolkit_theme.LOOK`: `slider.track.background` and `accent`)
    instead of the toast tokens. `setTextVisible(False)` is the only call the widget does not have - the toolkit bar
    paints no text at all, so `False` is the state it is already in; `True` is a splitter blocker.

    Deviation carried for the live proof: the toolkit rounds the ends by height/2 (a 12.5 px pill on a 25 px bar),
    RVE's QSS says 10 px, so the two end caps differ by ~2.5 px.
    """

    #: Measured under RVE's `QProgressBar` QSS: a stock bar hints 91x25 (Expanding/Fixed, so the width is the row's
    #: and 25 px is what the row was laid out against).
    WIDTH = 91
    HEIGHT = 25

    def __init__(self, parent=None) -> None:
        super().__init__(parent, height=self.HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QSize:                             # noqa: N802
        return QSize(self.WIDTH, self.HEIGHT)

    def minimumSizeHint(self) -> QSize:                      # noqa: N802
        return self.sizeHint()

    def setTextVisible(self, visible: bool) -> None:         # noqa: N802
        if visible:
            raise NotImplementedError("ToastProgressBar paints no text - a text progress bar stays QProgressBar")

    def setMinimum(self, minimum: int) -> None:              # noqa: N802
        self.setRange(int(minimum), self.maximum())

    def setMaximum(self, maximum: int) -> None:              # noqa: N802
        self.setRange(self.minimum(), int(maximum))

    def text(self) -> str:
        """Qt's accessor; the toolkit bar has no label to report."""
        return ""

    def _track_color(self) -> QColor:
        return _token("slider.track.background") or super()._track_color()

    def _fill_color(self) -> QColor:
        return _token("accent") or super()._fill_color()


def _left():
    from PySide6.QtCore import Qt

    return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter


_LEFT = _left()
