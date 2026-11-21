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

Not every class is here: `QSpinBox`/`QDoubleSpinBox` stay stock because the toolkit's `SpinBox`/`DoubleSpinBox` expose
no range API (`setMinimum`/`setMaximum`/`setSingleStep` are absent), and pixmap labels stay `QLabel` because `Text`
paints text only. Both are toolkit rows, not local workarounds.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtCore import QSize
from PySide6.QtWidgets import QSizePolicy
from PySide6.QtGui import QFont, QFontInfo, QFontMetrics

from sli_ui_toolkit.widgets import Button, CheckBox, ComboBox, CustomLineEdit, Text


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
        super().__init__(None, text=text, parent=parent, variant="surface", content_align=_LEFT,
                         content_padding=(5, 10, 5, 10), corner_radius=10)
        self._has_icon = False
        self._icon_box = QSize(0, 0)
        # The toolkit's Button base gives every instance a 36x36 minimum (its own default control size), which turned
        # RVE's 22x16 checkbox into a 36x36 square and clamped icon buttons. The Designer's own `setMinimumSize`
        # statements run after this constructor and still apply where the page declares one.
        self.setMinimumSize(0, 0)
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
        # had set is re-imposed after the pass.
        saved = _constraints(self)
        super().setText(text)
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
        if not text:
            if has_icon:
                return QSize(icon_size.width() + 20, icon_size.height() + 10)
            return super().sizeHint()
        fm = QFontMetrics(self.font())
        icon_room = (icon_size.width() + 6) if has_icon else 0
        return QSize(fm.horizontalAdvance(text) + 20 + icon_room, fm.height() + 10)

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

    def addItem(self, text, userData=None) -> None:      # noqa: N802
        super().addItem(text, userData)
        self._stock_hint = None

    def addItems(self, texts) -> None:                   # noqa: N802
        super().addItems(texts)
        self._stock_hint = None

    def setFont(self, font) -> None:                     # noqa: N802
        super().setFont(font)
        self._stock_hint = None

    def _stock_metrics(self):
        if self._stock_hint is None:
            from PySide6.QtWidgets import QComboBox

            oracle = QComboBox()
            oracle.setFont(self.font())
            for index in range(self.count()):
                oracle.addItem(self.itemText(index))
            self._stock_hint = oracle.sizeHint()
        return self._stock_hint

    def sizeHint(self) -> QSize:                          # noqa: N802
        # RVE's combo: `QComboBox{padding:5px 10px}`, so a field is its longest item plus 20 px, never under 80, and
        # 29 px tall (5+5 padding + 17 px of 12 px text + 2 border - measured on the Designer build).
        fm = QFontMetrics(self.font())
        widest = max((fm.horizontalAdvance(self.itemText(i)) for i in range(self.count())), default=0)
        return QSize(max(80, widest + 20), 29)

    def minimumSizeHint(self) -> QSize:                   # noqa: N802
        return self.sizeHint()


class QtLineEdit(CustomLineEdit):
    """`QLineEdit`'s shape over `CustomLineEdit`.

    RVE's line edit is 25 px tall for a 12 px font (`QLineEdit` with the page's 5 px padding); the toolkit field is
    32. Height is stated here, width stays expanding like Qt's.
    """

    def __init__(self, parent=None, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QSize:                 # noqa: N802
        return QSize(super().sizeHint().width(), QFontMetrics(self.font()).height() + 8)

    def minimumSizeHint(self) -> QSize:          # noqa: N802
        return self.sizeHint()


def _left():
    from PySide6.QtCore import Qt

    return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter


_LEFT = _left()
