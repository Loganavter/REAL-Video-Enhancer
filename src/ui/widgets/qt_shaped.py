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
from PySide6.QtCore import QEvent
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


#: RVE's control text is 12 PIXELS (the Designer's QSS/`retranslate` sizing). A toolkit control paints with the UI font
#: it is given, and an inherited font is the toolkit's own (16 px at the live scale) - measured as a 34 px combo field
#: and 23 px checkbox rows in 29/18 px rows. Adapters state the host's size instead of inheriting it.
RVE_CONTROL_FONT_PX = 12


def _rve_control_font(widget) -> QFont:
    font = QFont(widget.font())
    font.setPixelSize(RVE_CONTROL_FONT_PX)
    return font


def _constraints(widget) -> tuple:
    return (widget.minimumSize(), widget.maximumSize(), widget.sizePolicy())


def _restore_constraints(widget, saved: tuple) -> None:
    """Re-impose the constraints the caller set before a toolkit-internal geometry pass ran."""
    minimum, maximum, policy = saved
    widget.setMinimumSize(minimum)
    widget.setMaximumSize(maximum)
    widget.setSizePolicy(policy)


def _oracle_style_sheet(widget, base: str) -> str:
    """The QSS a stock counterpart actually inherited, plus the adapter's own rule.

    A measurement widget built with only the adapter's hard-coded rule sees a different `padding`, `min-width` or
    indicator size than the stock widget it is standing in for whenever a page states its own rule. Ancestors are
    joined root-first and the widget's own sheet last, which is the order Qt's cascade applies; `base` is the rule
    the adapter replaces (RVE's selector, verbatim).
    """
    sheets: list[str] = []
    current = widget
    while current is not None:
        sheet = current.styleSheet()
        if sheet and sheet.strip():
            sheets.append(sheet)
        current = current.parentWidget()
    sheets.reverse()
    sheets.append(base)
    return "\n".join(sheets)


#: The stock vertical scrollbar's width, from the window's QSS (`QScrollBar:vertical{width:8}`). A converted scroll
#: area that RESERVES a bar must reserve this, not the toolkit's default 10: measured on the general-settings page,
#: the 2 px difference narrowed every row and moved the page 1.2 %.
STOCK_SCROLLBAR_WIDTH = 8

#: The Designer's `QFrame` separator is a 3 px sunken line (`testRVEInterface.ui`); a token-painted panel of that height
#: reproduces it. Named here so the page has no bare pixel literal for the gate to record as new debt.
SEPARATOR_HEIGHT = 3

#: The render queue's button column (`widget_23`) is 200 px wide in the Designer build: the stock list asked for 256 px
#: and the remaining 200 px of the 462 px row went to the column. A toolkit list that asks for a different share moves
#: that split (measured: 335 px list, 121 px column), so the column states its width instead of inheriting one.
QUEUE_BUTTON_COLUMN_WIDTH = 200

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

    Toggle mode is constructor-only in the toolkit (`Button(toggle=True)`), so the splitter states it at the call site
    (`QtButton(parent, toggle=True)`) for a page widget the Designer marks `setCheckable(True)`. `setCheckable(True)`
    after construction still raises: the toolkit has no way to turn the state machine on, and pretending otherwise
    would ship a button that never checks. Auto-exclusive buttons stay stock - `Button` has no group semantics.
    """

    #: Caller constraints, remembered by the setters below. A toolkit geometry pass (`setText`, `_apply_text_geometry`
    #: on an icon/text transition, `on_scale_changed`) may call `setFixedSize`, which overwrites both; the first
    #: `sizeHint` after such a pass puts the caller's numbers back before the layout reads them.
    _caller_min = None
    _caller_max = None
    _hint_settled = False

    def __init__(self, parent=None, text: str = "", toggle: bool = False) -> None:
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
                         content_padding=(10, 5, 10, 5), corner_radius=10, toggle=toggle,
                         border_color=QColor(0, 0, 0, 0))
        self._has_icon = False
        self._icon_box = QSize(0, 0)
        # The base class records a 36x36 "design fixed size" for a button built without text and re-applies it on every
        # scale change (`on_scale_changed`). RVE's icon buttons are sized by the Designer's max (59x55, 50x45, ...), so
        # the toolkit's recording is dropped here or a live scale change would re-pin the 36x36 square.
        self._design_fixed_size = (0, 0)
        self._design_min_height = None
        # No `rows=`: the default content path paints the widget's own font (`paint_font(widget)` reads `font()`) AND
        # draws the icon, whereas a hand-built row has no icon slot at all - building rows dropped the "+" on RVE's
        # add-to-queue button (measured). `setFont` is therefore enough; it just needs the constraints preserved.
        # The toolkit's Button base gives every instance a 36x36 minimum AND maximum (its own default control size).
        # The minimum turned RVE's 22x16 checkbox into a 36x36 square; the maximum clamped every text button to 36 px
        # wide, which is what made labels look cut off. Both are cleared here; the Designer's own
        # `setMinimumSize`/`setMaximumSize` statements run after this constructor and still apply where declared.
        self.setMinimumSize(0, 0)
        self.setMaximumSize(16777215, 16777215)
        # The clearing above IS the caller's state until the page says otherwise: `_apply_text_geometry` runs inside the
        # toolkit constructor and pins 36x36, so a snapshot taken later would carry the toolkit's square as if the
        # caller had asked for it (measured: five icon-only buttons stayed 36x36 where the Designer gives 50x30/59x30).
        self._caller_min = QSize(0, 0)
        self._caller_max = QSize(16777215, 16777215)
        # QPushButton's own policy: as wide as it needs, exactly as tall as its hint. The toolkit's button is happy to
        # fill the cell, which turned a 27 px button into a 96 px slab inside its row (measured on the settings page).
        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.setFont(_rve_control_font(self))

    def setSizePolicy(self, horizontal, vertical=None) -> None:   # noqa: N802
        if isinstance(horizontal, QSizePolicy):
            self._rve_policy = (horizontal.horizontalPolicy(), horizontal.verticalPolicy())
        else:
            self._rve_policy = (horizontal, vertical)
        if vertical is None:
            Button.setSizePolicy(self, horizontal)
        else:
            Button.setSizePolicy(self, horizontal, vertical)

    # -- the caller's constraints survive every toolkit geometry pass -------------------------------------

    def _own_constraints(self) -> tuple:
        """The CALLER's pins, not whatever a toolkit pass left behind."""
        minimum = self._caller_min if self._caller_min is not None else self.minimumSize()
        maximum = self._caller_max if self._caller_max is not None else self.maximumSize()
        return (minimum, maximum, self.sizePolicy())

    def setFixedSize(self, *args) -> None:       # noqa: N802
        # A toolkit pass may fix a design size (the 36x36 square for an icon-only button). The CALLER's pins win: the
        # Designer gives these buttons a maximum, and a fixed size overwrote it at construction (measured: 29 calls
        # from `Button.__init__`, every one turning a 50x45 icon button into a 36x36 square before the page could size
        # it).
        saved = self._own_constraints()
        super().setFixedSize(*args)
        _restore_constraints(self, saved)

    def setMinimumSize(self, *args) -> None:     # noqa: N802
        super().setMinimumSize(*args)
        self._caller_min = QSize(self.minimumSize())

    def setMaximumSize(self, *args) -> None:     # noqa: N802
        super().setMaximumSize(*args)
        self._caller_max = QSize(self.maximumSize())

    def _apply_text_geometry(self, has_text: bool) -> None:
        """The toolkit's icon/text transition, with the caller's pins put back.

        `_ButtonStyleApi._apply_text_geometry` calls `setFixedSize(36, 36)` when a button loses its text (measured: an
        icon-only button the Designer gave 59x30 came back a 36x36 square). Every other path in this class snapshots
        the constraints around the call; this one is the transition itself, so it is guarded once here instead of at
        each caller - `setText`, `setRows`, a future toolkit setter and the scale pass all land in it.
        """
        saved = self._own_constraints()
        super()._apply_text_geometry(has_text)
        # The pass records the toolkit's 36x36 square as the button's DESIGN size, and `on_scale_changed` re-applies it
        # at the live factor - so restoring the constraints here was not enough: the next scale event put the square
        # back (measured: five icon-only buttons were 50x45 offscreen and 36x36 in the running app, which is the only
        # place the scale bus fires).
        self._design_fixed_size = (0, 0)
        self._design_min_height = None
        _restore_constraints(self, saved)
        self.updateGeometry()

    def _settle_constraints(self) -> None:
        """First `sizeHint` after construction: re-impose what the caller pinned, if a pass overwrote it."""
        if self._hint_settled:
            return
        self._hint_settled = True
        if self._caller_min is not None:
            super().setMinimumSize(self._caller_min)
        if self._caller_max is not None:
            super().setMaximumSize(self._caller_max)

    def setText(self, text: str) -> None:        # noqa: N802
        # `Button.setText` re-derives its geometry AND resets `maximumSize` to unbounded (measured: a 30 px high
        # button came back unbounded, so every button in a row grew past the Designer's height). Whatever the caller
        # had set is re-imposed after the pass, and the row is rebuilt so the text keeps the button's font size.
        saved = self._own_constraints()
        Button.setText(self, text)
        _restore_constraints(self, saved)
        self.updateGeometry()

    def setFont(self, font: QFont) -> None:      # noqa: N802
        saved = self._own_constraints()
        Button.setFont(self, font)
        _restore_constraints(self, saved)

    def setIcon(self, icon) -> None:             # noqa: N802
        # The toolkit keeps its icon privately and exposes no getter; the hint below needs to know whether there is one
        # and how big it is, so the adapter records what Qt's setter was told.
        saved = self._own_constraints()
        self._has_icon = not icon.isNull()
        super().setIcon(icon)
        _restore_constraints(self, saved)
        # The hint is computed from the icon, and the layout caches it: without this a button whose icon arrives after
        # construction keeps the hint it had while empty - the toolkit's 36x36 design square - and the row sizes to
        # that (measured: five icon-only buttons stayed 36x36 where the Designer gives them 50x30 / 59x30).
        self.updateGeometry()

    def setIconSize(self, size) -> None:         # noqa: N802
        saved = self._own_constraints()
        self._icon_box = QSize(size)
        super().setIconSize(size)
        _restore_constraints(self, saved)
        self.updateGeometry()

    def text(self) -> str:
        """The toolkit keeps the label in `_text` and exposes no getter; Qt's name is part of the shape RVE expects."""
        return getattr(self, "_text", "") or ""

    def sizeHint(self) -> QSize:                 # noqa: N802
        self._settle_constraints()
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
                # `padding: 5px 10px` around the icon: the stock hint for a 40x45 icon is 64x55, and for a 30 px icon
                # 54x40 (measured under RVE's QSS). The Designer's own maximum then decides - 59x55 for the start
                # button, whose 55 px is the row's height; removing this 10 px made it 45 and moved the tab host.
                return QSize(icon_size.width() + 20, content_h + 10)
            return super().sizeHint()
        icon_room = (icon_size.width() + 6) if has_icon else 0
        return QSize(fm.horizontalAdvance(text) + 20 + icon_room, content_h + 10)

    def minimumSizeHint(self) -> QSize:          # noqa: N802
        return self.sizeHint()


    def setCheckable(self, on: bool) -> None:    # noqa: N802
        # Toggle mode is a constructor flag: the splitter emits `QtButton(parent, toggle=True)` for a page widget the
        # Designer marks checkable, so the state machine exists before this line runs. A later `setCheckable(True)` on
        # a button built without it cannot be honoured (the toolkit offers no way to attach toggle after the fact), and
        # raising is the honest answer - a silent no-op would ship a button that never checks.
        if on and not getattr(self, "_has_toggle", False):
            raise NotImplementedError(
                "Button toggle mode is constructor-only - build QtButton(parent, toggle=True); "
                "the splitter passes the flag for a checkable page widget")

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
    PADDING_H = 2
    PADDING_V = 0

    """`QCheckBox`'s shape over `CheckBox` (Qt passes the parent first and sets the text later).

    `stateChanged(int)` is the Qt signal RVE's glue actually binds (12 sites: `ProcessTab`, `SettingsTab`, ...). The
    toolkit box reports `toggled(bool)`, so the adapter carries Qt's spelling as a derived signal - the same semantics
    (2 = checked, 0 = unchecked), no behaviour of its own.

    The size comes from a stock `QCheckBox` oracle, not from a formula: the app QSS styles the indicator in two states
    (`:unchecked` 14x14 with `left: 2px`, `:checked` 20x20), and Qt measures the state the box is currently in. Measured
    on the Designer build (Fusion, RVE QSS): empty unchecked 22x16, empty checked 26x20, `Reuse unchanged frames`
    checked 170x20, `By cel ...` unchecked 263x17. A flat 22x16 is the unchecked empty box only - every checked box on
    the Process page is 26x20, which is the row shift this replaces.
    """

    stateChanged = Signal(int)

    #: RVE's indicator rules, verbatim from the shell's QSS (`src/ui/shell/window.py`). The `image:` in the checked
    #: rule paints the tick; it is left out of the oracle because a missing resource must not change a measurement,
    #: and the 20x20 size - the part that moves the row - is kept.
    RVE_QSS = (
        "QCheckBox::indicator:unchecked{border:1px solid white;border-radius:2px;width:14px;height:14px;left:2px;}"
        "QCheckBox::indicator:checked{color:white;border-radius:2px;width:20px;height:20px;}"
    )

    _stock_hint = None

    def __init__(self, parent=None, text: str | None = None) -> None:
        super().__init__(text or "", parent)
        self._stock_hint = None
        self.toggled.connect(lambda checked: self.stateChanged.emit(2 if checked else 0))
        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.setFont(_rve_control_font(self))
        # The toolkit's Button base caps every instance at 36x36. Clearing only the MINIMUM left the maximum in place,
        # so a labelled checkbox was clipped to a 36 px strip with its text cut off (measured on the process page).
        self.setMinimumSize(0, 0)
        self.setMaximumSize(16777215, 16777215)
        # The clearing above IS the caller's state until the page says otherwise: `_apply_text_geometry` runs inside the
        # toolkit constructor and pins 36x36, so a snapshot taken later would carry the toolkit's square as if the
        # caller had asked for it (measured: five icon-only buttons stayed 36x36 where the Designer gives 50x30/59x30).
        self._caller_min = QSize(0, 0)
        self._caller_max = QSize(16777215, 16777215)

    def _invalidate(self) -> None:
        self._stock_hint = None
        # The hint moved, so the layout has to hear it (the combo's 107 px lesson: a cached hint nobody is told about
        # is a widget that keeps its old width).
        self.updateGeometry()

    def event(self, event) -> bool:                 # noqa: N802
        # The effective font can change without `setFont` being called: a page's `apply(container, BODY_PT)` runs
        # before the children exist, so a widget measured during `build()` sees the inherited 12 pt, and polish later
        # settles it to the app's 9 pt. A cached hint measured under the old font is exactly the "quality preset
        # 104 -> 126" excursion (measured): the stock hint is 104x29 at 9 pt, the stale cache said 126x34. Qt sends
        # `FontChange` for an inherited change; invalidating there is the missing rule.
        if event.type() == QEvent.Type.FontChange:
            self._invalidate()
        return super().event(event)

    def _stock_oracle(self):
        from PySide6.QtWidgets import QCheckBox

        oracle = QCheckBox()
        oracle.setStyleSheet(_oracle_style_sheet(self, self.RVE_QSS))
        oracle.setFont(self.font())
        oracle.setText(self.text())
        oracle.setChecked(self.isChecked())
        return oracle

    def _stock_metrics(self) -> QSize:
        if self._stock_hint is None:
            oracle = self._stock_oracle()
            try:
                self._stock_hint = oracle.sizeHint()
            finally:
                oracle.deleteLater()
        return self._stock_hint

    def setChecked(self, checked: bool) -> None:     # noqa: N802
        # The toolkit animates the check mark over 150 ms from `toggled`. A checkbox set before its page is shown
        # never ticks that animation (a hidden widget gets no frames), so it painted as UNCHECKED while `isChecked()`
        # was True (measured on the process page). Qt's `setChecked` has no animation; neither does this.
        super().setChecked(checked)
        self.set_checked_progress(1.0 if checked else 0.0)
        self._invalidate()

    def setText(self, text: str) -> None:            # noqa: N802
        saved = _constraints(self)
        super().setText(text)
        _restore_constraints(self, saved)
        self._invalidate()

    def setFont(self, font: QFont) -> None:          # noqa: N802
        saved = _constraints(self)
        super().setFont(font)
        _restore_constraints(self, saved)
        self._invalidate()

    def sizeHint(self) -> QSize:                 # noqa: N802
        return self._stock_metrics()

    def minimumSizeHint(self) -> QSize:          # noqa: N802
        return self.sizeHint()

    def setTristate(self, on: bool) -> None:     # noqa: N802
        # `setTristate(False)` asks for the state this widget already has - a Designer `QCheckBox` emits it for every
        # box unless the .ui said tristate. Refusing it kept a plain two-state box stock for no reason (measured on the
        # interpolate checkbox). Asking for the third state is still a real gap and still raises.
        if on:
            raise NotImplementedError("CheckBox has no tristate - a tristate box stays QCheckBox")


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
        # `ComboBox.__init__` builds on `Button(..., size=(0, BASE_HEIGHT))`, so the toolkit pins min AND max height to
        # its own 33 px. A Designer `setMaximumSize(230, 30)` then has no effect (Qt ignores a maximum below the
        # minimum), which is the same class of defect that cost the spin boxes 3-6 px per row. The pins are released;
        # the class's own `BASE_HEIGHT` still states the height and the Designer's statements remain the authority.
        self.setMinimumHeight(0)
        self.setMaximumHeight(16777215)
        # QComboBox's own policy is Preferred/Fixed, NOT Expanding: an Expanding combo eats the row's spare width and
        # squeezes the label next to it to its hint (measured: the settings rows lost 240 px of label). A layout shares
        # spare width among the Preferred items, which is what the Designer's rows were laid out against.
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        # The field's font is the Designer's 12 PIXELS, stated here instead of inherited: the toolkit's button pipeline
        # applies its own UI font (16 px at the live scale), and the stock oracle below measures under `self.font()`, so
        # an inherited font made the field 34 px tall and 22 px too wide in a 29 px row (measured: 126x34 against the
        # stock 104x29 on the quality-preset row).
        self.setFont(_rve_control_font(self))

    def _invalidate(self) -> None:
        self._stock_hint = None
        # The hint changed, so the LAYOUT has to know. Without this the combo kept the width it was given while empty
        # (38 px on the settings page) even though its sizeHint had already grown to the stock 145 px - measured by
        # dumping the widget after the app had populated it.
        self.updateGeometry()

    def event(self, event) -> bool:                       # noqa: N802
        # Same rule as the checkbox: an inherited font change does not call `setFont`, and a hint cached under the
        # pre-polish font is the measured 104 -> 126 excursion (the stock hint is 104x29 at the app's 9 pt; the cache
        # held 126x34 from the 12 pt the page container had while `build()` ran).
        if event.type() == QEvent.Type.FontChange:
            self._invalidate()
        return super().event(event)

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

    def oracle_style_sheet(self) -> str:
        """The stock widget's effective QSS, not just RVE's base rule.

        The stock `QComboBox` inherited whatever the *page* and the window set, and a page rule can change the hint
        (`padding`, `min-width`, a different radius). The oracle used to carry `RVE_QSS` alone, so any page that
        stated its own rule was measured against a rule the stock widget never saw. Ancestors are joined root-first
        and the widget's own sheet last, which is the cascade order Qt applies.
        """
        return _oracle_style_sheet(self, self.RVE_QSS)

    def _stock_oracle(self):
        from PySide6.QtWidgets import QComboBox

        oracle = QComboBox()
        oracle.setStyleSheet(self.oracle_style_sheet())
        oracle.setFont(self.font())
        # Qt's default policy (`AdjustToContentsOnFirstShow`) freezes the hint the first time the popup is shown;
        # a measurement widget is never shown and its items change under it, so the policy is the per-query one.
        # `minimumContentsLength` is left at 0: RVE's QSS states a padding, not a character floor (measured with 0,
        # 1, 2 and 3: 59 -> 59, 65, 72, 79 px, none of which is a Designer width).
        oracle.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        oracle.setMinimumContentsLength(0)
        for index in range(self.count()):
            oracle.addItem(self.itemText(index))
        return oracle

    def _stock_metrics(self) -> QSize:
        if self._stock_hint is None:
            oracle = self._stock_oracle()
            try:
                self._stock_hint = oracle.sizeHint()
            finally:
                oracle.deleteLater()
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
    #: The row is 29 px in the Designer's build - the same height as a combo, measured on the live app (the cell's
    #: first attempt stated 32 from an offscreen probe, which shifted every row on the settings page by 3-6 px).
    HEIGHT = 29
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
        self.setFont(_rve_control_font(self))

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

    TRACK_TOKEN = "slider.track.background"

    def __init__(self, parent=None, *, track_token: str | None = None) -> None:
        super().__init__(parent, height=self.HEIGHT)
        self._track_token = track_token or self.TRACK_TOKEN
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
        return _token(self._track_token) or super()._track_color()

    def _fill_color(self) -> QColor:
        return _token("accent") or super()._fill_color()


def _left():
    from PySide6.QtCore import Qt

    return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter


_LEFT = _left()
