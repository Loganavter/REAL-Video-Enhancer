"""The two preview bars: floating tools over the picture, the transport in the letterbox.

Layout E (owner-approved mockup, 2026-11-25; produced pill removed 2026-11-26):

* **top** - no common backing bar. The three view modes and the zoom row (`-` Fit `1:1` `+`) are each a CAPSULE (one
  SurfacePanel behind the group's ghost Buttons), and fullscreen is a standalone Button. A group's first/last buttons
  round their outer corners by the plate's radius minus its inset, so the capsule's corners share one shape; the
  checked view mode also wears an accent ring, so the current mode reads as a ring, not only as a fill. The compare is
  NOT a button: holding LMB on the picture is the gesture.
* **bottom** - video transport only: play/pause, step, content time `m:ss.mmm / m:ss.mmm`, and the scrubber. The
  run's own numbers (frame, percent, ETA, fps, status, elapsed) are the video-info pane's run card, not a repeat here.

Both bars are normal child widgets of the pane (the toolkit's time readout precedent: children of a `QRhiWidget`
composite above it). They sit in the pane's top/bottom inset and never in the middle. The overlay spans the pane but is
MASKED to the two bars, so the buttons take their presses and the picture's middle stays the pane's compare/pan.

`set_state(state)` is the ONE paint pass: it walks the catalog, applies `visible`/`enabled`/`checked` with signals
blocked (a state write must never look like a click), updates the time readout and hands the timeline its numbers.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QKeySequence, QPainter, QPen, QShortcut
from PySide6.QtWidgets import QHBoxLayout, QWidget

from sli_ui_toolkit.ui.managers.ui_scale import scaled_px
from sli_ui_toolkit.ui.widgets.atomic.surface_panel import SurfacePanel
from sli_ui_toolkit.ui.widgets.buttons.layers._base import outline_width
from sli_ui_toolkit.ui.widgets.buttons.layers.background import rounded_rect_path
from sli_ui_toolkit.ui.widgets.buttons.state import ButtonState
from sli_ui_toolkit.widgets import Button, ButtonFamily, Layer, Text

from .actions import ACTIONS, ActionSpec, actions_for, build_button, checked_for, enabled_for, visible_for
from .state import PreviewCommand, PreviewState
from .timeline import PreviewTimeline, format_content_time

#: Design px: the bars' inset from the pane edge, their gap, and their heights.
BAR_MARGIN = 12
BAR_GAP = 8
TOP_HEIGHT = 32
BOTTOM_HEIGHT = 42
#: Design px: a group capsule's plate radius, and the inset that separates its buttons from the plate (the same number
#: is the gap between the buttons). A nested button's outer corner is the difference, so its arc shares the plate's
#: centre and the gutter stays even at the top, bottom and outer edge.
GROUP_RADIUS = 10
GROUP_INSET = 2
NESTED_RADIUS = GROUP_RADIUS - GROUP_INSET
#: The time readout keeps a stable width, so a changing number cannot reflow the row.
TIME_MIN_WIDTH = 148


class CheckedOutlineLayer(Layer):
    """The checked toggle's accent ring, painted through the toolkit's own layer pipeline.

    `Button` paints its states through layers (`BackgroundLayer`, `BorderLayer`, ...); the toolkit's border layer is a
    *static* override, while a mode row needs the ring to follow the check. This layer is that stroke gated on
    `ButtonState.CHECKED` - the state `Button.setChecked` already writes - so hover/press keep the variant's fills and
    only the current group member is ringed. The colour is the theme accent; the width honours the toolkit's
    `borderWidth` property, exactly as `BorderLayer` reads it.
    """

    scope = "widget"

    def applies(self, ctx) -> bool:
        # A checked ring on a control that has nothing to act on is a promise: an empty pane has no mode, no zoom, so the
        # ring is suppressed while `outlineAllowed` is false (owner, 2026-11-26: "убирать обводку на выбранной кнопки когда
        # и показывать в превью ещё нечего").
        widget = getattr(ctx, "widget", None)
        if widget is not None and widget.property("outlineAllowed") is False:
            return False
        return ButtonState.CHECKED in ctx.states

    def draw(self, ctx, tm) -> None:
        painter = ctx.painter
        if not isinstance(painter, QPainter):
            return
        width = outline_width(ctx.widget)
        inset = width / 2.0
        painter.save()
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(QPen(tm.get_color("accent"), width))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(
                rounded_rect_path(QRectF(ctx.rect).adjusted(inset, inset, -inset, -inset), ctx.corner_radii))
        finally:
            painter.restore()


class PreviewControls(QWidget):
    """Both bars, built from the catalog and driven by `set_state`."""

    commandRequested = Signal(object, object)      # PreviewCommand, value

    def __init__(self, parent: QWidget, *, pane: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pane = pane if pane is not None else parent
        # The overlay spans the pane but only its two BARS may take the mouse: a widget mask (set in `_place`) is the
        # clip that keeps the picture's middle clickable. `WA_TransparentForMouseEvents` would make the bars and every
        # button in them dead (measured live 2026-11-25: the top tools and the transport ignored the pointer).
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._state = PreviewState()
        self._buttons: dict[str, Button] = {}
        self._specs: dict[str, ActionSpec] = {spec.id: spec for spec in ACTIONS}
        self._hidden_top: list[ActionSpec] = []
        self._groups: dict[str, ButtonFamily] = {}
        #: The buttons that wear the checked ring; `set_state` gates it on `state.has_picture()`.
        self._outline_buttons: list[Button] = []
        #: The visual capsules: one SurfacePanel per group, with the specs it holds (hidden together when it empties).
        self._group_panels: dict[str, SurfacePanel] = {}
        self._group_specs: dict[str, tuple[str, ...]] = {}

        self.top_bar = self._make_bar()
        self.bottom_bar = self._make_bar()
        self.top_bar.setObjectName("preview_top_bar")
        self.bottom_bar.setObjectName("preview_bottom_bar")
        self._build_top()
        self._build_bottom()
        self._build_shortcuts()

        self._pane.installEventFilter(self)
        self.setGeometry(self._pane.rect())
        self.top_bar.show()
        self.bottom_bar.show()
        self.show()
        self.raise_()
        self._place()

    # -- construction -----------------------------------------------------------------------------

    def _make_bar(self) -> QWidget:
        bar = QWidget(self)
        bar.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(scaled_px(BAR_GAP))
        bar._row = layout                     # noqa: SLF001 - the row is this module's private detail
        return bar

    def _add_button(self, layout: QHBoxLayout, spec: ActionSpec) -> Button:
        button = self._prepare_button(spec, layout.parentWidget())
        layout.addWidget(button)
        return button

    def _prepare_button(self, spec: ActionSpec, parent: QWidget, *,
                        corner_radii: tuple[int, int, int, int] | None = None,
                        outline_when_checked: bool = False) -> Button:
        """Build + wire one catalog entry. The layout that hosts it is the caller's (a bar row or a group capsule).

        `corner_radii` nests the button into its capsule (design px, tl/tr/br/bl); `outline_when_checked` installs the
        state-gated accent ring the group's current member wears.
        """
        extra_layers = [CheckedOutlineLayer()] if outline_when_checked else None
        button = build_button(spec, parent=parent, corner_radii=corner_radii, extra_layers=extra_layers)
        if outline_when_checked:
            # The ring follows the STATE, through one property the painter reads: `set_state` is its only writer.
            button.setProperty("outlineAllowed", False)
            self._outline_buttons.append(button)
            # The ring's weight is the toolkit's own `borderWidth` property (read by `outline_width`); 2 design px is
            # visible over the picture without turning the control into a plate.
            button.setProperty("borderWidth", float(scaled_px(2)))
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)   # the pane keeps its Space/arrows, see `_build_shortcuts`
        if spec.checkable:
            if spec.group:
                # The toolkit's Button is a QWidget, not a QAbstractButton, so a `QButtonGroup` cannot hold it. The
                # toolkit's own exclusive policy for standalone toggle buttons is `ButtonFamily.setExclusive(True)`
                # (the rank-pill idiom in BUTTON_API): checking one member unchecks the rest. The STATE is still the
                # source of truth (`set_state` re-applies `checked` every pass); the family is only widget exclusivity.
                family = self._groups.get(spec.group)
                if family is None:
                    family = ButtonFamily()
                    family.setExclusive(True)
                    self._groups[spec.group] = family
                family.addButton(button)
            button.toggled.connect(self._make_toggle_handler(spec))
        elif spec.hold:
            button.pressed.connect(lambda s=spec: self.commandRequested.emit(s.command, s.value))
            button.released.connect(
                lambda s=spec: self.commandRequested.emit(s.release_command or s.command, s.value))
        else:
            button.clicked.connect(lambda _=False, s=spec: self.commandRequested.emit(s.command, s.value))
        self._buttons[spec.id] = button
        return button

    def _add_group(self, layout: QHBoxLayout, name: str, spec_ids: tuple[str, ...], *,
                   outline_checked: bool = False) -> SurfacePanel:
        """A group capsule: one toolkit surface behind ghost buttons - the backing the owner asked for.

        The group is ONE unit for the overflow too: a capsule with half its buttons in a menu and half on the bar is
        not a control a person can read. The first and last buttons round their OUTER corners by the plate's radius
        minus the inset, so those arcs are concentric with the plate and the gutter reads even; the inner side stays
        flat, which makes the row read as one capsule.
        """
        # The plate wears the STANDALONE surface button's own fill and stroke (the fullscreen button's variant), not the
        # flyout colour: the modes and the zoom row are the same kind of control as the button beside them, and the owner
        # read the difference as two visual languages in one bar (2026-11-26: "2 группы контролов сверху нужно перекрасить
        # подложку в стиль кнопки fullscreen"). The tokens are the toolkit's, so a theme change moves both together.
        panel = SurfacePanel(self.top_bar,
                             background_token="button.dialog.default.background",
                             border_token="button.dialog.default.border",
                             radius=scaled_px(GROUP_RADIUS))
        row = QHBoxLayout(panel)
        inset = scaled_px(GROUP_INSET)
        row.setContentsMargins(inset, inset, inset, inset)
        row.setSpacing(inset)
        last = len(spec_ids) - 1
        for index, spec_id in enumerate(spec_ids):
            radii = None
            if last > 0:
                radii = (
                    NESTED_RADIUS if index == 0 else 0,
                    NESTED_RADIUS if index == last else 0,
                    NESTED_RADIUS if index == last else 0,
                    NESTED_RADIUS if index == 0 else 0,
                )
            row.addWidget(self._prepare_button(self._specs[spec_id], panel, corner_radii=radii,
                                               outline_when_checked=outline_checked))
        self._group_panels[name] = panel
        self._group_specs[name] = tuple(spec_ids)
        layout.addWidget(panel)
        return panel

    def _make_toggle_handler(self, spec: ActionSpec):
        def handle(checked: bool) -> None:
            if not checked:
                return
            self.commandRequested.emit(spec.command, spec.value)
        return handle

    def _build_top(self) -> None:
        layout = self.top_bar._row                   # noqa: SLF001
        layout.addStretch(1)
        self._add_group(layout, "view_mode", ("mode_single", "mode_sbs", "mode_split"), outline_checked=True)
        self._add_group(layout, "zoom", ("zoom_out", "zoom_fit", "zoom_one", "zoom_in"))

        # The overflow affordance for a narrow pane: the lowest-priority tools collapse here instead of shrinking.
        self.overflow_button = Button(text="\u2026", variant="surface", size=(scaled_px(30), scaled_px(30)),
                                      parent=self.top_bar)
        self.overflow_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.overflow_button.setToolTip("More preview tools")
        self.overflow_button.clicked.connect(self._show_overflow_menu)
        self.overflow_button.setVisible(False)
        layout.addWidget(self.overflow_button)

    def _build_bottom(self) -> None:
        layout = self.bottom_bar._row                # noqa: SLF001
        self._add_button(layout, self._specs["step_back"])
        self._add_button(layout, self._specs["play"])
        self._add_button(layout, self._specs["step_forward"])

        self.time_text = Text(format_content_time(0.0), variant="caption", minimum_width=scaled_px(TIME_MIN_WIDTH))
        self.time_text.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        layout.addWidget(self.time_text, 0, Qt.AlignmentFlag.AlignVCenter)

        self.timeline = PreviewTimeline(self.bottom_bar)
        #: Named for the control surface: a strip is a custom-painted widget, so a probe can only reach it by name and
        #: a point (`click:preview_timeline@x,y`) - and a seek is exactly the gesture that must be provable live, not
        #: inferred from a screenshot (2026-11-25: an unnamed strip made "the press did nothing" indistinguishable
        #: from "the press never landed").
        self.timeline.setObjectName("preview_timeline")
        self.timeline.seekRequested.connect(self._on_seek_requested)
        layout.addWidget(self.timeline, 1)

    def _build_shortcuts(self) -> None:
        """The catalog's keyboard routes, on the pane itself - the transport's keyboard contract.

        `WidgetShortcut` (not `WidgetWithChildren`): the slider keeps its own arrow navigation, and the pane only
        answers when it has the focus, exactly as it did for the strip's presses.
        """
        self._shortcuts: list[QShortcut] = []
        for spec in ACTIONS:
            if not spec.shortcut:
                continue
            shortcut = QShortcut(QKeySequence(spec.shortcut), self._pane)
            shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
            shortcut.activated.connect(lambda s=spec: self.commandRequested.emit(s.command, s.value))
            self._shortcuts.append(shortcut)

    # -- geometry ---------------------------------------------------------------------------------

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt spelling
        if watched is self._pane and event.type() in (QEvent.Type.Resize, QEvent.Type.Show):
            self.setGeometry(QRect(0, 0, self._pane.width(), self._pane.height()))
            self._place()
        return super().eventFilter(watched, event)

    def _place(self) -> None:
        width = max(0, self.width())
        height = max(0, self.height())
        margin = scaled_px(BAR_MARGIN)
        span = max(0, width - 2 * margin)
        self._apply_overflow(span)
        self.top_bar.setGeometry(margin, margin, span, scaled_px(TOP_HEIGHT))
        bottom = max(scaled_px(BOTTOM_HEIGHT), height - margin - scaled_px(BOTTOM_HEIGHT))
        self.bottom_bar.setGeometry(margin, bottom, span, scaled_px(BOTTOM_HEIGHT))
        self.top_bar.raise_()
        self.bottom_bar.raise_()
        # Only the bars take the mouse; the rest of the overlay is a hole the pane's compare/pan sees through.
        from PySide6.QtGui import QRegion

        self.setMask(QRegion(self.top_bar.geometry()) | QRegion(self.bottom_bar.geometry()))

    def _top_specs(self) -> list[ActionSpec]:
        return list(actions_for("top"))

    def _apply_overflow(self, available: int) -> None:
        """Collapse the lowest-priority tools into the "…" button when the row cannot hold them at this scale."""
        if available <= 0:
            return
        for spec in self._top_specs():
            self._buttons[spec.id].setVisible(visible_for(spec, self._state))
        for panel in self._group_panels.values():
            panel.setVisible(True)
        self._hidden_top = []
        needed = self.top_bar.sizeHint().width()
        if needed <= available:
            self.overflow_button.setVisible(False)
            return
        # Collapse by UNIT (a whole exclusive group moves together - half a mode row in a menu and half on the bar is
        # not a choice a person can read). Highest priority survives; zoom goes first, the modes last.
        units: dict[str, list[ActionSpec]] = {}
        for spec in self._top_specs():
            units.setdefault(spec.group or spec.id, []).append(spec)
        for _unit, specs in sorted(units.items(), key=lambda item: min(spec.priority for spec in item[1])):
            if needed <= available:
                break
            visible = [spec for spec in specs if not self._buttons[spec.id].isHidden()
                       and visible_for(spec, self._state)]
            if not visible:
                continue
            for spec in visible:
                self._buttons[spec.id].setVisible(False)
                self._hidden_top.append(spec)
            panel = self._group_panels.get(_unit)
            if panel is not None:
                panel.setVisible(False)     # the whole capsule went with its last visible button
            needed = self.top_bar.sizeHint().width()
        self.overflow_button.setVisible(bool(self._hidden_top))

    def _show_overflow_menu(self) -> None:
        """The collapsed tools, as a toolkit context menu anchored to the "…" button."""
        if not self._hidden_top:
            return
        from sli_ui_toolkit.widgets import entries_from_callbacks, popup_context_menu_for_anchor

        entries = []
        for spec in self._hidden_top:
            label = spec.text or spec.tip or spec.id
            entries.append((label, self._make_menu_handler(spec)))
        popup_context_menu_for_anchor(self.window(), self.overflow_button, entries_from_callbacks(entries))

    def _make_menu_handler(self, spec: ActionSpec):
        def handle() -> None:
            self.commandRequested.emit(spec.command, spec.value)
        return handle

    # -- the one state pass ------------------------------------------------------------------------

    def set_state(self, state: PreviewState) -> None:
        """Apply the state to every widget in ONE pass; signals blocked while `checked` is written."""
        self._state = state
        # The ring is a claim about the CURRENT mode; with nothing on the pane there is no mode to claim. One property
        # write per pass, read by `CheckedOutlineLayer.applies` (the painter's own gate).
        for outlined in self._outline_buttons:
            outlined.setProperty("outlineAllowed", bool(state.has_picture()))
        for spec in ACTIONS:
            button = self._buttons.get(spec.id)
            if button is None:
                continue
            button.setVisible(visible_for(spec, state))
            button.setEnabled(enabled_for(spec, state))
            if spec.checkable:
                checked = checked_for(spec, state)
                blocked = button.blockSignals(True)
                try:
                    if button.isChecked() != checked:
                        button.setChecked(checked)
                finally:
                    button.blockSignals(blocked)
            if spec.icon_when is not None:
                glyph = spec.icon_when(state)
                if glyph is not None:
                    button.setIcon(glyph)

        self.time_text.setText(
            f"{format_content_time(state.playhead)} / {format_content_time(state.duration)}")
        self.timeline.set_state(state)
        # Visibility changes above can widen the row; re-run the collapse so the bars keep their place in the pane.
        self._place()

    # -- transport ---------------------------------------------------------------------------------

    def _on_seek_requested(self, seconds: float) -> None:
        self.commandRequested.emit(PreviewCommand.SEEK_TO, float(seconds))

    def button(self, spec_id: str) -> Button | None:
        """The widget behind an id (tests and the live proof read the real button, not a look-alike)."""
        return self._buttons.get(spec_id)

    def hidden_tools(self) -> list[str]:
        return [spec.id for spec in self._hidden_top]
