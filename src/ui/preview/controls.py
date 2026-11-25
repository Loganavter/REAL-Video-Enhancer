"""The two preview bars: floating tools over the picture, the transport in the letterbox.

Layout E (owner-approved mockup, 2026-11-25):

* **top** - no common backing bar. A `produced 12.4 s · 60%` pill owns its own small surface; Compare (hold), the
  three view modes (exclusive toggle buttons), the zoom row (`-` Fit `1:1` `+`) and fullscreen are separate toolkit
  `Button`s, each with its own surface.
* **bottom** - video transport only: play/pause, step, content time `m:ss.mmm / m:ss.mmm`, source frame
  `f 100 / 493`, and the scrubber.

Both bars are normal child widgets of the pane (the toolkit's time readout precedent: children of a `QRhiWidget`
composite above it). They sit in the pane's top/bottom inset and never in the middle, so the pane's own LMB
compare/pan keeps the picture. The whole overlay is `WA_TransparentForMouseEvents`; only the controls take the mouse.

`set_state(state)` is the ONE paint pass: it walks the catalog, applies `visible`/`enabled`/`checked` with signals
blocked (a state write must never look like a click), updates the three readouts and hands the timeline its numbers.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QRect, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QHBoxLayout, QWidget

from sli_ui_toolkit.ui.managers.ui_scale import scaled_px
from sli_ui_toolkit.ui.widgets.atomic.surface_panel import SurfacePanel
from sli_ui_toolkit.widgets import Button, ButtonFamily, Text

from .actions import ACTIONS, ActionSpec, actions_for, build_button, checked_for, enabled_for, visible_for
from .state import PreviewCommand, PreviewState
from .timeline import PreviewTimeline, format_content_time

#: Design px: the bars' inset from the pane edge, their gap, and their heights.
BAR_MARGIN = 12
BAR_GAP = 8
TOP_HEIGHT = 32
BOTTOM_HEIGHT = 42
#: The pill and the two readouts keep a stable width, so a changing number cannot reflow the row.
TIME_MIN_WIDTH = 148
FRAME_MIN_WIDTH = 78


class PreviewControls(QWidget):
    """Both bars, built from the catalog and driven by `set_state`."""

    commandRequested = Signal(object, object)      # PreviewCommand, value

    def __init__(self, parent: QWidget, *, pane: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pane = pane if pane is not None else parent
        # The overlay itself and its empty areas must not eat the picture's mouse; the buttons and the slider are
        # children and are hit-tested first, so they keep their own presses.
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._state = PreviewState()
        self._buttons: dict[str, Button] = {}
        self._specs: dict[str, ActionSpec] = {spec.id: spec for spec in ACTIONS}
        self._hidden_top: list[ActionSpec] = []
        self._groups: dict[str, ButtonFamily] = {}

        self.top_bar = self._make_bar()
        self.bottom_bar = self._make_bar()
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
        bar.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        bar.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(scaled_px(BAR_GAP))
        bar._row = layout                     # noqa: SLF001 - the row is this module's private detail
        return bar

    def _add_button(self, layout: QHBoxLayout, spec: ActionSpec) -> Button:
        button = build_button(spec, parent=layout.parentWidget())
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
        layout.addWidget(button)
        return button

    def _make_toggle_handler(self, spec: ActionSpec):
        def handle(checked: bool) -> None:
            if not checked:
                return
            self.commandRequested.emit(spec.command, spec.value)
        return handle

    def _build_top(self) -> None:
        layout = self.top_bar._row                   # noqa: SLF001
        # The produced pill: the ONE tool with its own non-interactive surface (a bullet + one line of text).
        self.pill = SurfacePanel(self.top_bar, background_token="flyout.background", radius=scaled_px(10))
        pill_row = QHBoxLayout(self.pill)
        pill_row.setContentsMargins(scaled_px(8), 0, scaled_px(10), 0)
        pill_row.setSpacing(scaled_px(6))
        self.pill_dot = SurfacePanel(self.pill, background_token="accent", radius=scaled_px(4))
        self.pill_dot.setFixedSize(scaled_px(8), scaled_px(8))
        self.produced_text = Text("produced \u2014", variant="caption")
        self.produced_text.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        pill_row.addWidget(self.pill_dot, 0, Qt.AlignmentFlag.AlignVCenter)
        pill_row.addWidget(self.produced_text, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.pill)

        self._add_button(layout, self._specs["compare"])
        layout.addStretch(1)
        for spec_id in ("mode_single", "mode_sbs", "mode_split", "zoom_out", "zoom_fit", "zoom_one", "zoom_in",
                        "fullscreen"):
            self._add_button(layout, self._specs[spec_id])

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

        self.frame_text = Text("f \u2014", variant="caption", minimum_width=scaled_px(FRAME_MIN_WIDTH))
        self.frame_text.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        layout.addWidget(self.frame_text, 0, Qt.AlignmentFlag.AlignVCenter)

        self.timeline = PreviewTimeline(self.bottom_bar)
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

    def _top_specs(self) -> list[ActionSpec]:
        return list(actions_for("top"))

    def _apply_overflow(self, available: int) -> None:
        """Collapse the lowest-priority tools into the "…" button when the row cannot hold them at this scale."""
        if available <= 0:
            return
        for spec in self._top_specs():
            self._buttons[spec.id].setVisible(visible_for(spec, self._state))
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

        produced_seconds = max(0.0, float(state.produced))
        percent = int(round(max(0.0, min(1.0, state.progress_percent())) * 100))
        self.produced_text.setText(f"produced {produced_seconds:.1f} s \u00b7 {percent}%")
        self.pill.setVisible(state.duration > 0.0)
        self.time_text.setText(
            f"{format_content_time(state.playhead)} / {format_content_time(state.duration)}")
        self.frame_text.setText(state.frame_ratio())
        self.frame_text.setToolTip(
            f"output {int(state.output_frames)} frames (CFR)" if state.output_frames else "")
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
