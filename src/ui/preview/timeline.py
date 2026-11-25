"""The scrubber: the toolkit `Slider` with the preview's own track.

The track is a timeline, not a progress bar, and it carries three facts at once:

* the whole clip (the unfilled groove),
* the band the renderer has actually produced (`toast.progress.fill`, the render head is its right edge),
* the played fill up to the playhead (the accent), with the thumb standing where the playhead is.

The widget is a `sli_ui_toolkit` `Slider`, so the track/thumb geometry, the keyboard edit mode (`Enter`) and the
theme repaint are the toolkit's; only the band painting is ours (`setTrackPainter`). A press or drag emits
`seekRequested(seconds)`; a refusal is the controller's answer, never this widget's (it keeps the thumb where the
state puts it).
"""

from __future__ import annotations

import math

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen

from sli_ui_toolkit.theme import ThemeManager
from sli_ui_toolkit.ui.widgets.atomic.slider import Slider

from .state import PreviewState

#: The toolbar's slider range: a 0..1000 integer track is finer than a millisecond at any realistic clip length.
TRACK_STEPS = 1000
#: Rounded ends and a 2 px marker keep the band's edge legible without competing with the thumb.
MARKER_WIDTH = 2

#: The colours the scrubber paints, by role. Nothing here is a literal RGBA; a missing token falls back neutral.
TIMELINE_TOKENS = {
    "track": "slider.track.unfilled",
    "produced": "toast.progress.fill",
    "played": "accent",
    "marker": "slider.track.background",
}


def format_content_time(seconds: float | None) -> str:
    """`m:ss.mmm` - the mockup's content clock (one-minute field, milliseconds always shown)."""
    if seconds is None:
        return "\u2014:\u2014.\u2014\u2014\u2014"
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return "\u2014:\u2014.\u2014\u2014\u2014"
    if not math.isfinite(value) or value < 0.0:
        return "\u2014:\u2014.\u2014\u2014\u2014"
    total_ms = int(round(value * 1000.0))
    minutes, rest = divmod(total_ms, 60_000)
    whole_seconds, milliseconds = divmod(rest, 1_000)
    return f"{minutes:d}:{whole_seconds:02d}.{milliseconds:03d}"


def _token(name: str, fallback: str) -> QColor:
    try:
        return QColor(ThemeManager.get_instance().get_color(TIMELINE_TOKENS[name]))
    except Exception:
        return QColor(fallback)


class PreviewTimeline(Slider):
    """The bottom bar's scrubber: produced band + render head + played fill, thumb as the playhead."""

    seekRequested = Signal(float)      # content seconds

    def __init__(self, parent=None) -> None:
        super().__init__(Qt.Orientation.Horizontal, parent, show_value_fill=False, wheel_requires_focus=True)
        self.setRange(0, TRACK_STEPS)
        self._duration = 0.0
        self._produced_fraction = 0.0
        #: The played FILL and the THUMB are two facts (see `PreviewState.played`): the fill follows the picture, the
        #: thumb stands on the host's playhead - which is the requested time while a press has no answer yet.
        self._played_fraction = 0.0
        self._playhead_fraction = 0.0
        self._scrubbing = False
        #: The last value the widget ASKED for: a press and a release at one point are one seek, not two.
        self._last_emitted: int | None = None
        self.setTrackPainter(self._paint_track)
        self.sliderPressed.connect(self._on_pressed)
        self.sliderReleased.connect(self._on_released)
        self.sliderMoved.connect(self._on_moved)

    # -- state -----------------------------------------------------------------------------------

    def set_state(self, state: PreviewState) -> None:
        """Adopt the state's numbers; a drag in flight owns the thumb until it is released."""
        self._duration = max(0.0, float(state.duration))
        self._produced_fraction = max(0.0, min(1.0, state.progress_percent()))
        self._played_fraction = state.played_fraction()
        self._playhead_fraction = state.playhead_fraction()
        self.setEnabled(state.has_picture() and self._duration > 0.0)
        if not self._scrubbing:
            self._set_silent_value(round(self._playhead_fraction * TRACK_STEPS))
        self.update()

    def produced_fraction(self) -> float:
        return self._produced_fraction

    def playhead_fraction(self) -> float:
        return self._playhead_fraction

    def played_fraction(self) -> float:
        """The fill's share: the PICTURE's position, which lags the thumb while a seek is in flight."""
        return self._played_fraction

    def is_scrubbing(self) -> bool:
        return self._scrubbing

    # -- interaction -----------------------------------------------------------------------------

    def _on_pressed(self) -> None:
        self._scrubbing = True
        # The press ALWAYS asks: a refused press must be re-askable at the same point.
        self._emit_value(force=True)

    def _on_moved(self, value: int) -> None:
        if self._scrubbing:
            self._emit_value(value)

    def _on_released(self) -> None:
        # The release is the commit (the controller refuses what it cannot show, and the next state puts the thumb
        # back). A click that never moved asked once, at the press: asking again here sent the same seek twice and the
        # host answered it twice (two decodes, and the second one re-armed the seek marker).
        self._scrubbing = False
        self._emit_value()

    def _emit_value(self, value: int | None = None, *, force: bool = False) -> None:
        if self._duration <= 0.0:
            return
        raw = self.value() if value is None else int(value)
        if not force and self._last_emitted is not None and int(raw) == self._last_emitted:
            return
        self._last_emitted = int(raw)
        seconds = max(0.0, min(1.0, raw / float(TRACK_STEPS))) * self._duration
        self.seekRequested.emit(float(seconds))

    def _set_silent_value(self, value: int) -> None:
        value = max(self.minimum(), min(self.maximum(), int(value)))
        if value == self.value():
            return
        self.blockSignals(True)
        try:
            self.setValue(value)
        finally:
            self.blockSignals(False)

    # -- painting --------------------------------------------------------------------------------

    def _paint_track(self, painter: QPainter, rect: QRectF) -> None:
        """Whole clip, produced band, played fill, render head - in that order, inside the groove's clip."""
        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(_token("track", "#3a3d44")))
        painter.drawRect(rect)

        produced = max(0.0, min(1.0, self._produced_fraction))
        if produced > 0.0:
            painter.setBrush(QBrush(_token("produced", "#2f6f4f")))
            painter.drawRect(QRectF(rect.left(), rect.top(), rect.width() * produced, rect.height()))

        # The fill cannot run past the produced band: nothing exists beyond the edge the band marks, and a frame the
        # display route decoded ahead of it is one picture, not a stretch of played timeline.
        played = max(0.0, min(1.0, self._played_fraction))
        if produced > 0.0:
            played = min(played, produced)
        if played > 0.0:
            painter.setBrush(QBrush(_token("played", "#0078d4")))
            painter.drawRect(QRectF(rect.left(), rect.top(), rect.width() * played, rect.height()))

        if produced > 0.0:
            # The render head is the produced band's right edge: a 2 px vertical rule reads as "the renderer is here"
            # without a second marker colour competing with the thumb.
            x = rect.left() + rect.width() * produced
            painter.setPen(QPen(_token("marker", "#1f232a"), MARKER_WIDTH))
            painter.drawLine(int(x), int(rect.top()), int(x), int(rect.bottom()))
        painter.restore()
