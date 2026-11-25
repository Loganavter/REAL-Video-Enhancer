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
import time

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen

from sli_ui_toolkit.theme import ThemeManager
from sli_ui_toolkit.ui.widgets.atomic.slider import Slider

from ..video import telemetry as _telemetry
from .state import PreviewState

#: The toolbar's slider range: a 0..1000 integer track is finer than a millisecond at any realistic clip length.
TRACK_STEPS = 1000
#: Rounded ends and a 2 px marker keep the band's edge legible without competing with the thumb.
MARKER_WIDTH = 2

#: How many recent scrub actions the widget keeps for the control surface (`state:preview_timeline` -> `scrub`).
#: The preview's own telemetry is `preview-log.txt`; this is the same session's answer for THIS control, in memory, so a
#: diagnosis is one call instead of a screenshot argument ("the thumb did not move" vs "the press never landed").
RECENT_SCRUB_ACTIONS = 16

#: How long after asking a position a mismatch still counts as a refusal of THAT ask, seconds.
SNAP_BACK_WINDOW_S = 3.0

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
        #: (seconds, monotonic stamp) of the last ASK - what the host was asked to show.
        self._asked: tuple[float, float] | None = None
        #: The last value/fill this widget APPLIED from a state: what the user sees after this control's own state.
        self._applied_value: int | None = None
        self._applied_fill = 0.0
        self._applied_playhead = 0.0
        #: A small ring of what this control did, read back through the control surface.
        self._recent: list[dict] = []
        self.setTrackPainter(self._paint_track)
        self.sliderPressed.connect(self._on_pressed)
        self.sliderReleased.connect(self._on_released)
        self.sliderMoved.connect(self._on_moved)

    # -- state -----------------------------------------------------------------------------------

    def set_state(self, state: PreviewState) -> None:
        """Adopt the state's numbers; a drag in flight owns the thumb until it is released."""
        before_value = int(self.value())
        before_fill = float(self._played_fraction)
        self._duration = max(0.0, float(state.duration))
        self._produced_fraction = max(0.0, min(1.0, state.progress_percent()))
        self._played_fraction = state.played_fraction()
        self._playhead_fraction = state.playhead_fraction()
        self.setEnabled(state.has_picture() and self._duration > 0.0)
        if not self._scrubbing:
            self._set_silent_value(round(self._playhead_fraction * TRACK_STEPS))
        self.update()
        self._note_apply(state, before_value=before_value, before_fill=before_fill)

    def produced_fraction(self) -> float:
        return self._produced_fraction

    def playhead_fraction(self) -> float:
        return self._playhead_fraction

    def played_fraction(self) -> float:
        """The fill's share: the PICTURE's position, which lags the thumb while a seek is in flight."""
        return self._played_fraction

    def is_scrubbing(self) -> bool:
        return self._scrubbing

    # -- what this control did (the preview's `preview-log.txt`, one file for both) -----------------

    def scrub_report(self) -> dict:
        """The control surface's single answer: the two fractions, whether a drag is in flight, who asked what last,
        and the recent actions. Numbers, because a screenshot cannot tell "ignored" from "never delivered"."""
        asked_s = None if self._asked is None else float(self._asked[0])
        asked_age_ms = None if self._asked is None else (time.monotonic() - self._asked[1]) * 1000.0
        return {
            "duration": float(self._duration),
            "value": int(self.value()),
            "thumb": {"fraction": float(self._playhead_fraction), "seconds": self._playhead_fraction * self._duration},
            "fill": {"fraction": float(self._played_fraction), "seconds": self._played_fraction * self._duration},
            "produced": {"fraction": float(self._produced_fraction), "seconds": self._produced_fraction * self._duration},
            "scrubbing": bool(self._scrubbing),
            "asked": None if asked_s is None else {"seconds": asked_s, "age_ms": round(asked_age_ms or 0.0, 1)},
            "recent": [{key: value for key, value in entry.items() if key != "_t"} for entry in self._recent],
        }

    def _note(self, kind: str, **fields) -> None:
        """Record one action twice: the preview's own file (greppable, beside the preview's lines) and the widget's
        ring, which `state:preview_timeline` reads back."""
        clean = {key: value for key, value in fields.items() if value is not None}
        _telemetry.event(f"scrub {kind}", **clean)
        entry = {"kind": kind, "age_ms": 0.0}
        entry.update({key: (round(value, 4) if isinstance(value, float) else value) for key, value in clean.items()})
        now = time.monotonic()
        for old in self._recent:
            old["age_ms"] = round((now - float(old.get("_t", now))) * 1000.0, 1)
        entry["_t"] = now
        self._recent.append(entry)
        del self._recent[:-RECENT_SCRUB_ACTIONS]

    def _note_apply(self, state: PreviewState, *, before_value: int, before_fill: float) -> None:
        """Log what a STATE did to this control - only when it moved something.

        Two facts that used to be indistinguishable in a report: the fill/thumb the state asks for, and whether the
        thumb was allowed to follow (a drag in flight owns it). A press the host refuses shows up as `snapback`, naming
        the asked time, what the state showed instead, and the ceiling if the state carries one.
        """
        after_value = int(self.value())
        moved_thumb = bool(after_value != before_value)
        moved_fill = abs(float(self._played_fraction) - float(before_fill)) > 1e-6
        if not moved_thumb and not moved_fill:
            return
        self._applied_value = after_value
        self._applied_fill = float(self._played_fraction)
        self._applied_playhead = float(self._playhead_fraction)
        self._note("apply", value=after_value, thumb=round(self._playhead_fraction, 4),
                   fill=round(self._played_fraction, 4), produced=round(self._produced_fraction, 4),
                   scrubbing=int(bool(self._scrubbing)))
        if self._asked is None or self._scrubbing or not moved_thumb:
            return
        asked_s, asked_at = self._asked
        age = time.monotonic() - asked_at
        tolerance = max(0.05, state.frame_interval())
        shown = float(self._playhead_fraction) * float(self._duration)
        if age <= SNAP_BACK_WINDOW_S and abs(shown - asked_s) > tolerance:
            ceiling = state.seekable_ceiling()
            self._note("snapback", asked_s=round(float(asked_s), 3), shown_s=round(shown, 3),
                       age_ms=round(age * 1000.0, 1),
                       ceiling_s=None if ceiling is None else round(float(ceiling), 3),
                       produced_s=round(float(self._produced_fraction) * float(self._duration), 3),
                       why="ask-not-shown")

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
        self._asked = (float(seconds), time.monotonic())
        self._note("ask", value=int(raw), seconds=round(float(seconds), 3),
                   produced=round(float(self._produced_fraction), 4), scrubbing=int(bool(self._scrubbing)))
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
