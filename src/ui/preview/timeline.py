"""The scrubber: the toolkit `Slider` with the preview's own track.

The track is a timeline, not a progress bar, and it carries three facts at once:

* the whole clip (the unfilled groove, the theme's dark track),
* the band the renderer has actually produced - the same blue as the played fill at `PRODUCED_ALPHA`, so a fully
  produced clip reads as "the whole bar is tinted" and the render head is its right edge,
* the played fill up to the playhead, in full-strength blue, with the thumb standing where the playhead is.

Colours follow the approved layout-E mockup (`~/Загрузки/rve-preview-mockups/shared.css`): `--track: #2a303c`,
`.produced { background: rgba(91,157,255,.28) }`, `.fill { #4f8df0 -> #74adff }`. The code used to paint the played
fill with `accent`, which this app's theme maps to WHITE, and the produced band in solid blue - i.e. both roles swapped
against the design (owner, 2026-11-26: "у меня produced - всё, если ты смотрел на комментарий").

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
#: The render head: the mockup's `.ceiling` is a 1 px line at `rgba(255,255,255,.5)`, standing above and below the
#: groove. A dark rule on the finished band was invisible, which made "produced" look like nothing at all.
MARKER_WIDTH = 1
MARKER_ALPHA = 0.5

#: How many recent scrub actions the widget keeps for the control surface (`state:preview_timeline` -> `scrub`).
#: The preview's own telemetry is `preview-log.txt`; this is the same session's answer for THIS control, in memory, so a
#: diagnosis is one call instead of a screenshot argument ("the thumb did not move" vs "the press never landed").
RECENT_SCRUB_ACTIONS = 16

#: How long after asking a position a mismatch still counts as a refusal of THAT ask, seconds.
SNAP_BACK_WINDOW_S = 3.0

#: The colours the scrubber paints, by role. Nothing here is a literal RGBA; a missing token falls back neutral.
#: `played` is the theme's ACTIVE blue (the mockup's `#4f8df0 -> #74adff`); `produced` is the same blue at
#: `PRODUCED_ALPHA` (the mockup's `rgba(91,157,255,.28)`); the head is `accent` at `MARKER_ALPHA` (the mockup's
#: `rgba(255,255,255,.5)`). `accent` must never be the PLAYED fill: this app's theme resolves it to white, which is
#: why the bar used to show a white "played" bar and a solid blue band - both roles swapped against the design.
TIMELINE_TOKENS = {
    "track": "slider.track.background",
    "produced": "slider.track.filled",
    "played": "slider.track.filled",
    "marker": "accent",
}

#: The produced band's opacity: the mockup's `.produced` is `rgba(91,157,255,.28)`. One colour, two roles - the band is
#: the same blue as the played fill, only muted, so "produced but not played" reads as a tint rather than a second fill.
PRODUCED_ALPHA = 0.28


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
        self._state_finished = False
        self._state_fill_source = ""
        #: A small ring of what this control did, read back through the control surface.
        self._recent: list[dict] = []
        #: What the last `_paint_track` actually drew, role by role: the answer to "where is what drawn" without a
        #: screenshot. Filled by the painter, read by `scrub_report()`.
        self._last_paint: list[dict] = []
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
        self._state_finished = bool(state.finished())
        self._state_fill_source = str(getattr(state, "fill_source", "") or "")
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
            "finished": bool(self._state_finished),
            "paint": self._paint_summary(),
            "recent": [{key: value for key, value in entry.items() if key != "_t"} for entry in self._recent],
        }

    def _paint_summary(self) -> dict:
        """Where each thing is drawn and from which number - the report a screenshot cannot give.

        Four rects in paint order (track, produced band, played fill, render head), each with its px span, its fraction,
        the colour and the ALPHA actually used, and the state number it came from. `glued` says whether the fill and the
        thumb are the same point, and `why_not_glued` carries the page's reason (`fill_source`) when they are not -
        which is the difference between "the picture lags the request" and "the two are literally one number".
        """
        played = float(self._played_fraction)
        # The thumb the user SEES is the widget's value, not the last state's playhead: while the mouse is down the
        # widget owns it and a state cannot move it. Comparing the fill to the state's playhead would report "glued"
        # for a picture that is visibly apart from the thumb under the pointer.
        painted_thumb = max(0.0, min(1.0, float(self.value()) / float(TRACK_STEPS)))
        tolerance = 1.0 / float(TRACK_STEPS)
        glued = abs(played - painted_thumb) <= tolerance
        if glued:
            why = None
        elif self._scrubbing:
            why = "scrubbing (the mouse owns the thumb)"
        else:
            why = self._state_fill_source or "unknown"
        return {
            "width_px": round(float(self.width()), 1),
            "duration": round(float(self._duration), 3),
            "finished": bool(self._state_finished),
            "rects": list(self._last_paint),
            "glued": bool(glued),
            "why_not_glued": why,
            "state": {
                "thumb_value": int(self.value()),
                "thumb_fraction": round(painted_thumb, 4),
                "state_playhead_fraction": round(float(self._playhead_fraction), 4),
                "fill_fraction": round(played, 4),
                "produced_fraction": round(float(self._produced_fraction), 4),
            },
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
        glued = abs(float(self._played_fraction) - float(self._playhead_fraction)) <= 1.0 / float(TRACK_STEPS)
        self._note("apply", value=after_value, thumb=round(self._playhead_fraction, 4),
                   fill=round(self._played_fraction, 4), produced=round(self._produced_fraction, 4),
                   scrubbing=int(bool(self._scrubbing)), finished=int(bool(self._state_finished)),
                   glued=int(glued))
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
        """Whole clip, produced band, played fill, render head - in that order, inside the groove's clip.

        Every rect is recorded in `_last_paint` as it is drawn (role, px span, fraction, colour, alpha, source), so
        `scrub_report()` can answer "where is what drawn, and from which number" instead of a screenshot being read by
        eye. The paint is the only place that knows what the user actually sees.
        """
        drawn: list[dict] = []

        def record(role: str, fraction: float, span: QRectF, colour: QColor, source: str) -> None:
            drawn.append({
                "role": role,
                "x": round(float(span.left()), 1),
                "w": round(float(span.width()), 1),
                "fraction": round(float(fraction), 4),
                "colour": colour.name(),
                "alpha": round(float(colour.alphaF()), 2),
                "from": source,
            })

        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        track = _token("track", "#3a3d44")
        painter.setBrush(QBrush(track))
        painter.drawRect(rect)
        record("track", 1.0, rect, track, "the whole clip")

        produced = max(0.0, min(1.0, self._produced_fraction))
        if produced > 0.0:
            # One blue, two roles: the band is the played colour at 28 %, exactly the mockup's `rgba(91,157,255,.28)`.
            # A fully produced clip therefore tints the WHOLE bar and the played fill is the bright part of it.
            band = _token("produced", "#0096ff")
            band.setAlphaF(PRODUCED_ALPHA)
            span = QRectF(rect.left(), rect.top(), rect.width() * produced, rect.height())
            painter.setBrush(QBrush(band))
            painter.drawRect(span)
            record("produced", produced, span, band, "produced / duration")

        # The fill cannot run past the produced band: nothing exists beyond the edge the band marks, and a frame the
        # display route decoded ahead of it is one picture, not a stretch of played timeline.
        played = max(0.0, min(1.0, self._played_fraction))
        clamped = bool(produced > 0.0 and played > produced)
        if produced > 0.0:
            played = min(played, produced)
        if played > 0.0:
            fill = _token("played", "#0078d4")
            span = QRectF(rect.left(), rect.top(), rect.width() * played, rect.height())
            painter.setBrush(QBrush(fill))
            painter.drawRect(span)
            record("played", played, span, fill,
                   "played / duration" + (" (clamped to the band)" if clamped else ""))

        if produced > 0.0:
            # The render head is the produced band's right edge: a 1 px light rule reads as "the renderer is here"
            # without a second marker colour competing with the thumb.
            x = rect.left() + rect.width() * produced
            head = _token("marker", "#ffffff")
            head.setAlphaF(MARKER_ALPHA)
            painter.setPen(QPen(head, MARKER_WIDTH))
            painter.drawLine(int(x), int(rect.top()), int(x), int(rect.bottom()))
            record("head", produced, QRectF(x, rect.top(), 1.0, rect.height()), head, "produced / duration")
        painter.restore()
        self._last_paint = drawn
