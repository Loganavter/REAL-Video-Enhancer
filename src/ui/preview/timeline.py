"""The scrubber: the toolkit `Slider` with the preview's own track.

The track is a timeline, not a progress bar, and it carries three facts at once:

* the whole clip (the unfilled groove, the theme's dark track),
* the band the renderer has actually produced, in the theme's blue - ONE bar, no second fill, and the render head is
  its right edge,
* the thumb, standing where the playhead is.

Why one bar and not a "produced" tint plus a brighter "played" fill: the boundary between those two colours sat exactly
under the thumb, so every seek looked like the bar was split by the hand, and at produced=100 % the right half was a
dull tint (owner, 2026-11-26, two annotated screenshots: "сейчас так" = bright to the thumb then muted, "должно быть
так" = the same bright blue to the end). The played position is carried by the thumb and by the time readout; the
played FILL number survives in `scrub_report()` for diagnosis, it is simply not painted.

The widget is a `sli_ui_toolkit` `Slider`, so the track/thumb geometry, the keyboard edit mode (`Enter`) and the
theme repaint are the toolkit's; only the band painting is ours (`setTrackPainter`). A press or drag emits
`seekRequested(seconds)`; a refusal is the controller's answer, never this widget's (it never re-emits on a release).

The THUMB has three authorities, in this order: the pointer while a drag is in flight, the seek this control emitted
next (until the picture comes to it, the ask is past what the pane says it can show, or it stands unanswered past
`ASK_HOLD_WINDOW_S`), and the state's playhead last. The order is the fix for `rve-timeline-drag-glue-2026-11-27`: a
state pass used to be able to write the picture's old moment over a released drag (the thumb parked under the press
while the picture ran on), and a drag whose release never arrived owned the thumb for the rest of the session.
"""

from __future__ import annotations

import math
import time

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QGuiApplication, QPainter, QPen

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

#: How long an emitted ask may stand on the thumb when the state never agrees with it, seconds. Same window as the
#: snap-back report: an ask the pane will not show is NAMED there, it is not allowed to own the thumb for ever.
ASK_HOLD_WINDOW_S = SNAP_BACK_WINDOW_S

#: How many state passes a claimed drag survives with no mouse button down before it is treated as a lost release.
#: Two ticks (~200 ms at the page's 10 Hz): a real drag always carries the button through them, so this cannot cut a
#: gesture short, and a release that never arrived stops owning the thumb instead of pinning it for the session.
DRAG_LOST_TICKS = 2

#: The colours the scrubber paints, by role. Nothing here is a literal RGBA; a missing token falls back neutral.
#: `played` is the theme's ACTIVE blue (the mockup's `#4f8df0 -> #74adff`); `produced` is the same blue at
#: `PRODUCED_ALPHA` (the mockup's `rgba(91,157,255,.28)`); the head is `accent` at `MARKER_ALPHA` (the mockup's
#: `rgba(255,255,255,.5)`). `accent` must never be the PLAYED fill: this app's theme resolves it to white, which is
#: why the bar used to show a white "played" bar and a solid blue band - both roles swapped against the design.
TIMELINE_TOKENS = {
    # GREY is "not upscaled yet", blue is "produced": the pointer plays no part in either (owner, 2026-11-26). The
    # track token is the theme's own unfilled groove, not the darker background, so the unproduced part reads as grey
    # next to the blue instead of disappearing into the panel.
    "track": "slider.track.unfilled",
    "produced": "slider.track.filled",
    "played": "slider.track.filled",
    "marker": "accent",
}

#: The bar's opacity. 1.0 on purpose: the band IS the progress, and a tinted band put a second colour boundary under
#: the thumb (see the module docstring). Kept as a named constant so the decision is visible, not buried in a literal.
PRODUCED_ALPHA = 1.0


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
        #: `(track value, monotonic stamp, direction)` of the seek this control emitted and the state has not agreed
        #: with yet. THE OWNER OF THE THUMB after a release: the visible thumb stands on what the pane was asked for,
        #: not on whatever the last state pass says the picture is. Without it a drag that asked backward was undone
        #: by the next tick - the thumb went back to the picture's old moment and stayed there
        #: (`rve-timeline-drag-glue`). `direction` is +1 when the ask is AHEAD of the picture (playback will carry
        #: the picture onto it and past it) and -1 when it is BEHIND (the decoder has to come back to it).
        self._pending_ask: tuple[int, float, int] | None = None
        #: State passes since the last one that saw a mouse button down while a drag was claimed (`DRAG_LOST_TICKS`).
        self._drag_ticks_without_button = 0
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
        # The same fact from the other producer: a drag whose value moves without `sliderMoved` still asks (deduped).
        self.valueChanged.connect(self._on_value_changed)

    # -- state -----------------------------------------------------------------------------------

    def set_state(self, state: PreviewState) -> None:
        """Adopt the state's numbers; the thumb follows ONE authority at a time.

        The order is: the pointer (a drag in flight), then the last seek this control emitted (until the state agrees
        with it, or it is past what the pane can show, or it has stood unanswered past `ASK_HOLD_WINDOW_S`), then the
        state's playhead. It used to be only the last of those, with a `_scrubbing` exception - so the first pass after
        the release wrote the page's playhead into the widget, and while a press was unanswered that playhead was the
        REQUESTED time of the press (the page keeps the marker on the request), which is why the thumb parked under the
        press while the picture ran on.
        """
        before_value = int(self.value())
        before_fill = float(self._played_fraction)
        self._duration = max(0.0, float(state.duration))
        self._produced_fraction = max(0.0, min(1.0, state.progress_percent()))
        self._played_fraction = state.played_fraction()
        self._playhead_fraction = state.playhead_fraction()
        self._state_finished = bool(state.finished())
        self._state_fill_source = str(getattr(state, "fill_source", "") or "")
        self.setEnabled(state.has_picture() and self._duration > 0.0)
        self._reconcile_drag()
        if not self._scrubbing:
            self._set_silent_value(self._thumb_value(state))
        self.update()
        self._note_apply(state, before_value=before_value, before_fill=before_fill)

    def produced_fraction(self) -> float:
        return self._produced_fraction

    # -- who owns the thumb ------------------------------------------------------------------------

    def _thumb_value(self, state: PreviewState) -> int:
        """The track value the thumb stands on: the last ASK while it is outstanding, else the state's playhead.

        "Outstanding" ends in the three ways a seek can stop being a question: the picture reached the asked moment
        (within a tick's worth of frames, so the two are one number again and playback drives the thumb), the ask is
        past what the pane says it can show (a refusal - the snap-back report names it), or it has been outstanding
        for `ASK_HOLD_WINDOW_S` (nothing answered it either). A forward ask is also answered by the picture passing
        it: that is a seek that landed and kept playing.
        """
        ask = self._pending_ask
        if ask is None or self._duration <= 0.0:
            return round(self._playhead_fraction * TRACK_STEPS)
        value, stamp, direction = ask
        asked_s = float(value) / float(TRACK_STEPS) * self._duration
        shown_s = float(self._playhead_fraction) * self._duration
        # Three frames, not the session's two: the state arrives at 10 Hz, so a seek that landed perfectly can be a
        # tick old by the time the bar sees it (2.4 frames at 24 fps), and "the picture is at the ask" must not hinge
        # on which half of a tick it landed in.
        tolerance = max(0.05, 3.0 * state.frame_interval())
        if abs(shown_s - asked_s) <= tolerance:
            self._pending_ask = None            # landed: the state and the thumb are the same number from here on
            return round(self._playhead_fraction * TRACK_STEPS)
        if direction > 0 and shown_s > asked_s:
            # A forward ask the picture has already passed: playback answered it. (A backward ask is NOT closed by
            # the picture being ahead - that is the ordinary state before a backward seek lands.)
            self._pending_ask = None            # landed: the state and the thumb are the same number from here on
            return round(self._playhead_fraction * TRACK_STEPS)
        if asked_s > state.seekable_ceiling() + tolerance:
            self._pending_ask = None            # the pane cannot show it: the state wins, and `snapback` says so
            return round(self._playhead_fraction * TRACK_STEPS)
        if (time.monotonic() - float(stamp)) > ASK_HOLD_WINDOW_S:
            self._pending_ask = None            # nobody answered: bounded, so a dead ask cannot own the thumb
            return round(self._playhead_fraction * TRACK_STEPS)
        return int(value)

    def _reconcile_drag(self) -> None:
        """End a claimed drag whose release never arrived, and log it instead of silently owning the thumb.

        `_scrubbing` is set by one signal and cleared by another; a release lost between them (the pointer left the
        window, a grab was cancelled, an event went missing) left it true, and from then on no state could move the
        thumb - the permanent pin the owner saw. The pointer itself is the fact that cannot go missing: while a real
        drag is in flight Qt reports the left button down on every pass. `DRAG_LOST_TICKS` passes of "no button" is a
        drag nobody is holding, so it ends - and the last ask keeps the thumb where the finger left it.
        """
        if not self._scrubbing:
            self._drag_ticks_without_button = 0
            return
        buttons = QGuiApplication.mouseButtons()
        held = bool(buttons & Qt.MouseButton.LeftButton)
        self._drag_ticks_without_button = 0 if held else self._drag_ticks_without_button + 1
        if self._drag_ticks_without_button < DRAG_LOST_TICKS:
            return
        self._drag_ticks_without_button = 0
        self._scrubbing = False
        self._note("release-lost", value=int(self.value()))

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
            # `holding` is the thumb's authority, spelled out: True means the widget is standing the thumb on the
            # ask because the state has not agreed with it yet (a released drag, or a seek in flight) - the live proof
            # reads this instead of inferring "pinned" from a screenshot.
            "asked": None if asked_s is None else {
                "seconds": asked_s, "age_ms": round(asked_age_ms or 0.0, 1),
                "holding": bool(self._pending_ask is not None),
            },
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
        # ... and `glued` is a claim about ONE number being on screen: while a drag is in flight the pointer is moving
        # the thumb, so equal numbers are a coincidence, not a glue. It resumes when the drag ends and the picture has
        # come to the asked moment.
        glued = (not self._scrubbing) and abs(played - painted_thumb) <= tolerance
        if glued:
            why = None
        elif self._scrubbing:
            why = "scrubbing (the mouse owns the thumb)"
        else:
            why = self._state_fill_source or "unknown"
        return {
            "width_px": round(float(self.width()), 1),
            "one_bar": True,
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
        self._drag_ticks_without_button = 0
        # The press ALWAYS asks: a refused press must be re-askable at the same point.
        self._emit_value(force=True)

    def _on_moved(self, value: int) -> None:
        if self._scrubbing:
            self._emit_value(value)

    def _on_value_changed(self, value: int) -> None:
        """The second producer of "the pointer moved" - the toolkit's `setValue` during a drag.

        `sliderMoved` is how a drag *should* announce itself, but the base class emits it only while the toolkit's own
        pressed flag is up; a drag whose move reaches the value and not that flag moved the thumb and asked the host
        for nothing (in the live run the reported `value` never left the press). `_emit_value` dedupes on
        `_last_emitted`, so whichever of the two signals arrives first IS the ask and the second is a no-op.
        """
        if self._scrubbing:
            self._emit_value(value)

    def _on_released(self) -> None:
        # The release is the commit (the controller refuses what it cannot show, and the next state puts the thumb
        # back). A click that never moved asked once, at the press: asking again here sent the same seek twice and the
        # host answered it twice (two decodes, and the second one re-armed the seek marker).
        self._scrubbing = False
        self._drag_ticks_without_button = 0
        self._emit_value()

    def _emit_value(self, value: int | None = None, *, force: bool = False) -> None:
        if self._duration <= 0.0:
            return
        raw = self.value() if value is None else int(value)
        if not force and self._last_emitted is not None and int(raw) == self._last_emitted:
            return
        self._last_emitted = int(raw)
        seconds = max(0.0, min(1.0, raw / float(TRACK_STEPS))) * self._duration
        asked_at = time.monotonic()
        self._asked = (float(seconds), asked_at)
        # The thumb stands on this value until the state AGREES with it (`_thumb_value`): after a release the ask - not
        # the picture's last position - is what the user is looking at. The direction says which answer closes it: an
        # ask AHEAD of the picture is answered by the picture reaching or passing it (a forward seek while playing),
        # an ask BEHIND it only by the picture coming back (the decoder has to seek backwards).
        ahead = 1 if float(raw) / float(TRACK_STEPS) > float(self._played_fraction) else -1
        self._pending_ask = (int(raw), asked_at, ahead)
        # The press is the ONE moment the split is certain: the thumb goes to the asked position while the picture is
        # still where it was. Without the fill in this line the log showed a clean "apply" afterwards and nothing about
        # the 1.9 s in between (owner's log, 2026-11-26: press 15.3 %, landed after 1856 ms, no line said the picture
        # was still at 99.6 % while the thumb stood at 15.3 %).
        asked_fraction = max(0.0, min(1.0, raw / float(TRACK_STEPS)))
        fill_now = float(self._played_fraction)
        self._note("ask", value=int(raw), seconds=round(float(seconds), 3),
                   thumb=round(asked_fraction, 4), fill=round(fill_now, 4),
                   glued=int(abs(fill_now - asked_fraction) <= 1.0 / float(TRACK_STEPS)),
                   gap_s=round(abs(fill_now - asked_fraction) * float(self._duration), 3),
                   produced=round(float(self._produced_fraction), 4),
                   finished=int(bool(self._state_finished)), scrubbing=int(bool(self._scrubbing)))
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

        # NO second fill on purpose: the played fraction is a diagnostic number (it tells "the picture is behind the
        # request"), not a colour. Painting it put the only visible boundary of the bar under the thumb, which is the
        # seam the owner read as "someone glued the colours to the seeker".

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
