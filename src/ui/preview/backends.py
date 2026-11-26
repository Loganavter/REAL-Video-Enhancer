"""The playback strategies behind the preview transport.

The shape is a `PlaybackBackend`: whoever answers play/pause/seek/step for the pane at this moment. Three objects can
own the pane and each has its own protocol already - the live tail of a file still being written (`LiveTailThread`), the
finished file's decoder (`VideoDecoder`), and the export's display feed (one frame on demand through the shared block).
This module wraps those three as strategies so the controller can select one instead of branching inside a handler,
and so a fourth kind (no picture at all) refuses instead of crashing. The queue tab's own item is a fourth clock
(`QueueFileBackend`, plan P3/P5): a finished file whose decoder the QUEUE owns, so its own length and playhead are the
pane's axis there while the render behind it keeps its position for the hand-back.

The rules this module keeps, unchanged from the handlers they replace:

* **Play is the page's one entry** (`ProcessTab._on_preview_pause_toggled`): it is what starts a tail for a job that
  has none, stops a playback tail, holds the feed's `wanted` byte and repaints the button. The adapters therefore call
  it rather than re-implementing the owner selection - this is the "do not rewrite the tail/decoder logic" line.
* **Seek is clamped ONCE, in `PreviewSession`.** The controller hands the session the reach and the session refuses a
  target past it before any owner is asked; the adapters carry no ceiling of their own (a second clamp here is exactly
  the drift plan P1 removes). An adapter converts a time to the page's share and answers False when the page cannot
  take the press - it never snaps the target somewhere else.
* **Position/produced/duration are readings**, not cached state: the tail publishes its own edge, the decoder its own
  duration, and the page knows the clip's length.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .state import PreviewBackendKind


@runtime_checkable
class PlaybackBackend(Protocol):
    """What the controller needs from whoever owns the pane's clock."""

    kind: PreviewBackendKind

    def play(self) -> None: ...

    def pause(self) -> None: ...

    def seek_to(self, seconds: float) -> bool: ...

    def step(self, frames: int) -> bool: ...

    @property
    def position(self) -> float: ...

    @property
    def produced(self) -> float: ...

    @property
    def duration(self) -> float: ...

    @property
    def is_playing(self) -> bool: ...


class NoBackend:
    """Nothing can move the picture: every transport command is refused, queries answer zero."""

    kind = PreviewBackendKind.NONE

    def play(self) -> None:
        pass

    def pause(self) -> None:
        pass

    def seek_to(self, seconds: float) -> bool:
        return False

    def step(self, frames: int) -> bool:
        return False

    @property
    def position(self) -> float:
        return 0.0

    @property
    def produced(self) -> float:
        return 0.0

    @property
    def duration(self) -> float:
        return 0.0

    @property
    def is_playing(self) -> bool:
        return False


class _HostBackend:
    """Shared plumbing: play/pause go through the page's single transport entry.

    The page's handler is the only thing that knows which owner paints the pane (a playback tail vs a preview tail, a
    finished decoder vs a queue decoder, the feed's `wanted` byte). Calling it is delegation, not duplication.
    """

    kind = PreviewBackendKind.NONE

    def __init__(self, host: Any) -> None:
        self._host = host

    # -- play / pause ---------------------------------------------------------------------------

    def play(self) -> None:
        self._set_paused(False)

    def pause(self) -> None:
        self._set_paused(True)

    def _set_paused(self, paused: bool) -> None:
        setter = getattr(self._host, "_set_preview_paused", None)
        if callable(setter):
            setter(bool(paused))

    @property
    def is_playing(self) -> bool:
        return not bool(getattr(self._host, "_preview_paused", True))

    # -- readings -------------------------------------------------------------------------------

    @property
    def position(self) -> float:
        try:
            return max(0.0, float(getattr(self._host, "_preview_playhead", 0.0) or 0.0))
        except (TypeError, ValueError):
            return 0.0

    @property
    def duration(self) -> float:
        getter = getattr(self._host, "_clip_duration", None)
        if callable(getter):
            try:
                return max(0.0, float(getter() or 0.0))
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    @property
    def produced(self) -> float:
        duration = self.duration
        if duration <= 0.0:
            return 0.0
        getter = getattr(self._host, "_live_produced_fraction", None)
        if callable(getter):
            try:
                return max(0.0, min(1.0, float(getter() or 0.0))) * duration
            except (TypeError, ValueError):
                pass
        return duration

    def step(self, frames: int) -> bool:
        """Move by whole frames on the pane's own clock (the arrow keys' contract)."""
        if not frames:
            return False
        return self.seek_to(self.position + int(frames) * self._frame_interval())

    def seek_to(self, seconds: float) -> bool:
        """Seek through the page's existing press route, so one press keeps one instrument.

        `_on_preview_seek` is the handler the strip has always used: it picks the route (queue / on-demand / tail /
        decoder / export display), records the press and pauses a held pane. Re-implementing that here would be the
        second seek semantics the plan forbids. The REACH is not decided here: `PreviewSession` has already refused a
        target past it, so this only maps seconds to the page's share of the clip.
        """
        duration = self.duration
        handler = getattr(self._host, "_on_preview_seek", None)
        if callable(handler) and duration > 0.0:
            try:
                handler(max(0.0, min(1.0, float(seconds) / duration)))
            except Exception:
                return False
            return True
        return self._direct_seek(seconds)

    def _direct_seek(self, seconds: float) -> bool:
        return False

    def _frame_interval(self) -> float:
        fps = 0.0
        options = getattr(self._host, "currentRenderOptions", None)
        try:
            fps = float(getattr(options, "videoFps", 0.0) or 0.0)
        except (TypeError, ValueError):
            fps = 0.0
        if fps <= 0.0:
            try:
                fps = float(getattr(self._host, "fps", 0) or 0)
            except (TypeError, ValueError):
                fps = 0.0
        return 1.0 / fps if fps > 0.0 else 1.0 / 24.0


class HostPlaybackBackend(_HostBackend):
    """No owner selected yet: the PAGE picks one on play (it may start a tail for this job).

    This is the "play, and the pane has a picture but no clock" case the old handler answered by starting a playback
    tail. Refusing it would be a regression dressed as a state machine; the strategy here only delegates.
    """

    kind = PreviewBackendKind.NONE


class LiveTailBackend(_HostBackend):
    """A file the renderer is still writing: the tail is the clock and only the produced band exists."""

    kind = PreviewBackendKind.LIVE_TAIL

    def __init__(self, host: Any, tail: Any) -> None:
        super().__init__(host)
        self._tail = tail

    # `seek_to` is NOT overridden: a live tail is exactly the case `_HostBackend.seek_to` describes - the page decides,
    # and it is the only place that can tell "inside the produced band" (the tail repositions) from "past it" (the
    # renderer's one on-demand frame, or a visible refusal). Seeking `self._tail` directly skipped the press ticket,
    # the landing watch and the on-demand route: a press past the edge then failed inside the tail and the marker
    # snapped back with no line saying why (found 2026-11-26 with the scrubber's own `scrub snapback` report).

    def _direct_seek(self, seconds: float) -> bool:
        if self._tail is None:
            return False
        target = max(0.0, float(seconds))
        try:
            self._tail.seek(target, attempt_past_edge=False)
        except TypeError:                  # a tail older than the keyword
            try:
                self._tail.seek(target)
            except Exception:
                return False
        except Exception:
            return False
        return True

    @property
    def produced(self) -> float:
        if self._tail is not None:
            try:
                edge = float(getattr(self._tail, "produced_seconds", 0.0) or 0.0)
            except (TypeError, ValueError):
                edge = 0.0
            if edge > 0.0:
                return edge
        return super().produced

    @property
    def is_playing(self) -> bool:
        if self._tail is not None and bool(getattr(self._tail, "paused", True)):
            return False
        return super().is_playing


class FinishedFileBackend(_HostBackend):
    """A finished file: its own decoder plays the whole timeline at the media rate."""

    kind = PreviewBackendKind.FINISHED_FILE

    def __init__(self, host: Any, decoder: Any) -> None:
        super().__init__(host)
        self._decoder = decoder

    # `seek_to` is not overridden (see `LiveTailBackend`): the page's press route is the only one that opens a press
    # ticket, and the open ticket is what keeps the marker on the REQUESTED time until the frame that answers it is
    # painted. Seeking the decoder directly left the marker on the old frame for ~150 ms after every press, i.e. the
    # thumb jumped back and forth on a finished file too (found in the owner's log, 2026-11-26).

    def _direct_seek(self, seconds: float) -> bool:
        if self._decoder is None:
            return False
        try:
            self._decoder.seek(max(0.0, float(seconds)))
        except Exception:
            return False
        return True

    @property
    def duration(self) -> float:
        if self._decoder is not None:
            try:
                value = float(getattr(self._decoder, "duration", 0.0) or 0.0)
            except (TypeError, ValueError):
                value = 0.0
            if value > 0.0:
                return value
        return super().duration

    @property
    def produced(self) -> float:
        return self.duration

    @property
    def is_playing(self) -> bool:
        is_playing = getattr(self._decoder, "is_playing", None)
        if callable(is_playing):
            try:
                return bool(is_playing())
            except Exception:
                pass
        return super().is_playing


class QueueFileBackend(_HostBackend):
    """The Render Queue tab's own item: a finished file the QUEUE owns, and its own decoder is the pane's clock.

    Why this adapter exists (plan P3's note, closed in P5): `owner_of` names the queue item as the pane's one owner and
    hands the queue's decoder as `Owner.clock`, but `backend_for` had no `QUEUE_FILE` branch - so the session bound the
    HOST adapter and a press was judged against the render's position (`_preview_playhead`), not the item's. The seek
    itself still goes through the page's own press route (`_on_preview_seek`, whose queue branch opens the ticket's
    route facts and asks the decoder), because that route is what knows the queue's geometry; what changes here is
    that the clock the session sees is the queue's, and its readings are the item's file.
    """

    kind = PreviewBackendKind.QUEUE_FILE

    def __init__(self, host: Any, decoder: Any) -> None:
        super().__init__(host)
        self._decoder = decoder

    # `seek_to` is inherited: the page's press route is the only one that seeks the queue decoder AND records the
    # press (`ticket.decide(route="queue")`, `session.watch(seconds)`), so a direct decoder seek here would be the
    # second seek semantics plan P1 removes.

    def _direct_seek(self, seconds: float) -> bool:
        if self._decoder is None:
            return False
        try:
            self._decoder.seek(max(0.0, float(seconds)))
        except Exception:
            return False
        return True

    @property
    def position(self) -> float:
        """The ITEM's own playhead: the pane's shown time, else the queue path's remembered position.

        Never `_preview_playhead`: that is the RENDER's position behind the queue tab, and the hand-back must still
        find the render standing where it stood.
        """
        getter = getattr(self._host, "_queue_pane_playhead", None)
        if callable(getter):
            try:
                return max(0.0, float(getter() or 0.0))
            except (TypeError, ValueError):
                return 0.0
        return super().position

    @property
    def duration(self) -> float:
        if self._decoder is not None:
            try:
                value = float(getattr(self._decoder, "duration", 0.0) or 0.0)
            except (TypeError, ValueError):
                value = 0.0
            if value > 0.0:
                return value
        return super().duration

    @property
    def produced(self) -> float:
        # A queue item is a finished file the queue owns: its whole length exists the moment it opens.
        return self.duration

    @property
    def is_playing(self) -> bool:
        is_playing = getattr(self._decoder, "is_playing", None)
        if callable(is_playing):
            try:
                return bool(is_playing())
            except Exception:
                pass
        return super().is_playing


class DisplayFeedBackend(_HostBackend):
    """The export's feed: it can render ANY timeline frame on demand, so the whole clip is seekable."""

    kind = PreviewBackendKind.DISPLAY_FEED

    # `seek_to` is not overridden: the page decides between "the renderer answers one frame" and "nothing can answer".

    def _direct_seek(self, seconds: float) -> bool:
        duration = self.duration
        if duration <= 0.0:
            return False
        fraction = max(0.0, min(1.0, float(seconds) / duration))
        request = getattr(self._host, "_request_display_seek", None)
        if not callable(request):
            return False
        try:
            return bool(request(fraction, hold=True))
        except Exception:
            return False

    @property
    def produced(self) -> float:
        return self.duration


def backend_for(host: Any, kind: PreviewBackendKind, *, tail: Any = None, decoder: Any = None,
                host_playable: bool = False) -> PlaybackBackend:
    """Build the adapter for `kind` from the page's current owners.

    The page calls this on every state push; the objects are small, so a fresh adapter keeps "which tail/decoder" and
    "which strategy" from ever disagreeing (a stale adapter would seek a closed decoder). `host_playable` is the
    "a picture exists but no owner yet" case: the page's own transport entry selects one on play.
    """
    if kind is PreviewBackendKind.LIVE_TAIL and tail is not None:
        return LiveTailBackend(host, tail)
    if kind is PreviewBackendKind.QUEUE_FILE and decoder is not None:
        return QueueFileBackend(host, decoder)
    if kind is PreviewBackendKind.FINISHED_FILE and decoder is not None:
        return FinishedFileBackend(host, decoder)
    if kind is PreviewBackendKind.DISPLAY_FEED:
        return DisplayFeedBackend(host)
    if host_playable:
        return HostPlaybackBackend(host)
    return NoBackend()
