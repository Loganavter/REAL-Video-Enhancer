"""The preview control surface's state and commands.

One frozen state object is the single source of truth for every control: the transport, the view modes, the zoom, the
compare hold, the timeline bands and the frame statistics. Widgets never own any of it; they render a `PreviewState`
and send a `PreviewCommand`. The state has a `phase` and a `backend` kind instead of loose booleans, so "playing a live
tail at its produced edge" (BUFFERING) is a state, not a combination three handlers must agree on.

Frame statistics are TWO facts on purpose: `source_frames`/`frames_done` are the container's video packets (the reader
is passthrough; 493 on the owner's clip), `output_frames` is what the writer's CFR rebuild will emit (495, two of them
repeats across a PTS gap). They are never subtracted from each other - the `493/495` report (2026-11-25) was exactly
that subtraction.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class PreviewPhase(Enum):
    """IDLE = nothing to show; READY = a picture, not playing; PLAYING; BUFFERING = playing but at the produced edge
    of a file that is still being written; CLOSED = the pane was released."""
    IDLE = "idle"
    READY = "ready"
    PLAYING = "playing"
    BUFFERING = "buffering"
    CLOSED = "closed"


class PreviewBackendKind(Enum):
    """Who answers transport commands: a growing file's tail, a finished file's decoder, the export's display feed,
    or the Render Queue tab's own item.

    `QUEUE_FILE` is a finished file owned by the QUEUE, and its OWN length is the axis: the pane shows the queue item
    while its tab is current, so the bar's duration, band, reach and playhead must be the item's - not the render's
    numbers behind it (the defect this kind removes: the queue's bar showed the render's duration and produced band
    over the queue item's picture).
    """
    NONE = "none"
    LIVE_TAIL = "live_tail"
    FINISHED_FILE = "finished_file"
    DISPLAY_FEED = "display_feed"
    QUEUE_FILE = "queue_file"


class PreviewCommand(Enum):
    PLAY_PAUSE = "play_pause"
    STEP_BACK = "step_back"
    STEP_FORWARD = "step_forward"
    SEEK_TO = "seek_to"                    # value: seconds (content time)
    SEEK_FRACTION = "seek_fraction"        # value: 0..1 of the duration
    SET_MODE = "set_mode"                  # value: "single" | "sbs" | "split"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    ZOOM_FIT = "zoom_fit"
    ZOOM_ONE = "zoom_one"
    COMPARE_DOWN = "compare_down"
    COMPARE_UP = "compare_up"
    TOGGLE_FULLSCREEN = "toggle_fullscreen"


VIEW_MODES = ("single", "sbs", "split")
MIN_ZOOM = 0.02
MAX_ZOOM = 8.0


@dataclass(frozen=True)
class PreviewState:
    phase: PreviewPhase = PreviewPhase.IDLE
    backend: PreviewBackendKind = PreviewBackendKind.NONE
    mode: str = "single"
    zoom_fit: bool = True
    zoom: float = 1.0
    compare: bool = False
    playhead: float = 0.0
    #: Where the PICTURE is: the media time of the frame on screen. The playhead is where the host stands (the requested
    #: time while a press is unanswered); this one lags while the decoder catches up, and it is what the scrubber paints
    #: as played. `None` means "unknown", and the painter then falls back to the playhead.
    played: float | None = None
    #: WHY the fill and the thumb can differ right now, in the page's words: `"press-open"` (a press is unanswered, so
    #: the thumb is the request and the fill is the picture), `"shown"` (the fill IS the picture and they agree), or
    #: `"produced-edge"` (the fill was clamped to the band). Empty when nothing can say - the report then only states
    #: the numbers. Debug, not behaviour: nothing paints from this.
    fill_source: str = ""
    produced: float = 0.0
    duration: float = 0.0
    source_frames: int = 0
    frames_done: int = 0
    output_frames: int = 0
    fps: float = 0.0
    seek_ceiling: float | None = None
    fullscreen: bool = False

    def with_(self, **fields) -> "PreviewState":
        """A copy with the named fields replaced (frozen state; the controller is the only updater)."""
        allowed = set(self.__dataclass_fields__)
        unknown = set(fields) - allowed
        if unknown:
            raise KeyError(f"unknown PreviewState field(s): {', '.join(sorted(unknown))}")
        return replace(self, **fields)

    def frame_interval(self) -> float:
        return 1.0 / self.fps if self.fps > 0 else 1.0 / 24.0

    def has_picture(self) -> bool:
        return self.phase in (PreviewPhase.READY, PreviewPhase.PLAYING, PreviewPhase.BUFFERING)

    def playing(self) -> bool:
        return self.phase in (PreviewPhase.PLAYING, PreviewPhase.BUFFERING)

    def seekable_ceiling(self) -> float:
        """The furthest content time a press may ask for: the host's frame-on-demand ceiling, else the produced edge."""
        if self.seek_ceiling is not None:
            return max(0.0, self.seek_ceiling)
        return max(0.0, self.produced)

    def progress_percent(self) -> float:
        """The produced share, 0..1 (a growing file's band; a finished file reports the whole timeline)."""
        if self.duration <= 0.0:
            return 0.0
        return max(0.0, min(1.0, self.produced / self.duration))

    def playhead_fraction(self) -> float:
        """Where the playhead stands, 0..1 of the whole timeline (the scrubber's thumb)."""
        if self.duration <= 0.0:
            return 0.0
        return max(0.0, min(1.0, self.playhead / self.duration))

    def played_fraction(self) -> float:
        """How much of the timeline the PICTURE covers, 0..1 - the scrubber's played fill.

        Deliberately not the playhead: a press moves the thumb to the requested time at once, while the picture still
        shows the old frame for a few frames. Painting both from one number made a seek invisible until it landed (and
        made the fill jump with every stale frame); the two are separate facts.
        """
        if self.duration <= 0.0:
            return 0.0
        value = self.playhead if self.played is None else self.played
        return max(0.0, min(1.0, float(value) / self.duration))

    def finished(self) -> bool:
        """Whether the whole clip exists: everything is produced, so the band is the full track.

        Asked separately from `progress_percent()`, because "100 %" and "finished" are different claims: a live render
        at 100 % still has a moving band with a picture behind it, while a finished file has one band and the picture is
        wherever the decoder stands. The scrubber's report names this so a screenshot can be read without guessing.
        """
        return self.duration > 0.0 and self.produced >= self.duration - 1e-6

    def at_produced_edge(self, tolerance: float | None = None) -> bool:
        """Whether the playhead has reached the furthest moment this backend can show.

        A growing file's edge is the produced band; a finished file's edge is its duration. `tolerance` defaults to
        half a frame, so a player that lands on the last produced frame is at the edge instead of a hair before it.
        """
        span = self.frame_interval() / 2.0 if tolerance is None else max(0.0, float(tolerance))
        return self.playhead >= self.seekable_ceiling() - span

    def is_growing(self) -> bool:
        """A live tail over a file the renderer is still writing (the one backend with a real produced edge)."""
        return self.backend is PreviewBackendKind.LIVE_TAIL

    def frame_ratio(self) -> str:
        """`f <source frames processed> / <source frames>`.

        BOTH numbers are the source packet set (`frames_done`/`source_frames`). The writer's CFR count is a separate
        fact (`output_frames`) and is never a denominator here: subtracting the two populations is exactly the
        `493/495` report of 2026-11-25.
        """
        total = max(0, int(self.source_frames))
        done = max(0, min(total, int(self.frames_done))) if total else max(0, int(self.frames_done))
        if total <= 0:
            return "f \u2014"
        return f"f {done} / {total}"
