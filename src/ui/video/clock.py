"""Media time for a player with no audio: a PTS clock, not a wall clock.

A file player normally lets the audio device own time and drops or repeats video frames to match it.
The preview has no audio at all (the plan's non-goal), so the clock is the other half of that
bargain: media time is defined here, monotonic, and freezable, and ``due_pts`` says which frame on
the stream's grid is the one to draw. Everything the presenter does then follows from one number -
there is no second notion of "where are we" in the widget.

Why ``due_pts`` rounds **down** while the plan's prose says "round": a frame is due once its own PTS
has been reached, so asking for the next frame's PTS one frame early would hand the surface a frame
the queue does not have yet (``pop_due`` would return ``None`` and the last picture would stay up for
an extra frame - the classic half-frame stutter). Rounding down answers "the newest frame that must
already exist", which is exactly what the queue can serve.

No Qt and no PyAV here, so pause/resume/drift is testable with plain numbers.
"""

from __future__ import annotations

import math


class PlaybackClock:
    """Media position from a monotonic clock, anchored by ``start``/``resume`` and frozen by ``pause``.

    ``now`` is the caller's monotonic time in seconds (``time.monotonic()``, or a synthetic number in
    a test). The clock never reads the wall clock itself: the presenter decides what "now" is, so a
    paused window, a suspended process or a test can hand in any timeline it likes.
    """

    #: Float slack for the frame-grid division: 0.08 / 0.04 must land on 2, not 1.9999999 -> 1.
    _EPSILON = 1e-9

    def __init__(self) -> None:
        self._running = False
        self._origin = 0.0
        self._anchor_now = 0.0
        self._anchor_pos = 0.0

    def start(self, now: float, origin_pts: float = 0.0) -> None:
        """Begin running from ``origin_pts`` media seconds at wall time ``now``.

        This is also how a seek moves the clock: the frame grid is anchored at the new origin, so
        ``due_pts`` keeps landing on the stream's own frame boundaries after a jump.
        """
        self._origin = float(origin_pts)
        self._anchor_pos = float(origin_pts)
        self._anchor_now = float(now)
        self._running = True

    def pause(self, now: float) -> None:
        """Freeze the media position; calling it while already paused changes nothing."""
        if not self._running:
            return
        self._anchor_pos = self.position(now)
        self._anchor_now = float(now)
        self._running = False

    def resume(self, now: float) -> None:
        """Continue from the frozen position; calling it while running changes nothing."""
        if self._running:
            return
        self._anchor_now = float(now)
        self._running = True

    def position(self, now: float) -> float:
        """Media seconds at wall time ``now``; constant while paused, 0 before the first ``start``."""
        if not self._running:
            return self._anchor_pos
        # A `now` behind the anchor is a caller mistake (or a test looking backwards); media time
        # does not run backwards, so the anchor holds rather than going negative.
        return self._anchor_pos + max(0.0, float(now) - self._anchor_now)

    def due_pts(self, now: float, frame_interval: float) -> float:
        """The PTS of the newest frame that must already be decoded at ``now``.

        ``frame_interval`` is one frame in seconds (``1 / fps``); the grid starts at the clock's
        origin, so a stream whose first frame is at 5.0 s still gets frames at 5.0, 5.04, ... .
        """
        interval = float(frame_interval)
        if not interval > 0.0:
            raise ValueError(f"frame_interval must be positive, got {frame_interval!r}")
        pos = self.position(now)
        steps = math.floor((pos - self._origin) / interval + self._EPSILON)
        return self._origin + steps * interval

    def is_running(self) -> bool:
        return self._running
