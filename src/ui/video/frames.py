"""One decoded frame, and the bounded queue that paces them onto the preview surface.

Two consumers feed the same surface with frames of very different timbre, and the queue policy is
where that difference lives:

* the **live** stream (a growing preview file, or the shared-memory thumbnail) produces frames at the
  renderer's pace, not at a frame rate. Queueing them would fall further behind every second, so a
  full queue drops its **oldest** frame: latest wins, and the picture on screen is never older than
  the newest frame the renderer produced;
* **playback** of a finished file produces frames on a PTS grid. A frame whose PTS is already in the
  past by the time the surface asks is not worth an upload, so ``pop_due`` consumes every due frame
  and hands back the newest: drop-late, in one place, with no widget involved.

``FrameRef`` deliberately carries a plain ``bytes``/``memoryview`` and no Qt type: the decoder fills
it on its own thread, the surface uploads from it on the GUI thread, and nothing in between needs a
``QImage`` (which would be a copy per frame - the thing the plan's resource budget forbids).

This module imports neither Qt nor PyAV, on purpose: the queue policy is the half whose bugs look
like "the preview stutters", and it has to be testable on an interpreter that has neither installed
(this repo's test interpreter has no ``av``). ``pack_frame`` needs a *decoded* frame, but only
through the tiny duck-typed surface PyAV offers (``to_ndarray``, ``time``, ``width``, ``height``),
so a stub object exercises it in the tests.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The formats a ``FrameRef`` can carry today: packed 8-bit RGB, or the three YUV420 planes in one
#: buffer (Y, then U, then V - the same layout the shared-memory preview channel uses).
FORMATS = ("rgb24", "yuv420p")


@dataclass(frozen=True)
class FrameRef:
    """One decoded frame, owned by whoever produced it.

    ``buffer`` is the whole picture: ``stride * height`` bytes for ``rgb24``, or
    ``stride * height + 2 * chroma_stride * chroma_height`` for ``yuv420p`` where the chroma planes
    are stacked after the luma plane (``(width + 1) // 2`` columns each).
    ``pts`` is container time in seconds; ``index`` is the frame's number in the stream.
    """

    buffer: object
    width: int
    height: int
    format: str
    stride: int
    pts: float
    index: int


def frame_pts(frame) -> float:
    """A decoded frame's presentation time in seconds.

    Containers are allowed to leave a timestamp out (a broken or synthetic stream); a frame with no
    PTS then sorts at 0 rather than crashing the pump. The one rule lives here so the file decoder
    and the tail reader cannot disagree about it.
    """
    pts = getattr(frame, "time", None)
    return float(pts) if pts is not None else 0.0


def pack_frame(frame, index: int, pixel_format: str = "rgb24",
               pts: float | None = None, size: tuple[int, int] | None = None) -> FrameRef:
    """A :class:`FrameRef` from one decoded PyAV-style frame, with no copy beyond the pack itself.

    ``rgb24`` asks PyAV for a packed RGB picture; ``yuv420p`` asks for the three planes and keeps
    them stacked, which is a third of the traffic per 1080p frame (3.1 MB against 8.3 MB) at the
    price of the conversion happening in the fragment shader. Which one wins is P0's decision; both
    exist here so the surface can be pointed at either without touching the decoder.

    ``pts`` overrides the frame's own timestamp; a caller uses it for a container that left the
    timestamp out (the caller knows the previous frame's time and the stream's interval).
    """
    if pixel_format not in FORMATS:
        raise ValueError(f"unknown frame format {pixel_format!r}, expected one of {FORMATS}")
    if size is not None:
        # Convert AND scale in one step, at the size the pane can actually show. Measured cost of not doing it: an
        # RGB24 picture of a 2880x2160 upscale is 24.9 MB per frame, and the pane draws about 504x378 of it - the
        # packing and the QImage in the pane moved 43x more bytes than the screen can use (owner, 2026-11-25: the
        # playback path should be the cheap one). Never upscales: a frame already smaller than `size` is left alone.
        target_w, target_h = max(1, int(size[0])), max(1, int(size[1]))
        if target_w < int(frame.width) or target_h < int(frame.height):
            frame = frame.reformat(width=target_w, height=target_h)
    width, height = int(frame.width), int(frame.height)
    pts = frame_pts(frame) if pts is None else float(pts)
    if pixel_format == "rgb24":
        plane = frame.to_ndarray(format="rgb24")
        return FrameRef(
            buffer=plane.tobytes(),
            width=width,
            height=height,
            format="rgb24",
            stride=width * 3,
            pts=pts,
            index=int(index),
        )

    # PyAV hands yuv420p back as one array: `height` luma rows, then the two chroma planes in half
    # resolution. Odd sizes are legal in YUV, so the chroma width is ceil(width / 2) and the rows
    # are sliced to it (PyAV pads the array to the luma line size).
    plane = frame.to_ndarray(format="yuv420p")
    chroma_w, chroma_h = (width + 1) // 2, (height + 1) // 2
    luma = plane[:height, :width].tobytes()
    u = plane[height:height + chroma_h, :chroma_w].tobytes()
    v = plane[height + chroma_h:height + 2 * chroma_h, :chroma_w].tobytes()
    return FrameRef(
        buffer=luma + u + v,
        width=width,
        height=height,
        format="yuv420p",
        stride=width,
        pts=pts,
        index=int(index),
    )


class FrameQueue:
    """A fixed-depth queue where the newest frame wins and late frames are dropped on read."""

    def __init__(self, maxlen: int = 4) -> None:
        maxlen = int(maxlen)
        if maxlen < 1:
            raise ValueError(f"a frame queue needs room for at least one frame, got {maxlen}")
        self._maxlen = maxlen
        self._frames: list[FrameRef] = []

    @property
    def maxlen(self) -> int:
        return self._maxlen

    def push(self, frame: FrameRef) -> int:
        """Append a frame, dropping the oldest ones to stay within ``maxlen``.

        Returns how many frames were dropped: 0 in the normal case, 1 per step a producer is ahead
        of the consumer. Latest wins, because a preview that lags the renderer is worse than a
        preview that skips frames.
        """
        self._frames.append(frame)
        dropped = 0
        while len(self._frames) > self._maxlen:
            self._frames.pop(0)
            dropped += 1
        return dropped

    def pop_due(self, now: float) -> FrameRef | None:
        """The newest frame whose PTS has been reached, dropping every other due frame.

        ``now`` is media time in seconds (the PTS clock's position). Frames still in the future stay
        queued. ``None`` means "nothing to draw yet", not an error.
        """
        due: FrameRef | None = None
        for frame in self._frames:
            if frame.pts <= now and (due is None or frame.pts >= due.pts):
                due = frame
        if due is None:
            return None
        # Every frame at or before `now` is late now: the surface draws the newest of them and the
        # older ones must not be replayed when the clock stalls (the drop-late half of the policy).
        self._frames = [frame for frame in self._frames if frame.pts > now]
        return due

    def clear(self) -> None:
        """Forget every queued frame (a seek, a file change, a new render)."""
        self._frames.clear()

    def __len__(self) -> int:
        return len(self._frames)
