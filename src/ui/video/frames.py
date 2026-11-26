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

The YUV formats carry the three planes twice over: ``buffer`` keeps the historical STACKED layout
(Y, then U, then V) that the queue/legacy consumers read, and ``planes``/``strides`` name each plane
separately - which is what the GPU path uploads, one texture per plane, with no untangling and no
``to_ndarray``/``tobytes`` in the middle when the decoder's rows are already packed.

This module imports neither Qt nor PyAV, on purpose: the queue policy is the half whose bugs look
like "the preview stutters", and it has to be testable on an interpreter that has neither installed
(this repo's test interpreter has no ``av``). ``pack_frame``/``pack_planes`` need a *decoded* frame,
but only through the duck-typed surface PyAV offers (``planes`` with ``line_size``, or ``to_ndarray``,
plus ``time``, ``width``, ``height``), so a stub object exercises them in the tests.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The formats a ``FrameRef`` can carry today: packed 8-bit RGB, or the three YUV420 planes - 8-bit
#: (``yuv420p``, 1 byte/sample) and 10-bit little-endian (``yuv420p10le``, 2 bytes/sample, the code
#: value in the low 10 bits - exactly what PyAV hands over and what the writer emits for 10-bit).
FORMATS = ("rgb24", "yuv420p", "yuv420p10le")

#: Bytes per sample of the planar formats. A 10-bit sample occupies 16 bits; the shader divides by
#: the code maximum (1023), not by 65535, which is the one arithmetic detail a plane upload gets wrong
#: by default (an R16 texture samples ``code / 65535``).
SAMPLE_BYTES = {"yuv420p": 1, "yuv420p10le": 2}

#: The planar formats, as a tuple, for callers that branch on "is this a planes frame".
PLANAR_FORMATS = ("yuv420p", "yuv420p10le")


@dataclass(frozen=True)
class FrameRef:
    """One decoded frame, owned by whoever produced it.

    ``buffer`` is the whole picture: ``stride * height`` bytes for ``rgb24``, or
    ``stride * height + 2 * chroma_stride * chroma_height`` for the planar formats where the chroma
    planes are stacked after the luma plane (``(width + 1) // 2`` columns each).
    ``pts`` is container time in seconds; ``index`` is the frame's number in the stream.

    For the planar formats ``planes``/``strides`` name the three planes as the GPU uploads them
    (``Y`` full size, ``U``/``V`` half size in both axes) and ``stride`` is the LUMA ROW LENGTH IN
    BYTES (``width`` for 8-bit, ``width * 2`` for 10-bit). ``color_space``/``color_range`` carry the
    stream's own tags when the producer could read them (``"bt709"``/``"tv"``) - the renderer's
    matrix comes from these, never from a hard-coded BT.709.
    """

    buffer: object
    width: int
    height: int
    format: str
    stride: int
    pts: float
    index: int
    planes: tuple[object, ...] | None = None
    strides: tuple[int, ...] | None = None
    color_space: str | None = None
    color_range: str | None = None


def plane_geometry(width: int, height: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """The ``(luma, chroma)`` plane sizes of a 4:2:0 picture. Odd sizes are legal in YUV."""
    width, height = int(width), int(height)
    return (width, height), ((width + 1) // 2, (height + 1) // 2)


def frame_pts(frame) -> float:
    """A decoded frame's presentation time in seconds.

    Containers are allowed to leave a timestamp out (a broken or synthetic stream); a frame with no
    PTS then sorts at 0 rather than crashing the pump. The one rule lives here so the file decoder
    and the tail reader cannot disagree about it.
    """
    pts = getattr(frame, "time", None)
    return float(pts) if pts is not None else 0.0


def pack_frame(frame, index: int, pixel_format: str = "rgb24",
               pts: float | None = None, size: tuple[int, int] | None = None,
               color_space: str | None = None, color_range: str | None = None) -> FrameRef:
    """A :class:`FrameRef` from one decoded PyAV-style frame, with no copy beyond the pack itself.

    ``rgb24`` asks PyAV for a packed RGB picture; the planar formats ask for the three planes instead,
    which is 1.5 bytes/pixel (8-bit) or 3 (10-bit) against RGB's 4 at the same size, at the price of
    the colour conversion happening in the fragment shader. Scaling stays in swscale (``size``), the
    matrix does not: a planar request never makes the scaler convert colour, only resample.

    ``pts`` overrides the frame's own timestamp; a caller uses it for a container that left the
    timestamp out (the caller knows the previous frame's time and the stream's interval).
    ``color_space``/``color_range`` are the stream's own tags; the renderer's matrix is built from
    them (``None`` means "the producer could not read them" and the renderer falls back to bt709/tv,
    which is what this app's writer emits).
    """
    if pixel_format not in FORMATS:
        raise ValueError(f"unknown frame format {pixel_format!r}, expected one of {FORMATS}")
    if size is not None:
        # Convert AND scale in one step, at the size the pane can actually show. Measured cost of not doing it: an
        # RGB24 picture of a 2880x2160 upscale is 24.9 MB per frame, and the pane draws about 504x378 of it - the
        # packing and the QImage in the pane moved 43x more bytes than the screen can use (owner, 2026-11-25: the
        # playback path should be the cheap one). Never upscales: a frame already smaller than `size` is left alone.
        #
        # THE ASPECT IS KEPT, because `size` is the PANE's box, not the video's. Handing the box straight to
        # `reformat` squashed a 4:3 frame into the pane's 1260x896 shape (5.5 % vertically, measured live 2026-11-26),
        # and the pane then read that squashed size as the picture's aspect - so sbs drew both halves SMALLER during a
        # render (the result's 1260x896 became a 630x448 box and the 4:3 source letterboxed into it as 597x448) and
        # full-size after it (the finished file's frames are 4:3: 630x474). The owner's words: "во время рендера
        # почему-то оба видео меньше, а после - целиком заполняют необходимое пространство". `decode.py`'s fit and the
        # backend's `DisplayStager.fit_box` already work this way - this was the one copy that stretched.
        target_w, target_h = max(1, int(size[0])), max(1, int(size[1]))
        source_w, source_h = int(frame.width), int(frame.height)
        scale = (min(target_w / float(source_w), target_h / float(source_h))
                 if source_w > 0 and source_h > 0 else 1.0)
        if scale < 1.0:
            # Even sides: the planar formats' chroma planes are half-size, and an odd RGB row is pointless geometry.
            # `round`, not `int`: 504/1440 is 0.35 exactly, but 1440*0.35 is 503.99999999999994, and truncating it
            # cost the fit a whole pixel per side (`& ~1` then made it two).
            fit_w = min(target_w, max(2, int(round(source_w * scale)))) & ~1
            fit_h = min(target_h, max(2, int(round(source_h * scale)))) & ~1
            if (fit_w, fit_h) != (source_w, source_h):
                frame = frame.reformat(width=fit_w, height=fit_h)
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

    planes, strides = pack_planes(frame, pixel_format)
    # The STACKED buffer stays as it was: the queue pane and the display channel read one flat picture,
    # and the renderer reads `planes` - so a caller that only wants bytes keeps working unchanged.
    stacked = b"".join(bytes(plane) for plane in planes)
    return FrameRef(
        buffer=stacked,
        width=width,
        height=height,
        format=pixel_format,
        stride=strides[0],
        pts=pts,
        index=int(index),
        planes=planes,
        strides=strides,
        color_space=color_space,
        color_range=color_range,
    )


def pack_planes(frame, pixel_format: str = "yuv420p10le") -> tuple[tuple[object, ...], tuple[int, ...]]:
    """The three plane views and byte strides of a decoded frame, at the frame's own size.

    ``(Y, U, V)`` with ``strides`` = the row length in BYTES of each plane. PyAV's own ``planes`` are
    used when the frame already IS the requested format: a packed plane is then a ``memoryview`` over
    the decoder's buffer (no copy at all), a padded one is repacked row by row (the only copy left,
    and swscale pads rarely). A frame that is not the requested format is ``reformat``-ed first - a
    pixel-format conversion inside the YUV family, which is a bit-depth shift, not a colour matrix.

    A frame without ``planes`` (the duck-typed stub in the tests, and any object that only offers
    ``to_ndarray``) takes the array path: PyAV's planar array is ``height`` luma rows then the two
    chroma planes, and each plane is packed to its natural stride.
    """
    if pixel_format not in PLANAR_FORMATS:
        raise ValueError(f"pack_planes needs one of {PLANAR_FORMATS}, got {pixel_format!r}")
    sample_bytes = SAMPLE_BYTES[pixel_format]
    if _frame_format_name(frame) not in (None, pixel_format):
        frame = frame.reformat(format=pixel_format)
    width, height = int(frame.width), int(frame.height)
    _, chroma = plane_geometry(width, height)
    sizes = ((width, height), chroma, chroma)

    direct = _decoder_planes(frame, sizes, sample_bytes)
    if direct is not None:
        return direct
    return _array_planes(frame, pixel_format, sizes, sample_bytes)


def _frame_format_name(frame) -> str | None:
    """The decoded frame's pixel-format name (``"yuv420p10le"``), or None when the object has none."""
    name = getattr(getattr(frame, "format", None), "name", None)
    return name if isinstance(name, str) else None


def _decoder_planes(frame, sizes, sample_bytes):
    """Plane views over PyAV's own buffers, or None when this object has no ``planes``."""
    planes = getattr(frame, "planes", None)
    if planes is None:
        return None
    try:
        selected = list(planes)[:3]
        if len(selected) < 3:
            return None
        views: list[object] = []
        strides: list[int] = []
        for plane, (plane_w, plane_h) in zip(selected, sizes):
            row_bytes = int(plane_w) * sample_bytes
            view = memoryview(plane)
            stride = int(getattr(plane, "line_size", row_bytes) or row_bytes)
            if stride == row_bytes:
                views.append(view[:row_bytes * plane_h])
            else:
                # swscale padded the rows: the upload wants a contiguous picture, so repack. The common
                # case (line_size == width * bytes) never reaches this branch.
                views.append(memoryview(b"".join(
                    bytes(view[row * stride:row * stride + row_bytes]) for row in range(plane_h))))
            strides.append(row_bytes)
        return tuple(views), tuple(strides)
    except Exception:
        return None


def _array_planes(frame, pixel_format, sizes, sample_bytes):
    """The ``to_ndarray`` path: PyAV's planar array, sliced to the three planes and packed."""
    plane = frame.to_ndarray(format=pixel_format)
    (width, height), (chroma_w, chroma_h), _ = sizes
    # Odd sizes are legal in YUV, so the chroma width is ceil(width / 2) and the rows are sliced to
    # it (PyAV pads the array to the luma line size).
    luma = plane[:height, :width]
    u = plane[height:height + chroma_h, :chroma_w]
    v = plane[height + chroma_h:height + 2 * chroma_h, :chroma_w]
    views = tuple(memoryview(part.tobytes()) for part in (luma, u, v))
    strides = (width * sample_bytes, chroma_w * sample_bytes, chroma_w * sample_bytes)
    return views, strides


#: ``AVColorSpace`` / ``AVColorRange`` values PyAV may hand back as ints (the enums are not always
#: exposed as strings). Only the ones this app writes or reads today, plus the HDR family the tags
#: can name; anything unknown falls back to None and the renderer's default.
_COLOR_SPACES = {1: "bt709", 5: "bt470bg", 6: "smpte170m", 9: "bt2020nc", 10: "bt2020c"}
_COLOR_RANGES = {1: "tv", 2: "pc"}


def stream_color_tags(stream) -> tuple[str | None, str | None]:
    """The stream's ``(color_space, color_range)`` tags, or ``(None, None)`` when they are absent.

    PyAV's own attribute names for these have moved between releases, so every read is a ``getattr``
    on the stream and on its codec context, with the int enums mapped by hand. Absent tags are a real
    answer: a file written without ``-colorspace``/``-color_range`` has no declaration, and inventing
    one here would make the renderer's matrix look authoritative when nothing said so.
    """
    for holder in (stream, getattr(stream, "codec_context", None)):
        if holder is None:
            continue
        space = getattr(holder, "color_space", None)
        rng = getattr(holder, "color_range", None)
        if space is not None or rng is not None:
            return _tag_name(space, _COLOR_SPACES), _tag_name(rng, _COLOR_RANGES)
    return None, None


def _tag_name(value, table: dict[int, str]) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return table.get(int(value))
    except (TypeError, ValueError):
        return None


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
