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
plus ``time``, ``width``, ``height``), so a stub object exercises them in the tests. numpy is
imported lazily inside the >8-bit RGB conversion, which is the one place that needs it.

The RGB24 request has TWO conversions and they are not the same arithmetic (2026-11-27): an 8-bit
source keeps swscale (``to_ndarray(format="rgb24")``), a 10-bit source does not - swscale's 10-bit
yuv2rgb without ``SWS_ACCURATE_RND`` truncates the depth reduction before the range expansion and
lands white on 253 and mid-grey on 126 where H.273 says 255 and 127.5 (measured against zimg, and
against ffmpeg's own scaler). PyAV exposes no ``sws_flags`` argument, so the >8-bit depth is
converted here, in the same H.273 arithmetic the fragment shader uses - range affine separated from
the matrix (``_RGB_MATRICES``/``_RANGE_CODES_10``).
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

#: The H.273 matrices, rows `(Y, Cb, Cr) -> R|G|B`, on luma already expanded to 0..1 and chroma
#: centred on 0 - the range is an affine on the codes, the matrix itself is range-free (ITU-T H.273),
#: which is exactly how the shader path splits it (`preview_renderer.yuv_uniform_floats`). These are
#: the same numbers that module carries: it imports THIS module (a frame is what the renderer
#: uploads), so the numbers cannot be imported the other way round, and the pin test
#: `test_the_ten_bit_tables_are_the_ones_the_shader_uses` fails the moment the two diverge.
_RGB_MATRICES = {
    "bt709": ((1.0, 0.0, 1.5748), (1.0, -0.187324, -0.468124), (1.0, 1.8556, 0.0)),
    "smpte170m": ((1.0, 0.0, 1.402), (1.0, -0.344136, -0.714136), (1.0, 1.772, 0.0)),
    "bt470bg": ((1.0, 0.0, 1.402), (1.0, -0.344136, -0.714136), (1.0, 1.772, 0.0)),
    "bt2020nc": ((1.0, 0.0, 1.4746), (1.0, -0.164553, -0.571353), (1.0, 1.8814, 0.0)),
}

#: The fallback when the producer could not read the tags: what this app's own writer emits.
_DEFAULT_COLOR_SPACE = "bt709"
_DEFAULT_COLOR_RANGE = "tv"

#: `((luma offset, luma span), (chroma offset, chroma span))` at 10 bits, H.273 table 8. Limited
#: ("tv") 10-bit is 64..940 luma and 64..960 chroma; full ("pc") is 0..1023 and 0..1023.
_RANGE_CODES_10 = {
    "tv": ((64.0, 876.0), (512.0, 896.0)),
    "pc": ((0.0, 1023.0), (512.0, 1023.0)),
}

#: The planar >8-bit source formats the CPU conversion handles, with their chroma subsampling
#: `(horizontal, vertical)`. A format that is not here keeps the scaler it used before: swscale is
#: still the only path for the 8-bit formats, and this fix must not touch those.
_TEN_BIT_LAYOUTS = {
    "yuv420p10le": (2, 2),
    "yuv422p10le": (2, 1),
    "yuv444p10le": (1, 1),
}


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

    ``rgb24`` asks for a packed RGB picture - through the scaler for an 8-bit frame, through the H.273
    arithmetic in :func:`rgb24_buffer` for a 10-bit one (the scaler's 10-bit yuv2rgb is ~2.5 levels
    dark); the planar formats ask for the three planes instead, which is 1.5 bytes/pixel (8-bit) or 3
    (10-bit) against RGB's 4 at the same size, at the price of the colour conversion happening in the
    fragment shader. Scaling stays in swscale (``size``), the matrix does not: a planar request never
    makes the scaler convert colour, only resample.

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
        return FrameRef(
            buffer=rgb24_buffer(frame, color_space=color_space, color_range=color_range),
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


def rgb24_buffer(frame, color_space: str | None = None,
                 color_range: str | None = None) -> bytes:
    """The RGB24 bytes of one decoded frame: H.273 arithmetic for >8-bit YUV, swscale below it.

    The depth split is the whole point of this function. An 8-bit frame keeps the scaler
    (`to_ndarray("rgb24")`), whose 16..235 codes are already right on this build's output; a 10-bit
    frame does NOT, because swscale's fast yuv2rgb reduces the depth before the range expansion and
    loses ~2.5 levels on the way (measured 2026-11-27 with libswscale, the library PyAV calls: tv
    white 940 -> 253, mid-grey 502 -> 126, against H.273's 255 and 127.5). `SWS_ACCURATE_RND` fixes
    that scaler, but PyAV exposes no `sws_flags`, so the >8-bit depth is converted here - the same
    range-affine-then-matrix arithmetic the fragment shader uses, from the same tables.

    A frame the conversion does not recognise (no format name, a packed 10-bit format, an odd plane
    layout) falls back to the scaler rather than dropping the picture: a slightly dark preview is
    better than a blank one, and the fallback is the behaviour every caller had before.
    """
    layout = _TEN_BIT_LAYOUTS.get(_frame_format_name(frame) or "")
    if layout is not None:
        try:
            return _ten_bit_rgb24(frame, layout, color_space, color_range)
        except Exception:
            pass
    return frame.to_ndarray(format="rgb24").tobytes()


def _ten_bit_rgb24(frame, layout, color_space, color_range) -> bytes:
    """Range-expand one 10-bit YUV picture and apply its matrix, in one pass per channel."""
    numpy = _numpy()
    hsub, vsub = layout
    width, height = int(frame.width), int(frame.height)
    chroma_w, chroma_h = (width + hsub - 1) // hsub, (height + vsub - 1) // vsub
    y, cb, cr = _ten_bit_planes(frame, ((width, height), (chroma_w, chroma_h)))
    (luma_offset, luma_span), (chroma_offset, chroma_span) = _RANGE_CODES_10[_range_name(color_range)]
    luma = (y.astype(numpy.float32) - luma_offset) / luma_span
    cb = _upsampled(numpy, (cb.astype(numpy.float32) - chroma_offset) / chroma_span, hsub, vsub)
    cr = _upsampled(numpy, (cr.astype(numpy.float32) - chroma_offset) / chroma_span, hsub, vsub)
    # The upsampler returns whole chroma samples, so an odd picture size is cropped back here.
    cb, cr = cb[:height, :width], cr[:height, :width]
    matrix = _RGB_MATRICES.get(str(color_space or _DEFAULT_COLOR_SPACE), _RGB_MATRICES[_DEFAULT_COLOR_SPACE])
    channels = []
    for row in matrix:
        channel = row[0] * luma + row[1] * cb + row[2] * cr
        numpy.clip(channel, 0.0, 1.0, out=channel)
        channel *= 255.0
        channel += 0.5                    # round half up, which is how H.273's 127.5 becomes 128
        channels.append(channel.astype(numpy.uint8))
    return numpy.stack(channels, axis=-1).tobytes()


def _ten_bit_planes(frame, sizes):
    """The three ``uint16`` planes of a planar 10-bit frame, from PyAV's own planes or `to_ndarray`.

    PyAV's planes are used when they are there (no copy when the rows are packed, one repack when
    swscale padded them), the stacked array path otherwise. Both give the code value as an integer in
    the low 10 bits, which is the layout this app's writer emits and the one the shader divides by
    1023 - the code must never be treated as a 16-bit sample.
    """
    numpy = _numpy()
    (width, height), (chroma_w, chroma_h) = sizes
    wanted = ((width, height), (chroma_w, chroma_h), (chroma_w, chroma_h))
    planes = getattr(frame, "planes", None)
    if planes is not None:
        selected = list(planes)[:3]
        if len(selected) == 3:
            return tuple(_plane_codes(numpy, plane, plane_w, plane_h)
                         for plane, (plane_w, plane_h) in zip(selected, wanted))
    array = frame.to_ndarray(format=_frame_format_name(frame) or "yuv420p10le")
    return (numpy.asarray(array[:height, :width]),
            numpy.asarray(array[height:height + chroma_h, :chroma_w]),
            numpy.asarray(array[height + chroma_h:height + 2 * chroma_h, :chroma_w]))


def _plane_codes(numpy, plane, plane_w: int, plane_h: int):
    """One PyAV plane as a ``(rows, cols)`` array of 10-bit codes, dropping swscale's row padding.

    ``line_size`` is the ROW length in bytes; a plane that reports something the buffer cannot hold
    is treated as tightly packed rather than trusted into a bad reshape.
    """
    row_bytes = int(plane_w) * 2
    view = memoryview(plane)
    stride = int(getattr(plane, "line_size", row_bytes) or row_bytes)
    if stride == row_bytes or len(view) < stride * plane_h:
        return numpy.frombuffer(view[:row_bytes * plane_h], dtype="<u2").reshape(plane_h, plane_w)
    rows = numpy.frombuffer(view, dtype=numpy.uint8)[:stride * plane_h].reshape(plane_h, stride)
    return numpy.ascontiguousarray(rows[:, :row_bytes]).view("<u2").reshape(plane_h, plane_w)


def _upsampled(numpy, plane, hsub: int, vsub: int):
    """A chroma plane at luma resolution: 2x triangle on each axis, edges replicated.

    The weights are `3/4` current and `1/4` neighbour, which is the phase swscale's yuv2rgb uses
    horizontally (checked against its own output on a one-sample chroma step); vertically the
    symmetric triangle measured closer to zimg on real frames than the top-shifted phase did, so both
    axes use it. Nearest replication is NOT good enough here: the pane's pixel gate measured it at
    3.37 % of pixels more than 8 levels off on detailed frames against 1.43 % for a filtered tap
    (2026-11-25), and the RGBA path is the one the compare's source half draws through.
    """
    if hsub == 2:
        height, width = plane.shape
        padded = numpy.empty((height, width + 2), numpy.float32)
        padded[:, 1:-1] = plane
        padded[:, 0], padded[:, -1] = plane[:, 0], plane[:, -1]
        wide = numpy.empty((height, 2 * width), numpy.float32)
        wide[:, 0::2] = 0.75 * plane + 0.25 * padded[:, :-2]
        wide[:, 1::2] = 0.75 * plane + 0.25 * padded[:, 2:]
        plane = wide
    if vsub == 2:
        height, width = plane.shape
        padded = numpy.empty((height + 2, width), numpy.float32)
        padded[1:-1] = plane
        padded[0], padded[-1] = plane[0], plane[-1]
        tall = numpy.empty((2 * height, width), numpy.float32)
        tall[0::2] = 0.75 * plane + 0.25 * padded[:-2]
        tall[1::2] = 0.75 * plane + 0.25 * padded[2:]
        plane = tall
    return plane


def _range_name(color_range: str | None) -> str:
    """``"pc"``/``"jpeg"``/``"full"`` select the full range, everything else the limited one.

    The same rule the renderer's uniform bundle applies to a stream's tags - one spelling, so a tag
    read by two callers cannot mean two things.
    """
    return "pc" if str(color_range or _DEFAULT_COLOR_RANGE).lower() in ("pc", "jpeg", "full") else "tv"


def _numpy():
    """numpy, imported where it is used.

    At module scope it would make the queue policy (which never touches pixels) depend on it, and
    this module is deliberately importable on a bare interpreter - the docstring's reason.
    """
    import numpy

    return numpy


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
