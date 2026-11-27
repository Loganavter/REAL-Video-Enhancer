"""The RGB24 request at 10 bits: the one conversion swscale gets wrong, pinned by numbers.

Why this file exists (2026-11-27). A 10-bit source that goes through the RGBA path - the compare's
source half, the queue thumbnails, the `rgba8` fallback - came out ~2.5 levels dark. Measured against
zimg (the independent H.273 reference) and against libswscale itself, the library PyAV calls:

    code 940 (tv white)   swscale 253   H.273 255
    code 502 (mid-grey)   swscale 126   H.273 127.5
    code  64 (tv black)   swscale   0   H.273 0        (endpoints survive, the span between them shrinks)

swscale's fast yuv2rgb truncates the 10-bit -> 8-bit reduction before the range expansion; its
`SWS_ACCURATE_RND` mode converts correctly, but PyAV exposes no `sws_flags`. So the >8-bit depth is
converted in `frames.rgb24_buffer` with the H.273 arithmetic the fragment shader already uses - and
this file pins the numbers that decision is worth: a synthetic ramp, against the textbook formula
typed out here rather than against the implementation (a second copy of a wrong matrix agrees with
itself).

The 8-bit path is deliberately NOT changed by that fix, and its test below says so.

No PyAV on this interpreter (see `frames`' docstring): the frame is a stub with PyAV's duck-typed
surface, and the stub REFUSES `to_ndarray("rgb24")` - which is what makes the numeric tests fail on
the old code instead of silently measuring the scaler.
"""

from __future__ import annotations

import pytest


class _PlaneStub(bytes):
    """The bit of a PyAV ``VideoPlane`` the packer reads: the buffer protocol and ``line_size``."""

    def __new__(cls, data: bytes, line_size: int | None = None):
        self = super().__new__(cls, data)
        self.line_size = len(data) if line_size is None else int(line_size)
        return self


class _TenBitFrame:
    """A decoded 10-bit frame with PyAV's own plane views, and a scaler that would be a bug to use."""

    def __init__(self, width, height, y, cb, cr, format_name: str = "yuv420p10le", planes: bool = True):
        self.width, self.height, self.time = int(width), int(height), 0.0
        self.format = type("_Format", (), {"name": format_name})()
        self.planes = None if not planes else [
            _PlaneStub(y.tobytes(), width * 2),
            _PlaneStub(cb.tobytes(), max(1, (width + 1) // 2) * 2),
            _PlaneStub(cr.tobytes(), max(1, (width + 1) // 2) * 2)]
        self._stacked = {format_name: _stack(y, cb, cr)}
        self.requested: list[str] = []

    def to_ndarray(self, format):  # noqa: A002 - PyAV's keyword
        self.requested.append(format)
        if format == "rgb24":
            raise AssertionError("a 10-bit frame must not go through swscale: its yuv2rgb is 2.5 levels dark")
        return self._stacked[format]

    def reformat(self, **_kwargs):  # pragma: no cover - pack_frame only scales when `size=` is given
        raise AssertionError("no scaling in these tests")


class _EightBitFrame:
    """The 8-bit surface: swscale is the conversion, and the test needs to see that it still is."""

    def __init__(self, width, height, rgb, format_name: str = "yuv420p"):
        self.width, self.height, self.time = int(width), int(height), 0.0
        self.format = type("_Format", (), {"name": format_name})()
        self.planes = None
        self._rgb = rgb
        self.requested: list[str] = []

    def to_ndarray(self, format):  # noqa: A002 - PyAV's keyword
        self.requested.append(format)
        return self._rgb


def _stack(y, cb, cr):
    """PyAV's `to_ndarray` layout for a planar format: the planes stacked by rows, padded to the
    luma row length (which is what makes the fallback's `[:, :chroma_w]` slice necessary)."""
    numpy = pytest.importorskip("numpy")
    width = y.shape[1]
    planes = []
    for plane in (y, cb, cr):
        pad = width - plane.shape[1]
        planes.append(plane if pad == 0 else numpy.pad(plane, ((0, 0), (0, pad))))
    return numpy.concatenate(planes, axis=0)


def _grey(width, height, luma_code, *, chroma_code=512, dtype="uint16"):
    """A neutral flat picture at one 10-bit code, returned as `(y, cb, cr)` planes."""
    numpy = pytest.importorskip("numpy")
    kind = getattr(numpy, dtype)
    y = numpy.full((height, width), luma_code, kind)
    chroma = numpy.full(((height + 1) // 2, (width + 1) // 2), chroma_code, kind)
    return y, chroma, chroma


def _textbook(y, u, v, *, full_range=False):
    """ITU-T H.273, BT.709 non-constant-luminance, written out: code values -> RGB in 0..255."""
    y_offset, y_span = (0.0, 1023.0) if full_range else (64.0, 876.0)
    c_offset, c_span = (512.0, 1023.0) if full_range else (512.0, 896.0)
    yy = (float(y) - y_offset) / y_span
    cb = (float(u) - c_offset) / c_span
    cr = (float(v) - c_offset) / c_span
    r = yy + 1.5748 * cr
    g = yy - 0.187324 * cb - 0.468124 * cr
    b = yy + 1.8556 * cb
    return tuple(max(0.0, min(1.0, value)) * 255.0 for value in (r, g, b))


def _rgb(ref):
    numpy = pytest.importorskip("numpy")
    return numpy.frombuffer(ref.buffer, numpy.uint8).reshape(ref.height, ref.width, 3)


# -- the numbers the row is about ----------------------------------------------------------------


def test_a_ten_bit_limited_range_picture_lands_on_h273_white_grey_and_black():
    """940 -> 255, 502 -> 127.5, 64 -> 0. The old path produced 253 / 126 / 0."""
    from src.ui.video.frames import pack_frame

    for code, want in ((940, 255.0), (502, 127.5), (64, 0.0)):
        y, cb, cr = _grey(8, 8, code)
        frame = _TenBitFrame(8, 8, y, cb, cr)

        ref = pack_frame(frame, index=0, pixel_format="rgb24", color_space="bt709", color_range="tv")
        pixel = _rgb(ref)[4, 4].astype(float)

        assert pixel == pytest.approx((want, want, want), abs=0.5), f"code {code}: {pixel} != {want}"
        assert ref.format == "rgb24" and ref.stride == 8 * 3 and (ref.width, ref.height) == (8, 8)
        assert frame.requested == [], "the 10-bit conversion reads the planes, it never calls the scaler"


def test_the_array_fallback_gives_the_same_pixel_as_the_plane_views():
    """A frame without PyAV's plane views (P010-shaped sources, stubs) takes the stacked-array path."""
    from src.ui.video.frames import pack_frame

    y, cb, cr = _grey(8, 8, 940)
    frame = _TenBitFrame(8, 8, y, cb, cr, planes=False)

    pixel = _rgb(pack_frame(frame, 0, "rgb24", color_space="bt709", color_range="tv"))[4, 4]

    assert pixel.tolist() == [255, 255, 255]
    assert frame.requested == ["yuv420p10le"], "the array path asks for the source format, not rgb24"


def test_a_ten_bit_ramp_matches_the_textbook_formula_and_reaches_full_white():
    """Every step of 64..940 within one level of H.273, and the two ends are exactly 0 and 255."""
    numpy = pytest.importorskip("numpy")
    from src.ui.video.frames import pack_frame

    width, height = 876, 2
    codes = numpy.arange(64, 64 + width, dtype=numpy.uint16)          # 876 codes: tv black..tv white
    y = numpy.tile(codes, (height, 1))
    chroma = numpy.full((height // 2, width // 2), 512, numpy.uint16)
    frame = _TenBitFrame(width, height, y, chroma, chroma)

    ref = pack_frame(frame, index=0, pixel_format="rgb24", color_space="bt709", color_range="tv")
    got = _rgb(ref)[0, :, 0].astype(float)

    want = numpy.array([_textbook(int(code), 512, 512)[0] for code in codes])
    assert numpy.abs(got - want).max() <= 1.0, "the ramp must follow H.273 to within one level"
    assert got[0] == 0.0 and got[-1] == 255.0, "tv black and white are the ends of the scale, not 1 and 253"
    assert (numpy.diff(got) >= 0).all(), "a ramp may not wobble"


def test_full_range_ten_bit_uses_its_own_span():
    """pc range is a different affine: 0 -> 0 and 1023 -> 255, and mid-code is the scale midpoint."""
    from src.ui.video.frames import pack_frame

    for code, want in ((0, 0.0), (1023, 255.0), (512, 127.5)):
        y, cb, cr = _grey(8, 8, code)
        frame = _TenBitFrame(8, 8, y, cb, cr)

        pixel = _rgb(pack_frame(frame, 0, "rgb24", color_space="bt709", color_range="pc"))[4, 4].astype(float)

        assert pixel == pytest.approx((want, want, want), abs=0.5), f"pc code {code}: {pixel} != {want}"


def test_a_chroma_step_is_filtered_not_replicated():
    """The chroma tap is a 3/4-1/4 triangle: a one-sample step must show one intermediate column.

    Nearest replication was measured at 3.37 % of pixels more than 8 levels off the filtered tap's
    1.43 % (pane pixel gate, 2026-11-25), and the compare's source half draws through this path.
    """
    numpy = pytest.importorskip("numpy")
    from src.ui.video.frames import pack_frame

    width, height = 8, 4
    y = numpy.full((height, width), 502, numpy.uint16)
    cb = numpy.full((height // 2, width // 2), 200, numpy.uint16)
    cb[:, width // 4:] = 800                     # one hard chroma edge between columns 3 and 4
    cr = numpy.full((height // 2, width // 2), 512, numpy.uint16)
    frame = _TenBitFrame(width, height, y, cb, cr)

    blue = _rgb(pack_frame(frame, 0, "rgb24", color_space="bt709", color_range="tv"))[0, :, 2].astype(float)
    low, high = blue[0], blue[-1]

    assert low < high, "the edge must be visible at all"
    assert low < blue[width // 2 - 1] < high, "the sample next to the step is a mix, not a copy of either side"


def test_an_unknown_depth_layout_falls_back_to_the_scaler():
    """A format the conversion does not cover keeps the picture: the scaler is still reachable."""
    numpy = pytest.importorskip("numpy")
    from src.ui.video.frames import pack_frame

    rgb = numpy.full((4, 4, 3), 7, numpy.uint8)
    frame = _EightBitFrame(4, 4, rgb, format_name="yuv420p12le")     # not in the 10-bit table

    ref = pack_frame(frame, 0, "rgb24", color_space="bt709", color_range="tv")

    assert frame.requested == ["rgb24"], "an uncovered format must not lose its conversion"
    assert _rgb(ref)[0, 0].tolist() == [7, 7, 7]


def test_the_eight_bit_path_is_untouched():
    """The row's other half: 8-bit already maps 16..235 -> 0..255, so it keeps swscale exactly."""
    numpy = pytest.importorskip("numpy")
    from src.ui.video.frames import pack_frame

    rgb = numpy.zeros((4, 4, 3), numpy.uint8)
    rgb[..., 0] = 235
    frame = _EightBitFrame(4, 4, rgb)

    ref = pack_frame(frame, 0, "rgb24", color_space="bt709", color_range="tv")

    assert frame.requested == ["rgb24"], "an 8-bit frame is converted by the scaler, as before"
    assert _rgb(ref)[0, 0].tolist() == [235, 0, 0], "and its bytes pass through untouched"


def test_the_ten_bit_tables_are_the_ones_the_shader_uses():
    """The renderer owns the shader's numbers and imports this module, so these tables are a copy.

    A copy that may drift silently is the defect; this assertion is what makes it a pinned one.
    """
    pytest.importorskip("numpy")
    renderer = pytest.importorskip("src.ui.widgets.preview_renderer")
    from src.ui.video import frames

    assert frames._RGB_MATRICES == renderer.YUV_MATRICES
    assert frames._RANGE_CODES_10 == renderer._RANGE_CODES[10]
    assert frames._DEFAULT_COLOR_SPACE == renderer.DEFAULT_COLOR_SPACE
