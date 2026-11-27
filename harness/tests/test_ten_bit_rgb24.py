"""The RGB24 request at every planar depth: the conversion swscale gets wrong, pinned by numbers.

Why this file exists (2026-11-27, extended 2026-11-28). A 10-bit source that goes through the RGBA
path - the compare's source half, the queue thumbnails, the `rgba8` fallback - came out ~2.5 levels
dark. Measured against zimg (the independent H.273 reference) and against libswscale itself, the
library PyAV calls:

    code 940 (tv white)   swscale 253   H.273 255
    code 502 (mid-grey)   swscale 126   H.273 127.5
    code  64 (tv black)   swscale   0   H.273 0        (endpoints survive, the span between them shrinks)

swscale's fast yuv2rgb truncates the 10-bit -> 8-bit reduction before the range expansion; its
`SWS_ACCURATE_RND` mode converts correctly, but PyAV exposes no `sws_flags`. So the planar depths are
converted in `frames.rgb24_buffer` with the H.273 arithmetic the fragment shader already uses - and
this file pins the numbers that decision is worth: synthetic planes, against the textbook formula
typed out here rather than against the implementation (a second copy of a wrong matrix agrees with
itself).

The 8-bit half (2026-11-28) is about a flag, not a depth. Re-measured on this host's libswscale 10
(ffmpeg n9.1, one flat yuv420p frame, chroma 128):

    flags                     16    126    235
    default / bilinear         0    128    255     <- correct, and what this file used to assume
    accurate_rnd               0    126    253     <- the 1.6-level shrinkage the row was opened for

The same CLI flags reproduce the 10-bit numbers above exactly, so the instrument is the one the
10-bit row used. The app cannot choose the flag (PyAV's `VideoReformatter` exposes interpolation,
colorspace and range only), which is why both depths are converted from the codes: the result is then
a function of the picture rather than of a scaler build.

`p010le` carries its 10-bit code in the HIGH bits of a 16-bit word (`word >> 6`), unlike
`yuv420p10le`, whose code sits in the low ten bits. Reading a p010 white (940 << 6 = 60160) as a
16-bit sample is black, and reading it with a `>> 2` shift reads 3760 as code 940.

No PyAV on this interpreter (see `frames`' docstring): the frames are stubs with PyAV's duck-typed
surface, and a stub REFUSES `to_ndarray("rgb24")` - which is what makes the numeric tests fail on the
old code instead of silently measuring the scaler.
"""

from __future__ import annotations

import logging

import pytest

np = pytest.importorskip("numpy")


class _PlaneStub(bytes):
    """The bit of a PyAV ``VideoPlane`` the packer reads: the buffer protocol and ``line_size``."""

    def __new__(cls, data: bytes, line_size: int | None = None):
        self = super().__new__(cls, data)
        self.line_size = len(data) if line_size is None else int(line_size)
        return self


class _PlanarFrame:
    """A decoded planar frame with PyAV's own plane views, and a scaler that would be a bug to use."""

    def __init__(self, y, cb, cr, format_name: str = "yuv420p10le", planes: bool = True):
        self.height, self.width, self.time = int(y.shape[0]), int(y.shape[1]), 0.0
        self.format = type("_Format", (), {"name": format_name})()
        sample_bytes = y.dtype.itemsize
        self.planes = None if not planes else [
            _PlaneStub(plane.tobytes(), plane.shape[1] * sample_bytes) for plane in (y, cb, cr)]
        self._stacked = {format_name: _stack(y, cb, cr)}
        self.requested: list[str] = []

    def to_ndarray(self, format):  # noqa: A002 - PyAV's keyword
        self.requested.append(format)
        if format == "rgb24":
            raise AssertionError(
                f"a planar {self.format.name} frame must not go through swscale: its yuv2rgb is flag-dependent")
        return self._stacked[format]

    def reformat(self, **_kwargs):  # pragma: no cover - pack_frame only scales when `size=` is given
        raise AssertionError("no scaling in these tests")


class _ScalerFrame:
    """A frame the conversion does not cover: only the scaler can answer, and the test needs to see it."""

    def __init__(self, width, height, rgb, format_name: str):
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
    width = y.shape[1]
    planes = []
    for plane in (y, cb, cr):
        pad = width - plane.shape[1]
        planes.append(plane if pad == 0 else np.pad(plane, ((0, 0), (0, pad))))
    return np.concatenate(planes, axis=0)


def _grey(width, height, luma_code, *, chroma_code=None, dtype="uint16"):
    """A neutral flat picture at one luma code, returned as `(y, cb, cr)` planes.

    `chroma_code` defaults to the neutral code of the dtype's depth (128 at 8 bits, 512 at 10).
    """
    kind = getattr(np, dtype)
    if chroma_code is None:
        chroma_code = 128 if kind is np.uint8 else 512          # the neutral code at 8 / 10 bits
    y = np.full((height, width), luma_code, kind)
    chroma = np.full(((height + 1) // 2, (width + 1) // 2), chroma_code, kind)
    return y, chroma, chroma


def _textbook(y, u, v, *, full_range=False, depth=10):
    """ITU-T H.273, BT.709 non-constant-luminance, written out: code values -> RGB in 0..255."""
    if depth == 8:
        y_offset, y_span = (0.0, 255.0) if full_range else (16.0, 219.0)
        c_offset, c_span = (128.0, 255.0) if full_range else (128.0, 224.0)
    else:
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
    return np.frombuffer(ref.buffer, np.uint8).reshape(ref.height, ref.width, 3)


# -- the numbers the row is about ----------------------------------------------------------------


def test_a_ten_bit_limited_range_picture_lands_on_h273_white_grey_and_black():
    """940 -> 255, 502 -> 127.5, 64 -> 0. The scaler produced 253 / 126 / 0."""
    from src.ui.video.frames import pack_frame

    for code, want in ((940, 255.0), (502, 127.5), (64, 0.0)):
        y, cb, cr = _grey(8, 8, code)
        frame = _PlanarFrame(y, cb, cr)

        ref = pack_frame(frame, index=0, pixel_format="rgb24", color_space="bt709", color_range="tv")
        pixel = _rgb(ref)[4, 4].astype(float)

        assert pixel == pytest.approx((want, want, want), abs=0.5), f"code {code}: {pixel} != {want}"
        assert ref.format == "rgb24" and ref.stride == 8 * 3 and (ref.width, ref.height) == (8, 8)
        assert frame.requested == [], "the planar conversion reads the planes, it never calls the scaler"


def test_an_eight_bit_limited_range_picture_lands_on_h273_white_grey_and_black():
    """235 -> 255, 126 -> 127.5 (128 after round-half-up), 16 -> 0.

    The scaler reaches the same 235 on this build's default flags and 253 under `SWS_ACCURATE_RND`;
    this path is the one that answers the same numbers whichever flags a PyAV build passes. On the
    old code the stub's `to_ndarray("rgb24")` raises, so this test cannot pass by measuring swscale.
    """
    from src.ui.video.frames import pack_frame

    for code, want in ((235, 255.0), (126, 127.5), (16, 0.0)):
        y, cb, cr = _grey(8, 8, code, dtype="uint8")
        frame = _PlanarFrame(y, cb, cr, format_name="yuv420p")

        ref = pack_frame(frame, index=0, pixel_format="rgb24", color_space="bt709", color_range="tv")
        pixel = _rgb(ref)[4, 4].astype(float)

        assert pixel == pytest.approx((want, want, want), abs=1.0), f"8-bit code {code}: {pixel} != {want}"
        assert frame.requested == [], "the 8-bit conversion reads the planes, it never calls the scaler"


def test_an_eight_bit_chroma_centre_is_neutral_and_the_span_is_224_not_219():
    """Code 128 is chroma zero; 16 is the chroma minimum, and its 224-step span is what makes blue 18.

    The span is the one arithmetic detail a copy of the luma formula gets wrong: 219 instead of 224
    puts blue on 13 here, five levels dark, and no luma-only test can see it.
    """
    from src.ui.video.frames import pack_frame

    y, cb, cr = _grey(8, 8, 126, dtype="uint8", chroma_code=128)
    neutral = _rgb(pack_frame(_PlanarFrame(y, cb, cr, format_name="yuv420p"), 0, "rgb24",
                             color_space="bt709", color_range="tv"))[4, 4].astype(float)
    assert neutral == pytest.approx((128, 128, 128), abs=1.0), "a neutral picture keeps the luma value in all three"

    y = np.full((8, 8), 235, np.uint8)
    chroma = np.full((4, 4), 16, np.uint8)
    saturated = _rgb(pack_frame(_PlanarFrame(y, chroma, chroma, format_name="yuv420p"), 0, "rgb24",
                                color_space="bt709", color_range="tv"))[4, 4].astype(float)

    assert saturated == pytest.approx(_textbook(235, 16, 16, depth=8), abs=1.0), \
        f"chroma 16 at luma 235: {saturated.tolist()}"


def test_a_ten_bit_ramp_matches_the_textbook_formula_and_reaches_full_white():
    """Every step of 64..940 within one level of H.273, and the two ends are exactly 0 and 255."""
    from src.ui.video.frames import pack_frame

    width, height = 876, 2
    codes = np.arange(64, 64 + width, dtype=np.uint16)          # 876 codes: tv black..tv white
    y = np.tile(codes, (height, 1))
    chroma = np.full((height // 2, width // 2), 512, np.uint16)
    frame = _PlanarFrame(y, chroma, chroma)

    ref = pack_frame(frame, index=0, pixel_format="rgb24", color_space="bt709", color_range="tv")
    got = _rgb(ref)[0, :, 0].astype(float)

    want = np.array([_textbook(int(code), 512, 512)[0] for code in codes])
    assert np.abs(got - want).max() <= 1.0, "the ramp must follow H.273 to within one level"
    assert got[0] == 0.0 and got[-1] == 255.0, "tv black and white are the ends of the scale, not 1 and 253"
    assert (np.diff(got) >= 0).all(), "a ramp may not wobble"


def test_an_eight_bit_ramp_matches_the_textbook_formula_and_reaches_full_white():
    """Every step of 16..235 within one level of H.273, and the two ends are exactly 0 and 255."""
    from src.ui.video.frames import pack_frame

    width, height = 220, 2
    codes = np.arange(16, 16 + width, dtype=np.uint8)           # 220 codes: tv black..tv white
    y = np.tile(codes, (height, 1))
    chroma = np.full((height // 2, width // 2), 128, np.uint8)
    frame = _PlanarFrame(y, chroma, chroma, format_name="yuv420p")

    ref = pack_frame(frame, index=0, pixel_format="rgb24", color_space="bt709", color_range="tv")
    got = _rgb(ref)[0, :, 0].astype(float)

    want = np.array([_textbook(int(code), 128, 128, depth=8)[0] for code in codes])
    assert np.abs(got - want).max() <= 1.0, "the 8-bit ramp must follow H.273 to within one level"
    assert got[0] == 0.0 and got[-1] == 255.0, "16 and 235 are the ends of the scale"
    assert (np.diff(got) >= 0).all(), "a ramp may not wobble"


def test_eight_bit_422_and_444_are_covered_too():
    """The table's other 8-bit layouts: 4:2:2 chroma is half-width and full-height, 4:4:4 is neither.

    A subsampling borrowed from the 4:2:0 entry crops the chroma of a 4:2:2 frame to half its rows,
    which shows up as a green cast on the lower half - so each layout gets its own plane shape here.
    """
    from src.ui.video.frames import pack_frame

    for name, hsub, vsub in (("yuv422p", 2, 1), ("yuv444p", 1, 1)):
        width, height = 8, 4
        y = np.full((height, width), 235, np.uint8)
        chroma = np.full(((height + vsub - 1) // vsub, (width + hsub - 1) // hsub), 128, np.uint8)
        frame = _PlanarFrame(y, chroma, chroma, format_name=name)

        pixel = _rgb(pack_frame(frame, 0, "rgb24", color_space="bt709", color_range="tv"))[1, 1]

        assert pixel.tolist() == [255, 255, 255], f"{name}: {pixel.tolist()}"
        assert frame.requested == [], f"{name} must be converted from its planes"


def test_the_array_fallback_gives_the_same_pixel_as_the_plane_views():
    """A frame without PyAV's plane views takes the stacked-array path, at both depths."""
    from src.ui.video.frames import pack_frame

    for dtype, name, code in (("uint16", "yuv420p10le", 940), ("uint8", "yuv420p", 235)):
        y, cb, cr = _grey(8, 8, code, dtype=dtype)
        frame = _PlanarFrame(y, cb, cr, format_name=name, planes=False)

        pixel = _rgb(pack_frame(frame, 0, "rgb24", color_space="bt709", color_range="tv"))[4, 4]

        assert pixel.tolist() == [255, 255, 255], f"{name}: the array path must agree with the plane views"
        assert frame.requested == [name], "the array path asks for the source format, not rgb24"


def test_p010_is_read_as_a_ten_bit_code_in_the_high_bits():
    """p010le: code 940 is the word 60160 (940 << 6), not 3760 and not 940.

    A p010 white read as a 16-bit sample is black; read with a `>> 2` shift, 3760 becomes code 940 and
    a black word becomes a white one - both are pinned here so the shift cannot drift.
    """
    from src.ui.video.frames import pack_frame

    def p010_grey(code):
        # Every p010 sample is shifted, chroma included: neutral 10-bit code 512 is the word 32768.
        y, cb, cr = _grey(8, 8, code << 6, chroma_code=512 << 6, dtype="uint16")
        return _PlanarFrame(y, cb, cr, format_name="p010le")

    for code, want in ((940, 255.0), (502, 127.5), (64, 0.0)):
        frame = p010_grey(code)
        pixel = _rgb(pack_frame(frame, 0, "rgb24", color_space="bt709", color_range="tv"))[4, 4].astype(float)

        assert pixel == pytest.approx((want, want, want), abs=0.5), f"p010 code {code} (word {code << 6}): {pixel}"
        assert frame.requested == [], "p010 is covered by the conversion, not by the scaler"

    # 3760 is 940 << 2: a `>> 2` reader would call this tv white, a `>> 6` reader sees code 58 (black).
    black = _rgb(pack_frame(p010_grey(58), 0, "rgb24", color_space="bt709", color_range="tv"))[4, 4]
    assert black.tolist() == [0, 0, 0], "the low bits of a p010 word are not code bits"


def test_p010_array_fallback_applies_the_same_shift():
    """A p010 frame without plane views (`to_ndarray`) reads the words and shifts them too."""
    from src.ui.video.frames import pack_frame

    y, cb, cr = _grey(8, 8, 940 << 6, chroma_code=512 << 6, dtype="uint16")
    frame = _PlanarFrame(y, cb, cr, format_name="p010le", planes=False)

    pixel = _rgb(pack_frame(frame, 0, "rgb24", color_space="bt709", color_range="tv"))[4, 4]

    assert pixel.tolist() == [255, 255, 255]
    assert frame.requested == ["p010le"]


def test_full_range_ten_bit_uses_its_own_span():
    """pc range is a different affine: 0 -> 0 and 1023 -> 255, and mid-code is the scale midpoint."""
    from src.ui.video.frames import pack_frame

    for code, want in ((0, 0.0), (1023, 255.0), (512, 127.5)):
        y, cb, cr = _grey(8, 8, code)
        frame = _PlanarFrame(y, cb, cr)

        pixel = _rgb(pack_frame(frame, 0, "rgb24", color_space="bt709", color_range="pc"))[4, 4].astype(float)

        assert pixel == pytest.approx((want, want, want), abs=0.5), f"pc code {code}: {pixel} != {want}"


def test_full_range_eight_bit_uses_its_own_span():
    """8-bit pc: 0 -> 0 and 255 -> 255, with the chroma centred on 128 over a 255 span."""
    from src.ui.video.frames import pack_frame

    for code, want in ((0, 0.0), (255, 255.0)):
        y, cb, cr = _grey(8, 8, code, dtype="uint8")
        frame = _PlanarFrame(y, cb, cr, format_name="yuv420p")

        pixel = _rgb(pack_frame(frame, 0, "rgb24", color_space="bt709", color_range="pc"))[4, 4].astype(float)

        assert pixel == pytest.approx((want, want, want), abs=0.5), f"pc code {code}: {pixel} != {want}"


def test_a_chroma_step_is_filtered_not_replicated():
    """The chroma tap is a 3/4-1/4 triangle: a one-sample step must show one intermediate column.

    Nearest replication was measured at 3.37 % of pixels more than 8 levels off the filtered tap's
    1.43 % (pane pixel gate, 2026-11-25), and the compare's source half draws through this path.
    """
    from src.ui.video.frames import pack_frame

    width, height = 8, 4
    y = np.full((height, width), 502, np.uint16)
    cb = np.full((height // 2, width // 2), 200, np.uint16)
    cb[:, width // 4:] = 800                     # one hard chroma edge between columns 3 and 4
    cr = np.full((height // 2, width // 2), 512, np.uint16)
    frame = _PlanarFrame(y, cb, cr)

    blue = _rgb(pack_frame(frame, 0, "rgb24", color_space="bt709", color_range="tv"))[0, :, 2].astype(float)
    low, high = blue[0], blue[-1]

    assert low < high, "the edge must be visible at all"
    assert low < blue[width // 2 - 1] < high, "the sample next to the step is a mix, not a copy of either side"


def test_an_unknown_format_falls_back_to_the_scaler_and_says_so_once(caplog, monkeypatch):
    """An uncovered format keeps the picture and names itself ONCE, not once per frame at 30 fps."""
    from src.ui.video import frames
    from src.ui.video.frames import rgb24_buffer

    monkeypatch.setattr(frames, "_reported_fallbacks", set())
    rgb = np.full((4, 4, 3), 7, np.uint8)
    frame = _ScalerFrame(4, 4, rgb, format_name="yuv420p12le")     # not in the table

    with caplog.at_level(logging.INFO, logger="src.ui.video.frames"):
        first = rgb24_buffer(frame, color_space="bt709", color_range="tv")
        second = rgb24_buffer(frame, color_space="bt709", color_range="tv")

    assert frame.requested == ["rgb24", "rgb24"], "an uncovered format must not lose its conversion"
    assert first == second == rgb.tobytes()
    notes = [record for record in caplog.records if "yuv420p12le" in record.getMessage()]
    assert len(notes) == 1, f"one line per format, got {[record.getMessage() for record in notes]}"


def test_a_packed_rgb_frame_is_not_reported_as_a_fallback(caplog, monkeypatch):
    """`rgb24` in, `rgb24` out is a copy, not a dark conversion: it must not fill the log."""
    from src.ui.video import frames
    from src.ui.video.frames import rgb24_buffer

    monkeypatch.setattr(frames, "_reported_fallbacks", set())
    rgb = np.full((4, 4, 3), 9, np.uint8)
    frame = _ScalerFrame(4, 4, rgb, format_name="rgb24")

    with caplog.at_level(logging.INFO, logger="src.ui.video.frames"):
        assert rgb24_buffer(frame) == rgb.tobytes()

    assert not caplog.records, "a packed RGB frame needs no conversion and deserves no note"


def test_the_depth_tables_are_the_ones_the_shader_uses():
    """The renderer owns the shader's numbers and imports this module, so these tables are a copy.

    A copy that may drift silently is the defect; this assertion is what makes it a pinned one.
    """
    pytest.importorskip("numpy")
    renderer = pytest.importorskip("src.ui.widgets.preview_renderer")
    from src.ui.video import frames

    assert frames._RGB_MATRICES == renderer.YUV_MATRICES
    assert frames._RANGE_CODES_8 == renderer._RANGE_CODES[8]
    assert frames._RANGE_CODES_10 == renderer._RANGE_CODES[10]
    assert frames._DEFAULT_COLOR_SPACE == renderer.DEFAULT_COLOR_SPACE
