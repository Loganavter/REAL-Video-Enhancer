"""The YUV plane path: the packer, the shader's arithmetic, and the pane's additive ingestion method.

Why this file exists (2026-11-25). The playback pane used to be `swscale YUV->RGB24 -> to_ndarray -> tobytes -> QImage
-> RGBA8888 -> one 4-byte-per-pixel upload`: four copies and a colour conversion on the CPU, per frame. The plane path
keeps the decoder's Y/U/V as they are, uploads three textures (Y full size, U/V half) and converts in the fragment
shader. What is pinned here is everything that can be wrong WITHOUT a GPU:

* the packer's plane layout and strides at both depths (10-bit is TWO bytes per sample, and the code value lives in
  the low 10 bits - the arithmetic trap this path exists to get right);
* the colour arithmetic: a known YUV triple -> RGB within one level, at tv AND pc range, computed against the
  textbook H.273 formula rather than against the implementation (a second copy of a wrong matrix agrees with itself);
* the pane's new method being ADDITIVE: `set_frame(image, ...)` keeps its signature and its behaviour, the planes
  method reaches the renderer as a `PreviewFrame`, and the geometry reads the size of whichever shape arrived;
* the baked shaders existing and being valid `.qsb` files (a shader that fails to load is a black pane, live).

What this file CANNOT answer: whether the GPU samples the three textures as intended. Offscreen has no QRhi
(`VideoPreviewSurface.initialize` returns early, `grabFrame()` is a null readback - see
`test_two_surfaces_rhi.py`'s docstring). That proof is the supervisor's live run on a leased display: the RHI proof
and the pixel gate (A/A, then the worst frame of a spread sample, >8 levels in any channel <= 0.1 %).
"""

from __future__ import annotations

import inspect

import pytest

from src.ui.video.frames import (
    FORMATS,
    PLANAR_FORMATS,
    SAMPLE_BYTES,
    FrameRef,
    pack_frame,
    pack_planes,
    plane_geometry,
    stream_color_tags,
)

# -- the packer ------------------------------------------------------------------------------------


class _PlaneStub(bytes):
    """The bit of a PyAV ``VideoPlane`` the packer uses: the buffer protocol and ``line_size``."""

    def __new__(cls, data: bytes, line_size: int | None = None):
        self = super().__new__(cls, data)
        self.line_size = len(data) if line_size is None else int(line_size)
        return self


class _FrameStub:
    """A decoded frame with PyAV's own planes (no PyAV on this interpreter)."""

    def __init__(self, width: int, height: int, planes, time=0.0, format_name: str = "yuv420p10le") -> None:
        self.width, self.height, self.time = width, height, time
        self.planes = list(planes)
        self.format = type("_Format", (), {"name": format_name})()

    def reformat(self, **_kwargs):  # pragma: no cover - the tests only exercise the matching-format path
        raise AssertionError("the packer must not reformat a frame that already is the requested format")


class _ArrayFrame:
    """The duck-typed fallback surface: ``to_ndarray`` only, like the numpy path."""

    def __init__(self, width: int, height: int, array, time=0.0) -> None:
        self.width, self.height, self.time = width, height, time
        self._array = array
        self.formats: list[str] = []

    def to_ndarray(self, format: str):  # noqa: A002 - PyAV's keyword
        self.formats.append(format)
        return self._array


def test_the_two_planar_formats_exist_and_10_bit_is_two_bytes_a_sample():
    assert set(PLANAR_FORMATS) <= set(FORMATS)
    assert SAMPLE_BYTES == {"yuv420p": 1, "yuv420p10le": 2}


def test_plane_geometry_halves_the_chroma_and_rounds_odd_sizes_up():
    assert plane_geometry(3840, 2160) == ((3840, 2160), (1920, 1080))
    assert plane_geometry(5, 3) == ((5, 3), (3, 2)), "cel-legal odd sizes: chroma is ceil(w/2) x ceil(h/2)"


def test_pack_planes_reads_pyav_planes_directly_with_their_own_strides():
    """10-bit: each plane is a view over the decoder's buffer, and the stride is the plane's BYTE row length."""
    luma = bytes(range(128))                         # 8x8 picture, 2 bytes/sample -> 16-byte rows
    chroma = bytes([7]) * 32                          # 4x4 chroma, 2 bytes/sample -> 8-byte rows
    frame = _FrameStub(8, 8, (_PlaneStub(luma, 16), _PlaneStub(chroma, 8), _PlaneStub(chroma, 8)))

    planes, strides = pack_planes(frame, "yuv420p10le")

    assert len(planes) == 3
    assert strides == (16, 8, 8), "10-bit strides are in BYTES: width * 2, not width"
    assert bytes(planes[0]) == luma
    assert bytes(planes[1]) == chroma == bytes(planes[2])
    assert planes[0].obj is frame.planes[0], "a memoryview over PyAV's plane, not a copy of it"


def test_a_padded_plane_is_repacked_to_a_tight_stride():
    """swscale pads rows; the upload wants one contiguous picture per plane, so the pad is dropped once."""
    row = 8                                           # 8-bit, 8 px wide
    padded = b"".join(bytes([y]) * row + b"\xff" * 2 for y in range(4))   # 4 rows, 2 bytes of padding
    frame = _FrameStub(8, 4, (_PlaneStub(padded, row + 2), _PlaneStub(b"", 0), _PlaneStub(b"", 0)),
                       format_name="yuv420p")
    frame.reformat = lambda **_kwargs: frame          # the stub already IS yuv420p

    planes, strides = pack_planes(frame, "yuv420p")

    assert strides[0] == row, "the stride of the repacked plane is the tight row length"
    assert bytes(planes[0]) == b"".join(bytes([y]) * row for y in range(4))


def test_the_array_fallback_slices_luma_then_both_chroma_planes():
    numpy = pytest.importorskip("numpy")
    width, height = 6, 4
    chroma_w, chroma_h = width // 2, height // 2
    array = numpy.zeros((height + 2 * chroma_h, width), dtype=numpy.uint8)
    array[:height, :width] = 10
    array[height:height + chroma_h, :chroma_w] = 20
    array[height + chroma_h:, :chroma_w] = 30
    frame = _ArrayFrame(width, height, array)

    planes, strides = pack_planes(frame, "yuv420p")

    assert frame.formats == ["yuv420p"], "the request is passed through, never assumed"
    assert strides == (width, chroma_w, chroma_w)
    assert bytes(planes[0]) == bytes([10]) * (width * height)
    assert bytes(planes[1]) == bytes([20]) * (chroma_w * chroma_h)
    assert bytes(planes[2]) == bytes([30]) * (chroma_w * chroma_h)


def test_pack_frame_keeps_the_stacked_buffer_and_adds_the_plane_descriptors():
    numpy = pytest.importorskip("numpy")
    width, height = 4, 4
    array = numpy.zeros((height + 2 * ((height + 1) // 2), width), dtype=numpy.uint16)
    array[:height] = 1
    array[height:] = 2
    packed = pack_frame(_ArrayFrame(width, height, array), index=3, pixel_format="yuv420p10le",
                        color_space="bt709", color_range="tv")

    assert packed.format == "yuv420p10le"
    assert packed.stride == width * 2, "10-bit luma stride is in bytes"
    assert len(packed.buffer) == (width * height + 2 * (width // 2) * (height // 2)) * 2
    assert packed.planes is not None and packed.strides == (width * 2, width, width)
    assert (packed.color_space, packed.color_range) == ("bt709", "tv")
    assert packed.index == 3


def test_an_eight_bit_plane_set_costs_one_and_a_half_bytes_a_pixel():
    """The traffic claim the lane is measured on: 3 B/px for 10-bit planes, 1.5 for 8-bit, against RGB's 3 and RGBA's 4."""
    numpy = pytest.importorskip("numpy")

    def packed_bytes(fmt: str, bps: int) -> int:
        width, height = 320, 180
        rows = height + 2 * ((height + 1) // 2)
        array = numpy.zeros((rows, width), dtype=numpy.uint8 if bps == 1 else numpy.uint16)
        frame = _ArrayFrame(width, height, array)
        plane = pack_frame(frame, index=0, pixel_format=fmt)
        return sum(len(bytes(view)) for view in plane.planes)

    assert packed_bytes("yuv420p", 1) == 320 * 180 * 3 // 2
    assert packed_bytes("yuv420p10le", 2) == 320 * 180 * 3


def test_the_planes_of_a_frameref_default_to_none_so_older_constructors_still_work():
    ref = FrameRef(buffer=b"\x00" * 12, width=2, height=2, format="rgb24", stride=6, pts=0.0, index=0)
    assert ref.planes is None and ref.strides is None
    assert ref.color_space is None and ref.color_range is None


def test_stream_tags_are_read_from_the_codec_context_and_unknown_values_stay_none():
    stream = type("_Stream", (), {"color_space": 1, "color_range": 1})()
    assert stream_color_tags(stream) == ("bt709", "tv")
    tags = type("_Tags", (), {"color_space": "bt2020nc", "color_range": "pc"})()
    assert stream_color_tags(type("_S", (), {"codec_context": tags})()) == ("bt2020nc", "pc")
    assert stream_color_tags(type("_Empty", (), {})()) == (None, None)
    assert stream_color_tags(type("_Odd", (), {"color_space": 99, "color_range": 42})()) == (None, None)


# -- the shader's arithmetic (the CPU reference) ----------------------------------------------------


def _textbook_bt709(y_code: float, u_code: float, v_code: float, *, depth: int, full_range: bool):
    """The H.273 formula written out independently of the renderer: limited or full range, BT.709.

    Limited range at 10-bit is luma 64..940 and chroma 64..960; at 8-bit 16..235 and 16..240. The expansion is done
    first, then the FULL-RANGE matrix (this is how a decoder separates the two steps, and the reason the renderer's
    `row0..2` are range-free coefficients).
    """
    y_off, y_span = (16.0, 219.0) if depth == 8 else (64.0, 876.0)
    c_off, c_span = (128.0, 224.0) if depth == 8 else (512.0, 896.0)
    if full_range:
        yy = y_code / float((1 << depth) - 1)
        cb = (u_code - c_off) / float((1 << depth) - 1)
        cr = (v_code - c_off) / float((1 << depth) - 1)
    else:
        yy = (y_code - y_off) / y_span
        cb = (u_code - c_off) / c_span
        cr = (v_code - c_off) / c_span
    r = yy + 1.5748 * cr
    g = yy - 0.187324 * cb - 0.468124 * cr
    b = yy + 1.8556 * cb
    return tuple(max(0.0, min(1.0, value)) * 255.0 for value in (r, g, b))


def test_the_cpu_matrix_reproduces_bt709_limited_range_known_triples():
    """Black, white, and a saturated red - the three triples whose values a wrong matrix or range breaks first."""
    from src.ui.widgets.preview_renderer import yuv_code_to_rgb

    triples = {
        "black": (64.0, 512.0, 512.0),
        "white": (940.0, 512.0, 512.0),
        "red-ish": (409.0, 419.0, 713.0),           # the BT.709 limited-range code of pure red at 10 bits
    }
    for name, (y, u, v) in triples.items():
        got = yuv_code_to_rgb(y, u, v, color_space="bt709", color_range="tv", depth=10)
        want = _textbook_bt709(y, u, v, depth=10, full_range=False)
        assert got == pytest.approx(want, abs=1.0), f"{name}: {got} != {want} (within one level)"
    black = yuv_code_to_rgb(64.0, 512.0, 512.0, depth=10)
    white = yuv_code_to_rgb(940.0, 512.0, 512.0, depth=10)
    assert black == pytest.approx((0.0, 0.0, 0.0), abs=1.0)
    assert white == pytest.approx((255.0, 255.0, 255.0), abs=1.0)


def test_the_cpu_matrix_handles_pc_range_differently_from_tv():
    from src.ui.widgets.preview_renderer import yuv_code_to_rgb

    for depth, peak in ((10, 1023.0), (8, 255.0)):
        tv_black = yuv_code_to_rgb(16.0 * peak / 255.0 if depth == 8 else 64.0, 128.0 if depth == 8 else 512.0,
                                   128.0 if depth == 8 else 512.0, color_range="tv", depth=depth)
        pc_black = yuv_code_to_rgb(0.0, peak / 2.0, peak / 2.0, color_range="pc", depth=depth)
        assert tv_black == pytest.approx((0.0, 0.0, 0.0), abs=1.0)
        assert pc_black == pytest.approx((0.0, 0.0, 0.0), abs=1.0)

        tv_white = yuv_code_to_rgb(235.0 if depth == 8 else 940.0, 128.0 if depth == 8 else 512.0,
                                   128.0 if depth == 8 else 512.0, color_range="tv", depth=depth)
        pc_white = yuv_code_to_rgb(peak, peak / 2.0, peak / 2.0, color_range="pc", depth=depth)
        assert tv_white == pytest.approx((255.0, 255.0, 255.0), abs=1.0)
        assert pc_white == pytest.approx((255.0, 255.0, 255.0), abs=1.0)

        # The SAME code is a different colour in the two ranges - the tv/pc distinction must not be collapsed.
        grey_code = 128.0 if depth == 8 else 512.0
        assert yuv_code_to_rgb(grey_code, grey_code, grey_code, color_range="tv", depth=depth) == \
            pytest.approx((130.0, 130.0, 130.0), abs=1.0), "code 128/512 is video grey, one step above 0.5"
        assert yuv_code_to_rgb(grey_code, grey_code, grey_code, color_range="pc", depth=depth) == \
            pytest.approx((127.5, 127.5, 127.5), abs=1.0), "the same code is the code-space midpoint in pc range"
        # ... and tv black (64/16) is black while the same code in pc range is a lifted grey.
        black_tv = yuv_code_to_rgb(grey_code / 8.0, grey_code, grey_code, color_range="tv", depth=depth)
        black_pc = yuv_code_to_rgb(grey_code / 8.0, grey_code, grey_code, color_range="pc", depth=depth)
        assert black_tv == pytest.approx((0.0, 0.0, 0.0), abs=1.0)
        assert black_pc[0] > 10.0, "in pc range that code is a real (if dark) grey, not black"


def test_the_upload_bundle_normalises_by_the_code_maximum_not_by_65535():
    """The 10-bit trap, in one assertion: an R16 sample is code/65535, and the multiplier must undo exactly that."""
    from src.ui.widgets.preview_renderer import yuv_uniform_floats

    luma_x, luma_y = yuv_uniform_floats("bt709", "tv", depth=10)[:2]
    assert luma_x == pytest.approx(65535.0 / 876.0)
    assert luma_y == pytest.approx(-64.0 / 876.0)
    # 940 (limited white) must land on exactly 1.0 through the shader's `s * x + y`.
    assert (940.0 / 65535.0) * luma_x + luma_y == pytest.approx(1.0)
    eight_bit_x = yuv_uniform_floats("bt709", "tv", depth=8)[0]
    assert eight_bit_x == pytest.approx(255.0 / 219.0)
    assert (235.0 / 255.0) * eight_bit_x + yuv_uniform_floats("bt709", "tv", depth=8)[1] == pytest.approx(1.0)


def test_a_file_tagged_bt2020_gets_its_own_matrix():
    from src.ui.widgets.preview_renderer import yuv_uniform_floats

    bt709 = yuv_uniform_floats("bt709", "tv", depth=10)
    bt2020 = yuv_uniform_floats("bt2020nc", "tv", depth=10)
    assert bt709 != bt2020, "a hard-coded BT.709 would make these equal, which is the bug this test exists for"
    assert bt2020[8:11] == pytest.approx((1.0, 0.0, 1.4746))
    assert yuv_uniform_floats(None, None, depth=10) == bt709, "an untagged file of ours is bt709/tv"


def test_the_baked_shaders_are_present_and_valid():
    import os

    pytest.importorskip("PySide6")
    from PySide6.QtGui import QShader

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    shader_dir = os.path.join(root, "src", "ui", "widgets", "shaders")
    for name in ("yuv.vert.qsb", "yuv.frag.qsb"):
        path = os.path.join(shader_dir, name)
        assert os.path.isfile(path), f"{name} is missing: run harness/scripts/build_shaders.sh"
        with open(path, "rb") as handle:
            shader = QShader.fromSerialized(handle.read())
        assert shader.isValid(), f"{name} does not deserialize into a valid QShader (a black pane, live)"


# -- the widget bridge -----------------------------------------------------------------------------


@pytest.fixture()
def pane(qapp):
    pytest.importorskip("PySide6")
    from src.ui.widgets.video_preview import VideoPreviewSurface

    surface = VideoPreviewSurface()
    surface.resize(320, 180)
    yield surface
    surface.deleteLater()


def _planes_for(width: int, height: int, *, fill: int = 0) -> tuple[tuple[bytes, ...], tuple[int, ...]]:
    chroma_w, chroma_h = (width + 1) // 2, (height + 1) // 2
    planes = (bytes([fill]) * (width * height * 2),
              bytes([fill]) * (chroma_w * chroma_h * 2),
              bytes([fill]) * (chroma_w * chroma_h * 2))
    strides = (width * 2, chroma_w * 2, chroma_w * 2)
    return planes, strides


def test_the_pane_ingests_a_planes_frame_without_a_qimage(pane):
    from src.ui.widgets.preview_renderer import PreviewFrame

    planes, strides = _planes_for(64, 48, fill=17)
    pane.set_frame_planes(planes, strides, 64, 48, index=7, pts=1.25,
                          format="yuv420p10le", color_space="bt709", color_range="tv")

    assert pane.has_frame(), "a planes picture is a picture: the pane must not report an empty pane"
    assert pane.shown() == ("frame", 7)
    assert pane.shown_time() == pytest.approx(1.25)
    assert pane._image_size() == (64, 48), "geometry comes from the descriptor when there is no image"
    frame = pane._renderer._frame
    assert isinstance(frame, PreviewFrame)
    assert frame.format == "yuv420p10le"
    assert frame.strides == strides and len(frame.planes) == 3
    assert (frame.color_space, frame.color_range) == ("bt709", "tv")


def test_the_rgba_wrapper_is_unchanged_and_clears_a_standing_planes_frame(pane):
    from PySide6.QtGui import QImage, QColor

    planes, strides = _planes_for(32, 32)
    pane.set_frame_planes(planes, strides, 32, 32)
    assert pane._planes_size == (32, 32)

    image = QImage(20, 10, QImage.Format.Format_RGB888)
    image.fill(QColor(10, 20, 30))
    pane.set_frame(image, index=2, media_time=0.5)

    assert pane._planes_size is None, "an image frame replaces the planes picture instead of adding to it"
    assert pane._image_size() == (20, 10)
    assert pane._renderer._frame.format == "rgba8"
    assert pane.shown() == ("frame", 2)


def test_set_frame_keeps_its_public_signature():
    """The host calls this; the planes path is additive and may not reshape the existing API."""
    pytest.importorskip("PySide6")
    from src.ui.widgets.video_preview import VideoPreviewSurface

    names = list(inspect.signature(VideoPreviewSurface.set_frame).parameters)
    assert names == ["self", "image", "gui_ms", "index", "seq", "media_time", "generation"]
    assert all(inspect.signature(VideoPreviewSurface.set_frame).parameters[name].kind
               is inspect.Parameter.KEYWORD_ONLY for name in names[2:])
    planes_names = list(inspect.signature(VideoPreviewSurface.set_frame_planes).parameters)
    assert planes_names[:5] == ["self", "planes", "strides", "width", "height"]


def test_the_renderer_accepts_both_shapes_and_refuses_a_broken_plane_set():
    pytest.importorskip("PySide6")
    from src.ui.widgets.preview_renderer import PreviewFrame, PreviewRenderer

    renderer = PreviewRenderer()
    planes, strides = _planes_for(8, 8)
    renderer.set_frame(PreviewFrame(format="yuv420p10le", width=8, height=8, planes=planes, strides=strides))
    assert renderer._frame.format == "yuv420p10le"

    with pytest.raises(ValueError):
        renderer.set_frame(PreviewFrame(format="yuv420p10le", width=8, height=8, planes=planes[:2],
                                        strides=strides[:2]))
    with pytest.raises(ValueError):
        # A short plane is refused rather than uploaded as a half-black picture.
        renderer.set_frame(PreviewFrame(format="yuv420p10le", width=8, height=8,
                                        planes=(planes[0][:-8], planes[1], planes[2]), strides=strides))
    with pytest.raises(NotImplementedError):
        renderer.set_frame(PreviewFrame(format="nv12", width=8, height=8))
    with pytest.raises(ValueError):
        renderer.set_frame(PreviewFrame(format="rgba8", width=8, height=8))


def test_a_10_bit_plane_set_uploads_three_bytes_a_pixel_and_rgba_uploads_four():
    pytest.importorskip("PySide6")
    from src.ui.widgets.preview_renderer import PreviewFrame, PreviewRenderer

    planes, strides = _planes_for(320, 180)
    ten_bit = PreviewFrame(format="yuv420p10le", width=320, height=180, planes=planes, strides=strides)
    eight_bit = PreviewFrame(format="yuv420p", width=320, height=180, planes=planes, strides=strides)
    rgba = PreviewFrame(format="rgba8", width=320, height=180,
                        image=__import__("PySide6.QtGui", fromlist=["QImage"]).QImage(
                            320, 180, __import__("PySide6.QtGui", fromlist=["QImage"]).QImage.Format.Format_RGBA8888))

    assert PreviewRenderer._pixel_bytes(ten_bit) == 320 * 180 * 3
    assert PreviewRenderer._pixel_bytes(eight_bit) == 320 * 180 * 3 // 2
    assert PreviewRenderer._pixel_bytes(rgba) == 320 * 180 * 4


# -- the pass, against a fake RHI (offscreen has no QRhi at all) -----------------------------------
#
# What this can prove without a GPU: that a planes frame is uploaded as THREE plane entries with their own strides, that
# all of it rides ONE resource-update batch which the pass consumes, and that the pass draws once. What it cannot prove
# is what the sampler returns - that is the supervisor's live RHI proof.


class _Handle:
    """A stand-in for a QRhi object the renderer may destroy (a real texture/SRB/pipeline)."""

    def __init__(self) -> None:
        self.destroyed = False

    def destroy(self) -> None:
        self.destroyed = True


class _FakeTexture:
    def __init__(self, fmt, size, plane: bool = False) -> None:
        self._fmt, self._size = fmt, size
        self.plane = plane
        self.destroyed = False

    def format(self):
        return self._fmt

    def pixelSize(self):  # noqa: N802 - QRhi spelling
        return self._size

    def setPixelSize(self, size) -> None:  # noqa: N802
        self._size = size

    def create(self) -> bool:
        return True

    def destroy(self) -> None:
        self.destroyed = True


class _FakeBatch:
    def __init__(self) -> None:
        self.uploads: list[tuple[object, object]] = []
        self.uniform_writes: list[tuple[int, int, int]] = []

    def uploadTexture(self, texture, description) -> None:  # noqa: N802 - QRhi spelling
        self.uploads.append((texture, description))

    def updateDynamicBuffer(self, _buffer, offset, size, _data) -> None:  # noqa: N802
        self.uniform_writes.append((offset, size, len(_data)))

    def generateMips(self, texture) -> None:  # noqa: N802
        # A plane texture carries no mip chain: reaching this from the planes path is a bug, and the RGBA
        # path is expected to reach it once per upload.
        if getattr(texture, "plane", False):
            raise AssertionError("a plane texture has no mip chain: generateMips here is a bug")
        self.mips = getattr(self, "mips", 0) + 1


class _FakeRhi:
    def __init__(self) -> None:
        self.batches: list[_FakeBatch] = []

    def nextResourceUpdateBatch(self):  # noqa: N802
        batch = _FakeBatch()
        self.batches.append(batch)
        return batch


class _FakeTarget:
    def __init__(self, width: int, height: int) -> None:
        from PySide6.QtCore import QSize

        self._size = QSize(width, height)

    def pixelSize(self):  # noqa: N802
        return self._size


class _FakeCommandBuffer:
    def __init__(self) -> None:
        self.passes = 0
        self.draws = 0
        self.pipelines: list[object] = []
        self.bindings: list[object] = []

    def beginPass(self, *_args) -> None:  # noqa: N802
        self.passes += 1

    def endPass(self) -> None:  # noqa: N802
        self.passes += 1

    def setGraphicsPipeline(self, pipeline) -> None:  # noqa: N802
        self.pipelines.append(pipeline)

    def setViewport(self, *_args) -> None:  # noqa: N802
        pass

    def setScissor(self, *_args) -> None:  # noqa: N802
        pass

    def setShaderResources(self, bindings) -> None:  # noqa: N802
        self.bindings.append(bindings)

    def setVertexInput(self, *_args) -> None:  # noqa: N802
        pass

    def draw(self, count: int) -> None:
        self.draws += 1


def test_one_pass_uploads_the_three_planes_with_their_strides_and_draws_once():
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QRectF, QSize
    from PySide6.QtGui import QColor

    from src.ui.widgets.preview_renderer import PreviewFrame, PreviewRenderer, PreviewScene

    renderer = PreviewRenderer()
    width, height = 16, 12
    planes, strides = _planes_for(width, height, fill=9)
    frame = PreviewFrame(format="yuv420p10le", width=width, height=height,
                         planes=tuple(memoryview(plane) for plane in planes), strides=strides,
                         color_space="bt709", color_range="tv")
    fmt = __import__("PySide6.QtGui", fromlist=["QRhiTexture"]).QRhiTexture.Format.R16
    (chroma_w, chroma_h) = ((width + 1) // 2, (height + 1) // 2)
    renderer._rhi = _FakeRhi()
    renderer._deferred = False
    renderer._yuv_uniforms = _Handle()
    renderer._plane_textures = [_FakeTexture(fmt, QSize(width, height), plane=True),
                                _FakeTexture(fmt, QSize(chroma_w, chroma_h), plane=True),
                                _FakeTexture(fmt, QSize(chroma_w, chroma_h), plane=True)]
    sentinel = _Handle()
    renderer._yuv_bindings = sentinel
    renderer._yuv_bindings_nearest = sentinel
    renderer._yuv_pipeline = sentinel
    renderer._vertices = _Handle()
    renderer._hud_pipeline = None
    target = _FakeTarget(320, 180)
    renderer.set_frame(frame)
    renderer.set_scene(PreviewScene(
        target=target, rhi=renderer._rhi, mode="single",
        result_mvp=(1.0,) * 16, original_mvp=None, tint=(1.0, 1.0, 1.0, 1.0), nearest=False,
        result_rect=QRectF(0, 0, 320, 180), result_clip=QRectF(0, 0, 320, 180),
        original_clip=QRectF(), show_original=False, hud_vertices=(),
        viewport=(320.0, 180.0), dpr=1.0, clear=QColor(0, 0, 0)))

    command_buffer = _FakeCommandBuffer()
    renderer.render(command_buffer)

    batch = renderer._rhi.batches[-1]
    assert len(batch.uploads) == 3, "Y, U and V in one batch, not one texture RGB-packed"
    uploaded_strides = [description.entryAt(0).description().dataStride()
                        for _texture, description in batch.uploads]
    assert uploaded_strides == list(strides), "each plane keeps its own byte stride"
    assert batch.uniform_writes == [(0, 160, 160)], "mvp + tint + the matrix affine in one 160-byte write"
    assert command_buffer.passes == 2, "one beginPass and one endPass"
    assert command_buffer.draws == 1
    assert command_buffer.pipelines == [sentinel], "the planes pipeline, not the RGBA one"
    assert command_buffer.bindings == [sentinel]


def test_the_rgba_fallback_still_uploads_one_image_and_generates_mips():
    """The fallback path is byte-identical in shape: one image upload, its mip chain, one draw - no planes involved."""
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QRectF, QSize
    from PySide6.QtGui import QColor, QImage

    from src.ui.widgets.preview_renderer import PreviewFrame, PreviewRenderer, PreviewScene

    renderer = PreviewRenderer()
    sentinel = _Handle()
    renderer._rhi = _FakeRhi()
    renderer._deferred = False
    renderer._uniforms = _Handle()
    renderer._texture = _FakeTexture(None, QSize(16, 12))
    renderer._bindings = sentinel
    renderer._bindings_nearest = sentinel
    renderer._pipeline = sentinel
    renderer._vertices = _Handle()
    renderer._hud_pipeline = None
    image = QImage(16, 12, QImage.Format.Format_RGBA8888)
    image.fill(QColor(1, 2, 3))
    renderer.set_frame(PreviewFrame(format="rgba8", width=16, height=12, image=image))
    renderer.set_scene(PreviewScene(
        target=_FakeTarget(320, 180), rhi=renderer._rhi, mode="single",
        result_mvp=(1.0,) * 16, original_mvp=None, tint=(1.0, 1.0, 1.0, 1.0), nearest=False,
        result_rect=QRectF(0, 0, 320, 180), result_clip=QRectF(0, 0, 320, 180),
        original_clip=QRectF(), show_original=False, hud_vertices=(),
        viewport=(320.0, 180.0), dpr=1.0, clear=QColor(0, 0, 0)))
    command_buffer = _FakeCommandBuffer()
    renderer.render(command_buffer)

    batch = renderer._rhi.batches[-1]
    assert len(batch.uploads) == 1 and batch.uploads[0][0] is renderer._texture
    assert batch.uniform_writes == [(0, 64, 64), (64, 16, 16)], "the RGBA block is untouched (mvp + tint)"
    assert command_buffer.draws == 1


def test_the_compare_modes_draw_a_planes_result_and_an_rgb24_source_in_one_pass():
    """`sbs`/`split` may carry one of each shape: the pipeline is stated per draw, and both uploads share the batch.

    This is the seam a merge breaks first - the UI lane's compare keeps the SOURCE decoder on rgb24 while the result
    becomes planes, so the two halves of one pass genuinely have different SRB layouts.
    """
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QRectF, QSize
    from PySide6.QtGui import QColor, QImage

    from src.ui.widgets.preview_renderer import PreviewFrame, PreviewRenderer, PreviewScene

    renderer = PreviewRenderer()
    width, height = 16, 12
    planes, strides = _planes_for(width, height, fill=4)
    result = PreviewFrame(format="yuv420p10le", width=width, height=height,
                          planes=tuple(memoryview(plane) for plane in planes), strides=strides)
    image = QImage(width, height, QImage.Format.Format_RGBA8888)
    image.fill(QColor(3, 4, 5))
    original = PreviewFrame(format="rgba8", width=width, height=height, image=image)

    fmt = __import__("PySide6.QtGui", fromlist=["QRhiTexture"]).QRhiTexture.Format.R16
    renderer._rhi = _FakeRhi()
    renderer._deferred = False
    renderer._mode = "sbs"
    yuv_sentinel, rgba_sentinel = _Handle(), _Handle()
    renderer._yuv_uniforms = _Handle()
    renderer._original_yuv_uniforms = _Handle()
    renderer._uniforms = _Handle()
    renderer._original_uniforms = _Handle()
    renderer._plane_textures = [_FakeTexture(fmt, QSize(width, height), plane=True),
                                _FakeTexture(fmt, QSize((width + 1) // 2, (height + 1) // 2), plane=True),
                                _FakeTexture(fmt, QSize((width + 1) // 2, (height + 1) // 2), plane=True)]
    renderer._texture = _FakeTexture(None, QSize(width, height))
    renderer._original_texture = _FakeTexture(None, QSize(width, height))
    renderer._yuv_bindings = renderer._yuv_bindings_nearest = yuv_sentinel
    renderer._bindings = renderer._bindings_nearest = rgba_sentinel
    renderer._original_bindings = renderer._original_bindings_nearest = _Handle()
    renderer._yuv_pipeline = yuv_sentinel
    renderer._pipeline = rgba_sentinel
    renderer._vertices = _Handle()
    renderer._hud_pipeline = None
    renderer.set_frame(result)
    renderer.set_original(original)
    renderer.set_scene(PreviewScene(
        target=_FakeTarget(320, 180), rhi=renderer._rhi, mode="sbs",
        result_mvp=(1.0,) * 16, original_mvp=(1.0,) * 16, tint=(1.0, 1.0, 1.0, 1.0), nearest=False,
        result_rect=QRectF(160, 0, 160, 180), result_clip=QRectF(160, 0, 160, 180),
        original_clip=QRectF(0, 0, 160, 180), show_original=False, hud_vertices=(),
        viewport=(320.0, 180.0), dpr=1.0, clear=QColor(0, 0, 0)))
    command_buffer = _FakeCommandBuffer()
    renderer.render(command_buffer)

    batch = renderer._rhi.batches[-1]
    plane_uploads = [texture for texture, _desc in batch.uploads if getattr(texture, "plane", False)]
    image_uploads = [texture for texture, _desc in batch.uploads if not getattr(texture, "plane", False)]
    assert len(plane_uploads) == 3, "the result's three planes"
    assert image_uploads == [renderer._original_texture], "the source's one RGBA image, in the same batch"
    assert command_buffer.passes == 2 and command_buffer.draws == 2
    assert command_buffer.pipelines == [yuv_sentinel, rgba_sentinel], "the pipeline is stated per draw"


def test_the_chroma_sampler_is_never_nearest_even_when_the_luma_one_is():
    """A planes picture at 1:1 magnifies nothing in luma but still upsamples chroma 2x - so only luma may go nearest.

    The chroma plane is half the picture's size on every draw, so a nearest tap there paints 2x2 colour blocks.
    Measured at fit 1:1 on the 20 s preview (2026-11-25): nearest chroma cost 3.37% of pixels >8 levels off the RGBA
    path on detailed frames vs 1.43% filtered; against zimg 0.50% vs 0.40%. This pins the policy that the two
    samplers are chosen per role, not one per view.
    """
    pytest.importorskip("PySide6")
    from src.ui.widgets.preview_renderer import plane_samplers

    assert plane_samplers(nearest=True) == (True, False), "luma nearest, chroma filtered"
    assert plane_samplers(nearest=False) == (False, False), "the fit view filters both"
