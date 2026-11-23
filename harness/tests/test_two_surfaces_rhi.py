"""The two surfaces, proven in pixels: what `single`, `sbs` and `split` actually paint.

Why this gate exists (2026-11-24). `set_view_mode` shipped with 291 lines of offscreen tests
(`test_video_preview_two_surfaces.py`) that pin the arithmetic - which rect each picture gets, where the strip's hit test
lands, that both halves come from one zoom. None of it proves the part that can only be wrong in the GPU pass: the
second texture, the second SRB and the scissor. Offscreen has **no QRhi** (`VideoPreviewSurface.initialize` says so and
returns, `grabFrame()` is a null readback), so the suite was green while "the original's half samples the result's
texture", "the source was never uploaded", "the scissor leaked into the far half" were all untested.

What is measured, and with which instrument: `grabFrame()` -> `grabFramebuffer()` reads back the widget's own backing
texture (`harness/docs/LIVE-TESTING.md`; the preview plan's readback note in `video_preview.py`), i.e. the surface's
**content**. It is not the composite: a root-window screenshot is what proves the widget is placed and composited in
the real window, and this file cannot answer that (`harness/scripts/live.sh shot` does).

How the assertion is built so it cannot pass by accident:

* the pane is 640x180 and the frames are 320x180, so each `sbs` half is exactly the frame's own 16:9 - the sampled
  grid lies inside the picture, not in a letterbox band, and there is no guessed "half" anywhere: the rects come from
  the widget's own `_view_rect`/`_original_view_rect`/`_frame_rect`/`_original_clip_rect`;
* the two pictures are solid, distinguishable colours (result `rgb(200, 30, 30)`, source `rgb(30, 160, 40)`) and every
  one of 25 points per region must be the expected one, +-16 per channel for upload/readback rounding; a failure names
  the device pixel, the sampled rgb and the expected rgb;
* widget coordinates are converted to device pixels with the readback's own device pixel ratio, so the same test is
  valid under `QT_SCALE_FACTOR=2`;
* the strip is switched off for the measurement. It is chrome painted over the picture rect (its own HUD tests cover
  it); left on, it would mask picture pixels and turn a content assertion into a "mostly content" one.

The skip is deliberate and narrow: a cell without QRhi gets `pytest.skip("no QRhi on this platform...")` **before any
colour assertion**, and nothing else skips - an import error, a null-but-present readback of the wrong size, or a wrong
colour fails. The comparison itself has its own CPU red run in this file (`test_the_comparison_...`), so a green here
does not rest on an assertion nobody ever saw fail.

Run it on a leased display (never the owner's screen):

    DISPLAY=<leased> QT_QPA_PLATFORM=xcb LIBGL_ALWAYS_SOFTWARE=1 RVE_UI_RHI_BACKEND=opengl \\
    RVE_RHI_PROOF_DIR=/path/to/proof \\
    python3 -m pytest harness/tests/test_two_surfaces_rhi.py -q -o addopts="" -p no:cacheprovider

With `RVE_RHI_PROOF_DIR` set, each grabbed picture is saved as `<mode>-<dpr>x.png`; unset, nothing is written.
"""

from __future__ import annotations

import os
import time

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QRectF, QSize  # noqa: E402
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget  # noqa: E402

from src.ui.rhi_surface import configure_rhi_widget  # noqa: E402
from src.ui.widgets.video_preview import VIEW_MODES, VideoPreviewSurface  # noqa: E402

#: The two pictures, distinguishable at a glance and far outside the tolerance from each other.
RESULT_COLOUR = (200, 30, 30)
ORIGINAL_COLOUR = (30, 160, 40)

#: The frame each picture carries. 320x180 (16:9) is the shape a 720p/1080p render has.
FRAME_SIZE = (320, 180)

#: The pane. 640x180 is 32:9, so ONE HALF of `sbs` is exactly 320x180: each picture fills its half at scale 1.0 and the
#: sample grid never lands in a letterbox band. A different aspect would make the halves taller than the frame and the
#: corners of a half would legitimately be the clear colour, which is a weaker assertion than the one wanted here.
PANE_SIZE = (640, 180)

#: Points per axis per region: 25 samples, enough to catch a scissor that leaks, a texture that was never uploaded, or
#: a half drawn with the other picture - and cheap enough to run 3 modes x 2 scales.
GRID = 5

#: +-16 per channel: upload to RGBA8 and readback are exact for a solid colour, so this only absorbs a rounding step in
#: the compositing path. It is far below the distance between the two test colours (|200-30| = 170).
TOLERANCE = 16

#: How long `grabFrame()` may stay null or mis-sized before the verdict (llvmpipe's first frame is the slow one).
GRAB_TIMEOUT_S = 15.0

#: Platforms whose Qt integration has no RHI at all, so the readback is null by construction. Waiting the full timeout
#: there only costs 15 s per mode and tells nobody anything (`QT_QPA_PLATFORM=offscreen` is the cell's platform).
RHI_LESS_PLATFORMS = ("offscreen", "minimal", "minimalegl")

#: Sleep between readback attempts: the X server needs a round trip to expose/allocate the widget's target.
POLL_S = 0.02

#: The environment variable the supervisor uses to keep the proof pictures; unset means "write nothing".
PROOF_DIR_ENV = "RVE_RHI_PROOF_DIR"


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _picture(width: int, height: int, colour: tuple[int, int, int]) -> QImage:
    """A solid RGB32 picture - the format the pane is fed in the app, not the RGBA it uploads."""
    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(QColor(*colour))
    return image


def _settle(app: QApplication, turns: int = 5) -> None:
    """Let Qt deliver show/resize/expose and let X round-trip before a readback is asked for."""
    for _ in range(turns):
        app.processEvents()
        time.sleep(POLL_S)


def _device_ratio(image: QImage) -> float:
    """Widget pixels -> readback pixels.

    `QRhiWidget::grabFramebuffer()` returns the colour texture's pixels (widget size * device pixel ratio) and stamps
    that ratio on the QImage (`qrhiwidget.cpp`: `result.setDevicePixelRatio(q->devicePixelRatio())`). Reading it from
    the image rather than assuming 1.0 is what makes the same grid correct under `QT_SCALE_FACTOR=2`.
    """
    return float(image.devicePixelRatio() or 1.0)


def _grid_points(rect: QRectF, dpr: float, image: QImage, samples: int = GRID):
    """`samples` x `samples` device-pixel points inside a widget-space rect, with the sampled rgb.

    The points sit at the centre of equal cells, so no sample touches a rect edge: the edge is where the scissor and
    the divider live, and a pixel of rounding there would be an instrument artefact rather than a wrong picture.
    """
    points = []
    for row in range(samples):
        for column in range(samples):
            x = rect.left() + rect.width() * (column + 0.5) / samples
            y = rect.top() + rect.height() * (row + 0.5) / samples
            px = min(image.width() - 1, max(0, int(round(x * dpr))))
            py = min(image.height() - 1, max(0, int(round(y * dpr))))
            colour = image.pixelColor(px, py)
            points.append((px, py, (colour.red(), colour.green(), colour.blue())))
    return points


def _off_by(rgb: tuple[int, int, int], expected: tuple[int, int, int]) -> bool:
    return any(abs(sampled - wanted) > TOLERANCE for sampled, wanted in zip(rgb, expected))


def _mismatches(image: QImage, rect: QRectF, expected: tuple[int, int, int], dpr: float,
                label: str, samples: int = GRID) -> list[str]:
    """The comparison itself: one row per sample that is not the expected colour."""
    rows = []
    for px, py, rgb in _grid_points(rect, dpr, image, samples):
        if _off_by(rgb, expected):
            rows.append(f"{label}: device ({px}, {py}) samples rgb{rgb}, expected rgb{expected}")
    return rows


def _assert_region(image: QImage, rect: QRectF, expected: tuple[int, int, int], dpr: float,
                   label: str) -> None:
    """Every grid point inside `rect` is `expected`; failure reports each sampled rgb, not just a count."""
    assert not rect.isEmpty(), f"{label}: the widget gave an empty rect - there is nothing to sample"
    rows = _mismatches(image, rect, expected, dpr, label)
    assert not rows, (
        f"{label}: {len(rows)} of {GRID * GRID} samples are not rgb{expected} (tolerance +-{TOLERANCE})\n  "
        + "\n  ".join(rows))


def _dump_proof(image: QImage, mode: str, dpr: float) -> str | None:
    """Save the grabbed picture when the supervisor asked for artifacts; write nothing otherwise."""
    directory = os.environ.get(PROOF_DIR_ENV)
    if not directory:
        return None
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{mode}-{dpr:g}x.png")
    assert image.save(path), f"could not save the proof image to {path}"
    print(f"[rhi-proof] {mode} @ {dpr:g}x -> {path}", flush=True)
    return path


def _build_pane(app: QApplication, mode: str) -> tuple[QWidget, VideoPreviewSurface]:
    """A shown top-level window holding the pane at `PANE_SIZE`, fed with the two pictures.

    The surface is configured before it joins the layout: `QRhiWidget.setApi` only takes effect before the widget is
    added to a displayed hierarchy, which is what `RVE_UI_RHI_BACKEND=opengl` on a lease needs.
    """
    window = QWidget()
    window.setObjectName("twoSurfacesRhiProof")
    surface = VideoPreviewSurface()
    surface.setObjectName("previewLabel")
    configure_rhi_widget(surface)
    surface.set_frame(_picture(*FRAME_SIZE, RESULT_COLOUR))
    surface.set_original(_picture(*FRAME_SIZE, ORIGINAL_COLOUR))
    # The strip and the divider glyph are chrome over the picture rect; the picture content is what this file measures
    # (the HUD has its own tests). Leaving it on would put a 6 px track across the sampled area.
    surface.set_strip_visible(False)

    layout = QVBoxLayout(window)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    layout.addWidget(surface)

    window.resize(*PANE_SIZE)
    window.show()
    _settle(app)
    assert surface.size() == QSize(*PANE_SIZE), (
        f"the pane came out {surface.width()}x{surface.height()}, not {PANE_SIZE[0]}x{PANE_SIZE[1]}: the grid is built "
        "on a 32:9 pane whose halves are each the frame's own 16:9 (so no sample lands in a letterbox band)")
    surface.set_view_mode(mode)
    _settle(app, turns=2)
    return window, surface


def _grab_frame(app: QApplication, surface: VideoPreviewSurface) -> QImage:
    """Pump until the readback is a non-null image of the pane's own pixel size, or skip on "no QRhi".

    A null readback is the offscreen platform's honest answer (`initialize()` saw no QRhi and returned; `grabFramebuffer`
    cannot create a dedicated one either), so it is a SKIP - but only a null one. A non-null image of the wrong size is
    an instrument that is reading somebody else's target, and it fails.
    """
    # A platform without an RHI integration answers "null" immediately and forever; on a real display the first paint
    # can lose the backingstore-promotion race (`src/ui/rhi_surface.py`), so only there is the full timeout spent.
    deadline = time.monotonic() + (0.2 if QGuiApplication.platformName() in RHI_LESS_PLATFORMS else GRAB_TIMEOUT_S)
    last = None
    while time.monotonic() < deadline:
        app.processEvents()
        image = surface.grabFrame()
        if image is not None and not image.isNull():
            last = image
            dpr = _device_ratio(image)
            expected = (surface.width() * dpr, surface.height() * dpr)
            if abs(image.width() - expected[0]) <= 1 and abs(image.height() - expected[1]) <= 1:
                return image
        time.sleep(POLL_S)
    if last is None:
        pytest.skip("no QRhi on this platform - run on a lease display")
    dpr = _device_ratio(last)
    pytest.fail(
        f"grabFrame() returned {last.width()}x{last.height()} (device pixel ratio {dpr:g}) for a pane of "
        f"{surface.width()}x{surface.height()}: the readback is not the pane's own {surface.width() * dpr:g}x"
        f"{surface.height() * dpr:g} target")


def _assert_single(image: QImage, surface: VideoPreviewSurface, dpr: float) -> None:
    picture = surface._frame_rect()
    assert not picture.isEmpty(), "single: the pane reports no picture rect"
    assert picture.width() <= surface.width() + 0.5 and picture.height() <= surface.height() + 0.5, (
        f"single: the picture {picture} does not fit the {surface.width()}x{surface.height()} pane")
    _assert_region(image, picture, RESULT_COLOUR, dpr, "single: the result's picture")


def _assert_sbs(image: QImage, surface: VideoPreviewSurface, dpr: float) -> None:
    """The left surface is the source, the right one is the result - at the widget's own half rects.

    The rects are the pane's own geometry (`_original_view_rect`/`_view_rect`), not `width()/2` guessed here: if the
    pane ever lays the halves out differently, this test measures what it draws instead of what the test imagined. The
    picture rects are then required to FILL those halves, which is the premise that makes a full-half grid meaningful.
    """
    middle = surface.width() / 2.0
    left = surface._original_view_rect()
    right = surface._view_rect()
    assert left.right() <= middle + 0.5 and right.left() >= middle - 0.5, (
        f"sbs: the widget's own halves are {left} and {right}, which do not split the {surface.width()} px pane")
    assert surface._original_rect() == left, (
        f"sbs: the source is placed at {surface._original_rect()}, not its own surface {left} - at this pane size each "
        "picture is exactly its half, so a difference means placement changed")
    assert surface._frame_rect() == right, (
        f"sbs: the result is placed at {surface._frame_rect()}, not its own surface {right}")

    _assert_region(image, left, ORIGINAL_COLOUR, dpr, "sbs: the left surface (the source)")
    _assert_region(image, right, RESULT_COLOUR, dpr, "sbs: the right surface (the result)")


def _assert_split(image: QImage, surface: VideoPreviewSurface, dpr: float) -> None:
    """One picture, the source left of the divider: sample on both sides of the widget's own clip rect."""
    picture = surface._frame_rect()
    clip = surface._original_clip_rect()
    assert not picture.isEmpty() and not clip.isEmpty(), "split: the pane reports no picture or no clip"
    assert clip.left() == pytest.approx(picture.left()) and clip.top() == pytest.approx(picture.top()), (
        f"split: the source is clipped at {clip}, which does not start with the picture {picture}")
    assert clip.width() == pytest.approx(picture.width() / 2.0), (
        f"split: the source's clip is {clip.width():g} px wide, not half of the picture's {picture.width():g} px")

    right = QRectF(clip.right(), picture.top(), max(0.0, picture.right() - clip.right()), picture.height())
    _assert_region(image, clip, ORIGINAL_COLOUR, dpr, "split: left of the divider (the source)")
    _assert_region(image, right, RESULT_COLOUR, dpr, "split: right of the divider (the result)")


@pytest.mark.parametrize("mode", VIEW_MODES)
def test_the_two_surfaces_paint_their_own_pictures(mode):
    """The gap this file closes: each surface shows the picture it is named for, in every mode.

    Skips without QRhi (a cell); on a lease display it asserts 25 points per surface per mode, and under
    `QT_SCALE_FACTOR=2` it asserts the same points in device pixels.
    """
    app = _app()
    window, surface = _build_pane(app, mode)
    try:
        assert surface.view_mode() == mode
        image = _grab_frame(app, surface)
        dpr = _device_ratio(image)
        _dump_proof(image, mode, dpr)          # before the assertions: a red run still leaves the picture to look at
        if mode == "single":
            _assert_single(image, surface, dpr)
        elif mode == "sbs":
            _assert_sbs(image, surface, dpr)
        else:
            _assert_split(image, surface, dpr)
    finally:
        window.close()


# -- the comparison's own red run (pure CPU, no QRhi) ------------------------------------------------


def _two_halves(dpr: float = 1.0) -> QImage:
    """A synthetic picture: source on the left half, result on the right, exactly the sbs layout."""
    width, height = 200, 100
    image = QImage(int(width * dpr), int(height * dpr), QImage.Format.Format_RGBA8888)
    image.fill(QColor(*RESULT_COLOUR))
    painter = QPainter(image)
    painter.fillRect(0, 0, int(width * dpr / 2), int(height * dpr), QColor(*ORIGINAL_COLOUR))
    painter.end()
    image.setDevicePixelRatio(dpr)
    return image


def test_the_comparison_rejects_the_wrong_picture_on_a_synthetic_image():
    """A gate that cannot fail is a comment: the same `_assert_region` the RHI test uses must refuse a swapped half.

    This is the red run that does not need a GPU - the exact comparison, the exact `+-16` tolerance and the exact
    rect-to-device conversion, on a picture whose halves are known. Swap the two colours here and the failure names
    the sampled rgb, which is what an operator sees when the pane paints the wrong picture on a lease.
    """
    image = _two_halves()
    left = QRectF(0.0, 0.0, 100.0, 100.0)
    right = QRectF(100.0, 0.0, 100.0, 100.0)

    _assert_region(image, left, ORIGINAL_COLOUR, 1.0, "synthetic left")
    _assert_region(image, right, RESULT_COLOUR, 1.0, "synthetic right")

    with pytest.raises(AssertionError) as swapped:
        _assert_region(image, left, RESULT_COLOUR, 1.0, "synthetic left, the result colour")
    assert "samples rgb(30, 160, 40)" in str(swapped.value), (
        f"the refusal must name the sampled colour, it said: {swapped.value}")
    assert "expected rgb(200, 30, 30)" in str(swapped.value)

    with pytest.raises(AssertionError):
        _assert_region(image, right, ORIGINAL_COLOUR, 1.0, "synthetic right, the source colour")


def test_the_grid_reads_device_pixels_under_a_scale_factor_of_2():
    """`QT_SCALE_FACTOR=2` must not move the grid onto the neighbouring half: widget x 100..200 is device x 200..400.

    Without the ratio the same points would land at device x 100..200 - the source's half - so expecting the result
    there would fail on a correct picture. That is the exact false red this conversion exists to prevent.
    """
    image = _two_halves(dpr=2.0)
    assert image.width() == 400 and image.height() == 200, "the synthetic HiDPI picture is widget size x 2"
    widget_right = QRectF(100.0, 0.0, 100.0, 100.0)

    points = _grid_points(widget_right, 2.0, image)
    assert points, "the grid produced no points"
    assert all(200 <= px < 400 for px, _py, _rgb in points), (
        f"a widget point of the right half must read the device pixels 200..400, got {[px for px, _, _ in points]}")
    _assert_region(image, widget_right, RESULT_COLOUR, 2.0, "synthetic right half at 2x")

    with pytest.raises(AssertionError):
        _assert_region(image, widget_right, ORIGINAL_COLOUR, 2.0, "synthetic right half at 2x, the source colour")
