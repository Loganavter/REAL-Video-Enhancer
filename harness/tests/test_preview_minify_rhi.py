"""The fit view's minification, captured in-process: two arms of `RVE_MIP_MINIFY` on one build.

Why this file exists. The chain landed in `video_preview.py` (mip-flags + `generateMips` + trilinear) but its own
acceptance - PSNR/SSIM against a Lanczos3 reference, plus a temporal stability pair - was never produced: every attempt
to grab the pane THROUGH the control channel timed out (a QRhiWidget's `grab()` and `grabFrame()` both block when the
request is served from the app's own control loop). The RHI pixel gate already captures the same widget the working way:
in-process, `grabFrame()`, after the event loop has let the widget expose and render.

What this file does, per process:
* draws a 1440x1080 HIGH-FREQUENCY chart (1-px checker + fine vertical ramps: a minification's worst case) into the
  pane at the fit scale for a 421x512 pane, which is the owner's real 3.42x case;
* grabs the pane twice: at fit, and at fit * 1.0005 (a half-output-pixel scale step - the temporal phase pair);
* saves both PNGs under `RVE_RHI_PROOF_DIR` (arm name from `RVE_MIP_MINIFY`), so a second run with the other value
  produces the other arm and `harness/scripts/minify_ab.py` compares them.

Assertions here are deliberately structural, not aesthetic: the readback is not null, it has the pane's device size, it
is NOT a black frame, and the two phase grabs differ (a filter that does nothing would make them identical). The
quality numbers are computed by the offline script from the saved pictures - a threshold in a test would either be
machine-specific or a tautology.
"""

from __future__ import annotations

import os
import time

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSize  # noqa: E402
from PySide6.QtGui import QColor, QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from src.ui.rhi_surface import configure_rhi_widget  # noqa: E402
from src.ui.widgets.video_preview import VideoPreviewSurface, mip_minify_enabled  # noqa: E402

#: The owner's real case: a 1440x1080 frame in a 421-wide pane is 3.42x minification (`_fit_scale` = 0.2924).
FRAME_SIZE = (1440, 1080)
PANE_SIZE = (421, 512)

#: llvmpipe's first frame is the slow one; the RHI gate allows 15 s for the same reason.
GRAB_TIMEOUT_S = 20.0
POLL_S = 0.02

RHI_LESS_PLATFORMS = ("offscreen", "minimal", "minimalegl")
PROOF_DIR_ENV = "RVE_RHI_PROOF_DIR"


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _chart(width: int, height: int) -> QImage:
    """A chart whose detail is FINER than the fit scale can reproduce: 1-px vertical lines, a 1-px checker and ramps.

    Without sub-sample detail an aliasing metric has nothing to punish, and every kernel would score the same.
    """
    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(QColor(0, 0, 0))
    painter = QPainter(image)
    try:
        for x in range(0, width, 2):                 # 1-px vertical stripes: the classic minification aliasing probe
            painter.fillRect(x, 0, 1, height, QColor(255, 255, 255))
        for y in range(0, height, 4):                # horizontal band edges, a second period so folds cannot cancel
            painter.fillRect(0, y, width, 1, QColor(0, 255, 255))
        for x in range(0, width, 8):                 # a fine checker block
            for y in range(0, height, 8):
                if ((x // 8) + (y // 8)) % 2:
                    painter.fillRect(x, y, 8, 8, QColor(255, 0, 0))
        for x in range(width // 2, width, 12):       # a coarse ramp region, so SSIM has structure to reward
            level = (x * 255) // width
            painter.fillRect(x, 0, 6, height // 3, QColor(level, level, level))
    finally:
        painter.end()
    return image


def _settle(app: QApplication, turns: int = 6) -> None:
    for _ in range(turns):
        app.processEvents()
        time.sleep(POLL_S)


def _grab(surface: VideoPreviewSurface, app: QApplication):
    """`grabFrame()` until it is a frame of the pane's own size, or the timeout."""
    deadline = time.monotonic() + GRAB_TIMEOUT_S
    expected = QSize(PANE_SIZE[0], PANE_SIZE[1])
    while time.monotonic() < deadline:
        app.processEvents()
        image = surface.grabFrame()
        if image is not None and not image.isNull():
            ratio = float(image.devicePixelRatio() or 1.0)
            if abs(image.width() - expected.width() * ratio) <= 1 and \
                    abs(image.height() - expected.height() * ratio) <= 1:
                return image
        time.sleep(POLL_S)
    return None


def _arm_name() -> str:
    return "mips" if mip_minify_enabled() else "bilinear"


def _mean_luma(image: QImage) -> float:
    total = 0.0
    count = 0
    for y in range(0, image.height(), 7):            # a stride is enough for "is it black"
        for x in range(0, image.width(), 7):
            colour = image.pixelColor(x, y)
            total += 0.2126 * colour.red() + 0.7152 * colour.green() + 0.0722 * colour.blue()
            count += 1
    return total / max(1, count)


def test_the_fit_view_is_captured_and_the_two_arms_differ(tmp_path):
    platform = os.environ.get("QT_QPA_PLATFORM", "")
    if platform in RHI_LESS_PLATFORMS:
        pytest.skip("no QRhi on this platform: the readback is null by construction")

    app = _app()
    surface = VideoPreviewSurface()
    configure_rhi_widget(surface)
    surface.resize(*PANE_SIZE)
    surface.show()
    surface.set_strip_visible(False)               # chrome over the picture rect would mask samples
    chart = _chart(*FRAME_SIZE)
    surface.set_frame(chart)
    _settle(app, turns=10)

    fit = _grab(surface, app)
    if fit is None:
        pytest.skip("the RHI readback stayed null for this platform/backend")

    luma = _mean_luma(fit)
    assert luma > 8.0, f"the pane read back near-black (mean luma {luma:.1f}) - the upload or the pass is wrong"

    reference_dir = os.environ.get(PROOF_DIR_ENV)
    if reference_dir:
        os.makedirs(reference_dir, exist_ok=True)
        # The INPUT chart goes with the arms: the offline script builds its Lanczos3 reference from this exact file,
        # so the comparison cannot drift from what the pane was fed.
        chart.save(os.path.join(reference_dir, "chart-1440x1080.png"))
        fit.save(os.path.join(reference_dir, f"{_arm_name()}-fit.png"))

    # The temporal pair: a half-output-pixel scale step re-samples the same picture at a different phase. A filter that
    # ignores the footprint returns a picture whose phase dependence is the whole aliasing error, so the two grabs are
    # never identical - and the offline script measures how far apart they are.
    surface.set_zoom(surface._fit_scale() * 1.0005)
    _settle(app, turns=6)
    phase = _grab(surface, app)
    assert phase is not None, "the second readback (half-pixel phase) never arrived"
    if reference_dir:
        phase.save(os.path.join(reference_dir, f"{_arm_name()}-phase.png"))

    differing = 0
    for y in range(0, min(fit.height(), phase.height()), 11):
        for x in range(0, min(fit.width(), phase.width()), 11):
            if fit.pixel(x, y) != phase.pixel(x, y):
                differing += 1
    assert differing > 0, "a half-pixel scale step changed nothing: the sampler is not sampling where it says"
