"""The preview surface's geometry, without a GPU: fit, 1:1, zoom, the strip and its fractions.

Why these can be pinned without rendering: every number the shader receives is computed on the CPU (the mvp, the strip
rect, the HUD vertices), and those are exactly the numbers that were wrong during the build - a flipped texture, a
1:1 view that was not 1:1, a strip whose click mapped to the wrong time. The pixels themselves are proven on a live
display through `grabFramebuffer()` (harness/docs/LIVE-TESTING.md); this file keeps the arithmetic honest on any
interpreter with PySide6.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPointF, QRectF, QSize, Qt  # noqa: E402
from PySide6.QtGui import QColor, QImage, QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from src.ui.widgets.video_preview import (  # noqa: E402
    VideoPreviewSurface,
    samples_nearest_for,
    wheel_zoom_factor,
)


def _send(widget, kind, point: QPointF, *, held) -> None:
    """One mouse event at `point`; `held` is what `buttons()` reports (Qt's own convention)."""
    event = QMouseEvent(kind, QPointF(point), QPointF(point), held, held, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(widget, event)


def _release_with_clip(widget, point: QPointF) -> None:
    """A release carries `NoButton` in `buttons()`: the button that went up is in `button()` only."""
    event = QMouseEvent(QEvent.Type.MouseButtonRelease, QPointF(point), QPointF(point),
                        Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(widget, event)


@pytest.fixture()
def surface():
    """A fresh surface per test: the harness sweeps top-level widgets after each test, so a module-scoped one would be
    a deleted C++ object by the second test (measured: "Internal C++ object already deleted")."""
    _app = QApplication.instance() or QApplication([])
    widget = VideoPreviewSurface()
    widget.resize(800, 600)
    image = QImage(1600, 900, QImage.Format.Format_RGBA8888)
    image.fill(QColor("#204060"))
    widget.set_frame(image)
    return widget


def test_fit_keeps_the_frame_inside_the_pane_and_its_aspect(surface):
    rect = surface._frame_rect()
    assert rect.width() <= surface.width() + 0.5
    assert rect.height() <= surface.height() + 0.5
    assert abs(rect.width() / rect.height() - 1600 / 900) < 1e-6, "letterboxing must not change the aspect"
    assert abs(rect.center().x() - surface.width() / 2) < 0.5, "a fitted frame is centred"


def test_one_to_one_is_the_frames_own_size_and_may_overflow(surface):
    surface.reset_view(actual_size=True)
    rect = surface._frame_rect()
    assert (round(rect.width()), round(rect.height())) == (1600, 900)
    assert rect.width() > surface.width(), "1:1 on a smaller pane overflows instead of downscaling"
    surface.reset_view()


def test_zoom_is_clamped_and_anchored_at_the_cursor(surface):
    surface.reset_view()
    surface.set_zoom(100.0)
    assert surface.zoom() == pytest.approx(8.0), "the ceiling is 8x"
    surface.reset_view()
    surface.set_zoom(0.0001)
    assert surface.zoom() == pytest.approx(surface._fit_scale(), rel=1e-6), (
        "the wheel's floor is the fit scale: any smaller would leave gutters")
    surface.reset_view()
    surface.set_zoom(2.0, around=QRectF(QPointF(0, 0), QSize(1, 1)))
    after = surface._frame_rect()
    assert after.width() == pytest.approx(1600 * 2.0, rel=1e-6), (
        "zoom is a source-pixel scale: 2x a 1600 px frame is 3200 px on screen")
    surface.reset_view()


def test_a_wheel_notch_scales_by_its_magnitude_not_by_the_event():
    """A burst and N separate notches must give the same zoom: the step is proportional to the delta.

    The surface used to multiply by a fixed 1.15 per EVENT, so one notch and an OS-coalesced burst of five looked the
    same and every step was a 15 % jump - the "jerky" the owner reported, against BlinkTable
    (`ui/canvas_infra/viewport/zoom.py:135-147`: `1.1 ** (delta / 120)`). The per-notch step here is 1.2 rather than
    their 1.1 because the applied magnification is quantised to whole factors; with 1.1 a step would need four notches.
    """
    assert wheel_zoom_factor(120) == pytest.approx(1.2)
    assert wheel_zoom_factor(240) == pytest.approx(1.44)
    assert wheel_zoom_factor(60) == pytest.approx(1.2 ** 0.5)
    assert wheel_zoom_factor(-120) == pytest.approx(1 / 1.2)
    assert wheel_zoom_factor(360) == pytest.approx(wheel_zoom_factor(120) ** 3), \
        "three notches in one event are three notches"
    assert wheel_zoom_factor(0) is None, "a zero delta is not a zoom"
    assert wheel_zoom_factor(None) is None


def test_the_mip_chain_has_a_switch_so_an_ab_runs_on_one_build(monkeypatch):
    """`RVE_MIP_MINIFY` exists for the A/B, not for production: the default is on.

    The chain cannot be proven by a test - a texture only exists under a real QRhi, and the null backend refuses to
    instantiate (`QRhiWidget: QRhi is not supported on this platform`). What IS pinned here is the policy the two live
    arms depend on, so a stale env cannot silently turn the fix off.
    """
    from src.ui.widgets.video_preview import mip_minify_enabled

    monkeypatch.delenv("RVE_MIP_MINIFY", raising=False)
    assert mip_minify_enabled() is True, "the fix is on by default"
    monkeypatch.setenv("RVE_MIP_MINIFY", "1")
    assert mip_minify_enabled() is True
    monkeypatch.setenv("RVE_MIP_MINIFY", "0")
    assert mip_minify_enabled() is False, "the baseline arm"
    monkeypatch.setenv("RVE_MIP_MINIFY", " 0 ")
    assert mip_minify_enabled() is False, "whitespace must not resurrect the fix"


def test_the_zoom_is_continuous_and_nearest_covers_every_magnification():
    """BlinkTable's rule: `Nearest` always, and a continuous scale - its own constant says why.

    The ladder that made nearest "honest" at whole factors also made one wheel notch do nothing and the next one
    double the picture; that is the jerky zoom the owner reported ("рванное" = choppy, not jagged). Continuous plus
    nearest reproduces the reference viewer: the blocks widen smoothly as the wheel turns and the pixels are the real
    ones. Minification keeps the linear sampler - nearest there drops pixels.
    """
    assert samples_nearest_for(1.0) is True
    assert samples_nearest_for(1.87) is True, "magnified at any factor: the real pixels"
    assert samples_nearest_for(4.0) is True
    assert samples_nearest_for(0.5) is False, "minification averages instead"


def test_magnified_views_sample_nearest_and_fitted_ones_do_not(surface):
    """Crisp pixels when magnified, averaging when the pane is smaller than the frame."""
    surface.reset_view()
    assert surface.samples_nearest() is False, "the fit view MINIFIES: nearest there is aliasing, not sharpness"
    surface.set_zoom(2.0)
    assert surface.zoom() == 2.0 and surface.samples_nearest() is True
    surface.set_zoom(1.87)
    assert surface.zoom() == pytest.approx(1.87), "the scale is continuous - no ladder"
    assert surface.samples_nearest() is True, "BlinkTable's rule: nearest on every magnification"
    surface.reset_view(actual_size=True)
    assert surface.samples_nearest() is True, "exactly 1:1 is a source pixel per screen pixel"


def test_a_left_drag_pans_and_KEEPS_the_compare_on(surface):
    """Panning is on the LEFT button, and the compare stays on while it runs (owner, 2026-11-25).

    A drag moves the view over the same picture, so what the eye wants there is the ORIGINAL, not the upscale; the
    release ends the compare like any hold.
    """
    surface.set_zoom(surface.zoom() * 2.0)
    start = QPointF(200.0, 200.0)
    asked = []
    surface.originalRequested.connect(lambda pts: asked.append(pts))

    _send(surface, QEvent.Type.MouseButtonPress, start, held=Qt.MouseButton.LeftButton)
    assert surface._showing_original is True, "the press asks for the original at once - a hold shows the compare"
    assert asked, "and it asked exactly once"

    _send(surface, QEvent.Type.MouseMove, QPointF(240.0, 220.0), held=Qt.MouseButton.LeftButton)
    assert surface._panning is True, "a left drag pans"
    assert surface._showing_original is True, "and the compare stays on: the drag inspects the original"
    _release_with_clip(surface, QPointF(240.0, 220.0))

    assert surface._panning is False
    assert surface._pan != [0.0, 0.0], "the picture moved with the drag"
    assert surface._showing_original is False, "releasing the drag returns to the upscale by itself"
    assert surface._original is None, "and the held source frame is dropped, not kept for the next press"


def test_a_left_click_without_movement_still_shows_the_original(surface):
    """A press that does not move is the compare, not a pan: the two gestures share one button by distance alone."""
    start = QPointF(200.0, 200.0)
    asked = []
    surface.originalRequested.connect(lambda pts: asked.append(pts))

    _send(surface, QEvent.Type.MouseButtonPress, start, held=Qt.MouseButton.LeftButton)
    _send(surface, QEvent.Type.MouseMove, QPointF(201.0, 200.5), held=Qt.MouseButton.LeftButton)
    _release_with_clip(surface, QPointF(201.0, 200.5))

    assert surface._panning is False, "1 px is a click, not a drag"
    assert len(asked) == 1
    assert surface._pan == [0.0, 0.0]


def test_a_middle_drag_still_pans(surface):
    """The binding moved to the left button; the middle one keeps working, so nobody's habit breaks."""
    surface.set_zoom(surface.zoom() * 2.0)
    _send(surface, QEvent.Type.MouseButtonPress, QPointF(200.0, 200.0), held=Qt.MouseButton.MiddleButton)
    assert surface._panning is True


def test_pan_is_clamped_to_the_picture(surface):
    surface.reset_view(actual_size=True)
    surface.pan_by(10_000, 10_000)
    rect = surface._frame_rect()
    assert rect.left() <= 0.5 and rect.top() <= 0.5, "the picture cannot be dragged away from the top-left"
    assert rect.right() >= surface.width() - 0.5, "and never leaves a gutter on the right"
    surface.reset_view()


def test_the_strip_maps_a_click_to_the_same_fraction(surface):
    surface.reset_view()
    strip = surface._strip_rect()
    assert strip.width() > 0
    assert surface._fraction_at(strip.left()) == pytest.approx(0.0)
    assert surface._fraction_at(strip.left() + strip.width() * 0.25) == pytest.approx(0.25)
    assert surface._fraction_at(strip.right()) == pytest.approx(1.0)
    assert surface._fraction_at(strip.left() - 500) == 0.0, "outside the strip clamps, never goes negative"
    assert surface._fraction_at(strip.right() + 500) == 1.0


def test_the_hud_has_one_rect_per_marker_and_a_track(surface):
    surface.reset_view()
    surface.set_progress(rendered=5.0, total=20.0, playhead=2.5, duration=20.0)
    vertices = surface._hud_vertices()
    # track + rendered range + playhead halo + playhead line + handle halo + handle = 6 rects, 6 vertices, 6 floats
    assert len(vertices) == 6 * 6 * 6
    surface.set_progress(0.0, 0.0, 0.0, 0.0)
    assert len(surface._hud_vertices()) == 1 * 6 * 6, "with no progress only the track is drawn"


def test_the_strip_can_be_switched_off(surface):
    surface.set_strip_visible(False)
    assert surface._strip_rect().isEmpty()
    assert surface._hud_vertices() == []
    surface.set_strip_visible(True)


def test_the_original_flag_is_what_swaps_the_picture(surface):
    surface.set_original(QImage(1600, 900, QImage.Format.Format_RGBA8888))
    assert surface._original is not None
    surface.set_original(None)
    assert surface._original is None


def test_the_compare_keeps_the_sources_proportions(surface):
    """An upscaler can change the frame's aspect (border-detect crops). The source is stretched to the RENDER's size to
    make the compare pixel-aligned, but never distorted: with `IgnoreAspectRatio` a straight line would appear where the
    picture has a curve, and the viewer would be comparing its own artefact."""
    surface.reset_view()
    wide = QImage(800, 200, QImage.Format.Format_RGBA8888)
    wide.fill(QColor("white"))
    surface.set_original(wide)
    frame = surface._frame.size()
    expected_height = round(frame.width() * 200 / 800)   # 4:1 kept: width fills, height follows
    assert surface._original.size() == QSize(frame.width(), expected_height), (
        f"800x200 into a {frame.width()}x{frame.height()} render must letterbox, not stretch: "
        f"{surface._original.size().toTuple()}")
    surface.set_original(None)


def test_the_strip_refuses_a_seek_past_the_produced_edge(surface):
    surface.reset_view()
    surface.set_progress(rendered=12.0, total=20.0, playhead=6.0, duration=20.0)   # 60 % produced
    assert surface._seekable(0.5)
    assert not surface._seekable(0.75)
    assert surface._seekable(0.0)
    surface.set_progress(0.0, 0.0, 6.0, 20.0)   # a finished file: everything is there
    assert surface._seekable(1.0)


def test_hover_marks_the_unprocessed_part_with_a_forbidden_cursor(surface):
    surface.reset_view()
    surface.set_progress(rendered=12.0, total=20.0, playhead=6.0, duration=20.0)
    strip = surface._strip_rect()
    surface._update_hover(QPointF(strip.left() + strip.width() * 0.3, strip.center().y()))
    assert surface._hover_blocked is False
    assert surface.cursor().shape() == Qt.CursorShape.PointingHandCursor
    surface._update_hover(QPointF(strip.left() + strip.width() * 0.85, strip.center().y()))
    assert surface._hover_blocked is True
    assert surface.cursor().shape() == Qt.CursorShape.ForbiddenCursor, (
        "the pointer must say that nothing has been produced there")
    assert surface._hover_fraction is not None


def test_the_compare_puts_the_original_on_the_results_own_pixel_grid(surface):
    """Within rounding noise the source is scaled to the frame EXACTLY, so both pictures share one grid.

    The channel's payload box rounds each side to an even number independently: a 4:3 source in a 421x512 pane becomes a
    420x314 payload (0.32 % wide of 4:3), and `KeepAspectRatio` into it produced 418x314 that the draw then stretched
    back to 420 - a further 0.48 % (owner, 2026-11-25: "кадр чуть шире"). The measured case is pinned here.
    """
    surface.set_frame(QImage(420, 314, QImage.Format.Format_RGB32))    # the payload the pane receives
    surface.set_original(QImage(1440, 1080, QImage.Format.Format_RGB32))   # a 4:3 source
    assert surface._original.size().toTuple() == (420, 314), "same grid as the result, no extra stretch"


def test_a_really_different_aspect_is_letterboxed_and_named(surface):
    """A REAL difference (a border crop) must not be stretched: the proportions stay and the log says why."""
    surface.set_frame(QImage(420, 300, QImage.Format.Format_RGB32))
    surface.set_original(QImage(1440, 1080, QImage.Format.Format_RGB32))
    assert surface._original.size().toTuple() == (400, 300), "4:3 inside a 7:5 frame keeps its proportions"


def test_the_original_scales_to_a_planes_picture_size(surface):
    """The compare's source is scaled to the RESULT's size - including when the result is YUV planes.

    The YUV playback path has no `_frame` QImage, so `set_original` used to skip the scaling entirely and the sbs
    source half kept its native size (measured live 2026-11-25: the result half was the pane's fit, the source half a
    cropped native frame). The target is `_frame.size()` or the planes descriptor.
    """
    width, height = 320, 180
    planes = (bytes(width * 2 * height), bytes((width // 2) * 2 * (height // 2)),
              bytes((width // 2) * 2 * (height // 2)))
    surface.set_frame_planes(planes, (width * 2, width, width), width, height)
    source = QImage(640, 360, QImage.Format.Format_RGBA8888)
    source.fill(QColor("#204060"))
    surface.set_original(source)
    assert surface._original is not None
    assert (surface._original.width(), surface._original.height()) == (width, height)


def test_sbs_asks_for_the_source_on_the_yuv_path(surface):
    """The compare's standing original must be requested for a PLANES picture too.

    On the YUV playback path `_frame` is always None - the pixels are the three planes - and the request guard keyed on
    `_frame is None`, so sbs/split never asked the host for the source: the result half painted and the other half
    stayed empty (owner, 2026-11-26: "side by side мёртв"). The guard is about a PICTURE, not an image.
    """
    width, height = 64, 48
    planes = (bytes(width * 2 * height), bytes((width // 2) * 2 * (height // 2)),
              bytes((width // 2) * 2 * (height // 2)))
    surface.set_frame_planes(planes, (width * 2, width, width), width, height)
    surface.set_progress(rendered=5.0, total=10.0, playhead=2.0, duration=10.0)
    seen: list[float] = []
    surface.originalRequested.connect(seen.append)
    surface.set_view_mode("sbs")
    assert seen, "sbs never asked for the source on a planes picture"
    assert seen[-1] == pytest.approx(2.0)


def _split_at(surface, fraction: float) -> float:
    surface.set_view_mode("split")
    surface.set_split_fraction(fraction)
    x = surface._split_divider_x()
    assert x is not None, "split must have a divider"
    return float(x)


def test_the_split_divider_drags_and_does_not_pan_the_picture(surface):
    """The divider is the one control inside a compare mode: pressing it must move it, not pan (owner, 2026-11-26)."""
    x = _split_at(surface, 0.5)
    picture = surface._frame_rect()
    y = picture.top() + picture.height() / 2.0
    before_pan = list(surface._pan)
    _send(surface, QEvent.Type.MouseButtonPress, QPointF(x, y), held=Qt.MouseButton.LeftButton)
    _send(surface, QEvent.Type.MouseMove, QPointF(x + picture.width() * 0.25, y), held=Qt.MouseButton.LeftButton)
    _send(surface, QEvent.Type.MouseButtonRelease, QPointF(x + picture.width() * 0.25, y), held=Qt.MouseButton.LeftButton)
    assert surface.split_fraction() == pytest.approx(0.75, abs=0.02)
    assert surface._pan == before_pan, "dragging the divider must not pan the picture"


def test_a_press_away_from_the_divider_does_not_move_it(surface):
    x = _split_at(surface, 0.5)
    picture = surface._frame_rect()
    y = picture.top() + picture.height() / 2.0
    _send(surface, QEvent.Type.MouseButtonPress, QPointF(x - picture.width() * 0.3, y), held=Qt.MouseButton.LeftButton)
    _send(surface, QEvent.Type.MouseButtonRelease, QPointF(x - picture.width() * 0.3, y), held=Qt.MouseButton.LeftButton)
    assert surface.split_fraction() == pytest.approx(0.5)


def test_the_divider_is_clamped_so_neither_picture_disappears(surface):
    _split_at(surface, 0.5)
    assert surface.set_split_fraction(0.0) is None
    assert surface.split_fraction() == pytest.approx(0.05)
    surface.set_split_fraction(1.0)
    assert surface.split_fraction() == pytest.approx(0.95)


def test_the_divider_only_exists_in_split(surface):
    surface.set_view_mode("sbs")
    assert surface._split_divider_x() is None
    surface.set_view_mode("single")
    assert surface._split_divider_x() is None


def test_the_divider_fades_over_a_second_and_leaves_the_grip_at_twenty_percent(surface):
    """Owner, 2026-11-26: a rule that follows the pointer, then a 1 s fade that ends with only the grip at 20 %.

    The curve is a pure function of a timestamp (`divider_alpha(now)`), so the test drives it without a timer.
    """
    import time as _time

    from sli_ui_toolkit.ui.managers.ui_scale import scaled_px

    from src.ui.widgets.video_preview import DIVIDER_FADE_S, DIVIDER_REST_ALPHA

    source = QImage(640, 360, QImage.Format.Format_RGBA8888)
    source.fill(QColor("#204060"))
    surface.set_original(source)
    surface.set_view_mode("split")
    now = _time.monotonic()
    assert surface.divider_alpha(now) == pytest.approx(1.0), "the pointer is on it until it leaves"
    surface._divider_left_at = now
    assert surface.divider_alpha(now + DIVIDER_FADE_S / 2.0) == pytest.approx(0.5)
    assert surface.divider_alpha(now + DIVIDER_FADE_S + 0.01) == 0.0

    # At rest the rule is gone and the grip is still there at 20 %: one rect at the divider with that alpha.
    surface._divider_left_at = now - DIVIDER_FADE_S - 1.0
    floats = surface._hud_vertices()
    divider_x = surface._split_divider_x()
    near = [(floats[i], floats[i + 5]) for i in range(0, len(floats), 6)
            if abs(floats[i] - (divider_x if divider_x is not None else 0.0)) <= float(scaled_px(12))]
    assert near, "the divider left nothing at all"
    assert max(alpha for _x, alpha in near) == pytest.approx(DIVIDER_REST_ALPHA, abs=0.01), (
        f"the resting divider must be the grip at {DIVIDER_REST_ALPHA:.0%}, got {sorted({round(a, 2) for _x, a in near})}")
    assert float(scaled_px(9)) >= 9.0 - 0.01, "the grab radius is in toolkit pixels"


def test_the_fade_stops_repainting_once_it_reaches_the_rest_alpha(surface):
    """The 30 ms timer exists only while the rule is fading: a paused pane must not repaint for ever."""
    import time as _time

    source = QImage(640, 360, QImage.Format.Format_RGBA8888)
    source.fill(QColor("#204060"))
    surface.set_original(source)
    surface.set_view_mode("split")

    surface._start_divider_fade()
    assert surface._divider_timer is not None, "the fade needs a clock (a paused pane has none)"
    assert surface.divider_alpha() == pytest.approx(1.0, abs=0.05)
    surface._divider_left_at = _time.monotonic() - 3.0
    surface._tick_divider_fade()
    assert surface.divider_alpha() == 0.0
    assert not surface._divider_timer.isActive(), "the timer must stop when the rule is gone"

    # A hover cancels the fade and brings the rule back to full alpha.
    surface._on_hover_restore = None
    surface._divider_left_at = None
    assert surface.divider_alpha() == pytest.approx(1.0)


def test_the_divider_is_a_seam_not_a_highlight(surface):
    """Owner, 2026-11-26: "линия крайне яркая". The light edges stay under 0.45 and the dark core does the reading.

    A pin on the numbers, not on taste: the peak alpha of the light side is the thing that glared, and the grip must be
    softer than the rule so the line reads as a seam with a handle rather than a bright bar.
    """
    from sli_ui_toolkit.ui.managers.ui_scale import scaled_px

    source = QImage(640, 360, QImage.Format.Format_RGBA8888)
    source.fill(QColor("#204060"))
    surface.set_original(source)
    surface.set_view_mode("split")
    surface._divider_left_at = None          # full alpha: the brightest the rule ever is
    floats = surface._hud_vertices()
    divider_x = surface._split_divider_x()
    assert divider_x is not None
    near = [(round(floats[i], 1), round(floats[i + 5], 3)) for i in range(0, len(floats), 6)
            if abs(floats[i] - divider_x) <= float(scaled_px(12))]
    light = max(alpha for _x, alpha in near if alpha > 0.3)
    alphas = {round(alpha, 2) for _x, alpha in near}
    assert 0.38 in alphas, f"the light edges must be subdued at 0.38, got {sorted(alphas)}"
    assert 0.78 in alphas, f"the dark core must carry the reading at 0.78, got {sorted(alphas)}"
    assert max(alphas) <= 0.80, f"nothing on the divider may glare: {sorted(alphas)}"


def test_a_compare_mode_paints_only_when_the_two_halves_are_one_moment(surface):
    """Owner, 2026-11-26: "нам бы как-нибудь синхронить сик когда отображается одновременно два кадра ... прямо мерцание".

    Each half used to repaint on its own arrival, so a fast GPU showed the result at T+1 beside the source at T for ~40 ms
    and then both at T+1. The gate holds the newer half until its partner lands, with a timeout for the case where it
    never does.
    """
    import time as _time

    from src.ui.widgets.video_preview import PAIR_SYNC_HOLD_S, PAIR_SYNC_SLACK_S

    source = QImage(640, 360, QImage.Format.Format_RGBA8888)
    source.fill(QColor("#204060"))
    surface.set_original(source, media_time=5.0)
    surface.set_view_mode("split")
    def result_image() -> QImage:
        picture = QImage(640, 360, QImage.Format.Format_RGBA8888)
        picture.fill(QColor("#204060"))
        return picture

    surface.set_frame(result_image(), media_time=5.0)

    assert surface.pair_in_sync(), "the same moment on both sides is the normal case"
    assert surface.pair_paint_ready()

    # The result moves on one frame: the source has not caught up yet, so a paint now would show a mismatch.
    frame_seconds = 1.0 / 24.0
    surface.set_frame(result_image(), media_time=5.0 + frame_seconds)
    assert not surface.pair_in_sync()
    assert not surface.pair_paint_ready(), "a pair one frame apart is exactly the flicker: hold it"
    assert not surface.pair_paint_ready(now=_time.monotonic() + PAIR_SYNC_HOLD_S / 2.0)

    # The source lands: the pair is whole again and the paint is allowed.
    surface.set_original(source, media_time=5.0 + frame_seconds)
    assert surface.pair_in_sync()
    assert surface.pair_paint_ready()

    # Never held for ever: a decoder that does not answer must not freeze the picture.
    surface.set_frame(result_image(), media_time=5.0 + 3 * frame_seconds)
    blocked = _time.monotonic()
    surface._pair_blocked_at = blocked
    assert not surface.pair_paint_ready(now=blocked)
    assert surface.pair_paint_ready(now=blocked + PAIR_SYNC_HOLD_S + 0.001), "the hold window is the way out"
    assert PAIR_SYNC_SLACK_S < frame_seconds, "the slack is tighter than a frame, or a one-frame pair would pass"
