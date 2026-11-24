"""The preview's two surfaces: the render result and the source picture under one playhead.

`set_view_mode` gives the pane Topaz's three views - `single` (today), `sbs` (result right, original left) and
`split` (one picture, the original left of a divider). What is pinned here is the part the shader cannot answer:
which rectangle each picture gets, that both come from ONE zoom (the result's), where the strip's hit test lands
in every mode, and that the original is a standing pane outside `single` instead of a held button.

Everything runs offscreen and without media: this platform has **no QRhi** (`VideoPreviewSurface.initialize` says so
and returns), so `grabFrame()` is a null readback here and the pixels are proven on a live display
(`harness/docs/LIVE-TESTING.md`). What is left is exactly the arithmetic and the signal wiring - where every bug in
this file has been so far.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, QSize, Qt  # noqa: E402
from PySide6.QtGui import QColor, QImage, QMouseEvent, QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from src.ui.widgets.video_preview import VIEW_MODES, VideoPreviewSurface  # noqa: E402

FRAME_SIZE = (1600, 900)
PANE_SIZE = (800, 600)
#: a point above the strip (which sits in the bottom margin) - the picture, not the control
PICTURE_POINT = QPointF(PANE_SIZE[0] / 2.0, PANE_SIZE[1] / 3.0)


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _picture(width: int, height: int, colour: str = "#204060") -> QImage:
    image = QImage(width, height, QImage.Format.Format_RGBA8888)
    image.fill(QColor(colour))
    return image


@pytest.fixture()
def surface():
    """A fresh surface per test: the harness sweeps top-level widgets after each test, so a module-scoped one would
    be a deleted C++ object by the second test."""
    _app()
    widget = VideoPreviewSurface()
    widget.resize(*PANE_SIZE)
    widget.set_frame(_picture(*FRAME_SIZE))
    return widget


def _press(widget, point: QPointF, button=Qt.MouseButton.LeftButton) -> QMouseEvent:
    event = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(point), QPointF(point),
                        button, button, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(widget, event)
    return event


def _release(widget, point: QPointF, button=Qt.MouseButton.LeftButton) -> QMouseEvent:
    event = QMouseEvent(QEvent.Type.MouseButtonRelease, QPointF(point), QPointF(point),
                        button, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(widget, event)
    return event


def _wheel(widget, point: QPointF, delta: int = 120) -> QWheelEvent:
    event = QWheelEvent(QPointF(point), QPointF(point), QPoint(0, 0), QPoint(0, delta),
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                        Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(widget, event)
    return event


def test_the_default_mode_is_single_and_its_flash_is_unchanged(surface):
    """`single` is the pane this file shipped: an LMB press asks the host for the source at the playhead, the release
    drops it again. A mode API that changed that would have silently redefined the compare the owner knows."""
    assert VIEW_MODES[0] == "single"
    assert surface.view_mode() == "single"

    asked: list[float] = []
    surface.originalRequested.connect(asked.append)
    surface.set_progress(0.0, 0.0, playhead=4.0, duration=10.0)
    asked.clear()                                    # set_progress itself must not ask in `single`

    _press(surface, PICTURE_POINT)
    assert asked == [4.0], "the press asks for the source frame at the playhead"
    assert surface._showing_original is True
    surface.set_original(_picture(*FRAME_SIZE))     # the host answers
    assert surface._original is not None, "and the pane shows it as soon as the host hands it over"

    _release(surface, PICTURE_POINT)
    assert surface._showing_original is False
    assert surface._original is None, "one frame is held at a time: the release clears it"
    assert asked == [4.0], "the release asks for nothing"


def test_an_unknown_view_mode_is_refused_and_the_pane_keeps_its_mode(surface):
    for mode in ("", "side-by-side", "SBS", None):
        with pytest.raises(ValueError):
            surface.set_view_mode(mode)
    assert surface.view_mode() == "single", "a refused mode must not leave the pane half-switched"


def test_sbs_shows_the_original_without_a_held_button(surface):
    """Outside `single` the source is a STANDING pane: it survives a release, and a new result frame does not drop
    it. Only the host's own `set_original(None)` (or going back to `single`) removes it."""
    surface.set_progress(0.0, 0.0, playhead=2.0, duration=10.0)
    surface.set_original(_picture(*FRAME_SIZE))
    asked: list[float] = []
    surface.originalRequested.connect(asked.append)

    surface.set_view_mode("sbs")
    assert asked == [2.0], "opening a compare mode asks for the source at the playhead it will stand on"
    assert surface._showing_original is False, "nothing is being held"
    assert surface._original is not None

    _release(surface, PICTURE_POINT)
    assert surface._original is not None, "a release must not clear the standing pane"

    surface.set_frame(_picture(*FRAME_SIZE, "#3060a0"))
    assert surface._original is not None, "the next result frame does not take the original away"

    surface.set_original(None)
    assert surface._original is None, "the host clears it, the pane does not"


def test_one_strip_maps_to_the_same_fraction_in_every_mode(surface):
    """The strip is drawn once, under the whole widget, in every mode - including `sbs`, where a press at 25 % may
    land over either half. The gesture is the same seek in all three."""
    surface.set_progress(10.0, 10.0, playhead=3.0, duration=10.0)
    seeks: list[tuple[str, float]] = []
    surface.seekRequested.connect(lambda fraction: seeks.append((surface.view_mode(), fraction)))
    strips: dict[str, QRectF] = {}

    for mode in VIEW_MODES:
        surface.set_view_mode(mode)
        strip = surface._strip_rect()
        strips[mode] = QRectF(strip)
        assert strip.width() > 0
        x = strip.left() + strip.width() * 0.25
        assert surface._fraction_at(x) == pytest.approx(0.25)
        _press(surface, QPointF(x, strip.center().y()))
        _release(surface, QPointF(x, strip.center().y()))

    assert strips["sbs"] == strips["single"] == strips["split"], (
        "one strip for the whole widget: the compare modes do not move or narrow it")
    assert seeks == [(mode, pytest.approx(0.25)) for mode in VIEW_MODES], (
        f"each mode must seek to the same fraction, got {seeks}")


def test_an_empty_pane_in_sbs_does_not_crash_and_shows_no_picture_rects():
    """A compare mode can be selected before the first frame arrives (the page builds the pane empty). The paint path
    then clears once and draws nothing; the geometry must say so instead of inventing two halves."""
    _app()
    empty = VideoPreviewSurface()
    empty.resize(*PANE_SIZE)
    empty.set_view_mode("sbs")
    assert empty.view_mode() == "sbs"
    assert empty._strip_rect().isEmpty(), "no frame, no strip"
    assert empty._hud_vertices() == []
    assert empty._frame_rect().isEmpty()
    assert empty._original_rect().isEmpty()
    empty.grabFrame()                 # offscreen has no QRhi: a null readback, never an exception


def test_sbs_puts_the_result_right_and_the_original_left_at_one_scale(surface):
    surface.set_original(_picture(*FRAME_SIZE))
    surface.set_view_mode("sbs")

    result = surface._frame_rect()
    original = surface._original_rect()
    assert not result.isEmpty() and not original.isEmpty()
    assert result.left() >= PANE_SIZE[0] / 2.0 - 0.5, "the result owns the right half"
    assert original.right() <= PANE_SIZE[0] / 2.0 + 0.5, "the original owns the left half"
    assert result.width() == pytest.approx(original.width()), (
        "same source-pixel scale: both pictures are the same frame size, so they must be the same size on screen")
    assert surface._original_clip_rect() == QRectF(0.0, 0.0, PANE_SIZE[0] / 2.0, float(PANE_SIZE[1])), (
        "the original paints into its own half and nowhere else")

    # one strip: the track plus the boundary line, not a second copy of the markers
    assert len(surface._hud_vertices()) == 2 * 6 * 6
    surface.set_view_mode("single")
    assert len(surface._hud_vertices()) == 1 * 6 * 6, "single keeps exactly the track"


def test_sbs_letterboxes_each_picture_at_its_own_aspect(surface):
    """The source and the render normally share an aspect, but a render can change dimensions. Then the source keeps
    its own proportions inside its half instead of being stretched to it, and it still shows at the result's
    scale - a wider picture, never a taller one."""
    surface.set_original(_picture(800, 200))          # 4:1, upscaled to the render's size with proportions kept
    assert surface._original.size() == QSize(1600, 400)
    surface.set_view_mode("sbs")

    result = surface._frame_rect()
    original = surface._original_rect()
    assert original.width() == pytest.approx(result.width()), "both at the result's source-pixel scale"
    assert original.height() == pytest.approx(original.width() * 400 / 1600), "4:1 kept, not stretched to the half"
    assert original.height() < result.height(), "so the wide source is letterboxed inside its half"
    assert original.center().x() == pytest.approx(PANE_SIZE[0] * 0.25, abs=0.5)


def test_sbs_maps_the_two_quads_to_their_own_halves(surface):
    """The draw plan, in the numbers the shader receives: two quads, one per half. The translation is column 3 of
    the column-major mvp (elements 12/13), so its sign says which half each picture lands in."""
    surface.set_original(_picture(*FRAME_SIZE))
    surface.set_view_mode("sbs")
    result_mvp = surface._mvp_floats(surface._frame_rect())
    original_mvp = surface._mvp_floats(surface._original_rect())
    assert result_mvp[12] > 0.0, "the result's quad sits in the right half (+x in NDC)"
    assert original_mvp[12] < 0.0, "the original's quad sits in the left half"
    assert result_mvp[0] == pytest.approx(original_mvp[0]), "equal width on screen: the same x basis"
    surface.set_view_mode("split")
    assert surface._mvp_floats(surface._original_rect()) == surface._mvp_floats(surface._frame_rect()), (
        "split paints both pictures over one rect: one transform")

def test_split_clips_the_original_to_the_left_of_the_divider(surface):
    surface.set_original(_picture(*FRAME_SIZE))
    surface.set_view_mode("split")

    picture = surface._frame_rect()
    clip = surface._original_clip_rect()
    assert surface._original_rect() == picture, "one picture: the source overlays the result in place"
    assert clip.left() == pytest.approx(picture.left())
    assert clip.width() == pytest.approx(picture.width() / 2.0), "the divider sits in the middle of the picture"
    assert clip.top() == pytest.approx(picture.top())
    assert clip.height() == pytest.approx(picture.height())
    assert len(surface._hud_vertices()) == 2 * 6 * 6, "the track and the divider - the strip is not duplicated"
    assert surface._view_rect() == QRectF(0.0, 0.0, float(PANE_SIZE[0]), float(PANE_SIZE[1])), (
        "split fits the result into the whole pane, unlike sbs")


def test_the_wheel_ignores_the_original_half_and_zooms_the_result(surface):
    """The invariant: the original has no zoom of its own, it is drawn at the result's scale. So a wheel over its
    half must do nothing at all, and a wheel over the result must move both pictures together."""
    surface.set_original(_picture(*FRAME_SIZE))
    surface.set_view_mode("sbs")
    surface.reset_view()
    before = surface.zoom()

    _wheel(surface, QPointF(PANE_SIZE[0] * 0.25, PANE_SIZE[1] / 3.0))
    assert surface.zoom() == before, "the original's zoom is derived from the result; it has none to change"

    _wheel(surface, QPointF(PANE_SIZE[0] * 0.75, PANE_SIZE[1] / 3.0))
    assert surface.zoom() > before, "the wheel over the result still zooms the result"
    assert surface._original_rect().width() == pytest.approx(surface._frame_rect().width()), (
        "and the original follows at the same apparent scale")


def test_the_lmb_compare_asks_the_shown_pictures_time_not_the_stale_marker(surface):
    """`_progress` moves only when a frame paints, so a strip press leaves it behind; the picture names itself.

    Measured 2026-11-25 in the owner's log: "compare: asked for the source at 2.383s (pane playhead 9.827s)" - the
    compare decoded a frame seven seconds away from the picture on screen, which is the desync he reported.
    """
    asked: list[float] = []
    surface.originalRequested.connect(asked.append)
    surface.set_frame(_picture(*FRAME_SIZE), seq=1, media_time=3.2115)
    surface.set_progress(rendered=100.0, total=100.0, playhead=9.827, duration=20.646)

    _press(surface, QPointF(PANE_SIZE[0] * 0.5, PANE_SIZE[1] * 0.5))

    assert asked == [pytest.approx(3.2115)], "the ask is the picture's own time, not the marker's"


def test_a_middle_drag_pans_the_result_in_sbs_even_from_the_original_half(surface):
    """The pan belongs to the result; the original is placed from it, so dragging anywhere moves both together
    instead of giving the left half a private framing."""
    surface.set_original(_picture(*FRAME_SIZE))
    surface.set_view_mode("sbs")
    surface.set_zoom(0.5)                            # above the half's fit (0.25), so there is room to pan
    start = QPointF(PANE_SIZE[0] * 0.25, PANE_SIZE[1] / 3.0)
    before_result = surface._frame_rect().left()
    before_original = surface._original_rect().left()

    _press(surface, start, Qt.MouseButton.MiddleButton)
    moved = QPointF(start.x() - 40.0, start.y())
    QApplication.sendEvent(surface, QMouseEvent(
        QEvent.Type.MouseMove, moved, moved, Qt.MouseButton.NoButton,
        Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.NoModifier))
    _release(surface, moved, Qt.MouseButton.MiddleButton)

    assert surface._frame_rect().left() == pytest.approx(before_result - 40.0)
    assert surface._original_rect().left() == pytest.approx(before_original - 40.0), (
        "the same drag moves the original by the same amount")


def test_a_compare_mode_keeps_the_original_on_the_playhead(surface):
    """The standing pane is refreshed through the SAME `originalRequested` signal the flash uses: once when the mode
    opens, then once per moved playhead - not on a paused tick that republishes the same time."""
    asked: list[float] = []
    surface.originalRequested.connect(asked.append)
    surface.set_progress(0.0, 0.0, playhead=1.0, duration=10.0)
    assert asked == [], "single keeps the flash: no source request without a press"

    surface.set_view_mode("sbs")
    assert asked == [1.0], "opening a compare mode asks for the source at the playhead"
    surface.set_original(_picture(*FRAME_SIZE))

    surface.set_progress(10.0, 10.0, playhead=1.0, duration=10.0)
    assert asked == [1.0], "the same playhead is not requested twice"
    surface.set_progress(10.0, 10.0, playhead=2.5, duration=10.0)
    assert asked == [1.0, 2.5], "a moved playhead refreshes the standing original"

    surface.set_view_mode("single")
    surface.set_progress(10.0, 10.0, playhead=4.0, duration=10.0)
    assert asked == [1.0, 2.5], "single goes back to the flash: no standing requests"
