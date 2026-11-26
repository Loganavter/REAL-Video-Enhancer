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

from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, QSize, Qt  # noqa: E402
from PySide6.QtGui import QColor, QImage, QMouseEvent, QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from src.ui.widgets.video_preview import VIEW_MODES, VideoPreviewSurface  # noqa: E402
from src.ui.preview.source_register import ResultPicture, SourcePicture  # noqa: E402

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
    # The strip paints once and the divider is a CONTROL now (an outlined rule plus a grip, so it is visible over any
    # picture and can be dragged): count the DIVIDER, not a total that any grip detail would break.
    divider_floats = [v for v in surface._hud_vertices()]
    divider_x = surface._split_divider_x()
    assert divider_x is not None
    from sli_ui_toolkit.ui.managers.ui_scale import scaled_px
    rule = 2.0 * float(scaled_px(1)) + float(scaled_px(2))      # 1 px light, 2 px core, 1 px light
    xs = [divider_floats[i] for i in range(0, len(divider_floats), 6)]   # [x, y, r, g, b, a] per corner
    assert any(abs(x - (divider_x - rule / 2.0)) < 0.6 for x in xs), "the divider's left edge is not in the HUD"
    assert any(abs(x - (divider_x + rule / 2.0)) < 0.6 for x in xs), "the divider's right edge is not in the HUD"
    assert xs[0] < divider_x, "the strip is painted before the divider (one pass, strip first)"
    assert surface._view_rect() == QRectF(0.0, 0.0, float(PANE_SIZE[0]), float(PANE_SIZE[1])), (
        "split fits the result into the whole pane, unlike sbs")


def test_the_wheel_zooms_from_either_half_and_moves_both_pictures(surface):
    """REVISED, owner 2026-11-26: "в режиме side by side по оригиналу не работает зумирование, почему-то можно только
    на upscale".

    The invariant the old rule protected still holds - the original has no zoom of ITS OWN, it is drawn at the result's
    scale - and that is the reason the wheel should work over it: one scale means one gesture. Its anchor is mirrored
    into the result's half, so the same wheel over the mirrored points ends with the same zoom AND the same pan.
    """
    surface.set_original(_picture(*FRAME_SIZE))
    surface.set_view_mode("sbs")
    surface.reset_view()
    before = surface.zoom()

    _wheel(surface, QPointF(PANE_SIZE[0] * 0.25, PANE_SIZE[1] / 3.0))
    stepped = surface.zoom()
    pan = list(surface._pan)
    assert stepped > before, "a wheel over the original zooms, exactly like one over the result"
    assert surface._original_rect().width() == pytest.approx(surface._frame_rect().width()), (
        "and the original follows at the same apparent scale")

    surface.reset_view()
    _wheel(surface, QPointF(PANE_SIZE[0] * 0.75, PANE_SIZE[1] / 3.0))
    assert surface.zoom() == pytest.approx(stepped), "the mirrored point is the SAME gesture, not a second one"
    assert surface._pan == pytest.approx(pan), "and it leaves the shared pan where the first one left it"

def test_a_zoom_writes_a_draw_line_at_the_default_telemetry_level(surface, monkeypatch):
    """A gesture that changes the BOX leaves a line even at level 0: the pictures are the same, the geometry is not.

    Measured 2026-11-26: the wheel over the original answered `accepted`, the pane repainted, and no `draw` line followed -
    because the line followed the PICTURES only. "The gesture did nothing" and "the widget consumed it but the log follows
    pictures" were therefore indistinguishable at a normal telemetry level, which is how this very defect was reported.
    """
    from src.ui.video import telemetry
    from src.ui.widgets.preview_renderer import PreviewDrawn

    events = []
    monkeypatch.setattr(telemetry, "level", lambda: 0)
    monkeypatch.setattr(telemetry, "event", lambda name, **fields: events.append((name, fields)))

    surface.set_original(_picture(*FRAME_SIZE, colour="#402060"))
    surface.set_view_mode("sbs")
    painted = PreviewDrawn(mode="sbs", half="pair", drew_original=True)
    draws = lambda: [fields for name, fields in events if name == "draw"]  # noqa: E731 - a filter, not a helper

    surface._on_rhi_draw(painted)                                  # the opening box
    assert len(draws()) == 1

    surface._on_rhi_draw(painted)                                  # same pictures, same box
    assert len(draws()) == 1, "an unchanged box is not a new trace"

    _wheel(surface, QPointF(PANE_SIZE[0] * 0.2, PANE_SIZE[1] / 3.0))
    surface._on_rhi_draw(painted)

    assert len(draws()) == 2, "the zoom moved the box: it must be visible without level 2"
    assert draws()[-1]["scale"] != draws()[0]["scale"]

    # A PLANES result has no QImage, and the line used to infer "is there a picture" from `_frame`: every paint with a
    # source on the live YUV path wrote nothing (that is the state the compare actually runs in).
    planes = object()
    surface.set_frame_planes([b"\x00" * 8], [4], 2, 2, index=1, pts=0.5)
    surface._original = _picture(2, 2)
    surface._on_rhi_draw(painted)
    assert len(draws()) == 3, "a planes result with a source is a picture too"
    assert draws()[-1]["result_fp"] == "-", "the fingerprint says planes, and the line still has to exist"


def test_a_compare_opened_on_an_EMPTY_pane_still_asks_for_the_source_and_fits_it(surface):
    """The owner's log, 2026-11-26: `transport paused=1`, `produced=0.0000`, `hold on playhead=0.000`, then `view mode=sbs`
    and NOT ONE `compare:` line - the source half waited for a picture nobody had asked for.

    Two faults, both here: the ask required a result frame (`not self._has_picture()`), and with no result `_image_size`
    was (0, 0) so the fit had no box. The source is the one picture that CAN be shown.
    """
    asked: list[float] = []
    surface.originalRequested.connect(asked.append)
    surface.set_frame(None)                                    # an empty pane (a queue item, a stopped tick)
    assert not surface._has_picture()

    surface.set_view_mode("sbs")

    assert asked, "a compare mode asks for the source even before a result exists"
    source = _picture(*FRAME_SIZE, colour="#402060")
    surface.set_original(source, media_time=0.0)

    box = surface._frame_rect()
    half = surface._half_rects()[1]
    assert not box.isEmpty(), "the source defines the box while there is no result"
    assert half.contains(box) or abs(box.width() - half.width()) < 1.5, (
        f"the source is FITTED into its half, not drawn at native size: box {box}, half {half}")


def test_a_compare_with_no_result_paints_the_source_alone(surface):
    """The gate hands the renderer the source when there is no result, and `render()` must choose the source-only pass.

    The pair rule is not weakened: it still owns every paint where both halves exist. This is the third state - the one
    the owner's log hit (`transport paused=1`, `produced=0.0000`, `view mode=sbs`, no `compare:` line at all), where the
    pair could never form and the pane therefore painted NOTHING while a decoded source sat in hand.
    """
    surface.set_frame(None)
    surface.set_view_mode("sbs")
    surface.set_original(_picture(*FRAME_SIZE, colour="#402060"), media_time=0.0)

    painted = surface.painted_pair()
    assert painted is not None and painted.result is None, "no result is not a held pair: the source is committed"
    assert painted.lone_half == "source", "and the one picture is named as such, not sold as a pair"
    assert surface._renderer._original is not None, "the renderer got the source half"

    calls: list[str] = []
    surface._renderer._render_original_only = lambda *args, **_kwargs: calls.append("source-only")
    # The offscreen platform has no RHI, so `render()` bails on a missing scene/target before the branch: hand it a scene
    # whose target exists and let the patched pass record the routing (no command buffer is touched).
    surface._renderer._scene = SimpleNamespace(target=object(), clear=None)
    surface._renderer.render(object())
    assert calls == ["source-only"], "with no result the renderer draws the source alone"


def test_a_compare_with_a_result_still_paints_the_pair(surface):
    """The other direction, so the source-only branch cannot quietly take over the normal case.

    Asserted on the state the passes read (a result in the renderer, no source-only commit) rather than by calling
    `render()`: the offscreen platform has no RHI, so a real pass cannot run here and a patched one would only prove the
    patch.
    """
    surface.set_frame(_picture(*FRAME_SIZE), media_time=1.0)
    surface.set_original(_picture(*FRAME_SIZE, colour="#402060"), media_time=1.0)
    surface.set_view_mode("sbs")
    surface._pair.refresh()                      # what a mode change does: re-hand what the renderer holds

    assert surface._has_picture()
    assert surface._renderer._frame is not None, "the pair path keeps the RESULT in the renderer"
    assert surface._renderer._original is not None, "and the source beside it"
    painted = surface.painted_pair()
    assert painted is not None and painted.result is not None and painted.source is not None
    assert painted.lone_half is None, "both halves of one moment: not a held half"
    assert surface._view_mode == "sbs" and surface._renderer._mode == "sbs"


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


def test_an_lmb_drag_pans_a_compare_mode_and_shows_a_hand(surface):
    """Owner, 2026-11-26: "во всех режимах сравнения должна быть одинаковая политика взаимодействия - лкм - оригинал
    (если один кадр), лкм+зажатие - панарамирование со сменой курсора на руку". In `sbs` the source already stands, so
    the LMB is the drag: no flash to consume, and the hand says so while it runs."""
    surface.set_original(_picture(*FRAME_SIZE, colour="#402060"))
    surface.set_view_mode("sbs")
    surface.set_zoom(0.5)                                               # above the half's fit: room to pan
    start = QPointF(PANE_SIZE[0] * 0.25, PANE_SIZE[1] / 3.0)            # the ORIGINAL half
    before = surface._frame_rect().left()

    QApplication.sendEvent(surface, QMouseEvent(
        QEvent.Type.MouseMove, start, start, Qt.MouseButton.NoButton,
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier))
    assert surface.cursor().shape() != Qt.CursorShape.OpenHandCursor, (
        "no hover cursor on the picture: the hand is the DRAG's, not a promise made before it (owner, 2026-11-26)")

    _press(surface, start)
    assert surface.cursor().shape() == Qt.CursorShape.ClosedHandCursor, "the drag shows the hand"
    moved = QPointF(start.x() - 40.0, start.y())
    QApplication.sendEvent(surface, QMouseEvent(
        QEvent.Type.MouseMove, moved, moved, Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))
    _release(surface, moved)

    assert surface._frame_rect().left() == pytest.approx(before - 40.0), "an LMB drag pans, from either half"
    assert surface.cursor().shape() != Qt.CursorShape.ClosedHandCursor, "the release gives the pointer back"


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


def test_the_last_step_names_both_halves_in_one_line(surface):
    """The compare's DRAW is the only record of what was actually on screen - and labels alone cannot prove it.

    A writer that shifts its timeline (`-avoid_negative_ts make_zero` plus an encoder's B-frame delay) names every
    picture later than the frame it carries: measured 2026-11-25, the preview file starts at 0.083 s while its first
    frame is the source's 0.000 s frame, so a compare whose labels agree still shows two different frames. The pane
    therefore prints, at the draw: its own media time and index, the source's, a cheap pixel fingerprint of each, and
    the ask -> on-screen milliseconds. `result_t` is what makes "one frame behind" a subtraction.
    """
    from src.ui.video import telemetry

    lines: list[str] = []
    telemetry.set_writer(lambda line, _which: lines.append(line))
    try:
        surface.set_view_mode("single")
        surface.set_frame(_picture(*FRAME_SIZE), index=221, media_time=9.218)
        surface._showing_original = True
        # The source arrives for ITS OWN moment; the half-frame refusal is keyed on the result's moment, so clear it for
        # the handover. The draw line below still names both halves - that is the point of this test.
        surface._shown_time = None
        surface._note_original_ask(9.301)
        surface.set_original(_picture(*FRAME_SIZE, "#30a028"), media_time=9.301, index=223)
        surface._shown_time = 9.218
        surface._log_draw(half="original", picture=surface._original)
    finally:
        telemetry.set_writer(None)

    line = [one for one in lines if one.startswith("preview: draw ")][-1]
    for piece in ("mode=single", "half=original", "t=9.301", "idx=223",
                  "result_t=9.218", "result_idx=221", "result_moment=9.218", "generation=0",
                  "fp=", "ask_ms=", "ask_wall="):
        assert piece in line, f"{piece!r} is missing from the draw line: {line}"
    assert surface.original_shown_time() == 9.301
    assert surface.original_shown() == ("frame", 223), "the source half keeps the identity its decoder gave it"


def test_set_source_shows_exactly_that_moments_picture_or_a_gap(surface):
    """`set_source` is the pane's only source API, and it FAILS CLOSED.

    A snapshot whose moment differs from the result's own moment by more than half a frame is refused: the pane shows a
    GAP, never the previous picture (which would be a neighbour of the moment the result is showing). This is the guard
    that makes a stale `_source_shown_pts`-style memo impossible to reintroduce: a caller cannot assert "already shown",
    it can only name a moment.
    """
    surface.set_frame(_picture(*FRAME_SIZE), media_time=5.0)
    assert surface.shown_moment() == pytest.approx(5.0), "the result names its own moment"

    good = _picture(*FRAME_SIZE, "#402060")
    surface.set_source(SourcePicture(moment=5.0, picture=good, origin="block"))
    assert surface._original is not None, "the exact moment's picture is shown"
    assert surface.original_shown_time() == pytest.approx(5.0)
    assert surface._original.pixelColor(1, 1) == QColor("#402060")

    # The SAME moment again is one handover, not one repaint per producer arrival.
    before = surface._original
    surface.set_source(SourcePicture(moment=5.0, picture=good, origin="decoder"))
    assert surface._original is before

    # Half a frame of drift is still the same picture (the tolerance the register also uses).
    frame = 1.0 / 24.0
    surface.set_source(SourcePicture(moment=5.0 + frame / 2.0, picture=good, origin="decoder"))
    assert surface._original is not None

    # A snapshot for ANOTHER moment is refused, and the old picture goes with it: a gap, not a neighbour.
    surface.set_source(SourcePicture(moment=9.0, picture=_picture(*FRAME_SIZE, "#30a028"), origin="decoder"))
    assert surface._original is None
    assert surface.original_shown_time() is None

    # An explicit None is a gap too.
    surface.set_source(SourcePicture(moment=5.0, picture=good, origin="decoder"))
    surface.set_source(None)
    assert surface._original is None


def test_set_result_refuses_an_older_generation_and_keeps_the_last_picture(surface):
    """The result half carries its own identity: a frame from a previous render/file/seek cannot paint over the new one.

    A refusal keeps the picture that is on screen and marks the half as a GAP; nothing is queued for the refused
    result, so no source can pair with it.
    """
    surface.set_view_mode("split")
    first = _picture(*FRAME_SIZE, "#402060")
    surface.set_result(ResultPicture(moment=1.0, picture=first, generation=4))
    assert surface.shown_moment() == pytest.approx(1.0)
    assert surface._result_generation == 4
    assert surface._frame.pixelColor(1, 1) == QColor("#402060")

    stale = _picture(*FRAME_SIZE, "#30a028")
    surface.set_result(ResultPicture(moment=1.042, picture=stale, generation=3))

    assert surface._frame.pixelColor(1, 1) == QColor("#402060"), "the last picture stays"
    assert surface.shown_moment() == pytest.approx(1.0), "and so does its moment"
    assert surface._result_gap is True, "the half is a GAP after a refusal"
    held = surface._pair._pending_result
    assert held is not None and held.moment == pytest.approx(1.0), (
        "the picture waiting for a partner is the ACCEPTED one: a refused result is never offered for a pair")


def test_set_result_refuses_a_late_frame_within_one_generation(surface):
    """A frame older than the picture on screen, from the SAME generation, is late and must not paint."""
    surface.set_view_mode("split")
    surface.set_result(ResultPicture(moment=2.0, picture=_picture(*FRAME_SIZE, "#402060"), generation=7))
    surface.set_result(ResultPicture(moment=1.958, picture=_picture(*FRAME_SIZE, "#30a028"), generation=7))
    assert surface._frame.pixelColor(1, 1) == QColor("#402060")
    assert surface.shown_moment() == pytest.approx(2.0)
    assert surface._result_gap is True

    # A NEW generation resets the baseline: a lower moment is a legitimate seek result, not a late frame.
    surface.set_result(ResultPicture(moment=0.5, picture=_picture(*FRAME_SIZE, "#204020"), generation=8))
    assert surface.shown_moment() == pytest.approx(0.5)
    assert surface._result_generation == 8
    assert surface._result_gap is False


def test_set_result_refuses_a_nameless_picture_while_a_pair_is_required(surface):
    """`pts=None` must not inherit the previous pane moment and pair with a stale source."""
    surface.set_view_mode("split")
    before = surface._frame
    surface.set_result(ResultPicture(moment=None, picture=_picture(*FRAME_SIZE, "#402060"), generation=1))
    assert surface._frame is before, "nothing was accepted over the picture that was already there"
    assert surface._result_gap is True

    # `single` has no pair to form, so a nameless result is still the pane's one picture.
    surface.set_view_mode("single")
    surface.set_result(ResultPicture(moment=None, picture=_picture(*FRAME_SIZE, "#402060"), generation=1))
    assert surface._frame is not None
    assert surface.shown_moment() is None


def test_set_source_refuses_a_picture_from_another_generation(surface):
    """The source rule is identity too: same generation as the result, then its moment within half a frame."""
    surface.set_view_mode("split")
    surface.set_result(ResultPicture(moment=3.0, picture=_picture(*FRAME_SIZE), generation=5))

    stale = SourcePicture(moment=3.0, picture=_picture(*FRAME_SIZE, "#30a028"), origin="decoder", generation=4)
    surface.set_source(stale)
    assert surface._original is None, "a previous generation's picture is a gap, not an answer"

    current = SourcePicture(moment=3.0, picture=_picture(*FRAME_SIZE, "#204020"), origin="decoder", generation=5)
    surface.set_source(current)
    assert surface._original is not None
    assert surface._original.pixelColor(1, 1) == QColor("#204020")


def test_a_compare_mode_never_hands_the_renderer_a_lone_result(surface):
    """A compare mode paints a PAIR or nothing - the gate may not open because one half is missing.

    Audit 2026-11-27 (`video_preview.py:997`): the gate called `_commit_newest()` whenever `_original is None`, which is
    the state at every source gap and before the first source frame, so the upscale was painted alone and the pair
    followed one frame later. Owner: "оригинал не отрисовывается одновременно с апскейлом в режиме side by side".
    """
    surface.set_view_mode("sbs")
    surface.set_progress(0.0, 0.0, playhead=1.0, duration=10.0)
    surface.set_original(None)                       # the source half is a GAP
    committed: list[tuple[str, object]] = []
    surface._renderer.set_frame = lambda frame: committed.append(("frame", frame))
    surface._renderer.set_original = lambda original: committed.append(("orig", original))

    surface.set_result(ResultPicture(moment=1.0, picture=_picture(*FRAME_SIZE, "#3060a0"), generation=0))
    assert committed == [], "a result without its source must not be handed over at all - not alone, not 'first'"

    surface.set_source(SourcePicture(moment=1.0, picture=_picture(*FRAME_SIZE), origin="decoder", generation=0))
    kinds = [kind for kind, _ in committed]
    assert kinds == ["frame", "orig"] or kinds[-2:] == ["frame", "orig"], (
        "the pair is what reaches the renderer, together")


def test_a_half_painted_pair_is_not_a_pair(surface):
    """A predicate answering "in sync" about a picture with no source half was the lie that let the old gate commit the
    result alone; the pane now reports the painted pair, and one half is not one."""
    surface.set_view_mode("sbs")
    surface.set_frame(_picture(*FRAME_SIZE), media_time=1.0)
    surface.set_original(None)
    assert surface.painted_pair() is None, "one half is not a pair: a result with no source paints nothing"
    surface.set_original(_picture(*FRAME_SIZE), media_time=1.0)
    painted = surface.painted_pair()
    assert painted is not None and painted.source is not None and painted.lone_half is None, (
        "two halves of one moment are")


def test_a_cleared_pane_stays_cleared_when_a_late_source_arrives(surface):
    """A pane told to be empty must not be repainted by a leftover source (audit 2026-11-27 §6.1b, probe G2).

    `set_result(None)` reset the picture but kept the queued results, and with `shown_moment()` None the source guard
    had nothing to compare against, so a late source re-keyed itself by its own moment and painted half a picture onto
    the cleared pane - the "previous file's frame after the render ended" the owner saw.
    """
    surface.set_view_mode("sbs")
    surface.set_progress(0.0, 0.0, playhead=1.0, duration=10.0)
    surface.set_result(ResultPicture(moment=1.0, picture=_picture(*FRAME_SIZE), generation=0))
    surface.set_source(SourcePicture(moment=1.0, picture=_picture(*FRAME_SIZE), origin="decoder", generation=0))
    assert surface._original is not None, "a pair was painted before the clear"

    surface.set_result(None)                                  # the pane is told to be empty
    assert surface._frame is None
    painted = surface.painted_pair()
    assert painted is not None and painted.result is None and painted.source is None, (
        "the cleared pane keeps no half of the picture it no longer shows")
    assert surface._pair._pending_result is None, "and nothing queued for a pair"

    surface.set_source(SourcePicture(moment=0.5, picture=_picture(*FRAME_SIZE), origin="decoder", generation=0))
    assert surface._original is None, "a late source may not paint onto a pane that is empty"

    surface.set_result(ResultPicture(moment=2.0, picture=_picture(*FRAME_SIZE, "#3060a0"), generation=0))
    surface.set_source(SourcePicture(moment=2.0, picture=_picture(*FRAME_SIZE), origin="decoder", generation=0))
    assert surface._original is not None, "a new picture re-opens the pane, with its own source"
