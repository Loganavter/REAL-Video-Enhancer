"""The GPU preview surface: one QRhiWidget that paints a full-resolution frame with zoom and pan.

Scope of this file today: the frame. The strip, the playhead and the LMB compare overlay are separate layers that will
sit in the same pass (see plan-rve-video-preview-widget-2026-11-23); keeping them out until the frame is proven means the
first thing measured is the thing that can be wrong in the most ways (texture upload, colour, aspect, zoom).

The pane carries **two surfaces, one playhead** (`set_view_mode`, 2026-11-24): the render result and the source
picture, shown as `single` (today's pane), `sbs` (result right, original left, half the width each) or `split` (one
picture, the original left of a vertical divider). They are two draws inside the SAME RHI pass - one clear per frame.
The pane paints NO strip and takes NO seek: `src/ui/preview/controls.py` builds the one transport bar (its scrubber is
`PreviewTimeline`), and a seek reaches the host through the controller in seconds. The pane's own painted strip, its
four gesture guards and its fraction-valued `seekRequested`/`seekBlocked` were the second implementation of both and
are deleted (audit 2026-11-27 §6.6; CODE_PATTERNS §2 "one writer per fact").

Decisions encoded here, each from a measurement or an owner call:

* **fit to the pane on open.** "No downscale" in the plan is about the RENDER: the backend must not shrink frames to
  save work. The VIEW downscales freely to fit - that is what fit means - and `1:1` is only a zoom step the user can
  reach (`reset_view(actual_size=True)`), the range being [fit, 8x source pixels]. The texture always carries the
  frame's own pixels either way.
* **the transform lives on the CPU** (mvp), because the letterbox and the pan clamp are widget rules, not shading.
* **one upload per new frame**, and only when the frame actually changed (`set_frame` marks the texture dirty).
* **readback is the pixel instrument**: under llvmpipe a composited RHI widget comes out black in the window, while
  `grabFramebuffer()` shows the truth (harness/docs/LIVE-TESTING.md, 2026-11-23) - so `grabFrame()` exists and is what
  the tests use.

Since the phase-0 seam of `plan-rve-preview-controls-frame-path-2026-11-25` the RHI side lives in
`src/ui/widgets/preview_renderer.py`: `PreviewRenderer` owns the textures, samplers, SRBs, pipelines, uploads and the
pass; this class owns the widget API, the zoom/pan/mvp geometry and the frame/state bookkeeping. It hands the renderer
one `PreviewScene` per paint and receives what was drawn back through `on_draw`.

The pane accepts a RESULT through one guarded API, `set_result(ResultPicture(moment, picture, generation))`: the
generation must not be older than the picture on screen, its moment must not go backwards within one generation, and a
compare mode refuses a picture that names no moment (it must not inherit the previous clock). `set_frame` and
`set_frame_planes` remain as thin compatibility adapters that pass through the same guards - the planes path is the
YUV playback path's cheap one (no QImage, no `to_ndarray`, colour conversion in the fragment shader); `set_frame` is
also the page's pane-shape probe (`hasattr(pane, "set_frame")` tells this surface from the raster `ImageLabel`
fallback, `pages/proc_page.py`) and the only entry point a pre-`set_result` harness double gets. The SOURCE half has
its own guarded API, `set_source(SourcePicture)`, and `set_original` is that same guard in the old two-argument form.

What reaches the renderer is NOT decided here: `src/ui/preview/pair.py` owns the painted pair - a compare mode paints a
pair or holds the last one, one documented hard limit is the only way a half is shown alone, and nothing else in the
tree calls `renderer.set_frame`/`set_original`. This file keeps the identity guards, the geometry and the hand-over.
"""

from __future__ import annotations

import hashlib
import time

from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QRhiWidget

from sli_ui_toolkit.ui.managers.theme_manager import ThemeManager
from sli_ui_toolkit.ui.managers.ui_scale import scaled_px

from src.ui.video import telemetry
from src.ui.preview.pair import CommittedPair, Half, PairWriter
from src.ui.preview.source_register import ResultPicture, SourcePicture
from src.ui.widgets.playback_strip import rgba, strip_colours
from src.ui.widgets.preview_renderer import (
    PreviewDrawn,
    PreviewFrame,
    PreviewRenderer,
    PreviewScene,
    mip_minify_enabled,  # noqa: F401 - re-exported here: the harness imports it from this module
)

#: 8 source pixels per screen pixel at most. The floor is the FIT scale (computed per frame), not 1:1: the pane shows
#: the whole frame by default and the wheel can go back to that, while 1:1 is just a step inside the range. Zoom is a
#: source-pixel scale, so the same factor means the same detail whatever the pane size.
MAX_ZOOM = 8.0
#: One physical wheel notch. Qt reports wheel deltas in 120ths of a notch on every platform, so the factor is
#: `WHEEL_ZOOM_PER_NOTCH ** (delta / 120)` - a coalesced burst of N notches and N separate notches then produce the
#: same total zoom, which a fixed per-EVENT step cannot do (BlinkTable: `ui/canvas_infra/viewport/zoom.py:135-147`,
#: the reference the owner named). 1.1 per notch is that reference's step.
WHEEL_UNITS_PER_NOTCH = 120.0
WHEEL_ZOOM_PER_NOTCH = 1.2
#: How far the pointer must travel before an LMB press becomes a pan instead of the compare flash.
PAN_THRESHOLD_PX = 3.0
#: `mip_minify_enabled` (the `RVE_MIP_MINIFY` A/B switch) now lives in `preview_renderer.py`, which owns the texture
#: flags and the samplers; it is imported above and stays importable from this module (the harness does that).


#: NEAREST on every magnification, like the reference viewer. BlinkTable's own constant says why in one line:
#: `DEFAULT_INTERPOLATION_METHOD = "NEAREST"` - "it is the only filter whose result is the real pixels (blocks you can
#: count), and every other choice is a taste for smoothing". Its zoom is CONTINUOUS, so the blocks widen smoothly as
#: the wheel turns; a ladder is what makes that look jerky (owner, 2026-11-25: "почему у нас зум очень неплавный? У
#: нас заданы большие шаги" + "напомни как blinktable всегда делает nearest"). Minification (fit and below) keeps the
def pair_distance_frames(source_s: float, result_s: float, frame_seconds: float) -> float:
    """How far apart two moments are, in the PANE's own frames.

    The unit has to be the pane's cadence, not a constant: the counter used to multiply by 24.0, so on a 60 fps clip a
    two-frame desync read as 0.8 (0 when it rounded) and a lone-half paint never entered the denominator at all - the
    number quoted in every live proof could not see the failure it exists for (telemetry audit 2026-11-27).
    """
    return (float(source_s) - float(result_s)) / max(1e-3, float(frame_seconds))


#: LINEAR sampler: nearest there would drop pixels and shimmer.


def samples_nearest_for(scale: float) -> bool:
    """Whether this drawn scale is magnified (or exactly 1:1) - where `Nearest` shows the real pixels."""
    return float(scale) >= 1.0 - 1e-9


def wheel_zoom_factor(angle_delta: float | None) -> float | None:
    """The zoom factor ONE wheel event asks for, or None when it asks for nothing.

    Proportional to the magnitude: `120` is one notch (`x1.1`), `240` is two (`x1.21`), `60` is half (`x1.0488`). A
    fixed step per event - what this surface used to do - makes the gesture depend on how many notches the OS happened
    to bundle, which is exactly the "jerky" zoom the owner reported (owner, 2026-11-24: "посмотри почему у нас
    зуммирование какое-то рванное, эталон - blinktable").
    """
    if not angle_delta:
        return None
    return WHEEL_ZOOM_PER_NOTCH ** (float(angle_delta) / WHEEL_UNITS_PER_NOTCH)
MIN_ZOOM = 1.0

#: What the pane can show: the result alone, result + original side by side, or both under one divider. The names are
#: the API (`set_view_mode`), not a UI label - the host may put them on whichever control it likes.
VIEW_MODES = ("single", "sbs", "split")

#: How close to the `split` divider a press counts as grabbing it, in TOOLKIT pixels (`scaled_px`): wide enough for a
#: mouse, narrow enough that a press meant for the picture (pan/compare) stays one.
SPLIT_GRAB_PX = 9

#: A compare mode shows ONE moment, not two: a source this far from the result is still the same picture. One and a
#: half frames at 24 fps, and the number is a MEASURED trade-off, not a preference: at half a frame the exact source
#: arrives ~20 ms after its result, the gate holds, and the hold times out on the paints whose source is slower still
#: (`pair_off` 20-27 of 27 in the bad seconds); at one and a half the gate paints a one-frame-old neighbour instead
#: (`pair_sub` 7-9 of 20-27, `pair_ahead` 0 - a constant lag, never a wobble). It is the pane's own tolerance now;
#: WHEN a pair may paint at all is the policy of `src/ui/preview/pair.py`, not a number here.
PAIR_SYNC_SLACK_S = 0.0625

#: Half a frame at the scene's 24 fps: the pane REFUSES a source snapshot further than this from the result's own
#: moment. The register already answers with a bounded match, so this is the second, independent guard - and it is why
#: `set_source` can fail CLOSED (a gap) instead of drawing a neighbour when a caller gets the moment wrong. Half of
#: `PAIR_SYNC_SLACK_S / 3` (= one frame at 24 fps) rather than a second hand-picked number.
SOURCE_HALF_FRAME_S = PAIR_SYNC_SLACK_S / 3.0

#: The divider fades out this long after the pointer leaves it (owner, 2026-11-26: "фейд до 1 секунды срезать"), and the grip that stays is this visible: a rule at zero
#: alpha would leave nothing to aim at, and a rule at full alpha would sit over the picture for ever (owner, 2026-11-26:
#: "после того как курсор ушел - фейдом пропадало в течение 2 секунд (только ручка была видна с альфой 20%)").
DIVIDER_FADE_S = 1.0
DIVIDER_REST_ALPHA = 0.2

#: The shortest share of the picture each side keeps: a divider pushed to the very edge would hide one picture
#: completely, which is what `single` is for.
SPLIT_MIN_FRACTION = 0.05
SPLIT_MAX_FRACTION = 0.95


@dataclass(frozen=True)
class Frame:
    """One picture to show: RGBA8888 pixels, already packed, plus its identity."""

    image: QImage
    index: int = 0
    pts: float = 0.0


class VideoPreviewSurface(QRhiWidget):
    """The result picture (or the result and the source together), fitted to the pane and zoomable from there.

    The zoom/pan belongs to the RESULT: in `sbs` the original's half is placed from the same source-pixel scale (see
    `set_view_mode`), so the two pictures always show the same apparent scale.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        #: The RHI side of the pane: textures, samplers, SRBs, pipelines and the pass (`preview_renderer.py`). It
        #: receives every number it draws through a `PreviewScene` (see `_preview_scene`) and reports what it drew back
        #: through `_on_rhi_draw`, so the draw log keeps the widget's own bookkeeping.
        self._renderer = PreviewRenderer(on_draw=self._on_rhi_draw)
        self._frame: QImage | None = None
        #: The size of a planes picture (`set_frame_planes`): a YUV frame has no QImage, and the geometry/HUD must
        #: read its size from somewhere - this is that somewhere, and `set_frame(image)` clears it.
        self._planes_size: tuple[int, int] | None = None
        self._original: QImage | None = None
        self._zoom = MIN_ZOOM
        #: What the wheel is aiming at. The APPLIED zoom is quantised to `ZOOM_STEPS` (nearest needs whole factors), so
        #: without a separate target the wheel would multiply the snapped value and stall at 1x forever.
        self._zoom_target = MIN_ZOOM
        self._zoom_is_fit = True
        self._pan = [0.0, 0.0]
        #: straight-alpha multiply tint; (1,1,1,0.55) dims a stale frame while a seek is in flight
        self._tint = (1.0, 1.0, 1.0, 1.0)
        #: (rendered, total, playhead, duration) - the two time bases the plan keeps apart. The pane does not paint a
        #: strip: the transport bar (`preview/controls.py` + `preview/timeline.py`) shows these, and the pane keeps them
        #: for the HUD-less draw log and for the standing-source ask (`_request_standing_original`).
        self._progress = (0.0, 0.0, 0.0, 0.0)
        self._panning = False
        #: Where the LMB went down, while a drag could still turn into a pan (cleared once it does).
        self._left_press = None
        self._pan_anchor = QPointF()
        self._showing_original = False
        #: which surfaces the pane shows: "single" (today), "sbs" or "split" - see `set_view_mode`
        self._view_mode = "single"
        #: where the split divider sits, as a share of the picture's width; the pane's rule, not a per-frame value
        self._split_fraction = 0.5
        #: A drag in flight on the `split` divider: the only gesture that moves it, and it must never fall through to
        #: the picture's own press handling (which consumes compare presses).
        self._dragging_split = False
        #: When the pointer last left the divider (monotonic), or None while it is on/near it: the fade's clock.
        self._divider_left_at: float | None = None
        #: The ONE writer of the painted pair (`preview/pair.py`): the pane offers it halves, it decides what the
        #: renderer holds, and it is the only code in the tree that calls `set_frame`/`set_original`. The pair policy
        #: (a compare mode paints a pair or holds the last one; `MAX_PAIR_STALE_S` is the only way a half shows alone)
        #: lives there, in words and in one place. What the renderer holds - and which picture is on screen, not which
        #: half arrived last - is its state: `painted_pair()` is the only way to ask, because a repaint can come from
        #: anywhere (the compositor, a HUD fade, a resize) and must draw the pair that belonged together.
        self._pair = PairWriter(self._renderer)
        #: The generation of the result on screen. A new render/file/seek starts the next one; a result from an older
        #: generation is REFUSED (keep the last picture) instead of painting over the new one.
        self._result_generation = 0
        #: The moment of the result on screen, and the cadence the pane learned from consecutive results. The source
        #: tolerance is half of THIS interval (the pane's own fps), not a module constant.
        self._result_moment: float | None = None
        self._result_frame_seconds = 2.0 * SOURCE_HALF_FRAME_S
        #: True when a result was refused: the half is a GAP. The last picture stays on screen, but nothing may pair
        #: across the refusal.
        self._result_gap = False
        #: The fade needs repaints, and a paused pane has no other clock. 30 ms is smooth at this size and stops as soon
        #: as the rule reaches its rest alpha.
        self._divider_timer = None
        #: The identity of the source picture the host handed over (`("frame", n)`, its media time) and the moment the
        #: pane asked for it. The compare's "last step" is the DRAW, and a draw that cannot name both halves cannot
        #: answer "what was on screen when the offset happened" (owner, 2026-11-25).
        self._original_shown: tuple[str, int] | None = None
        self._original_time: float | None = None
        self._original_ask_time: float | None = None
        self._original_ask_wall: str | None = None
        self._original_ask_target: float | None = None
        self._original_changed = False
        #: Opaque clear colour for the pane. QRhiWidgets composite by OVERWRITING their rect, so a clear with alpha
        #: below 255 punches a hole (black on Wayland, white on xcb) instead of letting the page show through - the
        #: framework's `lifecycle-first-frame-signal-gating` and `painting-qrhiwidget-cannot-alpha-blend-siblings`
        #: precedents. The pane is therefore always opaque, with the page's own colour. (Written ONCE: the two-line
        #: duplicate of this paragraph and assignment was audit 2026-11-27 §6.10's dead copy.)
        self._clear_colour = self.palette().window().color()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # The `split` divider's hover (its cursor, and the fade once the pointer leaves) needs move events while NO
        # button is held: without tracking, `mouseMoveEvent` only arrives during a drag and the divider never shows as a
        # control. (The strip's hover used this too, and the strip is gone; the divider is what still needs it.)
        self.setMouseTracking(True)
        # The compare's seam/divider resolve their tokens at paint time (`strip_colours` in `_compare_boundary_rects`),
        # so a theme flip must repaint the pane.
        ThemeManager.get_instance().theme_changed.connect(self._on_theme_changed)

    def setAutoFillBackground(self, enabled: bool) -> None:  # noqa: N802 - Qt spelling
        """A no-op, on purpose.

        `qrhi-gotchas.md` §"QRhiWidget autofill": autofill's palette clear fights RHI texture compositing, so a widget
        that carries it shows a stale frame - BlinkTable's canvas answers the same call with `pass`. The pane's own
        colour is the opaque clear inside the RHI pass, never the palette.
        """

    def showEvent(self, event) -> None:  # noqa: N802 - Qt spelling
        """Re-post a paint after being shown: a widget that was hidden may come back with an undrawn target."""
        super().showEvent(event)
        self.update()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt spelling
        """Ask for a repaint after a resize - `initialize()` has just been called with a new target size."""
        super().resizeEvent(event)
        if telemetry.level() >= 1:
            telemetry.event("resize", w=self.width(), h=self.height(),
                            dpr=round(float(self.devicePixelRatioF()), 2))
        self.update()

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt spelling
        """A floor for the pane, in design px through the toolkit scale.

        Stated as a size hint rather than `setMinimumSize`: the ui-gates rule `raw-chrome-px` reads literal arguments of
        the chrome setters, and a scaled literal is still a literal to it - while a hint is exactly what this is.
        """
        return QSize(scaled_px(160), scaled_px(90))

    # -- data ------------------------------------------------------------

    def set_result(self, result: ResultPicture | None) -> None:
        """The pane's ONLY result API: accept `result`, or REFUSE it and keep the last picture.

        The result owns the pane's clock, so it cannot be checked against "its own" moment; what is checked is identity:

        * a generation OLDER than the one on screen is refused - a new render/file/seek resets the generation, and the
          previous job's frame must not paint over the new one;
        * within a generation, a moment OLDER than the picture on screen is refused - a late frame arriving after its
          successor;
        * a picture with NO moment while a compare mode requires a pair is refused - it must not inherit the previous
          pane moment and be paired with a stale source.

        A refusal keeps the last picture and marks the half as a GAP; nothing is offered to the pair writer for it, so no
        pair can form across it. The refusal is visible in the log with the register's own numbers.
        """
        if result is None:
            self._accept_result(None, None, int(self._result_generation))
            return
        moment = None if result.moment is None else float(result.moment)
        generation = int(result.generation)
        if result.picture is None and moment is not None:
            # An EMPTY publication that still names a moment: "nothing this tick", not "clear" (the empty right half of
            # sbs after a finished render was exactly this shape).
            return
        if result.picture is None:
            self._accept_result(None, None, generation)
            return
        reason = self._result_refusal(moment, generation)
        if reason is not None:
            self._refuse_result(moment, generation, reason)
            return
        self._accept_result(moment, result.picture, generation)

    def _result_refusal(self, moment: float | None, generation: int) -> str | None:
        """Why this result must not paint, or `None` when it is the next picture of this generation."""
        if generation < self._result_generation:
            return "older generation"
        if moment is None and self._view_mode != "single":
            return "no moment while a pair is required"
        if (generation == self._result_generation and self._result_moment is not None
                and moment is not None
                and moment < self._result_moment - self.source_half_frame_s()):
            return "older than the picture on screen"
        return None

    def source_half_frame_s(self) -> float:
        """Half the pane's own result cadence: the largest source distance that is still the same picture."""
        return 0.5 * max(1e-3, float(self._result_frame_seconds))

    def _refuse_result(self, moment: float | None, generation: int, reason: str) -> None:
        """Keep the last picture, mark the half as a GAP, and say why. Nothing is offered for a refused result."""
        self._result_gap = True
        if telemetry.level() >= 1:
            telemetry.event("result refused", why=reason,
                            moment="-" if moment is None else f"{float(moment):.3f}",
                            generation=int(generation), held_generation=int(self._result_generation),
                            held_moment="-" if self._result_moment is None else f"{self._result_moment:.3f}")
        self.update()

    def _accept_result(self, moment: float | None, image: QImage | None, generation: int, *,
                       planes_frame: PreviewFrame | None = None, index: int | None = None,
                       seq: int | None = None, gui_ms: float | None = None) -> None:
        """Store an accepted result and offer it to the pair writer. The one place the pane's clock is written."""
        cleared = image is None and planes_frame is None
        if cleared:
            self._frame = None
            self._planes_size = None
            self._pending_frame_obj = None
            self._original = None
            self._original_shown = None
            self._original_time = None
            self._shown = None
            # A CLEAR means the pane is EMPTY: the painter drops the pair and refuses a source offered for a moment
            # from before the clear. Keeping a queue across it meant a late source re-keyed itself by its own moment
            # (`shown_moment()` is None here, so `set_source`'s guard was skipped by construction) and painted half a
            # picture back onto a pane told to be empty (CODE_PATTERNS §12: a clear is a clear).
            self._pair.clear()
            self.update()
        elif planes_frame is not None:
            self._frame = None
            self._planes_size = (int(planes_frame.width), int(planes_frame.height))
            self._pending_frame_obj = planes_frame
        else:
            self._frame = image.convertToFormat(QImage.Format.Format_RGBA8888)
            # An image frame replaces any standing planes picture: the geometry reads `_planes_size` when there is no
            # image, and the two shapes must never both claim the same frame.
            self._planes_size = None
            self._pending_frame_obj = None
        if int(generation) > self._result_generation:
            # A new render/file/seek: the cadence and the "older than" baseline belong to the previous generation.
            self._result_generation = int(generation)
            self._result_moment = None
            self._result_frame_seconds = 2.0 * SOURCE_HALF_FRAME_S
        previous = self._result_moment
        if moment is not None and previous is not None and int(generation) == self._result_generation:
            delta = float(moment) - float(previous)
            if 0.0 < delta <= 0.5:
                # The pane's own cadence, learned from consecutive results: the source tolerance is half of it.
                self._result_frame_seconds = delta
        self._result_moment = None if moment is None else float(moment)
        self._result_gap = False
        # The name and the time FIRST: the gate reads the time, and the draw log must carry THIS picture.
        if index is not None:
            self._shown = ("frame", int(index))
        elif seq is not None:
            self._shown = ("seq", int(seq))
        self._shown_time = None if moment is None else float(moment)
        if not cleared:
            # The picture is OFFERED, not painted: what reaches the renderer (a pair, a hold or the stale-limit half) is
            # the pair writer's decision, made behind the pane's one render gate below.
            self._pair.offer_result(Half(
                frame=self._pending_frame_obj if planes_frame is not None else self._preview_frame(
                    self._frame, index=index, pts=moment),
                moment=moment,
                generation=int(generation),
                image=self._frame,
                name=getattr(self, "_shown", None),
            ))
            self._request_frame_paint()
        telemetry.note_frame(gui_ms=gui_ms, index=index, seq=seq, media_time=moment,
                             generation=int(generation), cleared=cleared)

    def set_frame(self, image: QImage | None, *, gui_ms: float | None = None,
                  index: int | None = None, seq: int | None = None,
                  media_time: float | None = None, generation: int | None = None) -> None:
        """Compatibility adapter: translate `(image, media_time)` into the result guards, keeping the identity fields.

        `generation` here is the historical seek-answer generation, NOT the result identity; identity defaults to the
        pane's current generation. A producer that knows the render/file/seek identity uses `set_result`.
        """
        del generation
        current = int(self._result_generation)
        if image is None and media_time is not None:
            return
        if image is None:
            self._accept_result(None, None, current)
            return
        reason = self._result_refusal(media_time, current)
        if reason is not None:
            self._refuse_result(media_time, current, reason)
            return
        self._accept_result(media_time, image, current, index=index, seq=seq, gui_ms=gui_ms)

    def set_frame_planes(self, planes, strides, width: int, height: int, *, index: int | None = None,
                         pts: float | None = None, format: str = "yuv420p10le",
                         color_space: str | None = None, color_range: str | None = None,
                         gui_ms: float | None = None, seq: int | None = None,
                         result_generation: int | None = None) -> None:
        """Show a picture that arrives as YUV planes: (Y, U, V) bytes, their strides, and the picture's size.

        The YUV path keeps its own thin adapter, but goes through the SAME result guards as `set_result`: a planes
        frame from an older generation, or one that arrives after its successor, cannot paint either.

        `format` is `"yuv420p10le"` (10-bit, R16 textures) or `"yuv420p"` (8-bit, R8); `color_space`/`color_range` are
        the file's own tags (`"bt709"`/`"tv"`) and the shader's matrix is built from them - a `None` here means "not
        declared", and the renderer uses this app's writer's tags (bt709/tv) rather than guessing per file.

        No QImage, no `to_ndarray`, no `tobytes`: the planes are the producer's bytes and the renderer uploads them as
        they are (`PreviewRenderer.set_frame(PreviewFrame(...))`). The geometry (zoom/pan/fit), the HUD and the compare
        arithmetic all read the picture's SIZE, which is kept here (`_planes_size`) instead of in an image.
        """
        frame = PreviewFrame(
            format=str(format),
            width=int(width),
            height=int(height),
            pts=None if pts is None else float(pts),
            index=None if index is None else int(index),
            planes=tuple(memoryview(plane) for plane in planes),
            strides=tuple(int(stride) for stride in strides),
            color_space=color_space,
            color_range=color_range,
        )
        generation = int(self._result_generation if result_generation is None else result_generation)
        reason = self._result_refusal(pts, generation)
        if reason is not None:
            self._refuse_result(pts, generation, reason)
            return
        self._accept_result(pts, None, generation, planes_frame=frame,
                            index=index, seq=seq, gui_ms=gui_ms)

    def has_frame(self) -> bool:
        return self._frame is not None or self._planes_size is not None

    def oldest_result_time(self) -> float | None:
        """The oldest result still waiting for a partner - the next picture the writer can paint."""
        return self._pair.oldest_result_time()

    def committed_time(self) -> float | None:
        """The media time of the pair the RENDERER holds - the picture actually on screen, not the newest arrival.

        The compare's SUPPLY is driven from this (decode ahead of what is displayed) while its PRESENTATION is driven from
        the newest result (`shown_time`). One clock for both was a permanent lag: the source was asked to lead the newest
        frame, the newest frame moved by the same amount, and the pair never closed (measured 2026-11-26, split during
        inference: `playhead=0.417 newest=0.375`, every paint a timeout).
        """
        return self._pair.committed_time()

    def painted_pair(self) -> CommittedPair | None:
        """What the renderer holds: both halves of one moment, or the one half the pair policy allows.

        The pane's answer to "what is on screen", derived from the pair writer's own state - never from a flag kept
        beside it (a predicate that answered "in sync" about a picture with no source half is exactly the lie the audit
        removed, 2026-11-27 §6.1).
        """
        return self._pair.painted()

    def shown_time(self) -> float | None:
        """The media time of the picture on screen, when the publisher named it."""
        return getattr(self, "_shown_time", None)

    def shown(self) -> tuple[str, int] | None:
        """What the pane is painting: ``("frame", n)`` or ``("seq", n)``, or None before the first picture."""
        return getattr(self, "_shown", None)

    def original_shown(self) -> tuple[str, int] | None:
        """What the SOURCE half carries, when its host named it (the compare's other side of the subtraction)."""
        return getattr(self, "_original_shown", None)

    def original_shown_time(self) -> float | None:
        """The media time of the source picture, straight from the decoder that produced it."""
        return getattr(self, "_original_time", None)

    def shown_fingerprint(self) -> str | None:
        """A cheap fingerprint of the RESULT picture, for a log line that can be compared with a decode's pixels."""
        if telemetry.level() < 1 or self._frame is None:
            return None
        return self._fingerprint(self._frame)

    def _note_original_ask(self, seconds: float) -> None:
        """Remember when the source half was asked for: the draw line reports ask -> on screen."""
        self._original_ask_time = time.perf_counter()
        self._original_ask_wall = telemetry.wall()
        self._original_ask_target = float(seconds)

    @staticmethod
    def _fingerprint(image: QImage) -> str:
        """A small content hash of a picture, for the LAST step of the compare.

        The labels are not the proof: a file whose timeline is shifted names a picture with the wrong time and a
        compare of labels agrees with itself (measured 2026-11-25: the preview file starts at 0.083 s while its first
        frame is the source's 0.000 s frame - see the writer's `-avoid_negative_ts make_zero`). A 32x18 grey hash costs
        ~0.1 ms, is stable frame to frame, and lets two logs be joined by PIXELS, not by the name printed beside them.
        """
        try:
            small = image.scaled(32, 18, Qt.AspectRatioMode.IgnoreAspectRatio,
                                 Qt.TransformationMode.FastTransformation)
            small = small.convertToFormat(QImage.Format.Format_Grayscale8)
            if small.isNull():
                return "-"
            bits = small.constBits()
            raw = bits.tobytes() if hasattr(bits, "tobytes") else bytes(bits)
            return hashlib.blake2b(raw[:small.sizeInBytes()], digest_size=6).hexdigest()
        except Exception:
            return "-"

    def _identity_fields(self, *, original: bool) -> dict[str, object]:
        """The names of one half, as the log prints them (a bare `-` where the producer gave no name)."""
        if original:
            kind = self._original_shown
            media = self._original_time
        else:
            kind = getattr(self, "_shown", None)
            media = getattr(self, "_shown_time", None)
        fields: dict[str, object] = {}
        fields["t"] = "-" if media is None else f"{float(media):.3f}"
        fields["idx"] = "-" if kind is None else int(kind[1])
        return fields

    def _log_draw(self, *, half: str, picture: QImage | None, level: int | None = None,
                  present: bool | None = None) -> None:
        """One line per picture actually DRAWN: what, where, and when, with the ask it answers.

        `single` draws one half and replaces the other in the same texture, so its `original` line also names the
        result it is standing over - that pair IS the LMB compare, and printing both makes "one frame behind" a
        subtraction. `sbs`/`split` draw both halves in one pass; the line names both.

        `present` is for the planes path: a YUV picture has no QImage to fingerprint (`fp` prints `-`), but it is still
        a picture that was drawn, and a draw log that goes silent the moment the pane switches format would hide
        exactly the frames a colour regression appears in.
        """
        if present is None:
            present = picture is not None and not picture.isNull()
        if not present:
            return
        mode = self._view_mode
        fields: dict[str, object] = {"mode": mode, "half": half}
        fields.update(self._identity_fields(original=(half == "original")))
        # The result's identity comes from the pane's register, never inferred from `_frame`: what the last accepted
        # result NAMED, and which render/file/seek it belonged to.
        fields["result_moment"] = "-" if self._result_moment is None else f"{self._result_moment:.3f}"
        fields["generation"] = int(self._result_generation)
        fields["fp"] = "-" if picture is None or picture.isNull() else self._fingerprint(picture)
        if half == "original" and mode == "single":
            other = self._identity_fields(original=False)
            fields["result_t"] = other["t"]
            fields["result_idx"] = other["idx"]
        if half == "original":
            asked = getattr(self, "_original_ask_target", None)
            asked_at = getattr(self, "_original_ask_time", None)
            if asked is not None:
                fields["asked"] = f"{float(asked):.3f}"
            if asked_at is not None:
                fields["ask_ms"] = f"{(time.perf_counter() - asked_at) * 1000.0:.1f}"
            if getattr(self, "_original_ask_wall", None) is not None:
                fields["ask_wall"] = self._original_ask_wall
        fields["wall"] = telemetry.wall()
        telemetry.event("draw", **fields)

    def _log_draw_two(self, result: QImage | None, original: QImage | None, *,
                      present: bool | None = None, source_moment: float | None = None,
                      source_name: tuple[str, int] | None = None) -> None:
        """The compare modes paint both halves in ONE pass: name both pictures in one line.

        `source_moment`/`source_name` are the identity of the picture the RENDERER holds. They are passed in because
        `_original` is the newest ACCEPTED snapshot, which during a hold is the one the pane refused: a draw line that
        named it would print a picture nobody can see.
        """
        if present is None:
            present = result is not None and not result.isNull()
        if not present:
            return
        fields: dict[str, object] = {"mode": self._view_mode, "half": "pair"}
        fields.update({f"result_{k}": v for k, v in self._identity_fields(original=False).items()})
        fields["result_moment"] = "-" if self._result_moment is None else f"{self._result_moment:.3f}"
        fields["generation"] = int(self._result_generation)
        fields["result_fp"] = "-" if result is None or result.isNull() else self._fingerprint(result)
        # GEOMETRY, because "one half is smaller during the render and full after it" is a number, not an impression
        # (owner, 2026-11-26: "во время рендера почему-то оба видео меньше, а после - целиком заполняют"). The pane fits
        # `_image_size()` into the view rect, so the drawn size depends on the PICTURE's pixel size: a live tail frame
        # and the finished file's frame can differ by a whole factor of two, and then the fit (and the letterbox the eye
        # reads as "smaller") changes with the source of the picture.
        box = self._frame_rect()
        fields["px"] = f"{self._image_size()[0]}x{self._image_size()[1]}"
        fields["rect"] = f"{box.left():.0f},{box.top():.0f},{box.width():.0f},{box.height():.0f}"
        fields["scale"] = f"{self._result_scale():.4f}"
        if original is not None and not original.isNull():
            fields["original_px"] = f"{original.width()}x{original.height()}"
        if original is not None and not original.isNull():
            fields.update({f"original_{k}": v for k, v in self._identity_fields(original=True).items()})
            fields["original_fp"] = self._fingerprint(original)
            # The PAINTED picture's own name wins over the newest snapshot's (`_identity_fields` read `_original_*`):
            # during a hold the newest snapshot is the one the pane refused and nobody can see.
            if source_moment is not None:
                fields["original_t"] = f"{float(source_moment):.3f}"
            if source_name is not None:
                fields["original_idx"] = int(source_name[1])
        fields["wall"] = telemetry.wall()
        telemetry.event("draw", **fields)

    def set_source(self, snapshot: SourcePicture | None) -> None:
        """The pane's ONLY source API: show `snapshot`, or a gap when it is not this pane's moment.

        A snapshot is accepted only when it belongs to the SAME generation as the result on screen AND its moment is
        the result's own moment within half a frame (the pane's cadence, not a module constant). Outside that it is
        REFUSED and the pane keeps a gap - never the previous picture, which would be a neighbour of the moment the
        result is showing. That refusal is the compare's identity rule at its last line: a producer cannot hand over a
        picture it merely believes belongs here.

        A gap is not a blank: the pair already painted stays on screen (`preview/pair.py`). Only a pane that was told to
        be EMPTY refuses a source outright, because there is no moment on it that a picture could belong to (audit
        2026-11-27 §6.1b).
        """
        if snapshot is not None:
            moment = self.shown_moment()
            if self._pair.cleared:
                # The pane was told to be EMPTY and nothing has been published since: a late source for a moment
                # before the clear must not re-key itself (`shown_moment()` is None, so the check below cannot see it)
                # and paint half a picture onto an empty pane (CODE_PATTERNS §12).
                snapshot = None
            same_generation = int(snapshot.generation) == int(self._result_generation) if snapshot is not None else False
            if snapshot is not None and not same_generation:
                snapshot = None
            elif snapshot is not None and moment is not None and abs(float(snapshot.moment) - float(moment)) > self.source_half_frame_s():
                snapshot = None
        self._apply_source_picture(snapshot)

    def set_original(self, image: QImage | None, *, media_time: float | None = None,
                     index: int | None = None) -> None:
        """Compatibility adapter for a host that still speaks the old two-argument form.

        It routes through the SAME guard `set_source` applies, so it is not a bypass: there is one source path, and
        this only translates `(image, media_time)` into a `SourcePicture` of the pane's current generation.
        """
        if image is None:
            self._apply_source_picture(None)
            return
        shown = self.shown_moment()
        moment = float(media_time) if media_time is not None else (0.0 if shown is None else float(shown))
        snapshot = SourcePicture(moment=moment, picture=image, origin="decoder",
                                 generation=int(self._result_generation))
        if shown is not None and abs(moment - float(shown)) > self.source_half_frame_s():
            self._apply_source_picture(None)
            return
        self._apply_source_picture(snapshot, index=index)

    def shown_moment(self) -> float | None:
        """The moment the RESULT picture is showing, or `None` before a publisher named one.

        Public because the page's one synchroniser asks THIS for the moment to look up. It is the pane's own state, not
        a tab clock, so no flag elsewhere can disagree with what is on screen.
        """
        moment = getattr(self, "_shown_time", None)
        return None if moment is None else float(moment)

    def _apply_source_picture(self, snapshot: SourcePicture | None, *, index: int | None = None) -> None:
        """Store the accepted picture and hand the renderer the pair it belongs to.

        The source is **stretched up to the rendered frame's size with nearest-neighbour** (`FastTransformation`): the
        cheapest interpolation there is, and the only one that adds no information. The comparison then shows two
        pictures of the same pixel size, so every difference on screen is one the upscaler produced rather than a
        smoothing artefact of the viewer, and the source pixel at (x, y) lands on the same screen point in both states
        (owner, 2026-11-23). In `sbs` that same step is what makes the half-width original the same size as the
        result, i.e. the same apparent scale with no second zoom to keep in sync.
        """
        if snapshot is None:
            if self._original is None:
                return
            self._original = None
            self._original_shown = None
            self._original_time = None
            self._original_changed = False
            # The GAP is handed over, not acted on: in a compare mode the pair already painted stays on screen (that is
            # what stops a strip drag blanking the left half), while `single` drops its LMB flash at once.
            self._pair.offer_source(None)
            self._request_frame_paint()
            return
        moment = float(snapshot.moment)
        if (self._original is not None and self._original_time is not None
                and abs(float(self._original_time) - moment) < 1e-6):
            # The pane already holds this moment's picture: one handover per picture, not one per producer arrival.
            return
        telemetry.note_frame()
        self._original_shown = ("frame", int(index)) if index is not None else None
        self._original_time = moment
        self._original_changed = True
        image = snapshot.picture
        picture = image.convertToFormat(QImage.Format.Format_RGBA8888)
        # The target is the RESULT's picture size: its QImage, or the planes descriptor when the playback path
        # carries YUV. A planes result has no `_frame`; without this the source half kept its native 1440x1080 and
        # the two halves were at different scales (measured live 2026-11-25: single held fine, sbs's source did
        # not match the result).
        target_size = None if self._frame is None else self._frame.size()
        if target_size is None and getattr(self, "_planes_size", None) is not None:
            target_size = QSize(int(self._planes_size[0]), int(self._planes_size[1]))
        if target_size is not None and picture.size() != target_size:
            target = target_size
            # TWO cases, and the difference is measurable. The channel's payload box rounds each side to an even
            # number independently (`fit_box`), so a 4:3 source in a 421x512 pane becomes a 420x314 payload whose
            # aspect is 0.32 % wide; scaling the source into it with `KeepAspectRatio` gave 418x314 and the draw
            # then stretched that back to 420 - a further 0.48 %. The original therefore sat on a DIFFERENT pixel
            # grid than the result, which reads as "the frame is a bit wider" (owner, 2026-11-25). Within rounding
            # noise the source is scaled to the frame EXACTLY: same grid, honest compare, no invented geometry.
            source_aspect = picture.width() / float(picture.height())
            frame_aspect = target.width() / float(target.height())
            relative = abs(source_aspect - frame_aspect) / frame_aspect
            if relative <= 0.01:
                picture = picture.scaled(target, Qt.AspectRatioMode.IgnoreAspectRatio,
                                         Qt.TransformationMode.FastTransformation)
            else:
                # A REAL difference (border crop, an odd scale): stretching would show a straight line where the
                # picture has a curve. Keep the proportions and say so once - a silent deformation is worse than
                # either.
                picture = picture.scaled(target, Qt.AspectRatioMode.KeepAspectRatio,
                                         Qt.TransformationMode.FastTransformation)
                if getattr(self, "_compare_aspect_noted", None) != (picture.size(), target):
                    self._compare_aspect_noted = (picture.size(), target)
                    telemetry.event("compare aspect", source=f"{image.width()}x{image.height()}",
                                    fitted=f"{picture.width()}x{picture.height()}",
                                    frame=f"{target.width()}x{target.height()}",
                                    diff_pct=f"{relative * 100.0:.2f}")
        self._original = picture
        # Offered by its OWN moment, not by the pane's: the writer pairs a source with the result of the same moment
        # within the tolerance the pane states, so a picture that merely believes it belongs here cannot be drawn as a
        # neighbour. The picture's own moment travels in the payload, so the draw line stays honest.
        self._pair.offer_source(Half(
            frame=self._preview_frame(picture, index=index, pts=moment),
            moment=moment,
            generation=int(snapshot.generation),
            image=picture,
            name=None if index is None else ("frame", int(index)),
        ))
        self._request_frame_paint()

    def set_view_mode(self, mode: str) -> None:
        """`single` (today) | `sbs` (result right, original left) | `split` (one picture, original left of a divider).

        The invariant of the compare modes, and the reason the wheel is ignored over the original's half: **both
        pictures are drawn at the RESULT's source-pixel scale**, so a feature is the same apparent size on both sides.
        The original's zoom is derived from the result - it is never stored separately - and neither picture is
        stretched into its half: each keeps its own aspect and is letterboxed on its own. A seek is the TRANSPORT bar's
        gesture (`preview/timeline.py`), outside this widget, so it means the same playhead in every mode without the
        pane owning a second strip.

        Outside "single" the source picture is a STANDING pane - there is no button to hold - so opening a compare mode
        asks the host for the source at the playhead (`originalRequested`), and the request is repeated when the
        playhead moves. "single" keeps the LMB flash unchanged: press = original, release = result.
        """
        if mode not in VIEW_MODES:
            raise ValueError(f"unknown view mode {mode!r}: expected one of {', '.join(VIEW_MODES)}")
        if mode == self._view_mode:
            return
        self._view_mode = mode
        telemetry.event("view", mode=self._view_mode)
        # No divider outside `split`: drop the fade's clock and its timer instead of repainting for nothing.
        self._stop_divider_fade(reset=True)
        # Leaving "single" ends a flash: the source becomes the standing pane instead of a held picture.
        self._showing_original = False
        # Both textures get re-uploaded once: `single` may have left the source in the shared one, and the compare
        # modes keep the result there. Cheap (one upload) and it removes a whole class of "stale half" readings. The
        # The renderer owns those flags and the mode they depend on, so it hears the mode change before any frame
        # arrives; the pair writer hears it too, because what a mode may paint is the writer's policy.
        self._renderer.set_view_mode(mode)
        self._pair.set_view_mode(mode)
        if mode != "single":
            self._request_standing_original()
        # RE-FEED the renderer, do not just ask for a repaint: a mode change made while the job that owned the pictures
        # had already released its RHI resources left the result half BLACK - the pane still held the picture (and its
        # size, which is why the rects were right) but the renderer's texture was gone and a repaint re-drew nothing
        # (owner, 2026-11-26: in sbs the right half stayed empty until a new frame arrived). Handing the painted pair
        # over again marks the upload dirty and rebuilds it.
        if self._pair.refresh():
            self.update()

    def split_fraction(self) -> float:
        """Where the `split` divider sits, as a share of the picture's width (0..1)."""
        return float(self._split_fraction)

    def set_split_fraction(self, fraction: float) -> None:
        """Move the `split` divider; clamped so neither picture is hidden (that is what `single` is for)."""
        value = max(SPLIT_MIN_FRACTION, min(SPLIT_MAX_FRACTION, float(fraction)))
        if value == self._split_fraction:
            return
        self._split_fraction = value
        self.update()

    def _split_divider_x(self) -> float | None:
        """The divider's x in widget coordinates, or None when there is no divider to grab."""
        if self._view_mode != "split":
            return None
        picture = self._frame_rect()
        if picture.isEmpty():
            return None
        return picture.left() + picture.width() * self._split_fraction

    def _grabs_split(self, position) -> bool:
        x = self._split_divider_x()
        if x is None:
            return False
        picture = self._frame_rect()
        if not (picture.top() <= position.y() <= picture.bottom()):
            return False
        return abs(position.x() - x) <= float(scaled_px(SPLIT_GRAB_PX))

    def _request_frame_paint(self) -> None:
        """The pane's ONE render gate: every accepted picture ends here, and the pair writer makes the decision.

        A second painter is how the compare's defects came back four times (rules §2/§5), so this method stays the only
        door - it now carries no policy of its own. `preview/pair.py` owns it: a compare mode paints a pair or holds the
        last one, `MAX_PAIR_STALE_S` is the only way a half is shown alone, and the writer is the only code that calls
        `renderer.set_frame`/`set_original`.
        """
        self._pair.set_tolerance(self.source_half_frame_s())
        # A hand-over is what asks for the repaint: the writer hands nothing while it holds the last pair, and a gap
        # notification must not repaint (or re-upload) a picture that has not changed.
        if self._pair.paint():
            self.update()

    def divider_alpha(self, now: float | None = None) -> float:
        """The divider's alpha right now: 1 while the pointer is on it, 0 after `DIVIDER_FADE_S` away from it."""
        left = self._divider_left_at
        if left is None:
            return 1.0
        moment = time.monotonic() if now is None else float(now)
        return max(0.0, min(1.0, 1.0 - (moment - left) / DIVIDER_FADE_S))

    def _divider_hovered(self, position) -> bool:
        if not self._grabs_split(position):
            return False
        # The pointer is on the divider: it is fully visible and the fade is cancelled.
        self._divider_left_at = None
        self._stop_divider_fade()
        return True

    def _start_divider_fade(self) -> None:
        """Begin (or keep) the fade after the pointer leaves the divider; repaints until the rest alpha is reached."""
        if self._view_mode != "split" or self._divider_left_at is not None:
            return
        self._divider_left_at = time.monotonic()
        if self._divider_timer is None:
            from PySide6.QtCore import QTimer

            self._divider_timer = QTimer(self)
            self._divider_timer.setInterval(30)
            self._divider_timer.timeout.connect(self._tick_divider_fade)
        # START it: creating a timer does not make it tick, and without the ticks there is no repaint, so the rule
        # stayed at full alpha for ever in the first cut (measured live 2026-11-26: identical pixels at T+0.4 s and T+3 s).
        if not self._divider_timer.isActive():
            self._divider_timer.start()

    def _tick_divider_fade(self) -> None:
        self.update()
        if self._divider_left_at is not None and self.divider_alpha() <= 0.0:
            self._stop_divider_fade()

    def _stop_divider_fade(self, *, reset: bool = False) -> None:
        timer = self._divider_timer
        if timer is not None:
            timer.stop()
        if reset:
            self._divider_left_at = None

    def view_mode(self) -> str:
        """Which surfaces the pane shows: "single", "sbs" or "split"."""
        return self._view_mode

    def _request_standing_original(self) -> None:
        """Ask the host for the source frame at the playhead (compare modes only).

        The original is a standing pane outside `single`, so nothing else would ever request it: the pane asks once
        when a compare mode opens and again whenever the playhead moves (`set_progress`). The host decodes ONE frame on
        its own decoder and coalesces repeated requests (last one wins), which is what keeps the two pictures on one
        playhead during playback without the pane knowing anything about decoding.
        """
        # A picture, not an image: on the YUV playback path `_frame` is ALWAYS None (the pixels are the three planes),
        # so keying this guard on `_frame` meant the compare modes never asked for the source there - the standing
        # original was simply never fetched and sbs/split painted the result half with nothing beside it (owner's
        # screenshot, 2026-11-26: "ещё side by side почему-то мертв").
        if self._view_mode == "single":
            return
        # NO `_has_picture()` GUARD. It used to require a result frame, so a compare opened on an EMPTY pane (a queue item
        # with `produced=0`, a finished render whose tick has stopped, a preview that never played) asked nobody: the
        # owner's log shows the mode switch and then not one `compare:` line, and the source half waited for a picture
        # that would never come (2026-11-26: "почему мы не можем отобразить одновременно два кадра, а очень долго ждем
        # оригинал"). The source is the one picture that CAN be shown; ask for it at the playhead the pane stands on.
        _, _, playhead, _ = self._progress
        playhead = float(playhead)
        # Suppress only when the pane HOLDS the moment it is standing on: the picture is the fact, not a remembered
        # request. `_original_requested_at` used to be that memory and it outlived the picture (a mode change or a clear
        # left it behind, and a PAUSED pane then never asked again - the fourth latch of the same shape, seam audit
        # 2026-11-26). The pane names its own source time, so ask THAT.
        pane_time = self.original_shown_time()
        if (self._original is not None and pane_time is not None
                and abs(float(pane_time) - playhead) <= self.source_half_frame_s()):
            return
        self._note_original_ask(playhead)
        self.originalRequested.emit(playhead)

    def set_tint(self, rgba: tuple[float, float, float, float]) -> None:
        self._tint = tuple(float(part) for part in rgba)
        self.update()

    def set_progress(self, rendered: float, total: float, playhead: float = 0.0,
                     duration: float = 0.0) -> None:
        """How much the renderer has produced, and where playback stands - published to the TRANSPORT bar.

        `rendered/total` is the render's progress (live: frames or seconds produced so far); `playhead/duration` is
        media time. The pane paints no strip of its own any more: the numbers are kept because the draw log and the
        standing-source ask read them, and the bar reads them off `PreviewState` (the page's one push).

        In a compare mode this is also the clock of the standing original: a playhead that moved means the source
        pictured on the left is one frame old, so the host is asked for the new one (the same `originalRequested`
        the LMB flash uses). `_refresh_held_progress` re-publishes the SAME playhead on a tick, so a paused preview
        does not spam the decoder.
        """
        previous_playhead = self._progress[2]
        self._progress = (float(rendered), float(total), float(playhead), float(duration))
        self.update()
        if float(playhead) != float(previous_playhead):
            self._request_standing_original()

    def set_background_colour(self, colour) -> None:
        """The opaque colour the pane shows where no frame is drawn (the page's surface token).

        A colour is not a picture: a theme change must not re-open a pane that was told to be empty (the old body reset
        the "told to be empty" latch here, which let a late source repaint a cleared pane after a theme flip).
        """
        from PySide6.QtGui import QColor as _QColor

        self._clear_colour = _QColor(colour)
        self.update()

    def set_strip_visible(self, visible: bool) -> None:
        """RETIRED with the pane's painted strip: inert on purpose, and the only survivor of that organ.

        The shipped window is the one `src/ui/preview/controls.py` builds - a transport bar with a real scrubber
        (`PreviewTimeline`) - and it is installed by `ProcessTab._install_preview_controls`, which still makes this
        call because that file belongs to another lane in this wave. Two RHI pixel tests also call it to keep chrome
        out of their sampled rect. Three callers, one method: it is kept as a NAMED entry point so no caller logs a
        failure, and it does nothing because this class paints no strip and has no readout to hide. A future caller
        that "needs" it is asking for the second bar back (CODE_PATTERNS §2).
        """
        del visible

    zoomChanged = Signal(float)       # the new zoom factor
    originalRequested = Signal(float)  # playhead seconds: the host decodes the SOURCE frame at this time

    # -- interaction -----------------------------------------------------

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt spelling
        angle = event.angleDelta().y()
        pixels = event.pixelDelta().y()
        if telemetry.level() >= 2:
            # Both raw deltas, because the step is proportional to them: a device that sends only `pixelDelta` would
            # otherwise look like "the wheel does nothing" with no evidence in the log.
            telemetry.event("zoom raw", angle=angle, pixel=pixels)
        factor = wheel_zoom_factor(angle)
        if factor is None:
            event.accept()
            return                      # a zero delta is not a zoom; a pixel-only device answers `zoom raw` above
        # ONE scale for both halves, so the wheel means the same thing wherever it is turned (owner, 2026-11-26: "в
        # режиме side by side по оригиналу не работает зумирование, почему-то можно только на upscale"). In `sbs` the
        # anchor is MIRRORED into the result's half: the pan is shared, so anchoring where the pointer is on the left
        # would slide the result's content out from under a pointer that is not on it - and the mirrored point keeps
        # the source pixel under the cursor put, which is exactly what the gesture promises.
        anchor = QPointF(event.position())
        if self._view_mode == "sbs":
            left, right = self._half_rects()
            if left.contains(event.position()):
                anchor = QPointF(float(right.left()) + (event.position().x() - float(left.left())),
                                 float(event.position().y()))
        # FROM THE SCALE THE PICTURE IS DRAWN AT, not from the stored factor: on a 1920x1440 payload in a 504-wide pane
        # `fit` is 0.26 while the stored factor starts at 1.0, so the FIRST notch - or one 16-unit nudge from a trackpad -
        # jumped straight to ~1.0, i.e. a 4x crop that reads as "zoomed twice" and stays until Fit is pressed (owner,
        # 2026-11-26: "картинка то ли кропается, то ли зумится в 2 раза и затем это невозможно убрать"). A notch from the
        # fitted view is now a notch FROM the fitted view.
        base = self._fit_scale() if self._zoom_is_fit else getattr(self, "_zoom_target", self._zoom)
        target = base * factor
        self.set_zoom(target, around=QRectF(anchor, QSize(1, 1)))
        self.zoomChanged.emit(self._zoom)
        event.accept()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt spelling
        position = event.position()
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_anchor = position
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            # The `split` divider is a CONTROL and the only thing in a compare mode that takes a press: grabbing it must
            # happen before the compare consumption below, or the divider can never be moved (owner, 2026-11-26:
            # "было бы неплохо добавить видимый сплитер который передвигать можно").
            if self._divider_hovered(position):
                self._dragging_split = True
                self.setCursor(Qt.CursorShape.SizeHorCursor)
                event.accept()
                return
            # A compare mode has nothing to "flash" - the source already stands beside the result - and there the
            # picture IS a draggable surface: ONE policy in every compare mode (owner, 2026-11-26: "во всех режимах
            # сравнения должна быть одинаковая политика взаимодействия - лкм - оригинал (если один кадр), лкм+зажатие -
            # панарамирование со сменой курсора на руку"). The middle button keeps working everywhere for the habit it
            # already built, and the divider (above) stays the one control that takes a press before the picture does.
            if self._view_mode != "single":
                self._panning = True
                self._pan_anchor = position
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                return
            # LMB = "show the original": the surface asks the host for the source frame at the playhead and draws it as
            # soon as `set_original` arrives. The request carries the media time, so the host decodes one frame, not a
            # stream - the renderer's own original is never involved (plan: 3b, GUI-only).
            self._showing_original = True
            # LMB drag pans (owner, 2026-11-24: "вместо СКМ перевесить на ЛКМ драг"; BlinkTable's `drag_pan` is the
            # reference). The press still ASKS for the original immediately - holding the button to compare is a
            # measured behaviour and a drag only cancels it once the pointer actually moves.
            self._left_press = QPointF(position)
            # Ask for the SHOWN picture's own media time, not the progress marker. `_progress` is refreshed only when a
            # frame paints, so after a strip press it still carries the previous clock (measured 2026-11-25: a press
            # logged "asked for the source at 2.383s (pane playhead 9.827s)"). The picture names itself through
            # `set_frame`, which is the same number the result half is drawn from.
            named = self.shown_time()
            ask = float(named) if named is not None else float(self._progress[2])
            telemetry.event("hold on", playhead=f"{ask:.3f}")
            self._note_original_ask(ask)
            self.originalRequested.emit(ask)
            self.update()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt spelling
        position = event.position()
        if not self._dragging_split and not self._panning and self._left_press is None:
            # The divider's own hover: full alpha while the pointer is on it, a fade once it leaves. The cursor says
            # "this can be dragged" before the press does. Skipped while a drag runs: the drag's own cursor is the truth
            # then, and re-deriving it from the pointer's position cleared the hand mid-pan.
            if self._divider_hovered(position):
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            elif self._view_mode == "split":
                self.unsetCursor()
                self._start_divider_fade()
        if self._dragging_split:
            picture = self._frame_rect()
            if not picture.isEmpty() and picture.width() > 0.0:
                self.set_split_fraction((position.x() - picture.left()) / picture.width())
            event.accept()
            return
        if self._panning:
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            delta = position - self._pan_anchor
            self._pan_anchor = position
            self.pan_by(delta.x(), delta.y())
            event.accept()
            return
        if self._left_press is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            moved = position - self._left_press
            if abs(moved.x()) > PAN_THRESHOLD_PX or abs(moved.y()) > PAN_THRESHOLD_PX:
                # The compare stays ON while the drag runs: panning moves the view over the same picture, so the
                # original is exactly what the eye wants to inspect there (owner, 2026-11-25: "чтобы во время
                # панорамирования показывался оригинал, а не апскейл"). The release ends it, like any hold; the
                # geometry is shared, so the result appears where the drag left the picture.
                self._left_press = None
                self._panning = True
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                self._pan_anchor = position
                # The distance already travelled counts: without this the first move of every drag was a dead zone,
                # which reads as "the picture sticks for a moment before it follows".
                self.pan_by(moved.x(), moved.y())
                event.accept()
                return
            event.accept()
            return
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt spelling
        """The pointer left the widget: the divider starts its fade (it was not "still hovered" any more)."""
        if not self._dragging_split:
            self.unsetCursor()
            self._start_divider_fade()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt spelling
        if event.button() == Qt.MouseButton.LeftButton and self._dragging_split:
            self._dragging_split = False
            self.unsetCursor()
            event.accept()
            return
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton) and self._panning:
            self._panning = False
            self._left_press = None
            self.unsetCursor()
            if self._showing_original:
                # The drag KEEPS the compare on (it inspects the original), so the release is what ends it. Without
                # this the pane stayed on the original for ever: `_panning` took the early return above, and the only
                # place that clears the flag and calls `set_original(None)` was never reached (owner, 2026-11-25:
                # "после отжимания панорамирования кадр к апскейлу не возвращается").
                self._showing_original = False
                telemetry.event("hold off")
                self.set_original(None)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._left_press = None
            if self._showing_original:
                self._showing_original = False
                telemetry.event("hold off")
                self.set_original(None)   # one frame is held at a time; the next press asks again
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt spelling
        self.reset_view()
        self.zoomChanged.emit(self._zoom)
        event.accept()

    def _on_theme_changed(self) -> None:
        """The compare boundary resolves its tokens at paint time; a light/dark switch only needs a repaint."""
        self.update()

    # -- view ------------------------------------------------------------

    def samples_nearest(self) -> bool:
        """Whether the view is magnified (or exactly 1:1) and therefore sampled with `Nearest`.

        Magnification is where the owner wants crisp source pixels; below 1 the linear sampler averages, because
        nearest-neighbour MINIFICATION is aliasing, not sharpness. The FIT view draws at `_fit_scale()`, not at
        `_zoom`, so it is the effective scale that decides - a fit of a frame SMALLER than the pane does magnify.
        """
        scale = self._fit_scale() if self._zoom_is_fit else float(self._zoom)
        return samples_nearest_for(scale)

    def zoom_is_fit(self) -> bool:
        """Whether the view is the FITTED one - the host mirrors this instead of assuming the last wheel left the fit."""
        return bool(self._zoom_is_fit)

    def zoom(self) -> float:
        """The scale the picture is DRAWN at: `_fit_scale()` while the view is fitted, else the stored factor.

        Reporting `_zoom` alone lied in the opening view - it said 1.0 while the pane drew the frame at 0.5 (measured
        2026-11-24: the zoom line read `zoom=1.000 fit=1` on a fitted 1600x900 frame in an 800x600 pane).
        """
        if self._zoom_is_fit:
            return float(self._fit_scale())
        return self._zoom

    def _min_zoom(self) -> float:
        """The wheel's floor: the whole frame visible. Zooming further out would leave gutters for no reason, and the
        initial view already is that scale."""
        return max(1e-6, self._fit_scale())

    def set_zoom(self, zoom: float, *, around: QRectF | None = None) -> None:
        """Set the zoom factor, clamped to [fit, 8]; `around` keeps that rect (the cursor area) put.

        The factor is a SOURCE-pixel scale, so 1.0 means one frame pixel per screen pixel; on a large frame inside a
        small pane `fit` is below 1 and the range is [fit, 8].
        """
        target = max(self._min_zoom(), min(MAX_ZOOM, float(zoom)))
        self._zoom_target = target
        # CONTINUOUS on purpose: the step size is the wheel's, not the ladder's. `samples_nearest` decides the filter
        # instead - crisp at whole factors, linear between them (a little soft, never ragged, never a 100 % jump).
        if around is not None:
            # Keep the frame point under the cursor where it is: solve for the pan that puts it back after the scale
            # change. The first version scaled the offset by a factor, which drifted the picture at the edges.
            width, height = self._image_size()
            if width and height:
                rect = self._frame_rect()
                anchor = around.center()
                u = (anchor.x() - rect.left()) / max(1e-6, rect.width())
                v = (anchor.y() - rect.top()) / max(1e-6, rect.height())
                drawn_w, drawn_h = width * target, height * target
                view = self._view_rect()
                self._pan[0] = anchor.x() - u * drawn_w - view.left() - (view.width() - drawn_w) / 2.0
                self._pan[1] = anchor.y() - v * drawn_h - view.top() - (view.height() - drawn_h) / 2.0
        self._zoom = target
        self._zoom_is_fit = False
        # After the flag flips: the line used to print the PREVIOUS state (`fit=1` on a zoom that just left fit).
        telemetry.event("zoom", zoom=f"{self._zoom:.3f}", fit=int(self._zoom_is_fit),
                        nearest=int(self.samples_nearest()))
        self._clamp_pan()
        self.update()

    def reset_view(self, *, actual_size: bool = False) -> None:
        """Back to the opening view: the whole frame fitted into the pane (or 1:1 when the user asks for it)."""
        self._zoom = 1.0
        self._zoom_target = self._fit_scale() if not actual_size else 1.0
        self._zoom_is_fit = not actual_size
        self._pan = [0.0, 0.0]
        self.update()

    def pan_by(self, dx: float, dy: float) -> None:
        self._pan[0] += float(dx)
        self._pan[1] += float(dy)
        self._clamp_pan()
        self.update()

    # -- geometry ---------------------------------------------------------

    def _image_size(self) -> tuple[int, int]:
        # Geometry always comes from the RENDERED frame, even while the original is shown: the compare must not move or
        # resize anything, or the eye compares two framings instead of two pictures (the source is drawn into the same
        # screen rect, which is also what "hold to see the original" means in the reference app).
        image = self._frame
        if image is None or image.isNull():
            # A planes frame carries no image: its size is the descriptor's, and it is the same number the RGBA path
            # would have produced for the same picture.
            planes_size = getattr(self, "_planes_size", None)
            if planes_size is not None:
                return (int(planes_size[0]), int(planes_size[1]))
            # BEFORE THE FIRST RESULT the source is the only picture: let it define the box, or the fit has no size to
            # work with (scale 1.0) and a 1440-wide source is drawn at native size, clipped to a corner of the half.
            source = self._source_size()
            if source is not None:
                return source
            return (0, 0)
        return (image.width(), image.height())

    def _source_size(self) -> tuple[int, int] | None:
        """The pixel size of the source half the RENDERER holds, or None when it holds none.

        Read from the pair writer, not from `_original`: while a strip drag leaves the newest snapshot a frame behind,
        `_original` is the picture the pane REFUSED and the renderer is still painting the last matched pair. The
        geometry must describe what is on screen, or the held source is laid out from a picture nobody can see.
        """
        pair = self._pair.painted()
        if pair is not None and pair.source is not None and pair.source.frame is not None:
            frame = pair.source.frame
            width = getattr(frame, "width", None)
            height = getattr(frame, "height", None)
            if width and height:
                return (int(width), int(height))
            image = pair.source.image
            if image is not None and not image.isNull():
                return (image.width(), image.height())
        newest = self._original
        if newest is not None and not newest.isNull():
            return (newest.width(), newest.height())
        return None

    def _fit_scale(self) -> float:
        width, height = self._image_size()
        if not width or not height:
            return 1.0
        view = self._view_rect()
        return min(view.width() / width, view.height() / height)

    def _result_scale(self) -> float:
        """The source-pixel scale the RESULT is drawn at: the fit scale on open, the wheel's factor after that."""
        return self._fit_scale() if self._zoom_is_fit else self._zoom

    def _half_rects(self) -> tuple[QRectF, QRectF]:
        """The two halves of `sbs`, as `(original, result)` - the source first, the render second.

        That order is the before/after reading Topaz's compare has, and it is the same order `split` uses (its left
        part is the original). ONE line: swapping these two rects is the whole layout decision.
        """
        half = self.width() / 2.0
        height = float(self.height())
        return QRectF(0.0, 0.0, half, height), QRectF(half, 0.0, half, height)

    def _view_rect(self) -> QRectF:
        """The widget rect the RESULT is laid out in: the whole pane, or the right half in `sbs`.

        One rect to fit, clamp and anchor into means one zoom for the result - the wheel and the middle drag address
        it even when the pointer is over the original's half.
        """
        if self._view_mode == "sbs":
            return self._half_rects()[1]
        return QRectF(0.0, 0.0, float(self.width()), float(self.height()))

    def _original_view_rect(self) -> QRectF:
        """The widget rect the ORIGINAL is laid out in: the left half in `sbs`; the picture's rect otherwise."""
        if self._view_mode == "sbs":
            return self._half_rects()[0]
        return self._frame_rect()

    def _frame_rect(self) -> QRectF:
        """The RESULT's rectangle in widget pixels at the current zoom/pan, centred in its view rect."""
        width, height = self._image_size()
        if not width or not height:
            return QRectF(0, 0, 0, 0)
        scale = self._result_scale()
        drawn_w, drawn_h = width * scale, height * scale
        view = self._view_rect()
        left = view.left() + (view.width() - drawn_w) / 2.0 + self._pan[0]
        top = view.top() + (view.height() - drawn_h) / 2.0 + self._pan[1]
        return QRectF(left, top, drawn_w, drawn_h)

    def _original_rect(self) -> QRectF:
        """The SOURCE picture's rectangle: the result's rect when they overlay, its own half in `sbs`.

        In `sbs` the placement comes from the source's own pixel size at the RESULT's source-pixel scale - never from
        the half's size. That is the invariant `set_view_mode` states: same apparent scale on both sides, each picture
        at its own aspect (a source with a different aspect letterboxes inside its half instead of being stretched to
        it). The pan is the result's, so panning either half shows the same region on both.
        """
        size = self._source_size()
        if size is None:
            return QRectF()
        if self._view_mode != "sbs":
            return self._frame_rect()
        scale = self._result_scale()
        view = self._original_view_rect()
        # THE RESULT'S SCREEN BOX, with the source's own ASPECT. `set_original` scales the source to the result when it
        # arrives, but a picture that arrived BEFORE a result existed (or before a size change) keeps its own pixel size,
        # and a source from the render block arrives at the block's allocation box - so the two halves were laid out at
        # wildly different sizes, each centred in its half: the "huge empty gap" between them (owner, 2026-11-26: "оба
        # видео сделают между собой огромный пустой промежуток места"). The invariant `set_view_mode` states is one
        # SCALE and each picture at its own aspect, so the box is the result's drawn size and a source with a different
        # aspect is fitted INTO that box (it letterboxes there instead of overflowing its half).
        result = self._frame_rect()
        width, height = size
        box_w, box_h = result.width(), result.height()
        if box_w <= 0.0 or box_h <= 0.0:
            box_w, box_h = width * scale, height * scale
        source_aspect = width / float(height or 1)
        box_aspect = box_w / float(box_h or 1)
        if abs(source_aspect - box_aspect) / box_aspect <= 0.01:
            drawn_w, drawn_h = box_w, box_h
        elif source_aspect > box_aspect:
            drawn_w, drawn_h = box_w, box_w / source_aspect
        else:
            drawn_w, drawn_h = box_h * source_aspect, box_h
        left = view.left() + (view.width() - drawn_w) / 2.0 + self._pan[0]
        top = view.top() + (view.height() - drawn_h) / 2.0 + self._pan[1]
        return QRectF(left, top, drawn_w, drawn_h)

    def _original_clip_rect(self) -> QRectF:
        """Where the original may paint: its half in `sbs`, the left of the divider in `split`.

        Scissoring to this rect is what lets a zoomed picture reach the edge of its own surface without leaking over
        the other one; in `sbs` it is also why an aspect-mismatched source cannot invade the result's half.
        """
        if self._view_mode == "sbs":
            return self._original_view_rect()
        if self._view_mode == "split":
            picture = self._frame_rect()
            if picture.isEmpty():
                return QRectF()
            divider = picture.left() + picture.width() * self._split_fraction
            return QRectF(picture.left(), picture.top(), max(0.0, divider - picture.left()),
                          picture.height())
        return QRectF()

    def _clamp_pan(self) -> None:
        """Keep the picture over the pane: no gutter when zoomed past the pane, no drifting it away when it fits."""
        width, height = self._image_size()
        if not width or not height:
            return
        scale = self._result_scale()
        drawn_w, drawn_h = width * scale, height * scale
        view = self._view_rect()
        max_x = max(0.0, (drawn_w - view.width()) / 2.0)
        max_y = max(0.0, (drawn_h - view.height()) / 2.0)
        self._pan[0] = max(-max_x, min(max_x, self._pan[0]))
        self._pan[1] = max(-max_y, min(max_y, self._pan[1]))

    # -- RHI ---------------------------------------------------------------
    #
    # The renderer (`preview_renderer.py`) owns every QRhi object, the uploads and the pass; this side owns the widget
    # contract. Three members are the whole interface between them: `_preview_frame` wraps a picture into the ingestion
    # contract, `_preview_scene` hands over one frame's geometry, and `_on_rhi_draw` turns "what was drawn" into the
    # draw log.

    def _preview_frame(self, image: QImage | None, *, index: int | None = None,
                       pts: float | None = None) -> PreviewFrame | None:
        """Wrap a picture into the renderer's ingestion contract (`rgba8`, the image already converted).

        This is the additive seam the YUV lane grows from: a producer with planes will build its own `PreviewFrame`
        (`format="yuv420p10le"`, `planes=`, `strides=`) and hand it to the renderer; the image path below stays the
        rgba8 wrapper, and `set_frame`/`set_original` keep the signatures their callers already use.
        """
        if image is None:
            return None
        return PreviewFrame(format="rgba8", width=image.width(), height=image.height(),
                            pts=None if pts is None else float(pts),
                            index=None if index is None else int(index), image=image)

    def _preview_scene(self, target) -> PreviewScene:
        """Everything the renderer draws this frame, computed here: geometry, clips, HUD, sampler, clear.

        The renderer never reads the widget: the mvp blocks are the same `_mvp_floats` the tests pin, the clip rects
        are the same `_view_rect`/`_original_clip_rect` the compare modes always used, and the HUD vertices are
        `_hud_vertices` unchanged. `target` is the CURRENT paint's render target - a resize replaces it, so it is
        passed per frame instead of cached.

        The source half is laid out from the picture the RENDERER holds (`_source_size`), not from the newest snapshot:
        while the writer holds the last matched pair through a source gap, the half on screen is the held one.
        """
        source = self._source_size()
        return PreviewScene(
            target=target,
            rhi=self.rhi(),
            mode=self._view_mode,
            result_mvp=tuple(self._mvp_floats(self._frame_rect())),
            original_mvp=None if source is None else tuple(self._mvp_floats(self._original_rect())),
            tint=tuple(self._tint),
            nearest=self.samples_nearest(),
            result_rect=self._frame_rect(),
            result_clip=self._view_rect(),
            original_clip=self._original_clip_rect(),
            show_original=bool(self._showing_original and source is not None),
            hud_vertices=tuple(self._hud_vertices()),
            viewport=(float(self.width()), float(self.height())),
            dpr=float(self.devicePixelRatioF() or 1.0),
            clear=self._opaque_clear(),
        )

    def _on_rhi_draw(self, drawn: PreviewDrawn) -> None:
        """One pass finished: write the draw line from the widget's own bookkeeping (`_log_draw`).

        The renderer reports WHAT it drew (which half, whether the source was in the pass); the identity, the media
        times, the fingerprints and the telemetry level all live here, so the lines are exactly the ones the old
        in-render logging wrote.
        """
        # GEOMETRY counts as a change: a zoom or a pan repaints the same pictures at a different box, and without this
        # line a gesture left no trace at the default telemetry level - so "the wheel did nothing over the original" and
        # "the picture moved" could not be told apart from the log at all (measured 2026-11-26: the wheel answered
        # `accepted`, the pane repainted, and no `draw` line followed because the PICTURES were the same).
        geometry = self._geometry_signature()
        geometry_changed = geometry != getattr(self, "_logged_geometry", None)
        if drawn.mode == "single":
            picture = self._original if drawn.half == "original" else self._frame
            present = self._has_picture() if drawn.half == "result" else picture is not None
            if drawn.half == "original" and (self._original_changed or geometry_changed or telemetry.level() >= 2):
                self._log_draw(half="original", picture=picture)
                self._original_changed = False
                self._logged_geometry = geometry
            elif drawn.half == "result" and (geometry_changed or telemetry.level() >= 2):
                self._log_draw(half="result", picture=picture, present=present)
                self._logged_geometry = geometry
            return
        # Both halves in one pass: one line names both pictures (a new original, a new box, or every paint at level 2).
        # The picture named is the one the RENDERER holds (the pair writer's source), not the newest snapshot: while a
        # drag leaves the arrival a frame behind, `_original` is a picture the pane refused and nobody can see.
        painted = self._pair.painted()
        source = None if painted is None else painted.source
        original = None if source is None else source.image
        if drawn.drew_original and (self._original_changed or geometry_changed or telemetry.level() >= 2):
            # `present` from the WIDGET, not inferred from `_frame`: a planes (YUV) result has no QImage, so the
            # inference in `_log_draw_two` answered "no picture" and EVERY paint with a source on the live path wrote
            # nothing - the draw log was blind exactly where the compare runs (measured 2026-11-26: the fit paint, with
            # no source yet, was logged; the same paint with a source was not, which is what made a zoom look untraceable).
            self._log_draw_two(self._frame, original, present=bool(original) or self._has_picture(),
                               source_moment=None if source is None else source.moment,
                               source_name=None if source is None else source.name)
            self._original_changed = False
            self._logged_geometry = geometry
        elif geometry_changed or telemetry.level() >= 2:
            # The box is the RESULT's, and a zoom/pan moves it whether or not the source half has arrived: a paused pane
            # with no source would otherwise leave a gesture with no trace at all (measured 2026-11-26).
            self._log_draw_two(self._frame, None, present=self._has_picture())
            self._logged_geometry = geometry

    def _geometry_signature(self) -> str:
        """What a zoom or a pan changes: the picture's pixel size and the box it is drawn into."""
        width, height = self._image_size()
        box = self._frame_rect()
        return (f"{width}x{height}|{box.left():.1f},{box.top():.1f},"
                f"{box.width():.1f},{box.height():.1f}|{self._result_scale():.4f}")

    def _has_picture(self) -> bool:
        """Whether the RESULT half carries a picture at all - an image, or the planes of a YUV frame."""
        return self._frame is not None or getattr(self, "_planes_size", None) is not None

    def initialize(self, command_buffer) -> None:  # noqa: N802 - Qt spelling
        """Start the telemetry session, then hand the resource lifetime to the renderer.

        Qt calls `initialize()`/`releaseResources()` on EVERY resize (the widget's backing target is reallocated at the
        new pixel size), not only on a real context change. The renderer's rule for that is unchanged: the same RHI
        keeps its resources and re-uploads the standing pictures, a new one releases and rebuilds. Releasing here
        unconditionally is the "the preview turns transparent at any sneeze" defect - the frame texture dies with the
        resource set, a resize does not mark it dirty, and a paused render never sends another frame. This side owns
        the session line and the no-RHI fallback message because both describe the widget, not the resources.
        """
        if not getattr(self, "_telemetry_session", False):
            self._telemetry_session = True
            telemetry.session_start(pane_w=self.width(), pane_h=self.height(),
                                    dpr=round(float(self.devicePixelRatioF()), 2), mode=self._view_mode)
        rhi = self.rhi()
        if rhi is None:
            # The platform has no QRhi at all (measured 2026-11-23 on the owner's Wayland/NVIDIA session: Qt prints
            # `QRhiWidget: No QRhi` after `eglSwapBuffers failed with 0x300d` and the widget is never composited).
            # Building anything here is meaningless and calling into a null RHI is how a paint path crashes; the pane
            # stays an empty plate and the log says which fallback exists (`RVE_UI_RASTER_PREVIEW=1`).
            if not getattr(self, "_no_rhi_logged", False):
                self._no_rhi_logged = True
                print("[preview] no QRhi on this platform: the pane cannot draw (set RVE_UI_RASTER_PREVIEW=1 "
                      "for the pixmap fallback)", flush=True)
            return
        kept = self._renderer.initialize(command_buffer, rhi, self.renderTarget())
        if kept:
            # Same RHI, just a new target size: every resource is still valid, so a repaint is enough - the renderer
            # has marked the standing pictures for re-upload.
            self.update()

    def releaseResources(self) -> None:  # noqa: N802 - Qt spelling
        self._pair.release()
        self._renderer.release()

    def render(self, command_buffer) -> None:  # noqa: N802 - Qt spelling
        """Hand this paint to the renderer: the scene first, then one pass into the current target."""
        target = self.renderTarget()
        if target is None or self.rhi() is None:
            return
        painted = self._pair.painted()
        if self._view_mode != "single" and painted is not None:
            # What this PAINT actually shows, both halves: the only number that can tell a consistent pair from a
            # flicker. The selection log says what was chosen; this says what reached the screen together. The COUNTER
            # is unconditional (two integer ops, and it rides the level-1 aggregate); the per-paint line is level 2.
            lone = getattr(painted, "lone_half", None)
            if lone:
                # A paint that showed ONE half is not a pair: it enters the denominator and has its own counter, or a
                # pane painting alone every second reads as perfectly paired (telemetry audit 2026-11-27).
                telemetry.note_pair(lone_half=str(lone))
            elif painted.source is not None and painted.moment is not None:
                result = painted.moment
                source = painted.source.moment
                if result is not None and source is not None:
                    period = max(1e-3, float(self._result_frame_seconds))
                    delta_frames = pair_distance_frames(source, result, period)
                    telemetry.note_pair(delta_frames=delta_frames)
                    if telemetry.level() >= 2:
                        newest = getattr(self, "_shown_time", None)
                        lag = "" if newest is None else round(pair_distance_frames(newest, result, period), 2)
                        telemetry.event("painted pair", result=round(float(result), 3),
                                        source=round(float(source), 3),
                                        delta_frames=round(delta_frames, 2), lag_frames=lag,
                                        frame_s=round(period, 5))
        self._renderer.set_scene(self._preview_scene(target))
        self._renderer.render(command_buffer)

    def _mvp_floats(self, rect: QRectF | None = None) -> list[float]:
        """Unit quad -> the picture's rect on the pane, column-major for the uniform buffer.

        Written out instead of composed from ``QMatrix4x4``: PySide6 hands that object back row-major, and a transposed
        transform in a shader is a silently mirrored picture - the kind of bug that reads as "the widget drew something"
        and is caught only by looking at the pixels. The rect comes from the caller in the compare modes (one picture
        per SRB); the default is the result's own placement.
        """
        rect = self._frame_rect() if rect is None else rect
        view = self.size()
        if rect.isEmpty() or not view.width() or not view.height():
            return [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        cx = (rect.left() + rect.right()) / 2.0
        cy = (rect.top() + rect.bottom()) / 2.0
        sx = rect.width() / 2.0
        sy = rect.height() / 2.0
        return [
            2.0 * sx / view.width(), 0.0, 0.0, 0.0,          # x basis
            0.0, -2.0 * sy / view.height(), 0.0, 0.0,        # y basis (NDC is up, pixels down)
            0.0, 0.0, 1.0, 0.0,
            2.0 * cx / view.width() - 1.0, 1.0 - 2.0 * cy / view.height(), 0.0, 1.0,
        ]

    def _opaque_clear(self) -> QColor:
        """The clear colour with alpha forced to 255: an RHI widget cannot alpha-blend with its siblings, so any
        transparency here is a hole, not a blend (framework precedent `painting-qrhiwidget-cannot-alpha-blend-siblings`)."""
        colour = QColor(self._clear_colour)
        colour.setAlpha(255)
        return colour

    def _hud_vertices(self) -> list[float]:
        """Rectangles for the HUD pass, in widget pixels: the compare modes' ONE boundary, and nothing else.

        The pane used to paint its own strip here as well (produced band, playhead handle, hover and blocked markers);
        that was the second bar implementation beside `PreviewTimeline` and it is deleted (audit 2026-11-27 §6.6). What
        remains is a control of the PICTURE: the `sbs` seam and the `split` divider with its grip, from
        `_compare_boundary_rects`. The HUD is a rect-only pass.
        """
        rects: list[tuple[float, float, float, float, tuple[float, float, float, float]]] = []
        self._compare_boundary_rects(rects, floor=float(self.height()))
        vertices: list[float] = []
        for x, y, w, h, colour in rects:
            corners = [(x, y), (x + w, y), (x, y + h), (x + w, y), (x + w, y + h), (x, y + h)]
            for cx, cy in corners:
                vertices.extend([cx, cy, *colour])
        return vertices

    def _compare_boundary_rects(self, rects, *, floor: float) -> None:
        """Append the compare modes' boundary: the `sbs` seam, or the `split` divider with its drag grip.

        `floor` is the lowest pixel a boundary may reach: the pane's own height, because the transport bar is a SIBLING
        widget under the picture rather than chrome this pass paints (`_hud_vertices` used to pass the strip's top here
        when the pane painted a strip of its own).
        """
        colours = strip_colours()
        divider = rgba(colours["divider"], 0.35)
        if self._view_mode == "sbs":
            middle = self.width() / 2.0
            picture = self._frame_rect()
            # A seam belongs to a PICTURE: an empty pane draws nothing at all (its own pin asserts exactly that).
            if floor > 0 and not picture.isEmpty():
                top = max(0.0, picture.top())
                bottom = min(floor, picture.bottom())
                if bottom > top:
                    rects.append((middle - 0.5, top, 1.0, bottom - top, divider))
        elif self._view_mode == "split" and self._source_size() is not None:
            picture = self._frame_rect()
            top = max(0.0, picture.top())
            bottom = min(floor, picture.bottom())
            if bottom > top:
                x = picture.left() + picture.width() * self._split_fraction
                # An OUTLINED rule in TOOLKIT pixels: 1 px light, 2 px dark core, 1 px light. Narrower than the first
                # cut (7/4 read heavy), scaled by `scaled_px` so a HiDPI window gets the same physical weight (owner,
                # 2026-11-26). It fades out after the pointer leaves and leaves only the grip at 20 %, so the picture is
                # never permanently divided by a line the owner is not using.
                alpha = self.divider_alpha()
                side = float(scaled_px(1))
                core = float(scaled_px(2))
                rule = side + core + side
                # A SEAM, not a highlight: a bright white rule at 0.90 glared over the picture (owner, 2026-11-26:
                # "линия крайне яркая"). The light edges carry the contrast at 0.38 and the dark core does the reading
                # at 0.78, so the line is legible over a bright sky and a dark tree without being the brightest thing
                # on screen - the GRIP is what says "control", and it is deliberately the softer of the two.
                if alpha > 0.0:
                    rects.append((x - rule / 2.0, top, side, bottom - top, rgba(colours["divider"], 0.38 * alpha)))
                    rects.append((x - core / 2.0, top, core, bottom - top,
                                  rgba(colours["divider_outline"], 0.78 * alpha)))
                    rects.append((x + core / 2.0, top, side, bottom - top, rgba(colours["divider"], 0.38 * alpha)))
                # The grip's own alpha runs from the owner's 20 % at rest to 55 % under the pointer: `max(0.2, alpha)`
                # multiplied the tone-down's 0.55 into 11 % and the resting handle stopped reading as one.
                grip_alpha = DIVIDER_REST_ALPHA + (0.55 - DIVIDER_REST_ALPHA) * alpha
                grip_h = min(float(scaled_px(36)), max(float(scaled_px(24)), (bottom - top) * 0.12))
                grip_w = float(scaled_px(14))
                grip_top = top + (bottom - top - grip_h) / 2.0
                outline = float(scaled_px(1))
                rects.append((x - grip_w / 2.0, grip_top, grip_w, grip_h, rgba(colours["divider"], grip_alpha)))
                rects.append((x - grip_w / 2.0 + outline, grip_top + outline, grip_w - 2 * outline, grip_h - 2 * outline,
                              rgba(colours["divider_outline"], grip_alpha * 0.78)))
                chevron = max(1.0, outline)
                for offset in (-float(scaled_px(4)), 0.0, float(scaled_px(4))):
                    rects.append((x - chevron / 2.0, grip_top + grip_h / 2.0 + offset - chevron / 2.0,
                                  chevron, chevron, rgba(colours["divider"], grip_alpha)))

    def grabFrame(self) -> QImage:  # noqa: N802 - Qt spelling is `grabFramebuffer`, this is the plan's name
        return self.grabFramebuffer()
