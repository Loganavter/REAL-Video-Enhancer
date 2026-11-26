"""The GPU preview surface: one QRhiWidget that paints a full-resolution frame with zoom and pan.

Scope of this file today: the frame. The strip, the playhead and the LMB compare overlay are separate layers that will
sit in the same pass (see plan-rve-video-preview-widget-2026-11-23); keeping them out until the frame is proven means the
first thing measured is the thing that can be wrong in the most ways (texture upload, colour, aspect, zoom).

The pane carries **two surfaces, one playhead** (`set_view_mode`, 2026-11-24): the render result and the source
picture, shown as `single` (today's pane), `sbs` (result right, original left, half the width each) or `split` (one
picture, the original left of a vertical divider). They are two draws inside the SAME RHI pass - one clear per frame -
and the strip/HUD is painted once for the whole widget, so a seek means the same playhead in every mode.

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
pass; this class owns the widget API, the zoom/pan/mvp geometry, the frame/state bookkeeping, the strip and its child
widgets. It hands the renderer one `PreviewScene` per paint and receives what was drawn back through `on_draw`.

The pane accepts a picture in two shapes: `set_frame(image, ...)` (RGBA8888, the fallback and the compare's source
path, unchanged) and `set_frame_planes(planes, strides, w, h, ...)` (YUV420 planes, the playback path's cheap one - no
QImage, no `to_ndarray`, the colour conversion in the fragment shader). Both feed the same geometry, HUD and log.
"""

from __future__ import annotations

import hashlib
import time

from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QRhiWidget

from sli_ui_toolkit.ui.managers.theme_manager import ThemeManager
from sli_ui_toolkit.ui.managers.ui_scale import UiScale, scaled_px

from src.ui.video import telemetry
from src.ui.widgets.playback_strip import (
    PlaybackTimeReadout,
    playhead_halo_radius,
    rgba,
    strip_colours,
    strip_handle_radius,
    strip_margin,
    strip_track_height,
)
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

#: A compare mode shows ONE moment, not two: a repaint is allowed only while the two halves are within this. One and a
#: half frames at 24 fps, and the number is a MEASURED trade-off, not a preference: at half a frame the exact source
#: arrives ~20 ms after its result, the gate holds, and the hold times out on the paints whose source is slower still
#: (`pair_off` 20-27 of 27 in the bad seconds); at one and a half the gate paints a one-frame-old neighbour instead
#: (`pair_sub` 7-9 of 20-27, `pair_ahead` 0 - a constant lag, never a wobble). The real fix is pairing by TIME in the
#: pane (a result queue and a source queue, painted when both hold the same moment), not a wider or narrower gate.
PAIR_SYNC_SLACK_S = 0.0625

#: ...but never held longer than this. A decoder that never answers, or a pair that is systematically a frame apart,
#: must not freeze the picture: a small mismatch is better than a still image.
PAIR_SYNC_HOLD_S = 0.25

#: How close two names must be to be the same moment, in milliseconds. The keys are both `round(time * 1000)`, so this
#: absorbs the rounding of two independently derived timestamps and nothing else.
PAIR_MATCH_MS = 2

#: A jump bigger than this in the result's own clock is a SEEK, not playback: the pair queues are flushed so a stale
#: half can never be paired with a fresh one. Measured live 2026-11-26 (split, seek during playback): the transition
#: painted `delta_frames=-80.06` twice - the valve firing while both the old and the new moments sat in the queues.
PAIR_JUMP_S = 0.5

#: How long the valve is held back after a SEEK (the queues were flushed, so the old source half is stale by definition).
#: Long enough for a keyframe rewind plus a couple of frames (~200 ms measured), short enough that a dead decoder cannot
#: freeze the picture: past this the valve paints whatever it has and says so in the log.
PAIR_JUMP_WAIT_S = 1.0

#: How many result and source pictures the pair queue may hold while a partner is missing. The rule is GStreamer's
#: `GstAggregator`: every input keeps its own queue, the aggregate step runs only when ALL inputs have a buffer, and a pad
#: that misses the bounded wait has its buffers SKIPPED rather than shifting the pair (`gstaggregator.c`:
#: `gst_aggregator_check_pads_ready`, `gst_aggregator_wait_and_check`, `gst_aggregator_pad_skip_buffers`).
PAIR_QUEUE_MAX = 8

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
        #: (rendered, total, playhead, duration) - the two time bases the plan keeps apart. `total`/`duration` of 0
        #: hide their marker; the strip stays visible as long as a frame is shown.
        self._progress = (0.0, 0.0, 0.0, 0.0)
        self._strip_visible = True
        self._hud_dirty = True
        self._scrubbing = False
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
        #: When the pair gate started holding a paint (monotonic), or None while a paint is allowed.
        self._pair_blocked_at: float | None = None
        #: When a seek flushed the queues (monotonic); the valve stands down for `PAIR_JUMP_WAIT_S` after it.
        self._pair_await_fresh_at: float | None = None
        #: The pair queue: `milliseconds -> picture`. The renderer is handed a RESULT and a SOURCE of the same moment,
        #: whichever arrived first, instead of the newest of each - that is what removes "one frame ahead, one frame
        #: behind" at the seam (owner, 2026-11-26).
        self._result_queue: dict[int, tuple] = {}
        self._source_queue: dict[int, tuple] = {}
        #: The time of the pair the RENDERER holds - not of the newest halves. A repaint can come from anywhere (the
        #: compositor, a HUD fade, a resize); those must draw the last pair that belonged together, not a half-updated
        #: one, or the very flicker the gate exists to remove comes back through the side door.
        self._committed_result_time: float | None = None
        self._committed_source_time: float | None = None
        #: The fade needs repaints, and a paused pane has no other clock. 30 ms is smooth at this size and stops as soon
        #: as the rule reaches its rest alpha.
        self._divider_timer = None
        #: the playhead the standing original was last asked for, so playback refreshes it once per move
        self._original_requested_at: float | None = None
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
        #: precedents. The pane is therefore always opaque, with the page's own colour.
        self._clear_colour = self.palette().window().color()
        #: strip geometry in widget px, from the toolkit's own `Slider` constants through `UiScale` (see
        #: `_apply_strip_metrics`); the HUD pass draws in widget pixels, so nothing here is a hand-picked number
        self._strip_margin = strip_margin()
        self._strip_height = strip_track_height()
        self._handle_radius = strip_handle_radius()
        self._playhead_halo_radius = playhead_halo_radius(self._handle_radius)
        #: hovered position on the strip (fraction 0..1), and whether it is past the produced edge
        self._hover_fraction: float | None = None
        self._hover_blocked = False
        #: how far the strip may seek, when the owner can render any frame on demand (an export's display feed).
        #: `None` = follow the produced edge (the live tail's honest limit); see `set_seek_ceiling`.
        self._seek_ceiling: float | None = None
        #: Opaque clear colour for the pane. QRhiWidgets composite by OVERWRITING their rect, so a clear with alpha
        #: below 255 punches a hole (black on Wayland, white on xcb) instead of letting the page show through - the
        #: framework's `lifecycle-first-frame-signal-gating` and `painting-qrhiwidget-cannot-alpha-blend-siblings`
        #: precedents. The pane is therefore always opaque, with the page's own colour.
        self._clear_colour = self.palette().window().color()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Hover feedback on the strip (marker + forbidden cursor over the unprocessed part) needs move events while no
        # button is held; without tracking, `mouseMoveEvent` only arrives during a drag (measured: the cursor never
        # changed in the hover test).
        self.setMouseTracking(True)
        #: The time display is a real widget, NOT part of the HUD: that pass is one rect-only draw and has no glyphs.
        #: A child of the pane composites above the RHI surface (the framework's precedent for a QRhiWidget: it
        #: overwrites its own rect and cannot alpha-blend, while its children are composed by Qt afterwards), and the
        #: pane owns both its text and its placement so a resize/seek keeps the two in step.
        self._time_readout = PlaybackTimeReadout(self)
        self._time_readout.setVisible(self._strip_visible)
        # The strip reads the theme at paint time, so a theme flip must repaint it; the readout recolours itself.
        ThemeManager.get_instance().theme_changed.connect(self._on_theme_changed)
        # The toolkit scale multiplies the strip's geometry; re-read the constants and re-place the caption.
        UiScale.get_instance().scale_changed.connect(self._on_scale_changed)

    def setAutoFillBackground(self, enabled: bool) -> None:  # noqa: N802 - Qt spelling
        """A no-op, on purpose.

        `qrhi-gotchas.md` §"QRhiWidget autofill": autofill's palette clear fights RHI texture compositing, so a widget
        that carries it shows a stale frame - BlinkTable's canvas answers the same call with `pass`. The pane's own
        colour is the opaque clear inside the RHI pass, never the palette.
        """

    def showEvent(self, event) -> None:  # noqa: N802 - Qt spelling
        """Re-post a paint after being shown: a widget that was hidden may come back with an undrawn target."""
        super().showEvent(event)
        self._position_readout()
        self.update()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt spelling
        """Ask for a repaint after a resize - `initialize()` has just been called with a new target size."""
        super().resizeEvent(event)
        self._position_readout()
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

    def set_frame(self, image: QImage | None, *, gui_ms: float | None = None,
                  index: int | None = None, seq: int | None = None,
                  media_time: float | None = None, generation: int | None = None) -> None:
        """Show this picture. A `None` clears the pane (no frame yet).

        The pixels go to the renderer as a `PreviewFrame` (`format="rgba8"`, the converted image); the widget keeps the
        same image for its geometry, its fingerprints and the draw log. Signatures here are the public API and do not
        change - a producer that speaks planes will hand the renderer a `PreviewFrame` directly (the YUV lane).
        """
        if image is None and media_time is not None:
            # A display feed can publish an EMPTY slot while still naming a moment (`_on_display_frame(None, ...)`): that
            # is "nothing to show this time", not "clear the pane". Queued, it became a result entry with no image AND no
            # planes, and `_commit_pair` then called `set_frame(None)` - the empty RIGHT half of sbs after a finished
            # render (owner, 2026-11-26: "оба видео сделают между собой огромный пустой промежуток"). The picture stays;
            # an explicit clear (a `None` with no name) still clears below.
            return
        self._frame = None if image is None else image.convertToFormat(QImage.Format.Format_RGBA8888)
        # An image frame replaces any standing planes picture: the geometry reads `_planes_size` when there is no
        # image, and the two shapes must never both claim the same frame.
        self._planes_size = None
        self._pending_frame_obj = None
        # NOT handed to the renderer here: `_request_frame_paint` commits the PAIR (see `_commit_pair`).
        self._cleared_empty = False
        self._refresh_readout()
        # The name and the time FIRST: the gate reads the time, and the draw log must carry THIS picture.
        if index is not None:
            self._shown = ("frame", int(index))
        elif seq is not None:
            self._shown = ("seq", int(seq))
        self._shown_time = None if media_time is None else float(media_time)
        self._enqueue(self._result_queue, media_time,
                      (self._frame, getattr(self, "_shown", None), media_time, index))
        self._request_frame_paint()
        telemetry.note_frame(gui_ms=gui_ms, index=index, seq=seq, media_time=media_time, generation=generation)

    def set_frame_planes(self, planes, strides, width: int, height: int, *, index: int | None = None,
                         pts: float | None = None, format: str = "yuv420p10le",
                         color_space: str | None = None, color_range: str | None = None,
                         gui_ms: float | None = None, seq: int | None = None) -> None:
        """Show a picture that arrives as YUV planes: (Y, U, V) bytes, their strides, and the picture's size.

        ADDITIVE to `set_frame(image, ...)`, which is unchanged: the RGBA path is the fallback, this is the one the
        playback host can choose, and it is driven directly by tests and by the RHI proof. `format` is
        `"yuv420p10le"` (10-bit, R16 textures) or `"yuv420p"` (8-bit, R8); `color_space`/`color_range` are the file's
        own tags (`"bt709"`/`"tv"`) and the shader's matrix is built from them - a `None` here means "not declared",
        and the renderer uses this app's writer's tags (bt709/tv) rather than guessing per file.

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
        self._frame = None
        self._planes_size = (int(width), int(height))
        self._pending_frame_obj = frame
        self._cleared_empty = False
        self._refresh_readout()
        if index is not None:
            self._shown = ("frame", int(index))
        elif seq is not None:
            self._shown = ("seq", int(seq))
        self._shown_time = None if pts is None else float(pts)
        self._enqueue(self._result_queue, pts, (None, getattr(self, "_shown", None), pts, index, frame))
        self._request_frame_paint()
        telemetry.note_frame(gui_ms=gui_ms, index=index, seq=seq, media_time=pts)

    def has_frame(self) -> bool:
        return self._frame is not None or self._planes_size is not None

    def oldest_result_time(self) -> float | None:
        """The oldest result still waiting for a partner - the next picture the queue can paint."""
        if not self._result_queue:
            return None
        return min(self._result_queue) / 1000.0

    def committed_time(self) -> float | None:
        """The media time of the pair the RENDERER holds - the picture actually on screen, not the newest arrival.

        The compare's SUPPLY is driven from this (decode ahead of what is displayed) while its PRESENTATION is driven from
        the newest result (`shown_time`). One clock for both was a permanent lag: the source was asked to lead the newest
        frame, the newest frame moved by the same amount, and the pair never closed (measured 2026-11-26, split during
        inference: `playhead=0.417 newest=0.375`, every paint a timeout).
        """
        return getattr(self, "_committed_result_time", None)

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
                      present: bool | None = None) -> None:
        """The compare modes paint both halves in ONE pass: name both pictures in one line."""
        if present is None:
            present = result is not None and not result.isNull()
        if not present:
            return
        fields: dict[str, object] = {"mode": self._view_mode, "half": "pair"}
        fields.update({f"result_{k}": v for k, v in self._identity_fields(original=False).items()})
        fields["result_fp"] = "-" if result is None or result.isNull() else self._fingerprint(result)
        if original is not None and not original.isNull():
            fields.update({f"original_{k}": v for k, v in self._identity_fields(original=True).items()})
            fields["original_fp"] = self._fingerprint(original)
        fields["wall"] = telemetry.wall()
        telemetry.event("draw", **fields)

    def set_original(self, image: QImage | None, *, media_time: float | None = None,
                     index: int | None = None) -> None:
        """The source frame for the compare: the LMB flash in `single`, a standing pane in the compare modes.

        The source is **stretched up to the rendered frame's size with nearest-neighbour** (`FastTransformation`): the
        cheapest interpolation there is, and the only one that adds no information. The comparison then shows two
        pictures of the same pixel size, so every difference on screen is one the upscaler produced rather than a
        smoothing artefact of the viewer, and the source pixel at (x, y) lands on the same screen point in both states
        (owner, 2026-11-23). In `sbs` that same step is what makes the half-width original the same size as the
        result, i.e. the same apparent scale with no second zoom to keep in sync.
        """
        if image is None:
            self._original = None
            self._original_shown = None
            self._original_time = None
            self._original_changed = False
        else:
            # The source half of a compare mode: a picture the pane accepted, counted in the window's `frames`.
            telemetry.note_frame()
            self._original_shown = ("frame", int(index)) if index is not None else None
            self._original_time = None if media_time is None else float(media_time)
            self._original_changed = True
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
        if image is not None and self._original is not None:
            self._enqueue(self._source_queue, media_time, (self._original, media_time, index))
        self._request_frame_paint()

    def set_view_mode(self, mode: str) -> None:
        """`single` (today) | `sbs` (result right, original left) | `split` (one picture, original left of a divider).

        The invariant of the compare modes, and the reason the wheel is ignored over the original's half: **both
        pictures are drawn at the RESULT's source-pixel scale**, so a feature is the same apparent size on both sides.
        The original's zoom is derived from the result - it is never stored separately - and neither picture is
        stretched into its half: each keeps its own aspect and is letterboxed on its own. The strip is painted once for
        the whole widget and drives the same `seekRequested`, so a seek means the same playhead in every mode.

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
        self._hud_dirty = True
        # Both textures get re-uploaded once: `single` may have left the source in the shared one, and the compare
        # modes keep the result there. Cheap (one upload) and it removes a whole class of "stale half" readings. The
        # renderer owns those flags and the mode they depend on, so it hears the mode change before any frame arrives.
        self._renderer.set_view_mode(mode)
        if mode != "single":
            self._request_standing_original()
        # RE-FEED the renderer, do not just ask for a repaint: a mode change made while the job that owned the pictures
        # had already released its RHI resources left the result half BLACK - the pane still held the picture (and its
        # size, which is why the rects were right) but the renderer's texture was gone and a repaint re-drew nothing
        # (owner, 2026-11-26: in sbs the right half stayed empty until a new frame arrived). Handing the newest pair over
        # again marks the upload dirty and rebuilds it.
        self._commit_newest()

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

    def _enqueue(self, queue: dict, media_time, payload) -> None:
        """Keep one picture in a pair queue, bounded: a picture whose partner never comes is dropped, not accumulated."""
        if media_time is None:
            return
        self._flush_on_jump(float(media_time))
        queue[int(round(float(media_time) * 1000.0))] = payload
        while len(queue) > PAIR_QUEUE_MAX:
            del queue[min(queue)]

    def _flush_on_jump(self, moment: float) -> None:
        """Drop both queues when the picture jumps - a seek must not pair the old moment with the new one.

        Playback advances by a frame at a time and never trips this; a press on the timeline does, and the pane then
        keeps showing the LAST consistent pair until the new moment's partner arrives (the hold's own window), instead of
        the valve painting `result=new source=old` after 500 ms (the `delta_frames=-80` transition measured live).
        """
        keys = list(self._result_queue) + list(self._source_queue)
        if self._committed_result_time is not None:
            # The committed moment is the picture ON SCREEN even when both queues are empty (a pair was painted and
            # pruned), and a seek away from it is exactly the case that must flush.
            keys.append(int(round(float(self._committed_result_time) * 1000.0)))
        if not keys:
            return
        # Compared against EVERYTHING the queues hold, not against the committed time: after the result of the new
        # moment has already been queued (the flush ran for it), its partner arrives and must NOT trip the flush again -
        # that threw away the very picture the pair was waiting for (caught by the regression test the first time it
        # ran, 2026-11-26).
        nearest = min(abs(key / 1000.0 - float(moment)) for key in keys)
        if nearest <= PAIR_JUMP_S:
            return
        self._result_queue.clear()
        self._source_queue.clear()
        self._pair_blocked_at = None
        self._pair_release_count = 0
        # ...and the valve is NOT allowed to paint the old source beside the new result: `_original` still holds the
        # picture of the moment we just left, so `_commit_newest` would pair the two across the jump. The pane keeps the
        # last consistent pair on screen until the new moment's partner arrives (bounded below, so a dead decode cannot
        # freeze it for ever).
        self._pair_await_fresh_at = time.monotonic()

    def _pick_pair(self):
        """The newest result that HAS a partner, as `(result_key, source_key)`; `None` while one is missing.

        GStreamer's aggregator rule, ported: the aggregate step runs only when every input has a buffer, and a pad that
        misses the wait has its buffers skipped rather than shifting the pair. A result older than the newest source by
        more than the tolerance will never get a partner, so it is dropped here instead of holding the picture.
        """
        if not self._result_queue or not self._source_queue:
            return None
        # EXACT, in milliseconds: the two keys are the same moment rounded independently, so a difference of more than a
        # couple of milliseconds is a different frame. The queue can afford this because it WAITS for the partner instead
        # of substituting a neighbour - which is the whole point of it (measured 2026-11-26: with the gate's 1.5-frame
        # tolerance the queue happily paired a frame with its predecessor, `delta_frames=-1.01` on 71 of 154 paints).
        slack = PAIR_MATCH_MS
        newest_source = max(self._source_queue)
        for key in sorted(self._result_queue, reverse=True):
            nearest = min(self._source_queue, key=lambda candidate: abs(candidate - key))
            if abs(nearest - key) <= slack:
                return key, nearest
            if key < newest_source - slack:
                del self._result_queue[key]
        return None

    def pair_in_sync(self) -> bool:
        """True when the two halves are the same picture (or there is only one picture to show)."""
        if self._view_mode == "single" or self._original is None:
            return True
        # TIME, not frame numbers: the tail's index and the source decoder's index do not share a base even with both on
        # content time (measured 2026-11-26: every paint refused when the gate compared indices), while the timestamps
        # agree to within the drift the slack below covers.
        result = getattr(self, "_shown_time", None)
        source = getattr(self, "_original_time", None)
        if result is None or source is None:
            # An unnamed picture cannot be compared, so it must not be held: the publisher decides, not the gate.
            return True
        return abs(float(result) - float(source)) <= PAIR_SYNC_SLACK_S

    def _commit_newest(self) -> None:
        """Hand the renderer the NEWEST of each half - `single`, and the timeout valve when no pair can be completed."""
        shown = getattr(self, "_shown", None)
        index = None if shown is None else shown[1]
        if self._frame is not None:
            self._renderer.set_frame(self._preview_frame(self._frame, index=index, pts=self._shown_time))
        elif getattr(self, "_pending_frame_obj", None) is not None:
            # The planes path: the producer's own `PreviewFrame` IS the result (no QImage exists for it).
            self._renderer.set_frame(self._pending_frame_obj)
        else:
            self._renderer.set_frame(None)
        self._renderer.set_original(self._preview_frame(self._original, index=index, pts=self._original_time))
        self._committed_result_time = getattr(self, "_shown_time", None)
        self._committed_source_time = getattr(self, "_original_time", None)
        self._pair_blocked_at = None
        self._pair_release_count = 0
        self.update()

    def _commit_pair(self) -> bool:
        """Hand the renderer a RESULT and a SOURCE of the same moment; False when one of them has not arrived yet.

        The RENDERER is what draws, so it is what must hold a matched pair: keeping `_frame`/`_original` as the newest
        for geometry and the readout is free, but handing the renderer a half whose partner has not arrived means every
        repaint from outside the gate (a compositor frame, the divider's fade) shows the mismatch. Painting the pair -
        whichever half arrived first - is what removes "one frame ahead, one frame behind" at the seam.
        """
        pair = self._pick_pair()
        if pair is None:
            return False
        result_key, source_key = pair
        entry = self._result_queue[result_key]
        picture, source_time, source_index = self._source_queue[source_key]
        result_image, _shown, result_time = entry[0], entry[1], entry[2]
        planes = entry[4] if len(entry) > 4 else None
        if result_image is not None:
            self._renderer.set_frame(self._preview_frame(result_image, index=source_index, pts=result_time))
        elif planes is not None:
            self._renderer.set_frame(planes)
        else:
            # Defensive: an entry with neither an image nor planes carries no picture at all. Dropping it is right - the
            # renderer keeps the last result - while `set_frame(None)` cleared the half (see `set_frame`).
            del self._result_queue[result_key]
            del self._source_queue[source_key]
            return False
        self._renderer.set_original(self._preview_frame(picture, index=source_index, pts=source_time))
        self._committed_result_time = None if result_time is None else float(result_time)
        self._committed_source_time = None if source_time is None else float(source_time)
        self._pair_await_fresh_at = None
        # Everything at or before the committed pair is spent: the queues stay small and the next pick starts after it.
        for key in [key for key in self._result_queue if key <= result_key]:
            del self._result_queue[key]
        for key in [key for key in self._source_queue if key <= source_key]:
            del self._source_queue[key]
        self._pair_blocked_at = None
        self._pair_release_count = 0
        self.update()
        return True

    def pair_paint_ready(self, now: float | None = None) -> bool:
        """Whether a repaint may happen NOW: in sync, or the hold window has run out."""
        if self.pair_in_sync():
            return True
        blocked = getattr(self, "_pair_blocked_at", None)
        if blocked is None:
            return False
        moment = time.monotonic() if now is None else float(now)
        return (moment - blocked) >= PAIR_SYNC_HOLD_S

    def _request_frame_paint(self) -> None:
        """Hand the renderer a pair that belongs together, or hold until it does.

        Without the gate each half repainted on its own arrival, so a fast GPU showed the result at T+1 beside the source
        at T for one frame, then both at T+1 - a 40 ms window of mismatched halves, repeated, which is the flicker the
        owner reports. The gate holds the newer half until its partner arrives; the timer is the way out when it never does.
        """
        if self._view_mode == "single" or self._original is None:
            self._commit_newest()
            return
        if self._commit_pair():
            return
        if getattr(self, "_pair_blocked_at", None) is None:
            self._pair_blocked_at = time.monotonic()
            if telemetry.level() >= 2:
                telemetry.event("pair hold", result=round(float(getattr(self, "_shown_time", 0.0) or 0.0), 3),
                                source=round(float(getattr(self, "_original_time", 0.0) or 0.0), 3))
            from PySide6.QtCore import QTimer

            QTimer.singleShot(int(PAIR_SYNC_HOLD_S * 1000) + 1, self._release_pair_paint)

    def _release_pair_paint(self) -> None:
        """The hold window ran out: REPEAT the pair already on screen, and commit a mismatch only after a second window.

        Showing the mismatched halves is the flicker itself; a picture that repeats for another 120 ms is not visible.
        The second window is the hard stop - a decoder that never answers must not freeze the pane, and a small
        mismatch is better than a still image.
        """
        if getattr(self, "_pair_blocked_at", None) is None:
            return
        fresh = getattr(self, "_pair_await_fresh_at", None)
        if fresh is not None and (time.monotonic() - float(fresh)) < PAIR_JUMP_WAIT_S:
            # A seek just happened: hold the old (consistent) pair a little longer rather than paint the new result
            # beside the source of the moment we left. Past `PAIR_JUMP_WAIT_S` the valve is allowed again, so a dead
            # source still cannot freeze the pane for ever.
            from PySide6.QtCore import QTimer

            QTimer.singleShot(int(PAIR_SYNC_HOLD_S * 1000) + 1, self._release_pair_paint)
            return
        self._pair_release_count = getattr(self, "_pair_release_count", 0) + 1
        if self._pair_release_count < 2:
            from PySide6.QtCore import QTimer

            QTimer.singleShot(int(PAIR_SYNC_HOLD_S * 1000) + 1, self._release_pair_paint)
            self.update()          # the committed pair again: no advance, and nothing to mismatch
            return
        self._pair_release_count = 0
        if telemetry.level() >= 2:
            telemetry.event("pair release", result=round(float(getattr(self, "_shown_time", 0.0) or 0.0), 3),
                            source=round(float(getattr(self, "_original_time", 0.0) or 0.0), 3))
        # The timeout's valve: a picture that cannot be paired is still shown, and the log says the pair was missed.
        self._commit_newest()

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
        if self._view_mode == "single" or not self._has_picture():
            return
        _, _, playhead, _ = self._progress
        playhead = float(playhead)
        if self._original is not None and playhead == self._original_requested_at:
            return
        self._original_requested_at = playhead
        self._note_original_ask(playhead)
        self.originalRequested.emit(playhead)

    def set_tint(self, rgba: tuple[float, float, float, float]) -> None:
        self._tint = tuple(float(part) for part in rgba)
        self.update()

    def set_progress(self, rendered: float, total: float, playhead: float = 0.0,
                     duration: float = 0.0) -> None:
        """Two markers, two bases: how much the renderer has produced, and where playback stands.

        `rendered/total` is the render's progress (live: frames or seconds produced so far); `playhead/duration` is
        media time. Either pair with a zero denominator hides its marker - the strip stays, the marker does not
        pretend to be at 0%.

        In a compare mode this is also the clock of the standing original: a playhead that moved means the source
        pictured on the left is one frame old, so the host is asked for the new one (the same `originalRequested`
        the LMB flash uses). `_refresh_held_progress` re-publishes the SAME playhead on a tick, so a paused preview
        does not spam the decoder.
        """
        previous_playhead = self._progress[2]
        self._progress = (float(rendered), float(total), float(playhead), float(duration))
        self._hud_dirty = True
        self._refresh_readout()
        self.update()
        if float(playhead) != float(previous_playhead):
            self._request_standing_original()

    def set_background_colour(self, colour) -> None:
        """The opaque colour the pane shows where no frame is drawn (the page's surface token)."""
        from PySide6.QtGui import QColor as _QColor

        self._clear_colour = _QColor(colour)
        self._cleared_empty = False
        self.update()

    def set_strip_visible(self, visible: bool) -> None:
        self._strip_visible = bool(visible)
        self._hud_dirty = True
        self._time_readout.setVisible(self._strip_visible)
        self._position_readout()
        self.update()

    def set_seek_ceiling(self, fraction: float | None) -> None:
        """How far the strip may seek: `None` = the produced edge, a fraction = the owner's own reach.

        The produced edge is the honest limit while the pane follows a live tail: nothing exists past it. An export's
        display feed is different - it renders any frame of the timeline on demand - so the host lifts the edge to
        1.0 there and restores `None` when the tail owns the pane again. The hover cursor follows, so the strip stops
        claiming the rest of the timeline is unreachable while the feed can serve it.
        """
        ceiling = None if fraction is None else max(0.0, min(1.0, float(fraction)))
        if ceiling == self._seek_ceiling:
            return                      # the host re-asserts this every tick; an unchanged edge repaints nothing
        self._seek_ceiling = ceiling
        self._hover_fraction = None
        self.unsetCursor()
        self.update()

    seekRequested = Signal(float)     # fraction 0..1 of the strip (inside the produced range, or the host's ceiling)
    seekBlocked = Signal(float)       # a seek was asked for past the reachable edge; nothing moved
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
        # In `sbs` the original is placed from the result's scale, so a wheel over that half has nothing of its own
        # to change: doing nothing there (instead of silently moving the far picture) is what keeps the two surfaces
        # comparable one gesture at a time.
        if self._view_mode == "sbs" and self._original_view_rect().contains(event.position()):
            event.accept()
            return
        # FROM THE SCALE THE PICTURE IS DRAWN AT, not from the stored factor: on a 1920x1440 payload in a 504-wide pane
        # `fit` is 0.26 while the stored factor starts at 1.0, so the FIRST notch - or one 16-unit nudge from a trackpad -
        # jumped straight to ~1.0, i.e. a 4x crop that reads as "zoomed twice" and stays until Fit is pressed (owner,
        # 2026-11-26: "картинка то ли кропается, то ли зумится в 2 раза и затем это невозможно убрать"). A notch from the
        # fitted view is now a notch FROM the fitted view.
        base = self._fit_scale() if self._zoom_is_fit else getattr(self, "_zoom_target", self._zoom)
        target = base * factor
        self.set_zoom(target, around=QRectF(event.position(), QSize(1, 1)))
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
            if self._strip_visible and self._strip_rect().contains(position):
                fraction = self._fraction_at(position.x())
                if not self._seekable(fraction):
                    # Past the reachable edge there is nothing to seek to. Refusing is the honest answer: emitting a
                    # seek the owner cannot satisfy would move the playhead over a picture that does not exist.
                    self.seekBlocked.emit(fraction)
                    event.accept()
                    return
                self._scrubbing = True
                telemetry.event("strip press", fraction=f"{fraction:.4f}")
                self.seekRequested.emit(fraction)
                event.accept()
                return
            # The `split` divider is a CONTROL and the only thing in a compare mode that takes a press: grabbing it must
            # happen before the compare consumption below, or the divider can never be moved (owner, 2026-11-26:
            # "было бы неплохо добавить видимый сплитер который передвигать можно").
            if self._divider_hovered(position):
                self._dragging_split = True
                self.setCursor(Qt.CursorShape.SizeHorCursor)
                event.accept()
                return
            # In a compare mode the source is already on screen: there is nothing to flash, and the picture is not a
            # control. The press is consumed so it cannot turn into a flash the user did not ask for.
            if self._view_mode != "single":
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
        if not self._dragging_split:
            # The divider's own hover: full alpha while the pointer is on it, a fade once it leaves. The cursor says
            # "this can be dragged" before the press does.
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
        if self._scrubbing:
            fraction = self._fraction_at(position.x())
            # A drag is clamped at the seek's own edge instead of jumping into the dark part: the playhead never
            # leaves frames the owner can actually show (the produced band, or the whole timeline on a feed).
            self.seekRequested.emit(min(fraction, self._seek_limit()))
            event.accept()
            return
        if self._panning:
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
                self._pan_anchor = position
                # The distance already travelled counts: without this the first move of every drag was a dead zone,
                # which reads as "the picture sticks for a moment before it follows".
                self.pan_by(moved.x(), moved.y())
                event.accept()
                return
            event.accept()
            return
        self._update_hover(position)
        super().mouseMoveEvent(event)

    def _update_hover(self, position) -> None:
        """Hover feedback on the strip: a marker where the pointer is, and a forbidden cursor past the produced edge."""
        strip = self._strip_rect()
        if strip.isEmpty() or not strip.contains(position):
            if self._hover_fraction is not None:
                self._hover_fraction = None
                self.unsetCursor()
                self.update()
            return
        fraction = self._fraction_at(position.x())
        blocked = not self._seekable(fraction)
        if fraction != self._hover_fraction or blocked != self._hover_blocked:
            self._hover_fraction = fraction
            self._hover_blocked = blocked
            self.setCursor(Qt.CursorShape.ForbiddenCursor if blocked else Qt.CursorShape.PointingHandCursor)
            self.update()

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
            if self._scrubbing:
                telemetry.event("strip release")
            self._scrubbing = False
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

    # -- strip geometry ---------------------------------------------------

    def _apply_strip_metrics(self) -> None:
        """Re-read the toolkit's strip constants (the scale changed under us) and re-place the caption."""
        self._strip_margin = strip_margin()
        self._strip_height = strip_track_height()
        self._handle_radius = strip_handle_radius()
        self._playhead_halo_radius = playhead_halo_radius(self._handle_radius)
        self._position_readout()
        self.update()

    def _on_scale_changed(self, *_args) -> None:
        """`UiScale.scale_changed`: the strip's geometry is design px times the factor - re-read, do not freeze."""
        self._apply_strip_metrics()

    def _on_theme_changed(self) -> None:
        """The strip resolves its tokens at paint time; a light/dark switch only needs a repaint."""
        self.update()

    def _refresh_readout(self) -> None:
        """Feed the widget from the pane's own progress (the playhead and duration it already keeps)."""
        if not self.has_frame():
            self._time_readout.set_times(None, None)
        else:
            _, _, playhead, duration = self._progress
            self._time_readout.set_times(playhead, duration if duration > 0 else None)
        self._position_readout()

    def _position_readout(self) -> None:
        """Put the caption just above the strip, right-aligned to the pane's margin.

        The strip occupies the bottom margin, so the caption's home is the band above it; before a frame (and
        therefore before a strip) it still shows its em dashes, at the same bottom-right corner.
        """
        readout = self._time_readout
        readout.adjustSize()
        margin = self._strip_margin
        strip = self._strip_rect()
        bottom = strip.top() - scaled_px(4) if not strip.isEmpty() else self.height() - margin
        x = max(margin, self.width() - margin - readout.width())
        y = max(margin, int(bottom) - readout.height())
        readout.move(x, y)
        readout.raise_()

    def _strip_rect(self) -> QRectF:
        if not self._strip_visible or not self.has_frame():
            return QRectF()
        margin = self._strip_margin
        height = self._strip_height
        width = max(1.0, self.width() - 2 * margin)
        top = self.height() - margin - height
        return QRectF(margin, top, width, height)

    def _produced_fraction(self) -> float:
        """The share of the strip the renderer has produced, i.e. the seekable edge while a render is running.

        `total == 0` means "this is a finished file": the whole strip is seekable. While `rendered < total` the area to
        the right of the produced edge has no frames - seeking there can only lie to the user.
        """
        rendered, total, _, _ = self._progress
        if total <= 0:
            return 1.0
        return max(0.0, min(1.0, rendered / total))

    def _seek_limit(self) -> float:
        """The edge the strip may seek to: the host's ceiling when it set one, else the produced share."""
        if self._seek_ceiling is not None:
            return self._seek_ceiling
        return self._produced_fraction()

    def _seekable(self, fraction: float) -> bool:
        return fraction <= self._seek_limit() + 1e-6

    def _fraction_at(self, x: float) -> float:
        rect = self._strip_rect()
        if rect.width() <= 0:
            return 0.0
        return max(0.0, min(1.0, (x - rect.left()) / rect.width()))

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
            return (0, 0) if planes_size is None else (int(planes_size[0]), int(planes_size[1]))
        return (image.width(), image.height())

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
        picture = self._original
        if picture is None or picture.isNull():
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
        box_w, box_h = result.width(), result.height()
        if box_w <= 0.0 or box_h <= 0.0:
            box_w, box_h = picture.width() * scale, picture.height() * scale
        source_aspect = picture.width() / float(picture.height() or 1)
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
        """
        original = self._original
        return PreviewScene(
            target=target,
            rhi=self.rhi(),
            mode=self._view_mode,
            result_mvp=tuple(self._mvp_floats(self._frame_rect())),
            original_mvp=None if original is None else tuple(self._mvp_floats(self._original_rect())),
            tint=tuple(self._tint),
            nearest=self.samples_nearest(),
            result_rect=self._frame_rect(),
            result_clip=self._view_rect(),
            original_clip=self._original_clip_rect(),
            show_original=bool(self._showing_original and original is not None),
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
        if drawn.mode == "single":
            picture = self._original if drawn.half == "original" else self._frame
            present = self._has_picture() if drawn.half == "result" else picture is not None
            if drawn.half == "original" and (self._original_changed or telemetry.level() >= 2):
                self._log_draw(half="original", picture=picture)
                self._original_changed = False
            elif drawn.half == "result" and telemetry.level() >= 2:
                self._log_draw(half="result", picture=picture, present=present)
            return
        # Both halves in one pass: one line names both pictures (a new original always, every paint at level 2).
        original = self._original if drawn.drew_original else None
        if drawn.drew_original and (self._original_changed or telemetry.level() >= 2):
            self._log_draw_two(self._frame, original)
            self._original_changed = False
        elif telemetry.level() >= 2:
            self._log_draw_two(self._frame, None, present=self._has_picture())

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
        self._renderer.release()

    def render(self, command_buffer) -> None:  # noqa: N802 - Qt spelling
        """Hand this paint to the renderer: the scene first, then one pass into the current target."""
        target = self.renderTarget()
        if target is None or self.rhi() is None:
            return
        if self._view_mode != "single" and self._original is not None:
            # What this PAINT actually shows, both halves: the only number that can tell a consistent pair from a
            # flicker. The selection log says what was chosen; this says what reached the screen together. The COUNTER
            # is unconditional (two integer ops, and it rides the level-1 aggregate); the per-paint line is level 2.
            result = getattr(self, "_committed_result_time", None)
            source = getattr(self, "_committed_source_time", None)
            if result is not None and source is not None:
                delta_frames = (float(source) - float(result)) * 24.0
                telemetry.note_pair(delta_frames=delta_frames)
                if telemetry.level() >= 2:
                    newest = getattr(self, "_shown_time", None)
                    lag = "" if newest is None else round((float(newest) - float(result)) * 24.0, 2)
                    telemetry.event("painted pair", result=round(float(result), 3), source=round(float(source), 3),
                                    delta_frames=round(delta_frames, 2), lag_frames=lag)
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
        """Rectangles for the HUD pass, in widget pixels: the strip (when the pane paints its own) and the compare
        modes' ONE boundary.

        The strip is a *timeline* - produced band, playhead handle, hover and blocked markers - borrowing the toolkit
        slider's groove/thumb language, and every colour comes from ``STRIP_TOKENS`` through `ThemeManager` at paint
        time. The HUD is a rect-only pass, which is why the time text lives in a widget above it instead.

        The compare boundary is NOT conditional on the strip: the host's controls bar hides the pane's own strip
        (`set_strip_visible(False)`), and an early return on an empty strip rect meant the `split` divider and the `sbs`
        seam were never drawn in the real window - the "invisible splitter" (owner, 2026-11-26). The divider is a control
        of the PICTURE.
        """
        rects: list[tuple[float, float, float, float, tuple[float, float, float, float]]] = []
        strip = self._strip_rect()
        if not strip.isEmpty():
            rendered, total, playhead, duration = self._progress
            colours = strip_colours()
            track = rgba(colours["track"])
            produced = rgba(colours["produced"])
            playhead_fill = rgba(colours["playhead"])
            outline = rgba(colours["playhead_outline"], 0.70)
            divider = rgba(colours["divider"], 0.35)
            hover = rgba(colours["hover"], 0.65)
            blocked = rgba(colours["blocked"], 0.55)
            # The DARK band is what the renderer has not produced yet, the ACCENT part is what it has. No extra tick on the
            # fill's edge: the edge itself is the render head.
            rects.append((strip.left(), strip.top(), strip.width(), strip.height(), track))
            if total > 0:
                filled = max(0.0, min(1.0, rendered / total))
                if filled > 0:
                    rects.append((strip.left(), strip.top(), strip.width() * filled, strip.height(), produced))
            if duration > 0:
                fraction = max(0.0, min(1.0, playhead / duration))
                x = strip.left() + strip.width() * fraction
                handle = self._handle_radius
                halo = self._playhead_halo_radius
                # A backdrop behind the line and the handle: the thumb is white in light mode and dark in dark mode, so a
                # token-coloured plate (not a fixed black or white) is what stays legible in both themes.
                rects.append((x - 2.5, strip.top() - 6.0, 5.0, strip.height() + 12.0, outline))
                rects.append((x - 1.0, strip.top() - 5.0, 2.0, strip.height() + 10.0, produced))
                rects.append((x - halo, strip.top() - 6.0 - halo, halo * 2.0, halo * 2.0, outline))
                # The handle is the slider's thumb, from the toolkit's own thumb token.
                rects.append((x - handle, strip.top() - 6.0 - handle, handle * 2.0, handle * 2.0, playhead_fill))
            if self._hover_fraction is not None and not self._scrubbing:
                x = strip.left() + strip.width() * self._hover_fraction
                colour = blocked if self._hover_blocked else hover
                rects.append((x - 0.5, strip.top() - 3.0, 1.0, strip.height() + 6.0, colour))
            strip_top = float(strip.top())
        else:
            strip_top = float(self.height())
        self._compare_boundary_rects(rects, strip_top=strip_top)
        vertices: list[float] = []
        for x, y, w, h, colour in rects:
            corners = [(x, y), (x + w, y), (x, y + h), (x + w, y), (x + w, y + h), (x, y + h)]
            for cx, cy in corners:
                vertices.extend([cx, cy, *colour])
        return vertices

    def _compare_boundary_rects(self, rects, *, strip_top: float) -> None:
        """Append the compare modes' boundary: the `sbs` seam, or the `split` divider with its drag grip.

        `strip_top` is where the pane's own strip begins (the pane's height when it paints none), so a boundary never
        runs under the strip's controls when the strip is visible.
        """
        colours = strip_colours()
        divider = rgba(colours["divider"], 0.35)
        if self._view_mode == "sbs":
            middle = self.width() / 2.0
            picture = self._frame_rect()
            # A seam belongs to a PICTURE: an empty pane draws nothing at all (its own pin asserts exactly that).
            if strip_top > 0 and not picture.isEmpty():
                top = max(0.0, picture.top())
                bottom = min(strip_top, picture.bottom())
                if bottom > top:
                    rects.append((middle - 0.5, top, 1.0, bottom - top, divider))
        elif (self._view_mode == "split" and self._original is not None
              and not self._original.isNull()):
            picture = self._frame_rect()
            top = max(0.0, picture.top())
            bottom = min(strip_top, picture.bottom())
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
