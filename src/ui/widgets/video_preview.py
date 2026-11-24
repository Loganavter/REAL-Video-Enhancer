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
"""

from __future__ import annotations

import os
import time

from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QImage,
    QRhi,
    QRhiBuffer,
    QRhiDepthStencilClearValue,
    QRhiGraphicsPipeline,
    QRhiSampler,
    QRhiShaderResourceBinding,
    QRhiScissor,
    QRhiShaderResourceBindings,
    QRhiShaderStage,
    QRhiTexture,
    QRhiVertexInputAttribute,
    QRhiViewport,
    QRhiVertexInputBinding,
    QRhiVertexInputLayout,
    QShader,
)
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

SHADER_DIR = "shaders"
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
#: Mipminification on/off. The chain is the whole fix for the fit view (a 1440x1080 frame in a 421x512 pane is a
#: 3.42x minification through ONE bilinear tap, which aliases); the switch exists so an A/B runs on ONE build and the
#: owner can compare the two arms himself. `RVE_MIP_MINIFY=0` keeps the old single-level texture.
def mip_minify_enabled() -> bool:
    import os

    return str(os.environ.get("RVE_MIP_MINIFY", "1")).strip() != "0"


#: The magnification ladder for NEAREST sampling. `Nearest` duplicates a source pixel by a WHOLE number of screen
#: pixels; at a fractional factor the blocks are 3 px wide in one place and 4 px in the next, which is the ragged
#: "рванина" the owner refused (2026-11-24: "мне нужны по ближайшим, но при этом качество не рванина"). So the applied
#: zoom is always one of these, and the wheel moves a target between them. libplacebo calls the same mode
#: `oversample` ("aspect-ratio preserving nearest neighbour sampling").
ZOOM_STEPS = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0)


def nearest_zoom_step(zoom: float) -> float:
    """The allowed magnification step closest to `zoom`, compared in log space so the feel is even.

    Below 1.0 the value is returned untouched: that is MINIFICATION (fit and tighter), where `Nearest` would alias
    rather than preserve pixels, so the caller keeps the linear sampler there until the mip chain lands.
    """
    value = float(zoom)
    if value < 1.0:
        return value
    import math

    return min(ZOOM_STEPS, key=lambda step: abs(math.log(value / step)))


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
        self._frame: QImage | None = None
        self._original: QImage | None = None
        self._texture_dirty = False
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
        #: the source picture has its OWN texture, used by the compare modes; "single" keeps drawing it through
        #: `_texture` (the flash replaces the frame, so one texture is enough there)
        self._original_texture_dirty = False
        #: where the split divider sits, as a share of the picture's width; the pane's rule, not a per-frame value
        self._split_fraction = 0.5
        #: the playhead the standing original was last asked for, so playback refreshes it once per move
        self._original_requested_at: float | None = None
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
        """Show this picture. A `None` clears the pane (no frame yet)."""
        self._frame = None if image is None else image.convertToFormat(QImage.Format.Format_RGBA8888)
        self._texture_dirty = self._frame is not None
        self._cleared_empty = False
        if self._texture_dirty and getattr(self, "_texture", None) is not None:
            # Create the texture OUTSIDE a frame: QRhi resources are created between frames, and a create() issued
            # while a frame is being recorded is refused by the backend without an exception (the frame then samples
            # the 1x1 placeholder - opaque black, which reads exactly like "the upload was skipped").
            self._ensure_texture(self._frame.size())
        self._refresh_readout()
        self.update()
        # The pane is the only place that knows what it PAINTS: a decoder's frame number when the producer has one,
        # otherwise the display channel's sequence. Recorded here, read back through `shown()` - a seek's landing must
        # be the picture on screen, not a counter that happens to sit next to it.
        if index is not None:
            self._shown = ("frame", int(index))
        elif seq is not None:
            self._shown = ("seq", int(seq))
        self._shown_time = None if media_time is None else float(media_time)
        telemetry.note_frame(gui_ms=gui_ms, index=index, seq=seq, media_time=media_time, generation=generation)

    def has_frame(self) -> bool:
        return self._frame is not None

    def shown_time(self) -> float | None:
        """The media time of the picture on screen, when the publisher named it."""
        return getattr(self, "_shown_time", None)

    def shown(self) -> tuple[str, int] | None:
        """What the pane is painting: ``("frame", n)`` or ``("seq", n)``, or None before the first picture."""
        return getattr(self, "_shown", None)

    def set_original(self, image: QImage | None) -> None:
        if image is not None:
            telemetry.note_frame()      # the source half of a compare mode: a picture the pane accepted too
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
        else:
            picture = image.convertToFormat(QImage.Format.Format_RGBA8888)
            if self._frame is not None and picture.size() != self._frame.size():
                target = self._frame.size()
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
        if self._view_mode == "single":
            # The flash replaces the frame in the one shared texture.
            self._texture_dirty = True
        else:
            # The compare modes draw both pictures at once, so the source has its own texture and the result's
            # upload is untouched by an arriving source frame.
            self._original_texture_dirty = self._original is not None
            if self._original is not None and getattr(self, "_texture", None) is not None:
                self._ensure_original_texture(self._original.size())
        self.update()

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
        # Leaving "single" ends a flash: the source becomes the standing pane instead of a held picture.
        self._showing_original = False
        self._hud_dirty = True
        # Both textures get re-uploaded once: `single` may have left the source in the shared one, and the compare
        # modes keep the result there. Cheap (one upload) and it removes a whole class of "stale half" readings.
        self._texture_dirty = self._frame is not None
        self._original_texture_dirty = self._original is not None
        if mode != "single":
            self._request_standing_original()
        self.update()

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
        if self._view_mode == "single" or self._frame is None:
            return
        _, _, playhead, _ = self._progress
        playhead = float(playhead)
        if self._original is not None and playhead == self._original_requested_at:
            return
        self._original_requested_at = playhead
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
        target = getattr(self, "_zoom_target", self._zoom) * factor
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
            self.originalRequested.emit(ask)
            self.update()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt spelling
        position = event.position()
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

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt spelling
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton) and self._panning:
            self._panning = False
            self._left_press = None
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
        return scale >= 1.0 - 1e-9

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
        # Magnification is a ladder of whole factors: `Nearest` at 1.87x draws uneven pixel blocks. Minification
        # (fit and tighter) stays continuous, because there the sampler averages.
        target = nearest_zoom_step(target)
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
            return (0, 0)
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
        drawn_w, drawn_h = picture.width() * scale, picture.height() * scale
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

    def initialize(self, command_buffer) -> None:  # noqa: N802 - Qt spelling
        """Build the resource set - but not on a mere resize.

        Qt calls `initialize()`/`releaseResources()` on EVERY resize (the widget's backing target is reallocated at the
        new pixel size), not only on a real context change - the framework's `qrhi-gotchas.md` §"Canvas resize tears down
        the whole tile array on every resize" measured it at ~1 s per resize in BlinkTable. Releasing here unconditionally
        is worse than slow for a still picture: the frame texture is destroyed with it, the next `render()` builds a fresh
        texture and only uploads when `_texture_dirty` is set - which a resize does not set - so the pane draws an
        undrawn texture until the NEXT frame. A finished or paused render never sends one: the pane stays see-through
        after any window resize or sibling repaint. That is the "the preview turns transparent at any sneeze" defect.
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
        same_context = rhi is getattr(self, "_rhi", None) and getattr(self, "_pipeline", None) is not None
        if same_context:
            # Same RHI, just a new target size: keep every resource and re-upload the frame we already hold, so the
            # pane shows the picture again on the very next paint instead of an empty texture.
            self._rhi = rhi
            if self._frame is not None:
                self._texture_dirty = True
            if self._original is not None:
                self._original_texture_dirty = True
            self.update()
            return
        self._release()
        self._rhi = rhi
        if self._frame is None:
            # Nothing to draw yet. Building a 1x1 texture and then resizing it forces an SRB and pipeline rebuild in
            # the middle of the first real frame - the one path that produced a black pane in every probe. Deferring
            # the whole resource set to the first frame removes that path instead of patching it.
            self._deferred = True
            return
        self._create_resources(command_buffer)

    def _create_resources(self, command_buffer) -> None:
        """All RHI resources for the CURRENT frame size, built once per size change.

        `command_buffer` is needed for the static vertex upload; `initialize()` and `render()` both have one, which is
        why this is a method taking it rather than a constructor-time step.
        """
        rhi = self._rhi
        initial_size = self._frame.size() if self._frame is not None else QSize(1, 1)
        # `MipMapped` makes Qt allocate the chain (`mipLevelsForSize` = 11 for 1440x1080) and requires the levels to be
        # generated after each upload; `UsedWithGenerateMips` is what the backend needs to accept `generateMips`.
        flags = QRhiTexture.Flag.UsedAsTransferSource
        if mip_minify_enabled():
            flags |= QRhiTexture.Flag.MipMapped | QRhiTexture.Flag.UsedWithGenerateMips
        self._texture = rhi.newTexture(QRhiTexture.Format.RGBA8, initial_size, 1, flags)
        if not self._texture.create():
            raise RuntimeError(f"frame texture {initial_size.width()}x{initial_size.height()} was refused by QRhi")
        # The THIRD filter is the mipmap mode, and it must be `None_` for a single-level texture: with a mipmap filter
        # set, sampling returned opaque black on llvmpipe (measured: a 4x4 RGBA texture uploaded with the right colours
        # sampled (0,0,0,255) with `Nearest` and the correct colours with `None_`). This is the kind of failure that
        # looks like a missing upload, so the trap is named here.
        self._sampler_nearest = rhi.newSampler(QRhiSampler.Filter.Nearest, QRhiSampler.Filter.Nearest,
                                               QRhiSampler.Filter.None_, QRhiSampler.AddressMode.ClampToEdge,
                                               QRhiSampler.AddressMode.ClampToEdge)
        self._sampler_nearest.create()
        # The THIRD argument is the mipmap mode. `Linear` there is trilinear: the fit view samples the two levels the
        # derivative-chosen LOD falls between, which is what turns the aliasing tap into an average. Left `None_` when
        # the chain is off, because a mipmap mode over a single-level texture sampled opaque black on llvmpipe once.
        mip_mode = QRhiSampler.Filter.Linear if mip_minify_enabled() else QRhiSampler.Filter.None_
        self._sampler_linear = rhi.newSampler(QRhiSampler.Filter.Linear, QRhiSampler.Filter.Linear,
                                              mip_mode, QRhiSampler.AddressMode.ClampToEdge,
                                              QRhiSampler.AddressMode.ClampToEdge)
        self._sampler_linear.create()

        self._uniforms = rhi.newBuffer(QRhiBuffer.Type.Dynamic, QRhiBuffer.UsageFlag.UniformBuffer, 64 + 16)
        self._uniforms.create()
        # The source picture's own uniform buffer: same block, a different mvp (its half), and the same tint. Two
        # buffers instead of one with two offsets keeps each SRB a plain whole-buffer binding.
        self._original_uniforms = rhi.newBuffer(QRhiBuffer.Type.Dynamic, QRhiBuffer.UsageFlag.UniformBuffer,
                                                64 + 16)
        self._original_uniforms.create()

        # A unit quad: two triangles, positions in [-1,1] and matching uv.
        # NDC -1 is the bottom of the pane while a QImage row 0 is its top, so the texture's v axis is flipped here
        # rather than in the shader: v = 0 is the image's bottom row, which is the quad's bottom-left corner.
        quad = [
            -1.0, -1.0, 0.0, 0.0,
            1.0, -1.0, 1.0, 0.0,
            -1.0, 1.0, 0.0, 1.0,
            1.0, -1.0, 1.0, 0.0,
            1.0, 1.0, 1.0, 1.0,
            -1.0, 1.0, 0.0, 1.0,
        ]
        self._vertices = rhi.newBuffer(QRhiBuffer.Type.Immutable, QRhiBuffer.UsageFlag.VertexBuffer,
                                       len(quad) * 4)
        self._vertices.create()
        updates = rhi.nextResourceUpdateBatch()
        updates.uploadStaticBuffer(self._vertices, _floats(quad))
        command_buffer.resourceUpdate(updates)

        self._build_bindings()

        # HUD: its own 16-byte viewport uniform and a dynamic vertex buffer (rects are rebuilt only when they change).
        self._hud_uniforms = rhi.newBuffer(QRhiBuffer.Type.Dynamic, QRhiBuffer.UsageFlag.UniformBuffer, 16)
        self._hud_uniforms.create()
        self._hud_capacity = 64 * 6 * 6 * 4          # 64 rectangles x 6 vertices x 6 floats
        self._hud_buffer = rhi.newBuffer(QRhiBuffer.Type.Dynamic, QRhiBuffer.UsageFlag.VertexBuffer,
                                         self._hud_capacity)
        self._hud_buffer.create()
        self._hud_bindings = rhi.newShaderResourceBindings()
        self._hud_bindings.setBindings([
            QRhiShaderResourceBinding.uniformBuffer(0, QRhiShaderResourceBinding.StageFlag.VertexStage,
                                                    self._hud_uniforms),
        ])
        self._hud_bindings.create()
        hud_layout = QRhiVertexInputLayout()
        hud_layout.setBindings([QRhiVertexInputBinding(6 * 4)])
        hud_layout.setAttributes([
            QRhiVertexInputAttribute(0, 0, QRhiVertexInputAttribute.Format.Float2, 0),
            QRhiVertexInputAttribute(0, 1, QRhiVertexInputAttribute.Format.Float4, 2 * 4),
        ])
        self._hud_pipeline = rhi.newGraphicsPipeline()
        self._hud_pipeline.setShaderStages([
            QRhiShaderStage(QRhiShaderStage.Type.Vertex, _shader("hud.vert.qsb")),
            QRhiShaderStage(QRhiShaderStage.Type.Fragment, _shader("hud.frag.qsb")),
        ])
        self._hud_pipeline.setVertexInputLayout(hud_layout)
        self._hud_pipeline.setShaderResourceBindings(self._hud_bindings)
        self._hud_pipeline.setRenderPassDescriptor(self.renderTarget().renderPassDescriptor())
        self._hud_pipeline.setDepthTest(False)
        self._hud_pipeline.setDepthWrite(False)
        hud_blend = QRhiGraphicsPipeline.TargetBlend()
        hud_blend.enable = True
        hud_blend.srcColor = QRhiGraphicsPipeline.BlendFactor.SrcAlpha
        hud_blend.dstColor = QRhiGraphicsPipeline.BlendFactor.OneMinusSrcAlpha
        hud_blend.srcAlpha = QRhiGraphicsPipeline.BlendFactor.One
        hud_blend.dstAlpha = QRhiGraphicsPipeline.BlendFactor.OneMinusSrcAlpha
        self._hud_pipeline.setTargetBlends([hud_blend])
        self._hud_pipeline.create()
        self._hud_dirty = True

        self._deferred = False
        self._texture_dirty = self._frame is not None
        self._pipeline = None
        self._original_texture = None
        if self._original is not None and not self._original.isNull():
            # A compare mode was opened before the first frame (or the resources were rebuilt while it stood). The
            # source's texture is created exactly like the frame's, outside a frame, and its own upload follows.
            self._original_texture_dirty = True
            self._ensure_original_texture(self._original.size())

    def _build_pipeline(self) -> None:
        """Build the graphics pipeline for the CURRENT SRB. Re-run whenever the texture (and so its SRB) is re-created:
        a pipeline compiled against an older SRB layout is not the same layout object, and the GL backend then binds
        nothing (samples black) instead of raising."""
        rhi = self._rhi
        layout = QRhiVertexInputLayout()
        layout.setBindings([QRhiVertexInputBinding(4 * 4)])
        layout.setAttributes([
            QRhiVertexInputAttribute(0, 0, QRhiVertexInputAttribute.Format.Float2, 0),
            QRhiVertexInputAttribute(0, 1, QRhiVertexInputAttribute.Format.Float2, 2 * 4),
        ])
        self._pipeline = rhi.newGraphicsPipeline()
        self._pipeline.setShaderStages([
            QRhiShaderStage(QRhiShaderStage.Type.Vertex, _shader("frame.vert.qsb")),
            QRhiShaderStage(QRhiShaderStage.Type.Fragment, _shader("frame.frag.qsb")),
        ])
        self._pipeline.setVertexInputLayout(layout)
        self._pipeline.setShaderResourceBindings(self._bindings)
        self._pipeline.setRenderPassDescriptor(self.renderTarget().renderPassDescriptor())
        # Without this flag QRhi IGNORES `setScissor` on the pipeline: the scissor state is only read by a pipeline
        # compiled with it. Both compare modes depend on that clipping (the split divider, and a zoomed `sbs` half that
        # must not leak into the other one), and the absence was invisible while nothing sampled the pixels: the split
        # mode painted the ORIGINAL over the whole picture rect (the pixel gate, 2026-11-24).
        self._pipeline.setFlags(QRhiGraphicsPipeline.Flag.UsesScissor)
        self._pipeline.setDepthTest(False)
        self._pipeline.setDepthWrite(False)
        blend = QRhiGraphicsPipeline.TargetBlend()
        blend.enable = True
        blend.srcColor = QRhiGraphicsPipeline.BlendFactor.SrcAlpha
        blend.dstColor = QRhiGraphicsPipeline.BlendFactor.OneMinusSrcAlpha
        blend.srcAlpha = QRhiGraphicsPipeline.BlendFactor.One
        blend.dstAlpha = QRhiGraphicsPipeline.BlendFactor.OneMinusSrcAlpha
        self._pipeline.setTargetBlends([blend])
        self._pipeline.create()

    def _ensure_texture(self, size) -> None:
        if getattr(self, "_texture", None) is None:
            return
        if self._texture.pixelSize() == size:
            return
        self._texture.destroy()
        self._texture.setPixelSize(size)
        if not self._texture.create():
            raise RuntimeError(f"could not resize the frame texture to {size.width()}x{size.height()}")
        self._build_bindings()
        pipeline = getattr(self, "_pipeline", None)
        if pipeline is not None:
            pipeline.destroy()
        self._pipeline = None

    def _ensure_original_texture(self, size) -> None:
        """Create or resize the SOURCE picture's own texture (the compare modes draw both pictures in one pass).

        Same rules as `_ensure_texture`, and the same trap: an SRB holds the texture's native handle, so a texture that
        is destroyed and re-created at a new size silently invalidates every binding that names it. The pipeline is
        built against a specific SRB object, so it is dropped too and rebuilt on the next render.
        """
        if getattr(self, "_texture", None) is None or getattr(self, "_rhi", None) is None:
            # The resource set is built at the first real frame; before that there is nothing to attach a texture to.
            return
        texture = getattr(self, "_original_texture", None)
        if texture is not None and texture.pixelSize() == size:
            return
        if texture is not None:
            texture.destroy()
            self._original_texture = None
        texture = self._rhi.newTexture(QRhiTexture.Format.RGBA8, size, 1,
                                       QRhiTexture.Flag.UsedAsTransferSource)
        if not texture.create():
            raise RuntimeError(f"original texture {size.width()}x{size.height()} was refused by QRhi")
        self._original_texture = texture
        self._build_bindings()
        pipeline = getattr(self, "_pipeline", None)
        if pipeline is not None:
            pipeline.destroy()
        self._pipeline = None

    def _build_bindings(self) -> None:
        """(Re)create the SRBs: the result's and, when it exists, the source's.

        The sampler is part of the SRB, so the 1:1 (nearest) and zoomed (linear) views are two bindings that differ
        only by the sampler; and a texture that is destroyed and re-created at a new size invalidates the native handle
        the SRB holds, so this runs again after such a re-create. The two pictures share one binding LAYOUT (a uniform
        buffer at 0, a sampled texture at 1) and therefore one pipeline: QRhi compares the layout, not the object.
        """
        rhi = self._rhi
        # The UBO is read by BOTH stages (`mvp` in the vertex shader, `tint` in the fragment one), and the sampler's
        # binding index must equal the `binding = 1` the fragment shader declares: QRhi assigns texture units by SRB
        # binding index, so a mismatch reads an unbound unit and returns (0,0,0,0) - a transparent "frame" that looks
        # like a missing upload.
        flags = (QRhiShaderResourceBinding.StageFlag.VertexStage
                 | QRhiShaderResourceBinding.StageFlag.FragmentStage)
        frag = QRhiShaderResourceBinding.StageFlag.FragmentStage
        specs = (
            ("_bindings", self._uniforms, self._texture, self._sampler_linear),
            ("_bindings_nearest", self._uniforms, self._texture, self._sampler_nearest),
            ("_original_bindings", self._original_uniforms,
             getattr(self, "_original_texture", None), self._sampler_linear),
            ("_original_bindings_nearest", self._original_uniforms,
             getattr(self, "_original_texture", None), self._sampler_nearest),
        )
        for attribute, uniforms, texture, sampler in specs:
            current = getattr(self, attribute, None)
            if current is not None:
                current.destroy()
                setattr(self, attribute, None)
            if texture is None:
                continue
            bindings = rhi.newShaderResourceBindings()
            bindings.setBindings([
                QRhiShaderResourceBinding.uniformBuffer(0, flags, uniforms),
                QRhiShaderResourceBinding.sampledTexture(1, frag, texture, sampler),
            ])
            bindings.create()
            setattr(self, attribute, bindings)

    def _release(self) -> None:
        for name in ("_pipeline", "_bindings", "_bindings_nearest", "_vertices", "_uniforms",
                     "_sampler_nearest", "_sampler_linear", "_texture", "_original_texture",
                     "_original_bindings", "_original_bindings_nearest", "_original_uniforms",
                     "_hud_pipeline", "_hud_bindings", "_hud_buffer", "_hud_uniforms"):
            item = getattr(self, name, None)
            if item is not None:
                try:
                    item.destroy()
                except Exception:
                    pass
                setattr(self, name, None)

    def releaseResources(self) -> None:  # noqa: N802 - Qt spelling
        self._release()

    def render(self, command_buffer) -> None:  # noqa: N802 - Qt spelling
        target = self.renderTarget()
        if target is None or self.rhi() is None:
            return
        if self._frame is None:
            # Nothing to show yet: an OPAQUE clear on EVERY render, so the pane reads as an empty plate rather than a
            # hole. It used to clear once (`_cleared_empty`) - a latch that is false on the next paint, which leaves the
            # region undrawn after any repaint and shows the window behind it (the "transparent at any sneeze" half).
            pass_started = time.perf_counter()
            command_buffer.beginPass(target, self._opaque_clear(), QRhiDepthStencilClearValue(1.0, 0))
            command_buffer.endPass()
            # An empty pane still paints: counting it keeps the cadence honest before the first frame arrives, and
            # `frames=0` in a stats line is itself an answer ("the render never fed the pane").
            telemetry.note_paint(pass_ms=(time.perf_counter() - pass_started) * 1000.0)
            telemetry.maybe_stats()
            return
        if getattr(self, "_deferred", False) or getattr(self, "_rhi", None) is None:
            self._release()
            self._rhi = self.rhi()
            self._create_resources(command_buffer)
        if self._view_mode == "single":
            self._render_single(command_buffer, target)
            return
        self._render_two_surfaces(command_buffer, target)

    def _render_single(self, command_buffer, target) -> None:
        """Today's pane: one picture per frame, in one pass (the LMB flash replaces it in place)."""
        picture = self._original if (self._showing_original and self._original is not None) else self._frame
        if picture is None:
            return
        if self._texture_dirty and picture is not None:
            self._ensure_texture(picture.size())
        if getattr(self, "_pipeline", None) is None:
            self._build_pipeline()
            if self._pipeline is None:
                return
        updates = self.rhi().nextResourceUpdateBatch()
        if self._texture_dirty:
            upload_started = time.perf_counter()
            updates.uploadTexture(self._texture, picture)
            if mip_minify_enabled():
                # Same batch as the upload, BEFORE the pass: the chain is built from level 0 on the queue, and
                # `resourceUpdate` is only legal outside a render pass. Cost is one-time per uploaded frame (the chain
                # of an 1440x1080 RGBA8 is ~2.1 MB - noise against the 6.2 MB level 0 that already crossed the bus).
                updates.generateMips(self._texture)
            telemetry.note_upload(upload_ms=(time.perf_counter() - upload_started) * 1000.0,
                                  size_bytes=picture.width() * picture.height() * 4)
            self._texture_dirty = False
        updates.updateDynamicBuffer(self._uniforms, 0, 64, _floats(self._mvp_floats()))
        updates.updateDynamicBuffer(self._uniforms, 64, 16, _floats(list(self._tint)))
        # The HUD's buffers are filled into the SAME batch, before the pass: `QRhiCommandBuffer::resourceUpdate` must be
        # called outside a render pass, so building the strip inside one (the first version did) is not allowed.
        hud_vertices = self._hud_vertices()
        if hud_vertices and getattr(self, "_hud_pipeline", None) is not None:
            updates.updateDynamicBuffer(self._hud_uniforms, 0, 16,
                                        _floats([float(self.width()), float(self.height()),
                                                 float(self.devicePixelRatioF() or 1.0), 0.0]))
            updates.updateDynamicBuffer(self._hud_buffer, 0, len(hud_vertices) * 4, _floats(hud_vertices))
        pass_started = time.perf_counter()
        command_buffer.beginPass(target, self._opaque_clear(), QRhiDepthStencilClearValue(1.0, 0), updates)
        command_buffer.setGraphicsPipeline(self._pipeline)
        command_buffer.setViewport(QRhiViewport(0, 0, target.pixelSize().width(), target.pixelSize().height()))
        sample_at_one = self.samples_nearest()
        # The pipeline declares `UsesScissor` (the compare modes need it), so the scissor state is LIVE here: a frame
        # must state its own scissor before drawing, or it inherits the last one set on the command buffer - an empty
        # one clipped the whole picture away in `single` after a compare mode had run (pixel gate, 2026-11-24).
        command_buffer.setScissor(
            QRhiScissor(0, 0, target.pixelSize().width(), target.pixelSize().height()))
        command_buffer.setShaderResources(self._bindings_nearest if sample_at_one else self._bindings)
        command_buffer.setVertexInput(0, [(self._vertices, 0)])
        command_buffer.draw(6)
        if hud_vertices and getattr(self, "_hud_pipeline", None) is not None:
            command_buffer.setGraphicsPipeline(self._hud_pipeline)
            command_buffer.setShaderResources(self._hud_bindings)
            command_buffer.setVertexInput(0, [(self._hud_buffer, 0)])
            command_buffer.draw(len(hud_vertices) // 6)
        command_buffer.endPass()
        telemetry.note_paint(pass_ms=(time.perf_counter() - pass_started) * 1000.0)
        telemetry.maybe_stats()

    def _render_two_surfaces(self, command_buffer, target) -> None:
        """The result and the source in ONE pass: one clear, then up to two draws, each clipped to its own surface.

        `sbs` lays each picture into its half, scissored so a zoomed picture cannot leak across the boundary; `split`
        paints both over the same picture rect and clips the original to the left of the divider. Two textures, two
        SRBs that share one pipeline layout, one `beginPass`/`endPass` - the pane's rect is overwritten exactly once,
        which is the whole reason a QRhiWidget cannot alpha-blend with its siblings in the first place. The HUD is
        drawn after the scissor is released, so the strip is the same single object under every mode.
        """
        frame = self._frame
        if frame is None:
            return
        self._ensure_texture(frame.size())
        original = self._original if (self._original is not None and not self._original.isNull()) else None
        if original is not None:
            self._ensure_original_texture(original.size())
        if getattr(self, "_pipeline", None) is None:
            self._build_pipeline()
            if self._pipeline is None:
                return
        updates = self.rhi().nextResourceUpdateBatch()
        if self._texture_dirty:
            upload_started = time.perf_counter()
            updates.uploadTexture(self._texture, frame)
            if mip_minify_enabled():
                updates.generateMips(self._texture)
            telemetry.note_upload(upload_ms=(time.perf_counter() - upload_started) * 1000.0,
                                  size_bytes=frame.width() * frame.height() * 4)
            self._texture_dirty = False
        tint = list(self._tint)
        updates.updateDynamicBuffer(self._uniforms, 0, 64, _floats(self._mvp_floats(self._frame_rect())))
        updates.updateDynamicBuffer(self._uniforms, 64, 16, _floats(tint))
        draw_original = original is not None and getattr(self, "_original_texture", None) is not None
        if draw_original:
            if self._original_texture_dirty:
                upload_started = time.perf_counter()
                updates.uploadTexture(self._original_texture, original)
                telemetry.note_upload(upload_ms=(time.perf_counter() - upload_started) * 1000.0,
                                      size_bytes=original.width() * original.height() * 4)
                self._original_texture_dirty = False
            updates.updateDynamicBuffer(self._original_uniforms, 0, 64,
                                        _floats(self._mvp_floats(self._original_rect())))
            updates.updateDynamicBuffer(self._original_uniforms, 64, 16, _floats(tint))
        hud_vertices = self._hud_vertices()
        if hud_vertices and getattr(self, "_hud_pipeline", None) is not None:
            updates.updateDynamicBuffer(self._hud_uniforms, 0, 16,
                                        _floats([float(self.width()), float(self.height()),
                                                 float(self.devicePixelRatioF() or 1.0), 0.0]))
            updates.updateDynamicBuffer(self._hud_buffer, 0, len(hud_vertices) * 4, _floats(hud_vertices))
        pass_started = time.perf_counter()
        command_buffer.beginPass(target, self._opaque_clear(), QRhiDepthStencilClearValue(1.0, 0), updates)
        command_buffer.setGraphicsPipeline(self._pipeline)
        command_buffer.setViewport(QRhiViewport(0, 0, target.pixelSize().width(), target.pixelSize().height()))
        sample_at_one = self.samples_nearest()
        result_bindings = self._bindings_nearest if sample_at_one else self._bindings
        result_clip = self._view_rect()
        if result_bindings is not None and not result_clip.isEmpty() and self._frame_rect().intersects(result_clip):
            command_buffer.setShaderResources(result_bindings)
            command_buffer.setVertexInput(0, [(self._vertices, 0)])
            self._scissor(command_buffer, result_clip)
            command_buffer.draw(6)
        if draw_original:
            original_bindings = (self._original_bindings_nearest if sample_at_one
                                 else self._original_bindings)
            original_clip = self._original_clip_rect()
            if original_bindings is not None and not original_clip.isEmpty():
                command_buffer.setShaderResources(original_bindings)
                command_buffer.setVertexInput(0, [(self._vertices, 0)])
                self._scissor(command_buffer, original_clip)
                command_buffer.draw(6)
        # The HUD spans the whole widget; a scissor left over from one half would clip the far end of the strip.
        # `QRhiCommandBuffer.setScissor` takes a `QRhiScissor`, not a `QRect`: PySide6 6.11 rejected the implicit rect
        # with `TypeError: setScissor(QRect)` while 6.8 still accepted it, so `sbs`/`split` drew nothing there and said
        # so only inside a paint (the two-surface pixel gate, 2026-11-24, was the first thing to see it).
        command_buffer.setScissor(
            QRhiScissor(0, 0, target.pixelSize().width(), target.pixelSize().height()))
        if hud_vertices and getattr(self, "_hud_pipeline", None) is not None:
            command_buffer.setGraphicsPipeline(self._hud_pipeline)
            command_buffer.setShaderResources(self._hud_bindings)
            command_buffer.setVertexInput(0, [(self._hud_buffer, 0)])
            command_buffer.draw(len(hud_vertices) // 6)
        command_buffer.endPass()
        telemetry.note_paint(pass_ms=(time.perf_counter() - pass_started) * 1000.0)
        telemetry.maybe_stats()

    def _scissor(self, command_buffer, rect: QRectF) -> None:
        """Confine the next draw to a widget-space rect (QRhi scissors are in framebuffer pixels)."""
        target = self.renderTarget()
        if target is None:
            return
        pixels = target.pixelSize()
        dpr = float(self.devicePixelRatioF() or 1.0)
        left = max(0.0, rect.left() * dpr)
        top = max(0.0, rect.top() * dpr)
        right = min(float(pixels.width()), rect.right() * dpr)
        bottom = min(float(pixels.height()), rect.bottom() * dpr)
        command_buffer.setScissor(QRhiScissor(int(left), int(top),
                                             max(0, int(right - left)), max(0, int(bottom - top))))

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
        """Rectangles for the strip, in widget pixels: track, produced band, playhead, handle, hover marker.

        The design language's numbers and colours, not this file's: the track is the toolkit slider's track height,
        the handle is its thumb radius (inside the timeline's playhead handle), and every colour is resolved from
        ``STRIP_TOKENS`` through `ThemeManager` at paint time. The strip is a *timeline* - it keeps its produced band,
        hover marker and blocked marker - but it borrows the slider's groove/thumb language so it reads as the same
        control family. The HUD is a rect-only pass, which is why the time text lives in a widget above it instead.
        """
        strip = self._strip_rect()
        if strip.isEmpty():
            return []
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
        rects: list[tuple[float, float, float, float, tuple[float, float, float, float]]] = [
            (strip.left(), strip.top(), strip.width(), strip.height(), track),
        ]
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
        # The two-surface modes draw ONE boundary, not a second HUD: a faint line between the halves in `sbs`, and
        # the divider itself in `split`. Both stop above the strip so the playhead stays readable across the width.
        strip_top = strip.top()
        if self._view_mode == "sbs":
            middle = self.width() / 2.0
            if strip_top > 0:
                rects.append((middle - 0.5, 0.0, 1.0, strip_top, divider))
        elif (self._view_mode == "split" and self._original is not None
              and not self._original.isNull()):
            picture = self._frame_rect()
            top = max(0.0, picture.top())
            bottom = min(strip_top, picture.bottom())
            if bottom > top:
                x = picture.left() + picture.width() * self._split_fraction
                rects.append((x - 1.0, top, 2.0, bottom - top, rgba(colours["divider"], 0.85)))
        vertices: list[float] = []
        for x, y, w, h, colour in rects:
            corners = [(x, y), (x + w, y), (x, y + h), (x + w, y), (x + w, y + h), (x, y + h)]
            for cx, cy in corners:
                vertices.extend([cx, cy, *colour])
        return vertices

    def grabFrame(self) -> QImage:  # noqa: N802 - Qt spelling is `grabFramebuffer`, this is the plan's name
        return self.grabFramebuffer()


def _floats(values) -> bytes:
    import struct

    flat: list[float] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            flat.extend(float(part) for part in value)
        else:
            flat.append(float(value))
    return struct.pack(f"<{len(flat)}f", *flat)


def _shader(name: str) -> QShader:
    import os

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), SHADER_DIR, name)
    with open(path, "rb") as handle:
        shader = QShader.fromSerialized(handle.read())
    if not shader.isValid():
        # A missing or half-built .qsb must be loud: a silent blank video pane is the worst failure mode here.
        raise RuntimeError(f"shader {path} is not a valid .qsb (rebuild with harness/scripts/build_shaders.sh)")
    return shader
