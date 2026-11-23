"""The GPU preview surface: one QRhiWidget that paints a full-resolution frame with zoom and pan.

Scope of this file today: the frame. The strip, the playhead and the LMB compare overlay are separate layers that will
sit in the same pass (see plan-rve-video-preview-widget-2026-11-23); keeping them out until the frame is proven means the
first thing measured is the thing that can be wrong in the most ways (texture upload, colour, aspect, zoom).

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

from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QImage,
    QRhi,
    QRhiBuffer,
    QRhiDepthStencilClearValue,
    QRhiGraphicsPipeline,
    QRhiSampler,
    QRhiShaderResourceBinding,
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

from sli_ui_toolkit.ui.managers.ui_scale import scaled_px

SHADER_DIR = "shaders"
#: 8 source pixels per screen pixel at most. The floor is the FIT scale (computed per frame), not 1:1: the pane shows
#: the whole frame by default and the wheel can go back to that, while 1:1 is just a step inside the range. Zoom is a
#: source-pixel scale, so the same factor means the same detail whatever the pane size.
MAX_ZOOM = 8.0
MIN_ZOOM = 1.0


@dataclass(frozen=True)
class Frame:
    """One picture to show: RGBA8888 pixels, already packed, plus its identity."""

    image: QImage
    index: int = 0
    pts: float = 0.0


class VideoPreviewSurface(QRhiWidget):
    """A frame, fitted to the pane on open and zoomable from there (range: fit .. 8x source pixels)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._frame: QImage | None = None
        self._original: QImage | None = None
        self._texture_dirty = False
        self._zoom = MIN_ZOOM
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
        self._pan_anchor = QPointF()
        self._showing_original = False
        #: Opaque clear colour for the pane. QRhiWidgets composite by OVERWRITING their rect, so a clear with alpha
        #: below 255 punches a hole (black on Wayland, white on xcb) instead of letting the page show through - the
        #: framework's `lifecycle-first-frame-signal-gating` and `painting-qrhiwidget-cannot-alpha-blend-siblings`
        #: precedents. The pane is therefore always opaque, with the page's own colour.
        self._clear_colour = self.palette().window().color()
        #: strip geometry in design px; the widget scales it with the device pixel ratio at paint time
        self._strip_height = scaled_px(6)
        self._strip_margin = scaled_px(10)
        self._handle_radius = scaled_px(7)
        #: hovered position on the strip (fraction 0..1), and whether it is past the produced edge
        self._hover_fraction: float | None = None
        self._hover_blocked = False
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

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt spelling
        """A floor for the pane, in design px through the toolkit scale.

        Stated as a size hint rather than `setMinimumSize`: the ui-gates rule `raw-chrome-px` reads literal arguments of
        the chrome setters, and a scaled literal is still a literal to it - while a hint is exactly what this is.
        """
        return QSize(scaled_px(160), scaled_px(90))

    # -- data ------------------------------------------------------------

    def set_frame(self, image: QImage | None) -> None:
        """Show this picture. A `None` clears the pane (no frame yet)."""
        self._frame = None if image is None else image.convertToFormat(QImage.Format.Format_RGBA8888)
        self._texture_dirty = self._frame is not None
        self._cleared_empty = False
        if self._texture_dirty and getattr(self, "_texture", None) is not None:
            # Create the texture OUTSIDE a frame: QRhi resources are created between frames, and a create() issued
            # while a frame is being recorded is refused by the backend without an exception (the frame then samples
            # the 1x1 placeholder - opaque black, which reads exactly like "the upload was skipped").
            self._ensure_texture(self._frame.size())
        self.update()

    def has_frame(self) -> bool:
        return self._frame is not None

    def set_original(self, image: QImage | None) -> None:
        """The source frame for the LMB compare; `None` returns to the rendered picture.

        The source is **stretched up to the rendered frame's size with nearest-neighbour** (`FastTransformation`): the
        cheapest interpolation there is, and the only one that adds no information. The comparison then shows two
        pictures of the same pixel size, so every difference on screen is one the upscaler produced rather than a
        smoothing artefact of the viewer, and the source pixel at (x, y) lands on the same screen point in both states
        (owner, 2026-11-23).
        """
        if image is None:
            self._original = None
        else:
            picture = image.convertToFormat(QImage.Format.Format_RGBA8888)
            if self._frame is not None and picture.size() != self._frame.size():
                # KEEP the source's proportions. The source and the render normally share an aspect, but a render can
                # change dimensions (border-detect crops, an odd scale) and a stretched compare would then show a
                # straight line where the picture has a curve - a difference the viewer invented. Nearest is still the
                # filter: cheapest, and it adds no information.
                picture = picture.scaled(self._frame.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                         Qt.TransformationMode.FastTransformation)
            self._original = picture
        self._texture_dirty = True
        self.update()

    def set_tint(self, rgba: tuple[float, float, float, float]) -> None:
        self._tint = tuple(float(part) for part in rgba)
        self.update()

    def set_progress(self, rendered: float, total: float, playhead: float = 0.0,
                     duration: float = 0.0) -> None:
        """Two markers, two bases: how much the renderer has produced, and where playback stands.

        `rendered/total` is the render's progress (live: frames or seconds produced so far); `playhead/duration` is
        media time. Either pair with a zero denominator hides its marker - the strip stays, the marker does not
        pretend to be at 0%.
        """
        self._progress = (float(rendered), float(total), float(playhead), float(duration))
        self._hud_dirty = True
        self.update()

    def set_background_colour(self, colour) -> None:
        """The opaque colour the pane shows where no frame is drawn (the page's surface token)."""
        from PySide6.QtGui import QColor as _QColor

        self._clear_colour = _QColor(colour)
        self._cleared_empty = False
        self.update()

    def set_strip_visible(self, visible: bool) -> None:
        self._strip_visible = bool(visible)
        self._hud_dirty = True
        self.update()

    seekRequested = Signal(float)     # fraction 0..1 of the strip (only inside the produced range)
    seekBlocked = Signal(float)       # a seek was asked for past the produced edge; nothing moved
    zoomChanged = Signal(float)       # the new zoom factor
    originalRequested = Signal(float)  # playhead seconds: the host decodes the SOURCE frame at this time

    # -- interaction -----------------------------------------------------

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt spelling
        delta = event.angleDelta().y()
        if not delta:
            return
        factor = 1.15 if delta > 0 else 1 / 1.15
        self.set_zoom(self._zoom * factor, around=QRectF(event.position(), QSize(1, 1)))
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
                    # Past the produced edge there is nothing to seek to. Refusing is the honest answer: emitting a
                    # seek the renderer cannot satisfy would move the playhead over a picture that does not exist.
                    self.seekBlocked.emit(fraction)
                    event.accept()
                    return
                self._scrubbing = True
                self.seekRequested.emit(fraction)
                event.accept()
                return
            # LMB = "show the original": the surface asks the host for the source frame at the playhead and draws it as
            # soon as `set_original` arrives. The request carries the media time, so the host decodes one frame, not a
            # stream - the renderer's own original is never involved (plan: 3b, GUI-only).
            self._showing_original = True
            _, _, playhead, _ = self._progress
            self.originalRequested.emit(float(playhead))
            self.update()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt spelling
        position = event.position()
        if self._scrubbing:
            fraction = self._fraction_at(position.x())
            # A drag is clamped at the produced edge instead of jumping into the dark part: the playhead never leaves
            # frames that exist.
            self.seekRequested.emit(min(fraction, self._produced_fraction()))
            event.accept()
            return
        if self._panning:
            delta = position - self._pan_anchor
            self._pan_anchor = position
            self.pan_by(delta.x(), delta.y())
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
        if event.button() == Qt.MouseButton.MiddleButton and self._panning:
            self._panning = False
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._scrubbing = False
            if self._showing_original:
                self._showing_original = False
                self.set_original(None)   # one frame is held at a time; the next press asks again
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt spelling
        self.reset_view()
        self.zoomChanged.emit(self._zoom)
        event.accept()

    # -- strip geometry ---------------------------------------------------

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

    def _seekable(self, fraction: float) -> bool:
        return fraction <= self._produced_fraction() + 1e-6

    def _fraction_at(self, x: float) -> float:
        rect = self._strip_rect()
        if rect.width() <= 0:
            return 0.0
        return max(0.0, min(1.0, (x - rect.left()) / rect.width()))

    # -- view ------------------------------------------------------------

    def zoom(self) -> float:
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
                self._pan[0] = anchor.x() - u * drawn_w - (self.width() - drawn_w) / 2.0
                self._pan[1] = anchor.y() - v * drawn_h - (self.height() - drawn_h) / 2.0
        self._zoom = target
        self._zoom_is_fit = False
        self._clamp_pan()
        self.update()

    def reset_view(self, *, actual_size: bool = False) -> None:
        """Back to the opening view: the whole frame fitted into the pane (or 1:1 when the user asks for it)."""
        self._zoom = 1.0
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
        view = self.size()
        return min(view.width() / width, view.height() / height)

    def _frame_rect(self) -> QRectF:
        """The picture's rectangle in widget pixels at the current zoom/pan, centred."""
        width, height = self._image_size()
        if not width or not height:
            return QRectF(0, 0, 0, 0)
        scale = self._fit_scale() if self._zoom_is_fit else self._zoom
        drawn_w, drawn_h = width * scale, height * scale
        left = (self.size().width() - drawn_w) / 2.0 + self._pan[0]
        top = (self.size().height() - drawn_h) / 2.0 + self._pan[1]
        return QRectF(left, top, drawn_w, drawn_h)

    def _clamp_pan(self) -> None:
        """Keep the picture over the pane: no gutter when zoomed past the pane, no drifting it away when it fits."""
        width, height = self._image_size()
        if not width or not height:
            return
        scale = self._fit_scale() if self._zoom_is_fit else self._zoom
        drawn_w, drawn_h = width * scale, height * scale
        max_x = max(0.0, (drawn_w - self.size().width()) / 2.0)
        max_y = max(0.0, (drawn_h - self.size().height()) / 2.0)
        self._pan[0] = max(-max_x, min(max_x, self._pan[0]))
        self._pan[1] = max(-max_y, min(max_y, self._pan[1]))

    # -- RHI ---------------------------------------------------------------

    def initialize(self, command_buffer) -> None:  # noqa: N802 - Qt spelling
        self._release()
        rhi = self.rhi()
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
        self._texture = rhi.newTexture(QRhiTexture.Format.RGBA8, initial_size, 1,
                                       QRhiTexture.Flag.UsedAsTransferSource)
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
        self._sampler_linear = rhi.newSampler(QRhiSampler.Filter.Linear, QRhiSampler.Filter.Linear,
                                              QRhiSampler.Filter.None_, QRhiSampler.AddressMode.ClampToEdge,
                                              QRhiSampler.AddressMode.ClampToEdge)
        self._sampler_linear.create()

        self._uniforms = rhi.newBuffer(QRhiBuffer.Type.Dynamic, QRhiBuffer.UsageFlag.UniformBuffer, 64 + 16)
        self._uniforms.create()

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

    def _build_bindings(self) -> None:
        """(Re)create the two SRBs. The sampler is part of the SRB, so the 1:1 (nearest) and zoomed (linear) views are
        two bindings that differ only by the sampler; and a texture that is destroyed and re-created at a new size
        invalidates the native handle the SRB holds, so this runs again after such a re-create."""
        rhi = self._rhi
        # The UBO is read by BOTH stages (`mvp` in the vertex shader, `tint` in the fragment one), and the sampler's
        # binding index must equal the `binding = 1` the fragment shader declares: QRhi assigns texture units by SRB
        # binding index, so a mismatch reads an unbound unit and returns (0,0,0,0) - a transparent "frame" that looks
        # like a missing upload.
        flags = (QRhiShaderResourceBinding.StageFlag.VertexStage
                 | QRhiShaderResourceBinding.StageFlag.FragmentStage)
        frag = QRhiShaderResourceBinding.StageFlag.FragmentStage
        for attribute, sampler in (("_bindings", self._sampler_linear),
                                   ("_bindings_nearest", self._sampler_nearest)):
            current = getattr(self, attribute, None)
            if current is not None:
                current.destroy()
            bindings = rhi.newShaderResourceBindings()
            bindings.setBindings([
                QRhiShaderResourceBinding.uniformBuffer(0, flags, self._uniforms),
                QRhiShaderResourceBinding.sampledTexture(1, frag, self._texture, sampler),
            ])
            bindings.create()
            setattr(self, attribute, bindings)

    def _release(self) -> None:
        for name in ("_pipeline", "_bindings", "_bindings_nearest", "_vertices", "_uniforms",
                     "_sampler_nearest", "_sampler_linear", "_texture",
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
        if target is None:
            return
        if self._frame is None:
            # Nothing to show yet: one OPAQUE clear, so the pane reads as an empty plate rather than a black hole.
            if not getattr(self, "_cleared_empty", False):
                self._cleared_empty = True
                command_buffer.beginPass(target, self._opaque_clear(), QRhiDepthStencilClearValue(1.0, 0))
                command_buffer.endPass()
            return
        if getattr(self, "_deferred", False) or getattr(self, "_rhi", None) is None:
            self._release()
            self._rhi = self.rhi()
            self._create_resources(command_buffer)
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
            updates.uploadTexture(self._texture, picture)
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
        command_buffer.beginPass(target, self._opaque_clear(), QRhiDepthStencilClearValue(1.0, 0), updates)
        command_buffer.setGraphicsPipeline(self._pipeline)
        command_buffer.setViewport(QRhiViewport(0, 0, target.pixelSize().width(), target.pixelSize().height()))
        sample_at_one = (not self._zoom_is_fit) and abs(self._zoom - 1.0) < 1e-6
        command_buffer.setShaderResources(self._bindings_nearest if sample_at_one else self._bindings)
        command_buffer.setVertexInput(0, [(self._vertices, 0)])
        command_buffer.draw(6)
        if hud_vertices and getattr(self, "_hud_pipeline", None) is not None:
            command_buffer.setGraphicsPipeline(self._hud_pipeline)
            command_buffer.setShaderResources(self._hud_bindings)
            command_buffer.setVertexInput(0, [(self._hud_buffer, 0)])
            command_buffer.draw(len(hud_vertices) // 6)
        command_buffer.endPass()

    def _mvp_floats(self) -> list[float]:
        """Unit quad -> the picture's rect on the pane, column-major for the uniform buffer.

        Written out instead of composed from ``QMatrix4x4``: PySide6 hands that object back row-major, and a transposed
        transform in a shader is a silently mirrored picture - the kind of bug that reads as "the widget drew something"
        and is caught only by looking at the pixels.
        """
        rect = self._frame_rect()
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
        """Rectangles for the strip, in widget pixels: track, rendered range, playhead, render head, handle."""
        strip = self._strip_rect()
        if strip.isEmpty():
            return []
        rendered, total, playhead, duration = self._progress
        # Two colours, by the owner's rule (2026-11-23): the DARK band is what the renderer has not produced yet, the
        # WHITE part is what it has. No accent colour on the strip - the fill's own edge is the render head, so a tick
        # there was the same information drawn twice.
        rects: list[tuple[float, float, float, float, tuple[float, float, float, float]]] = [
            (strip.left(), strip.top(), strip.width(), strip.height(), (1.0, 1.0, 1.0, 0.14)),
        ]
        if total > 0:
            filled = max(0.0, min(1.0, rendered / total))
            if filled > 0:
                rects.append((strip.left(), strip.top(), strip.width() * filled, strip.height(),
                              (1.0, 1.0, 1.0, 0.92)))
        if duration > 0:
            fraction = max(0.0, min(1.0, playhead / duration))
            x = strip.left() + strip.width() * fraction
            handle = self._handle_radius
            # A dark halo behind the playhead: a white marker disappears on the white "processed" part, and a dark one
            # disappears on the dark part - the pair is visible on both.
            rects.append((x - 2.5, strip.top() - 6.0, 5.0, strip.height() + 12.0, (0.0, 0.0, 0.0, 0.45)))
            rects.append((x - 1.0, strip.top() - 5.0, 2.0, strip.height() + 10.0, (1.0, 1.0, 1.0, 0.95)))
            rects.append((x - handle - 1.5, strip.top() - 6.0 - handle - 1.5, (handle + 1.5) * 2.0, (handle + 1.5) * 2.0,
                          (0.0, 0.0, 0.0, 0.45)))
            rects.append((x - handle, strip.top() - 6.0 - handle, handle * 2.0, handle * 2.0,
                          (1.0, 1.0, 1.0, 0.95)))
        if self._hover_fraction is not None and not self._scrubbing:
            x = strip.left() + strip.width() * self._hover_fraction
            colour = (1.0, 0.45, 0.45, 0.55) if self._hover_blocked else (1.0, 1.0, 1.0, 0.55)
            rects.append((x - 0.5, strip.top() - 3.0, 1.0, strip.height() + 6.0, colour))
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
