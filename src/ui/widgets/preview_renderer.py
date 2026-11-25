"""The preview pane's GPU renderer: the QRhi resources, the per-frame uploads and the pass.

This is the seam of `plan-rve-preview-controls-frame-path-2026-11-25` (phase 0): `video_preview.py` used to hold three
concerns in one class - the widget API, this RHI renderer and the painted chrome. The renderer moved here unchanged, so
the pane can keep its public API while the frame path grows a YUV branch (the next lane) without touching the widget.

What the renderer owns, and nothing else:

* every QRhi object: the result and source textures, the two samplers, the four SRBs, the pipelines (frame and HUD),
  the uniform and vertex buffers, the HUD geometry buffer, and their `create()`/`destroy()` lifetime;
* the per-frame uploads (`uploadTexture`, `generateMips`, the dynamic uniform/HUD buffers) and the one pass that draws
  them (`beginPass`/`endPass`);
* the resource-update batch and the paint/upload telemetry that belongs to it.

What it does NOT own, and must not start owning:

* geometry. The zoom/pan/fit arithmetic, the mvp matrices, the clip rects, the split fraction and the HUD vertex list
  all stay in `VideoPreviewSurface`; they arrive per frame in a `PreviewScene` (see its docstring for the contract);
* frame bookkeeping and the draw log. The widget decides what a drawn picture means (which half, which media time, the
  fingerprint line); the renderer only reports that it drew a half through the `on_draw` callback (`PreviewDrawn`);
* the clear colour's policy. `set_background_colour`/`_opaque_clear` stay on the widget - the scene carries the final
  opaque `QColor`.

The one lifetime rule that was expensive to learn and stays exactly as it was: `initialize()` runs on EVERY resize, so
releasing the resources there unconditionally leaves the pane sampling an undrawn texture after any window resize (the
"preview turns transparent at any sneeze" defect). Same context -> keep the resources and re-upload the standing
pictures; new context -> release and rebuild.

Frame ingestion is `PreviewFrame` (frozen): `format="rgba8"` with `image=` today, and the `yuv420p10le`/`yuv420p`
plane descriptors are the contract the next lane fills in. This file implements `rgba8` only and says so loudly for any
other format instead of drawing a black pane.
"""

from __future__ import annotations

import time

from dataclasses import dataclass

from PySide6.QtCore import QRectF, QSize
from PySide6.QtGui import (
    QColor,
    QImage,
    QRhi,
    QRhiBuffer,
    QRhiDepthStencilClearValue,
    QRhiGraphicsPipeline,
    QRhiRenderTarget,
    QRhiSampler,
    QRhiScissor,
    QRhiShaderResourceBinding,
    QRhiShaderStage,
    QRhiTexture,
    QRhiVertexInputAttribute,
    QRhiVertexInputBinding,
    QRhiVertexInputLayout,
    QRhiViewport,
    QShader,
)

from src.ui.video import telemetry

SHADER_DIR = "shaders"


def mip_minify_enabled() -> bool:
    """Mipminification on/off. The chain is the whole fix for the fit view (a 1440x1080 frame in a 421x512 pane is a
    3.42x minification through ONE bilinear tap, which aliases); the switch exists so an A/B runs on ONE build and the
    owner can compare the two arms himself. `RVE_MIP_MINIFY=0` keeps the old single-level texture."""
    import os

    return str(os.environ.get("RVE_MIP_MINIFY", "1")).strip() != "0"


@dataclass(frozen=True)
class PreviewFrame:
    """One picture to ingest: its pixel format, its size and its identity.

    `rgba8` carries the pixels as a `QImage` (already converted by the caller). The YUV formats carry the planes and
    their byte strides instead - the next lane (the YUV path) fills those in; the renderer refuses anything that is not
    `rgba8` today rather than sampling an unbound texture.

    `width`/`height` describe the PICTURE. For `rgba8` the texture is sized from the image itself (implicit-sharing
    cheap), so a descriptor that disagrees with its image still uploads what it draws.
    """

    format: str                    # "rgba8" | "yuv420p10le" | "yuv420p"
    width: int
    height: int
    pts: float | None = None
    index: int | None = None
    image: QImage | None = None    # rgba8 only
    planes: tuple[memoryview, ...] | None = None   # Y, U, V
    strides: tuple[int, ...] | None = None


@dataclass(frozen=True)
class PreviewScene:
    """One frame's draw description, computed entirely by the widget.

    The renderer never reads `VideoPreviewSurface`: every number it needs arrives here, in widget pixels (the scissor
    helper multiplies by `dpr` and clamps to the target's pixel size, exactly as before).

    * `target` - the render target of THIS paint (`QRhiWidget.renderTarget()`), never cached across paints: a resize
      replaces it and a pass must go into the current one.
    * `rhi` - the widget's current QRhi device (`QRhiWidget.rhi()`): the deferred first-frame path builds its resources
      against the device of the paint they are built for, exactly as the inline path did.
    * `mode` - `"single"` | `"sbs"` | `"split"`, as `set_view_mode` defines them.
    * `result_mvp` / `original_mvp` - the 16-float column-major mvp blocks (`VideoPreviewSurface._mvp_floats`), one per
      picture. `original_mvp is None` means there is no source picture to place.
    * `tint` - the straight-alpha multiply, `(r, g, b, a)`.
    * `nearest` - which sampler the magnified view wants (`samples_nearest_for`).
    * `result_rect` / `result_clip` - where the result is drawn and where it may paint (the full pane in `single`, its
      half in `sbs`).
    * `original_clip` - where the source may paint (`sbs`: its half, `split`: left of the divider; empty: nowhere).
    * `show_original` - the `single`-mode LMB flash: the SHARED texture carries the source picture.
    * `hud_vertices` - the strip's rectangles, 6 floats per vertex (`_hud_vertices`), already in widget pixels.
    * `viewport` / `dpr` - the widget size in device-independent px and its device pixel ratio, for the HUD's viewport
      uniform and the scissor conversion.
    * `clear` - the opaque clear colour (`_opaque_clear`); the pass always clears once.
    """

    target: QRhiRenderTarget
    rhi: QRhi
    mode: str
    result_mvp: tuple[float, ...]
    original_mvp: tuple[float, ...] | None
    tint: tuple[float, ...]
    nearest: bool
    result_rect: QRectF
    result_clip: QRectF
    original_clip: QRectF
    show_original: bool
    hud_vertices: tuple[float, ...]
    viewport: tuple[float, float]
    dpr: float
    clear: QColor


@dataclass(frozen=True)
class PreviewDrawn:
    """What one pass actually drew, as the widget's draw log needs it.

    `half` is `"result"` or `"original"` in `single` (the shared texture carries one picture), `"pair"` in the compare
    modes (one pass, both halves). `drew_original` is the compare modes' "the source was part of this pass" flag - the
    same condition the old inline log used (an original frame exists and its texture does), so the log lines stay
    identical whether or not the clip rect turned out empty.
    """

    mode: str
    half: str
    drew_original: bool = False


class PreviewRenderer:
    """The pane's QRhi renderer: resources, uploads, and the pass. See the module docstring for the split."""

    def __init__(self, on_draw=None) -> None:
        #: Called once per pass with a `PreviewDrawn`; the widget owns the log format (fingerprints, identity, level).
        self._on_draw = on_draw
        #: Geometry for the paint that is about to run; `set_scene` fills it in every frame.
        self._scene: PreviewScene | None = None
        self._target: QRhiRenderTarget | None = None
        #: The ingested pictures. `rgba8` only today; the descriptors are the frame-ingestion contract
        #: (`PreviewFrame`).
        self._frame: PreviewFrame | None = None
        self._original: PreviewFrame | None = None
        #: The view mode as `set_view_mode` last stated it: `set_original` reads it to decide WHICH texture is dirty
        self._mode = "single"
        self._rhi = None
        #: No resource set yet (a context change, or the first frame has not arrived): the next `render` builds it.
        self._deferred = True
        #: One flag per picture: whether the next pass must upload it. Set outside a frame (a texture is created and an
        #: upload queued between frames, never inside a pass).
        self._result_dirty = False
        self._original_dirty = False
        # Everything below is None until `_create_resources` ran; `release` puts it back to None.
        self._texture = None
        self._original_texture = None
        self._sampler_nearest = None
        self._sampler_linear = None
        self._uniforms = None
        self._original_uniforms = None
        self._vertices = None
        self._bindings = None
        self._bindings_nearest = None
        self._original_bindings = None
        self._original_bindings_nearest = None
        self._pipeline = None
        self._hud_uniforms = None
        self._hud_buffer = None
        self._hud_bindings = None
        self._hud_pipeline = None
        self._hud_capacity = 0

    # -- ingestion -------------------------------------------------------

    def set_frame(self, frame: PreviewFrame | None) -> None:
        """Show this picture; `None` forgets it (the pass then just clears)."""
        self._check_frame(frame)
        self._frame = frame
        self._result_dirty = frame is not None
        if self._result_dirty and self._texture is not None:
            # Create the texture OUTSIDE a frame: QRhi resources are created between frames, and a create() issued
            # while a frame is being recorded is refused by the backend without an exception (the frame then samples
            # the 1x1 placeholder - opaque black, which reads exactly like "the upload was skipped").
            self._ensure_texture(self._frame_size(frame))

    def set_original(self, frame: PreviewFrame | None) -> None:
        """The source picture for the compare; `None` drops it.

        In `single` it replaces the frame in the ONE shared texture (the LMB flash); in the compare modes it has its own
        texture so an arriving source frame never clobbers the result's upload.
        """
        self._check_frame(frame)
        self._original = frame
        if self._mode == "single":
            self._result_dirty = True
        else:
            self._original_dirty = frame is not None
            if frame is not None and self._texture is not None:
                self._ensure_original_texture(self._frame_size(frame))

    def set_view_mode(self, mode: str) -> None:
        """State the view mode; both textures are re-uploaded once on a real change.

        `single` may have left the source in the shared texture, and the compare modes keep the result there. Cheap
        (one upload) and it removes a whole class of "stale half" readings.
        """
        if mode == self._mode:
            return
        self._mode = mode
        self._result_dirty = self._frame is not None
        self._original_dirty = self._original is not None

    @staticmethod
    def _check_frame(frame: PreviewFrame | None) -> None:
        """Refuse a frame this renderer cannot upload yet - loudly, not as a silently black pane."""
        if frame is None:
            return
        if frame.format != "rgba8":
            raise NotImplementedError(
                f"PreviewRenderer: frame format {frame.format!r} is not implemented yet (only 'rgba8'; the YUV planes "
                "path is the next lane)")
        if frame.image is None:
            raise ValueError("PreviewRenderer: an 'rgba8' PreviewFrame must carry its QImage in `image`")

    @staticmethod
    def _frame_size(frame: PreviewFrame) -> QSize:
        """The texture size for a picture: its image's own size when it has one, the descriptor's otherwise."""
        if frame.image is not None:
            return frame.image.size()
        return QSize(int(frame.width), int(frame.height))

    # -- lifetime --------------------------------------------------------

    def initialize(self, command_buffer, rhi, target: QRhiRenderTarget | None) -> bool:
        """Build the resource set - but not on a mere resize; returns True when the existing set was kept.

        Qt calls `initialize()`/`releaseResources()` on EVERY resize (the widget's backing target is reallocated at the
        new pixel size), not only on a real context change. Same RHI and a live pipeline therefore means "new target
        size": keep every resource, mark the standing pictures dirty and let the next paint re-upload them. That is
        what keeps a paused frame on screen across a resize.
        """
        self._target = target
        same_context = rhi is self._rhi and self._pipeline is not None
        if same_context:
            self._rhi = rhi
            if self._frame is not None:
                self._result_dirty = True
            if self._original is not None:
                self._original_dirty = True
            return True
        self.release()
        self._rhi = rhi
        if self._frame is None:
            # Nothing to draw yet. Building a 1x1 texture and then resizing it forces an SRB and pipeline rebuild in
            # the middle of the first real frame - the one path that produced a black pane in every probe. Deferring
            # the whole resource set to the first frame removes that path instead of patching it.
            self._deferred = True
            return False
        self._create_resources(command_buffer)
        return False

    def release(self) -> None:
        """Destroy every QRhi object. Idempotent: the names stay (as None) for the next `initialize`."""
        for name in ("_pipeline", "_bindings", "_bindings_nearest", "_vertices", "_uniforms",
                     "_sampler_nearest", "_sampler_linear", "_texture", "_original_texture",
                     "_original_bindings", "_original_bindings_nearest", "_original_uniforms",
                     "_hud_pipeline", "_hud_bindings", "_hud_buffer", "_hud_uniforms"):
            item = getattr(self, name, None)
            if item is not None:
                try:
                    item.destroy()
                except Exception:  # a resource the context already took: teardown must not raise
                    pass
                setattr(self, name, None)

    # -- the pass --------------------------------------------------------

    def set_scene(self, scene: PreviewScene) -> None:
        """Hand the renderer this frame's geometry (see `PreviewScene`); `render` consumes it."""
        self._scene = scene
        self._target = scene.target

    def render(self, command_buffer) -> None:
        """Draw one frame into `scene.target`: an opaque clear alone when nothing is ingested yet."""
        scene = self._scene
        if scene is None or scene.target is None:
            return
        target = scene.target
        if self._frame is None:
            # Nothing to show yet: an OPAQUE clear on EVERY render, so the pane reads as an empty plate rather than a
            # hole. It used to clear once (`_cleared_empty`) - a latch that is false on the next paint, which leaves the
            # region undrawn after any repaint and shows the window behind it (the "transparent at any sneeze" half).
            pass_started = time.perf_counter()
            command_buffer.beginPass(target, scene.clear, QRhiDepthStencilClearValue(1.0, 0))
            command_buffer.endPass()
            # An empty pane still paints: counting it keeps the cadence honest before the first frame arrives, and
            # `frames=0` in a stats line is itself an answer ("the render never fed the pane").
            telemetry.note_paint(pass_ms=(time.perf_counter() - pass_started) * 1000.0)
            telemetry.maybe_stats()
            return
        if self._deferred or self._rhi is None:
            self._rhi = scene.rhi
            self.release()
            if self._rhi is None:
                return
            self._create_resources(command_buffer)
        if self._mode == "single":
            self._render_single(command_buffer, target, scene)
            return
        self._render_two_surfaces(command_buffer, target, scene)

    def _render_single(self, command_buffer, target, scene: PreviewScene) -> None:
        """The one-picture pane: one pass per frame, the LMB flash replacing the picture in place."""
        showing_original = bool(scene.show_original and self._original is not None)
        picture = self._original if showing_original else self._frame
        if picture is None:
            return
        if self._result_dirty:
            self._ensure_texture(self._frame_size(picture))
        if self._pipeline is None:
            self._build_pipeline()
            if self._pipeline is None:
                return
        updates = self._rhi.nextResourceUpdateBatch()
        if self._result_dirty:
            upload_started = time.perf_counter()
            updates.uploadTexture(self._texture, picture.image)
            if mip_minify_enabled():
                # Same batch as the upload, BEFORE the pass: the chain is built from level 0 on the queue, and
                # `resourceUpdate` is only legal outside a render pass. Cost is one-time per uploaded frame (the chain
                # of an 1440x1080 RGBA8 is ~2.1 MB - noise against the 6.2 MB level 0 that already crossed the bus).
                updates.generateMips(self._texture)
            telemetry.note_upload(upload_ms=(time.perf_counter() - upload_started) * 1000.0,
                                  size_bytes=self._pixel_bytes(picture))
            self._result_dirty = False
        updates.updateDynamicBuffer(self._uniforms, 0, 64, _floats(scene.result_mvp))
        updates.updateDynamicBuffer(self._uniforms, 64, 16, _floats(list(scene.tint)))
        # The HUD's buffers are filled into the SAME batch, before the pass: `QRhiCommandBuffer::resourceUpdate` must be
        # called outside a render pass, so building the strip inside one (the first version did) is not allowed.
        hud_vertices = list(scene.hud_vertices)
        if hud_vertices and self._hud_pipeline is not None:
            updates.updateDynamicBuffer(self._hud_uniforms, 0, 16,
                                        _floats([float(scene.viewport[0]), float(scene.viewport[1]),
                                                 float(scene.dpr), 0.0]))
            updates.updateDynamicBuffer(self._hud_buffer, 0, len(hud_vertices) * 4, _floats(hud_vertices))
        pass_started = time.perf_counter()
        command_buffer.beginPass(target, scene.clear, QRhiDepthStencilClearValue(1.0, 0), updates)
        command_buffer.setGraphicsPipeline(self._pipeline)
        command_buffer.setViewport(QRhiViewport(0, 0, target.pixelSize().width(), target.pixelSize().height()))
        # The pipeline declares `UsesScissor` (the compare modes need it), so the scissor state is LIVE here: a frame
        # must state its own scissor before drawing, or it inherits the last one set on the command buffer - an empty
        # one clipped the whole picture away in `single` after a compare mode had run (pixel gate, 2026-11-24).
        command_buffer.setScissor(
            QRhiScissor(0, 0, target.pixelSize().width(), target.pixelSize().height()))
        command_buffer.setShaderResources(self._bindings_nearest if scene.nearest else self._bindings)
        command_buffer.setVertexInput(0, [(self._vertices, 0)])
        command_buffer.draw(6)
        if hud_vertices and self._hud_pipeline is not None:
            command_buffer.setGraphicsPipeline(self._hud_pipeline)
            command_buffer.setShaderResources(self._hud_bindings)
            command_buffer.setVertexInput(0, [(self._hud_buffer, 0)])
            command_buffer.draw(len(hud_vertices) // 6)
        command_buffer.endPass()
        telemetry.note_paint(pass_ms=(time.perf_counter() - pass_started) * 1000.0)
        # The LAST step, named: a press is one line (the original it put on screen, plus the result it stands over);
        # level 2 additionally names every result picture, so a shift is visible frame by frame.
        self._note_draw(PreviewDrawn(mode="single", half="original" if showing_original else "result"))
        telemetry.maybe_stats()

    def _render_two_surfaces(self, command_buffer, target, scene: PreviewScene) -> None:
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
        self._ensure_texture(self._frame_size(frame))
        original = self._original
        if original is not None and _null_picture(original):
            # The old inline path treated a null source image as "no source half" - it must not reach
            # `_ensure_original_texture` with a 0x0 size, which QRhi refuses.
            original = None
        if original is not None:
            self._ensure_original_texture(self._frame_size(original))
        if self._pipeline is None:
            self._build_pipeline()
            if self._pipeline is None:
                return
        updates = self._rhi.nextResourceUpdateBatch()
        if self._result_dirty:
            upload_started = time.perf_counter()
            updates.uploadTexture(self._texture, frame.image)
            if mip_minify_enabled():
                updates.generateMips(self._texture)
            telemetry.note_upload(upload_ms=(time.perf_counter() - upload_started) * 1000.0,
                                  size_bytes=self._pixel_bytes(frame))
            self._result_dirty = False
        tint = list(scene.tint)
        updates.updateDynamicBuffer(self._uniforms, 0, 64, _floats(scene.result_mvp))
        updates.updateDynamicBuffer(self._uniforms, 64, 16, _floats(tint))
        draw_original = (original is not None and self._original_texture is not None
                         and scene.original_mvp is not None)
        if draw_original:
            if self._original_dirty:
                upload_started = time.perf_counter()
                updates.uploadTexture(self._original_texture, original.image)
                telemetry.note_upload(upload_ms=(time.perf_counter() - upload_started) * 1000.0,
                                      size_bytes=self._pixel_bytes(original))
                self._original_dirty = False
            updates.updateDynamicBuffer(self._original_uniforms, 0, 64, _floats(scene.original_mvp))
            updates.updateDynamicBuffer(self._original_uniforms, 64, 16, _floats(tint))
        hud_vertices = list(scene.hud_vertices)
        if hud_vertices and self._hud_pipeline is not None:
            updates.updateDynamicBuffer(self._hud_uniforms, 0, 16,
                                        _floats([float(scene.viewport[0]), float(scene.viewport[1]),
                                                 float(scene.dpr), 0.0]))
            updates.updateDynamicBuffer(self._hud_buffer, 0, len(hud_vertices) * 4, _floats(hud_vertices))
        pass_started = time.perf_counter()
        command_buffer.beginPass(target, scene.clear, QRhiDepthStencilClearValue(1.0, 0), updates)
        command_buffer.setGraphicsPipeline(self._pipeline)
        command_buffer.setViewport(QRhiViewport(0, 0, target.pixelSize().width(), target.pixelSize().height()))
        result_bindings = self._bindings_nearest if scene.nearest else self._bindings
        result_clip = scene.result_clip
        if result_bindings is not None and not result_clip.isEmpty() and scene.result_rect.intersects(result_clip):
            command_buffer.setShaderResources(result_bindings)
            command_buffer.setVertexInput(0, [(self._vertices, 0)])
            self._scissor(command_buffer, result_clip, target, scene.dpr)
            command_buffer.draw(6)
        if draw_original:
            original_bindings = (self._original_bindings_nearest if scene.nearest
                                 else self._original_bindings)
            original_clip = scene.original_clip
            if original_bindings is not None and not original_clip.isEmpty():
                command_buffer.setShaderResources(original_bindings)
                command_buffer.setVertexInput(0, [(self._vertices, 0)])
                self._scissor(command_buffer, original_clip, target, scene.dpr)
                command_buffer.draw(6)
        # The HUD spans the whole widget; a scissor left over from one half would clip the far end of the strip.
        # `QRhiCommandBuffer.setScissor` takes a `QRhiScissor`, not a `QRect`: PySide6 6.11 rejected the implicit rect
        # with `TypeError: setScissor(QRect)` while 6.8 still accepted it, so `sbs`/`split` drew nothing there and said
        # so only inside a paint (the two-surface pixel gate, 2026-11-24, was the first thing to see it).
        command_buffer.setScissor(
            QRhiScissor(0, 0, target.pixelSize().width(), target.pixelSize().height()))
        if hud_vertices and self._hud_pipeline is not None:
            command_buffer.setGraphicsPipeline(self._hud_pipeline)
            command_buffer.setShaderResources(self._hud_bindings)
            command_buffer.setVertexInput(0, [(self._hud_buffer, 0)])
            command_buffer.draw(len(hud_vertices) // 6)
        command_buffer.endPass()
        telemetry.note_paint(pass_ms=(time.perf_counter() - pass_started) * 1000.0)
        # Both halves in one pass: one line names both pictures (a new original always, every paint at level 2). The
        # flag is the TEXTURE condition, not the clip's: the old inline log named the pair whenever a source frame and
        # its texture existed, and the draw log keeps exactly that reading.
        self._note_draw(PreviewDrawn(mode=self._mode, half="pair", drew_original=draw_original))
        telemetry.maybe_stats()

    def _note_draw(self, drawn: PreviewDrawn) -> None:
        if self._on_draw is not None:
            self._on_draw(drawn)

    # -- resources -------------------------------------------------------

    def _create_resources(self, command_buffer) -> None:
        """All RHI resources for the CURRENT frame size, built once per size change.

        `command_buffer` is needed for the static vertex upload; `initialize()` and `render()` both have one, which is
        why this is a method taking it rather than a constructor-time step.
        """
        rhi = self._rhi
        initial_size = self._frame_size(self._frame) if self._frame is not None else QSize(1, 1)
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
        self._hud_pipeline.setRenderPassDescriptor(self._target.renderPassDescriptor())
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

        self._deferred = False
        self._result_dirty = self._frame is not None
        self._pipeline = None
        self._original_texture = None
        if self._original is not None and not _null_picture(self._original):
            # A compare mode was opened before the first frame (or the resources were rebuilt while it stood). The
            # source's texture is created exactly like the frame's, outside a frame, and its own upload follows.
            self._original_dirty = True
            self._ensure_original_texture(self._frame_size(self._original))

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
        self._pipeline.setRenderPassDescriptor(self._target.renderPassDescriptor())
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
        if self._texture is None:
            return
        if self._texture.pixelSize() == size:
            return
        self._texture.destroy()
        self._texture.setPixelSize(size)
        if not self._texture.create():
            raise RuntimeError(f"could not resize the frame texture to {size.width()}x{size.height()}")
        self._build_bindings()
        pipeline = self._pipeline
        if pipeline is not None:
            pipeline.destroy()
        self._pipeline = None

    def _ensure_original_texture(self, size) -> None:
        """Create or resize the SOURCE picture's own texture (the compare modes draw both pictures in one pass).

        Same rules as `_ensure_texture`, and the same trap: an SRB holds the texture's native handle, so a texture that
        is destroyed and re-created at a new size silently invalidates every binding that names it. The pipeline is
        built against a specific SRB object, so it is dropped too and rebuilt on the next render.
        """
        if self._texture is None or self._rhi is None:
            # The resource set is built at the first real frame; before that there is nothing to attach a texture to.
            return
        texture = self._original_texture
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
        pipeline = self._pipeline
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
            ("_original_bindings", self._original_uniforms, self._original_texture, self._sampler_linear),
            ("_original_bindings_nearest", self._original_uniforms, self._original_texture,
             self._sampler_nearest),
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

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _pixel_bytes(frame: PreviewFrame) -> int:
        """The bytes one uploaded picture costs, for the upload telemetry (RGBA8888: 4 per pixel)."""
        size = PreviewRenderer._frame_size(frame)
        return size.width() * size.height() * 4

    @staticmethod
    def _scissor(command_buffer, rect: QRectF, target, dpr: float) -> None:
        """Confine the next draw to a widget-space rect (QRhi scissors are in framebuffer pixels)."""
        pixels = target.pixelSize()
        left = max(0.0, rect.left() * dpr)
        top = max(0.0, rect.top() * dpr)
        right = min(float(pixels.width()), rect.right() * dpr)
        bottom = min(float(pixels.height()), rect.bottom() * dpr)
        command_buffer.setScissor(QRhiScissor(int(left), int(top),
                                             max(0, int(right - left)), max(0, int(bottom - top))))


def _null_picture(frame: PreviewFrame) -> bool:
    """Whether this frame is an `rgba8` descriptor whose image is null (the old inline path drew nothing for it)."""
    return frame.format == "rgba8" and frame.image is not None and frame.image.isNull()


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
