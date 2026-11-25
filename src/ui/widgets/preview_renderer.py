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

Frame ingestion is `PreviewFrame` (frozen): `format="rgba8"` with `image=`, or one of the planar formats
(`yuv420p10le`, `yuv420p`) with `planes=`/`strides=`. The RGBA path is the fallback and is byte-identical to what it
was; the planes path uploads three textures (Y full size, U/V half) in ONE resource-update batch and converts colour in
the fragment shader (`yuv.frag`), with the matrix and the range taken from the frame's own tags (`color_space`,
`color_range`) rather than from a constant. `yuv_code_to_rgb` below is the CPU mirror of that shader for tests and for
the pixel gate's reference.
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
    QRhiTextureUploadDescription,
    QRhiTextureUploadEntry,
    QRhiTextureSubresourceUploadDescription,
    QRhiVertexInputAttribute,
    QRhiVertexInputBinding,
    QRhiVertexInputLayout,
    QRhiViewport,
    QShader,
)

from src.ui.video import telemetry
from src.ui.video.frames import PLANAR_FORMATS, SAMPLE_BYTES, plane_geometry

SHADER_DIR = "shaders"

#: The planar formats this renderer uploads as three textures.
YUV_FORMATS = PLANAR_FORMATS

#: The planes uniform block: mat4 mvp (64) + vec4 tint (16) + luma (16) + chroma (16) + three matrix
#: rows (48). std140 packs vec4 members on 16-byte boundaries, so this is also the byte size.
YUV_UNIFORM_BYTES = 160

#: The full-range matrices (luma already expanded to 0..1, chroma centred on 0). Rows are
#: `(Y, Cb, Cr) -> R|G|B`. These are the standard full-range coefficients of each matrix: the
#: LIMITED range is handled by the luma/chroma affine below, so the matrix itself is range-free -
#: exactly how a decoder separates "range expansion" from "matrix" (ITU-T H.273).
YUV_MATRICES = {
    "bt709": ((1.0, 0.0, 1.5748), (1.0, -0.187324, -0.468124), (1.0, 1.8556, 0.0)),
    "smpte170m": ((1.0, 0.0, 1.402), (1.0, -0.344136, -0.714136), (1.0, 1.772, 0.0)),
    "bt470bg": ((1.0, 0.0, 1.402), (1.0, -0.344136, -0.714136), (1.0, 1.772, 0.0)),
    "bt2020nc": ((1.0, 0.0, 1.4746), (1.0, -0.164553, -0.571353), (1.0, 1.8814, 0.0)),
}
#: Every matrix this app's writer can tag, and the fallback when a file declares none. The writer
#: emits `-colorspace bt709 -color_range tv`, so bt709/tv is what an untagged file of ours is.
DEFAULT_COLOR_SPACE = "bt709"
DEFAULT_COLOR_RANGE = "tv"

#: Per-depth code ranges of the two ranges (H.273): luma `(offset, span)`, chroma `(offset, span)`.
#: Limited ("tv") 10-bit is 64..940 luma and 64..960 chroma; 8-bit is 16..235 and 16..240.
_RANGE_CODES = {
    8: {"tv": ((16, 219), (128, 224)), "pc": ((0, 255), (128, 255))},
    10: {"tv": ((64, 876), (512, 896)), "pc": ((0, 1023), (512, 1023))},
}


def yuv_uniform_floats(color_space: str | None = None, color_range: str | None = None,
                       *, depth: int = 10) -> tuple[float, ...]:
    """The 20 floats the YUV shader reads: luma/chroma affine, then the three matrix rows.

    Layout (std140, four `vec4`s after the mvp/tint): `luma = (multiplier, offset, 0, 0)`,
    `chroma = (multiplier, offset, 0, 0)`, then `row0/row1/row2 = (Y, Cb, Cr, 0)`.

    The multipliers fold the sample normalisation IN: an R16 texture samples `code / 65535`, an R8
    texture `code / 255`, and the shader only ever writes `s * multiplier + offset`. So the one
    arithmetic trap of a 10-bit plane upload (dividing by 65535 instead of 1023) is answered here,
    once, where a test can read it.
    """
    matrix = YUV_MATRICES.get(str(color_space or DEFAULT_COLOR_SPACE), YUV_MATRICES[DEFAULT_COLOR_SPACE])
    rng = "pc" if str(color_range or DEFAULT_COLOR_RANGE).lower() in ("pc", "jpeg", "full") else "tv"
    (y_off, y_span), (c_off, c_span) = _RANGE_CODES[int(depth)][rng]
    sample_max = 65535.0 if int(depth) > 8 else 255.0
    luma = (sample_max / float(y_span), -float(y_off) / float(y_span), 0.0, 0.0)
    chroma = (sample_max / float(c_span), -float(c_off) / float(c_span), 0.0, 0.0)
    floats: list[float] = [*luma, *chroma]
    for row in matrix:
        floats.extend([float(row[0]), float(row[1]), float(row[2]), 0.0])
    return tuple(floats)


def yuv_code_to_rgb(y: float, u: float, v: float, *, color_space: str | None = None,
                    color_range: str | None = None, depth: int = 10) -> tuple[float, float, float]:
    """The CPU mirror of `yuv.frag`: code values at `depth` -> RGB in 0..255 (clamped).

    Same numbers the shader gets, applied in Python: the uniform floats are scaled back into "code
    space" by the sample maximum, which is exactly what `s * multiplier + offset` does when
    `s = code / sample_max`. Used by the tests' reference and by the pixel gate's comparison.
    """
    sample_max = 65535.0 if int(depth) > 8 else 255.0
    params = yuv_uniform_floats(color_space, color_range, depth=depth)
    luma_x, luma_y = params[0], params[1]
    chroma_x, chroma_y = params[4], params[5]
    rows = YUV_MATRICES.get(str(color_space or DEFAULT_COLOR_SPACE), YUV_MATRICES[DEFAULT_COLOR_SPACE])
    yy = (float(y) / sample_max) * luma_x + luma_y
    cb = (float(u) / sample_max) * chroma_x + chroma_y
    cr = (float(v) / sample_max) * chroma_x + chroma_y
    return tuple(  # type: ignore[return-value]
        max(0.0, min(1.0, row[0] * yy + row[1] * cb + row[2] * cr)) * 255.0 for row in rows)


def mip_minify_enabled() -> bool:
    """Mipminification on/off. The chain is the whole fix for the fit view (a 1440x1080 frame in a 421x512 pane is a
    3.42x minification through ONE bilinear tap, which aliases); the switch exists so an A/B runs on ONE build and the
    owner can compare the two arms himself. `RVE_MIP_MINIFY=0` keeps the old single-level texture."""
    import os

    return str(os.environ.get("RVE_MIP_MINIFY", "1")).strip() != "0"


@dataclass(frozen=True)
class PreviewFrame:
    """One picture to ingest: its pixel format, its size and its identity.

    `rgba8` carries the pixels as a `QImage` (already converted by the caller). The planar formats
    (`yuv420p10le`, `yuv420p`) carry the three planes and their byte strides (`Y` full size, `U`/`V`
    half size) plus the stream's colour tags, and the renderer converts colour in the shader.

    `width`/`height` describe the PICTURE. For `rgba8` the texture is sized from the image itself
    (implicit-sharing cheap), so a descriptor that disagrees with its image still uploads what it
    draws.
    """

    format: str                    # "rgba8" | "yuv420p10le" | "yuv420p"
    width: int
    height: int
    pts: float | None = None
    index: int | None = None
    image: QImage | None = None    # rgba8 only
    planes: tuple[memoryview, ...] | None = None   # Y, U, V
    strides: tuple[int, ...] | None = None
    #: The stream's own colour tags, when its producer could read them (`"bt709"`, `"tv"`). `None`
    #: means "not declared": the renderer then uses bt709/tv, which is what this app's writer emits.
    #: Never hard-coded for a file that DOES declare another space (H.273 tags are read, not assumed).
    color_space: str | None = None
    color_range: str | None = None


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
        #: The ingested pictures. `rgba8` uses one texture, the planar formats three.
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
        #: The planar pictures' own textures: three per picture (Y, U, V), R16 for 10-bit and R8 for
        #: 8-bit, created/resized as a set so a mid-playback format change cannot leave a half-bound
        #: SRB (the same rule the RGBA texture learned the expensive way).
        self._plane_textures = None
        self._original_plane_textures = None
        self._sampler_nearest = None
        self._sampler_linear = None
        #: The planes samplers: always single-level (`None_` mip filter), because a mipmap filter over
        #: a texture with no chain sampled opaque black on llvmpipe once - see `_create_resources`.
        self._sampler_yuv_nearest = None
        self._sampler_yuv_linear = None
        self._uniforms = None
        self._original_uniforms = None
        self._yuv_uniforms = None
        self._original_yuv_uniforms = None
        self._vertices = None
        self._bindings = None
        self._bindings_nearest = None
        self._original_bindings = None
        self._original_bindings_nearest = None
        self._yuv_bindings = None
        self._yuv_bindings_nearest = None
        self._original_yuv_bindings = None
        self._original_yuv_bindings_nearest = None
        self._pipeline = None
        self._yuv_pipeline = None
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
        if self._result_dirty:
            # Create the texture OUTSIDE a frame: QRhi resources are created between frames, and a create() issued
            # while a frame is being recorded is refused by the backend without an exception (the frame then samples
            # the 1x1 placeholder - opaque black, which reads exactly like "the upload was skipped").
            if frame.format in YUV_FORMATS:
                if self._yuv_uniforms is not None:
                    self._ensure_plane_textures(frame, original=False)
            elif self._texture is not None:
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
            if frame is not None:
                if frame.format in YUV_FORMATS:
                    if self._yuv_uniforms is not None:
                        self._ensure_plane_textures(frame, original=True)
                elif self._texture is not None:
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
        """Refuse a frame this renderer cannot upload - loudly, not as a silently black pane."""
        if frame is None:
            return
        if frame.format in YUV_FORMATS:
            if not frame.planes or frame.strides is None or len(frame.planes) != 3 or len(frame.strides) != 3:
                raise ValueError(f"PreviewRenderer: a {frame.format!r} PreviewFrame needs three planes and three "
                                 "strides (Y, U, V)")
            if int(frame.width) < 1 or int(frame.height) < 1:
                raise ValueError(f"PreviewRenderer: a {frame.format!r} frame needs a positive size, got "
                                 f"{frame.width}x{frame.height}")
            sample_bytes = SAMPLE_BYTES[frame.format]
            luma, chroma = plane_geometry(frame.width, frame.height)
            for index, (plane, stride, (plane_w, plane_h)) in enumerate(
                    zip(frame.planes, frame.strides, (luma, chroma, chroma))):
                row_bytes = int(plane_w) * sample_bytes
                if int(stride) < row_bytes or len(plane) < int(stride) * int(plane_h):
                    raise ValueError(
                        f"PreviewRenderer: plane {index} of a {frame.format!r} frame carries {len(plane)} bytes at "
                        f"stride {stride}; {plane_w}x{plane_h} needs at least {row_bytes} per row")
            return
        if frame.format != "rgba8":
            raise NotImplementedError(
                f"PreviewRenderer: frame format {frame.format!r} is not implemented (expected 'rgba8', "
                f"{', '.join(repr(name) for name in YUV_FORMATS)})")
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
        for attr in ("_plane_textures", "_original_plane_textures"):
            self._destroy_textures(getattr(self, attr, None))
            setattr(self, attr, None)
        for name in ("_pipeline", "_yuv_pipeline", "_bindings", "_bindings_nearest", "_vertices", "_uniforms",
                     "_sampler_nearest", "_sampler_linear", "_texture", "_original_texture",
                     "_original_bindings", "_original_bindings_nearest", "_original_uniforms",
                     "_yuv_bindings", "_yuv_bindings_nearest", "_original_yuv_bindings",
                     "_original_yuv_bindings_nearest", "_yuv_uniforms", "_original_yuv_uniforms",
                     "_sampler_yuv_nearest", "_sampler_yuv_linear",
                     "_hud_pipeline", "_hud_bindings", "_hud_buffer", "_hud_uniforms"):
            item = getattr(self, name, None)
            if item is not None:
                try:
                    item.destroy()
                except Exception:  # a resource the context already took: teardown must not raise
                    pass
                setattr(self, name, None)

    @staticmethod
    def _destroy_textures(textures) -> None:
        for texture in textures or ():
            try:
                texture.destroy()
            except Exception:  # a resource the context already took
                pass

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
        is_yuv = picture.format in YUV_FORMATS
        if self._result_dirty:
            if is_yuv:
                self._ensure_plane_textures(picture, original=False)
            else:
                self._ensure_texture(self._frame_size(picture))
        pipeline = self._pipeline_for(picture)
        if pipeline is None:
            pipeline = self._build_pipeline_for(picture)
            if pipeline is None:
                return
        updates = self._rhi.nextResourceUpdateBatch()
        if self._result_dirty:
            upload_started = time.perf_counter()
            if is_yuv:
                self._upload_planes(updates, picture, self._plane_textures)
            else:
                updates.uploadTexture(self._texture, picture.image)
                if mip_minify_enabled():
                    # Same batch as the upload, BEFORE the pass: the chain is built from level 0 on the queue, and
                    # `resourceUpdate` is only legal outside a render pass. Cost is one-time per uploaded frame (the chain
                    # of an 1440x1080 RGBA8 is ~2.1 MB - noise against the 6.2 MB level 0 that already crossed the bus).
                    updates.generateMips(self._texture)
            telemetry.note_upload(upload_ms=(time.perf_counter() - upload_started) * 1000.0,
                                  size_bytes=self._pixel_bytes(picture))
            self._result_dirty = False
        if is_yuv:
            updates.updateDynamicBuffer(
                self._yuv_uniforms, 0, YUV_UNIFORM_BYTES,
                self._yuv_uniform_block(picture, scene.result_mvp, scene.tint))
            bindings = self._yuv_bindings_nearest if scene.nearest else self._yuv_bindings
        else:
            updates.updateDynamicBuffer(self._uniforms, 0, 64, _floats(scene.result_mvp))
            updates.updateDynamicBuffer(self._uniforms, 64, 16, _floats(list(scene.tint)))
            bindings = self._bindings_nearest if scene.nearest else self._bindings
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
        command_buffer.setGraphicsPipeline(pipeline)
        command_buffer.setViewport(QRhiViewport(0, 0, target.pixelSize().width(), target.pixelSize().height()))
        # The pipeline declares `UsesScissor` (the compare modes need it), so the scissor state is LIVE here: a frame
        # must state its own scissor before drawing, or it inherits the last one set on the command buffer - an empty
        # one clipped the whole picture away in `single` after a compare mode had run (pixel gate, 2026-11-24).
        command_buffer.setScissor(
            QRhiScissor(0, 0, target.pixelSize().width(), target.pixelSize().height()))
        command_buffer.setShaderResources(bindings)
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
        if frame.format in YUV_FORMATS:
            self._ensure_plane_textures(frame, original=False)
        else:
            self._ensure_texture(self._frame_size(frame))
        original = self._original
        if original is not None and _null_picture(original):
            # The old inline path treated a null source image as "no source half" - it must not reach
            # `_ensure_original_texture` with a 0x0 size, which QRhi refuses.
            original = None
        if original is not None:
            if original.format in YUV_FORMATS:
                self._ensure_plane_textures(original, original=True)
            else:
                self._ensure_original_texture(self._frame_size(original))
        if self._pipeline_for(frame) is None and self._build_pipeline_for(frame) is None:
            return
        if original is not None and self._pipeline_for(original) is None:
            if self._build_pipeline_for(original) is None:
                return
        updates = self._rhi.nextResourceUpdateBatch()
        if self._result_dirty:
            upload_started = time.perf_counter()
            if frame.format in YUV_FORMATS:
                self._upload_planes(updates, frame, self._plane_textures)
            else:
                updates.uploadTexture(self._texture, frame.image)
                if mip_minify_enabled():
                    updates.generateMips(self._texture)
            telemetry.note_upload(upload_ms=(time.perf_counter() - upload_started) * 1000.0,
                                  size_bytes=self._pixel_bytes(frame))
            self._result_dirty = False
        tint = list(scene.tint)
        if frame.format in YUV_FORMATS:
            updates.updateDynamicBuffer(self._yuv_uniforms, 0, YUV_UNIFORM_BYTES,
                                        self._yuv_uniform_block(frame, scene.result_mvp, tint))
        else:
            updates.updateDynamicBuffer(self._uniforms, 0, 64, _floats(scene.result_mvp))
            updates.updateDynamicBuffer(self._uniforms, 64, 16, _floats(tint))
        draw_original = (original is not None
                         and (self._original_texture is not None or self._original_plane_textures is not None)
                         and scene.original_mvp is not None)
        if draw_original:
            if self._original_dirty:
                upload_started = time.perf_counter()
                if original.format in YUV_FORMATS:
                    self._upload_planes(updates, original, self._original_plane_textures)
                else:
                    updates.uploadTexture(self._original_texture, original.image)
                telemetry.note_upload(upload_ms=(time.perf_counter() - upload_started) * 1000.0,
                                      size_bytes=self._pixel_bytes(original))
                self._original_dirty = False
            if original.format in YUV_FORMATS:
                updates.updateDynamicBuffer(self._original_yuv_uniforms, 0, YUV_UNIFORM_BYTES,
                                            self._yuv_uniform_block(original, scene.original_mvp, tint))
            else:
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
        command_buffer.setViewport(QRhiViewport(0, 0, target.pixelSize().width(), target.pixelSize().height()))
        result_bindings = self._bindings_for(frame, original=False, nearest=scene.nearest)
        result_clip = scene.result_clip
        if result_bindings is not None and not result_clip.isEmpty() and scene.result_rect.intersects(result_clip):
            # Each half may carry a different pixel format (a YUV result against an RGB24 source), so the
            # pipeline is stated per draw - the two SRB layouts differ in their texture bindings.
            command_buffer.setGraphicsPipeline(self._pipeline_for(frame))
            command_buffer.setShaderResources(result_bindings)
            command_buffer.setVertexInput(0, [(self._vertices, 0)])
            self._scissor(command_buffer, result_clip, target, scene.dpr)
            command_buffer.draw(6)
        if draw_original:
            original_bindings = self._bindings_for(original, original=True, nearest=scene.nearest)
            original_clip = scene.original_clip
            if original_bindings is not None and not original_clip.isEmpty():
                command_buffer.setGraphicsPipeline(self._pipeline_for(original))
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
        initial_size = (self._frame_size(self._frame)
                        if self._frame is not None and self._frame.format == "rgba8" else QSize(1, 1))
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
        # The planes' samplers are a SEPARATE pair with a `None_` mip filter in both: the plane textures carry no chain
        # (a mip chain over half-size chroma is a blur nobody asked for), and a mipmap filter over a single-level
        # texture is the documented llvmpipe black-sample trap above. Linear for the fit view and the chroma upsample,
        # nearest for the magnified 1:1 view - the same rule as the RGBA pair, one mip filter less.
        self._sampler_yuv_nearest = rhi.newSampler(QRhiSampler.Filter.Nearest, QRhiSampler.Filter.Nearest,
                                                   QRhiSampler.Filter.None_,
                                                   QRhiSampler.AddressMode.ClampToEdge,
                                                   QRhiSampler.AddressMode.ClampToEdge)
        self._sampler_yuv_nearest.create()
        self._sampler_yuv_linear = rhi.newSampler(QRhiSampler.Filter.Linear, QRhiSampler.Filter.Linear,
                                                  QRhiSampler.Filter.None_,
                                                  QRhiSampler.AddressMode.ClampToEdge,
                                                  QRhiSampler.AddressMode.ClampToEdge)
        self._sampler_yuv_linear.create()

        self._uniforms = rhi.newBuffer(QRhiBuffer.Type.Dynamic, QRhiBuffer.UsageFlag.UniformBuffer, 64 + 16)
        self._uniforms.create()
        # The source picture's own uniform buffer: same block, a different mvp (its half), and the same tint. Two
        # buffers instead of one with two offsets keeps each SRB a plain whole-buffer binding.
        self._original_uniforms = rhi.newBuffer(QRhiBuffer.Type.Dynamic, QRhiBuffer.UsageFlag.UniformBuffer,
                                                64 + 16)
        self._original_uniforms.create()
        # The planes' block: mvp + tint + the luma/chroma affine + three matrix rows (`yuv.frag`'s std140 layout).
        # Kept apart from the RGBA block so the RGBA shaders stay byte-for-byte what they were.
        self._yuv_uniforms = rhi.newBuffer(QRhiBuffer.Type.Dynamic, QRhiBuffer.UsageFlag.UniformBuffer,
                                           YUV_UNIFORM_BYTES)
        self._yuv_uniforms.create()
        self._original_yuv_uniforms = rhi.newBuffer(QRhiBuffer.Type.Dynamic, QRhiBuffer.UsageFlag.UniformBuffer,
                                                    YUV_UNIFORM_BYTES)
        self._original_yuv_uniforms.create()

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
        self._yuv_pipeline = None
        self._original_texture = None
        self._plane_textures = None
        self._original_plane_textures = None
        if self._frame is not None and self._frame.format in YUV_FORMATS:
            # The standing picture came in as planes (a context change or a resize while it stood): build its
            # textures now, outside any frame, exactly like the RGBA path's own texture.
            self._ensure_plane_textures(self._frame, original=False)
        if self._original is not None and not _null_picture(self._original):
            # A compare mode was opened before the first frame (or the resources were rebuilt while it stood). The
            # source's texture is created exactly like the frame's, outside a frame, and its own upload follows.
            self._original_dirty = True
            if self._original.format in YUV_FORMATS:
                self._ensure_plane_textures(self._original, original=True)
            else:
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

    def _build_yuv_pipeline(self) -> None:
        """The planes pipeline: three samplers and the 160-byte block, everything else as the RGBA one.

        A separate pipeline rather than a branch in `frame.frag`: an SRB layout with three textures is not the same
        layout as one with a single texture, and QRhi compares layouts, not objects - so the format decides the
        pipeline. Both are `UsesScissor`, both write straight alpha, and the compare modes may draw one of each in the
        same pass (a planes result against an RGB24 source).
        """
        rhi = self._rhi
        layout = QRhiVertexInputLayout()
        layout.setBindings([QRhiVertexInputBinding(4 * 4)])
        layout.setAttributes([
            QRhiVertexInputAttribute(0, 0, QRhiVertexInputAttribute.Format.Float2, 0),
            QRhiVertexInputAttribute(0, 1, QRhiVertexInputAttribute.Format.Float2, 2 * 4),
        ])
        bindings = self._yuv_bindings or self._original_yuv_bindings
        pipeline = rhi.newGraphicsPipeline()
        pipeline.setShaderStages([
            QRhiShaderStage(QRhiShaderStage.Type.Vertex, _shader("yuv.vert.qsb")),
            QRhiShaderStage(QRhiShaderStage.Type.Fragment, _shader("yuv.frag.qsb")),
        ])
        pipeline.setVertexInputLayout(layout)
        pipeline.setShaderResourceBindings(bindings)
        pipeline.setRenderPassDescriptor(self._target.renderPassDescriptor())
        pipeline.setFlags(QRhiGraphicsPipeline.Flag.UsesScissor)
        pipeline.setDepthTest(False)
        pipeline.setDepthWrite(False)
        blend = QRhiGraphicsPipeline.TargetBlend()
        blend.enable = True
        blend.srcColor = QRhiGraphicsPipeline.BlendFactor.SrcAlpha
        blend.dstColor = QRhiGraphicsPipeline.BlendFactor.OneMinusSrcAlpha
        blend.srcAlpha = QRhiGraphicsPipeline.BlendFactor.One
        blend.dstAlpha = QRhiGraphicsPipeline.BlendFactor.OneMinusSrcAlpha
        pipeline.setTargetBlends([blend])
        pipeline.create()
        self._yuv_pipeline = pipeline

    def _pipeline_for(self, frame: PreviewFrame):
        """The pipeline for this picture's format, or None when it is not built yet."""
        return self._yuv_pipeline if frame.format in YUV_FORMATS else self._pipeline

    def _build_pipeline_for(self, frame: PreviewFrame):
        """Build the pipeline this picture needs (and only that one), returning it or None on refusal."""
        if frame.format in YUV_FORMATS:
            if (self._yuv_bindings is None and self._original_yuv_bindings is None) or self._yuv_pipeline is not None:
                return self._yuv_pipeline
            self._build_yuv_pipeline()
            return self._yuv_pipeline
        if self._pipeline is not None:
            return self._pipeline
        self._build_pipeline()
        return self._pipeline

    def _bindings_for(self, frame: PreviewFrame, *, original: bool, nearest: bool):
        """The SRB for one half at one sampler: planes and RGBA have different layouts."""
        if frame.format in YUV_FORMATS:
            if original:
                return self._original_yuv_bindings_nearest if nearest else self._original_yuv_bindings
            return self._yuv_bindings_nearest if nearest else self._yuv_bindings
        if original:
            return self._original_bindings_nearest if nearest else self._original_bindings
        return self._bindings_nearest if nearest else self._bindings

    @staticmethod
    def _yuv_uniform_block(frame: PreviewFrame, mvp, tint) -> bytes:
        """mvp + tint + the matrix affine, 160 bytes, packed once per draw."""
        depth = 10 if frame.format == "yuv420p10le" else 8
        return _floats(list(mvp) + list(tint)) + _floats(
            yuv_uniform_floats(frame.color_space, frame.color_range, depth=depth))

    @staticmethod
    def _upload_planes(updates, frame: PreviewFrame, textures) -> None:
        """One batch entry per plane, raw bytes with their own stride - no QImage in the path.

        `QRhiTextureSubresourceUploadDescription` takes the plane's bytes directly, which is what makes a 10-bit plane
        upload possible at all (`uploadTexture(texture, QImage)` would have to invent a QImage format for a 16-bit
        single-channel picture). Three entries, one `uploadTexture` call each, all in the ONE batch the pass consumes.

        `bytes(plane)` is the batch's own copy - QRhi stores the data until the pass runs - and it is the ONLY copy in
        this path: no numpy array, no QImage, no `convertToFormat`, no scaler pass.
        """
        if textures is None:
            return
        for texture, plane, stride in zip(textures, frame.planes, frame.strides):
            description = QRhiTextureSubresourceUploadDescription(bytes(plane))
            description.setDataStride(int(stride))
            updates.uploadTexture(
                texture, QRhiTextureUploadDescription(QRhiTextureUploadEntry(0, 0, description)))

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

    def _ensure_plane_textures(self, frame: PreviewFrame, *, original: bool) -> None:
        """Create or resize one picture's three plane textures (R16 for 10-bit, R8 for 8-bit).

        A set, not three independent textures: a format change (a 10-bit file replaced by an 8-bit one) must replace
        all three together, and any SRB naming the old set is invalid the moment a texture is destroyed - so the
        bindings and the planes pipeline are dropped here, exactly like the RGBA path does for its one texture.
        """
        if self._rhi is None or self._yuv_uniforms is None:
            return
        attr = "_original_plane_textures" if original else "_plane_textures"
        existing = getattr(self, attr, None)
        fmt = QRhiTexture.Format.R16 if frame.format == "yuv420p10le" else QRhiTexture.Format.R8
        (luma_w, luma_h), (chroma_w, chroma_h) = plane_geometry(frame.width, frame.height)
        sizes = (QSize(luma_w, luma_h), QSize(chroma_w, chroma_h), QSize(chroma_w, chroma_h))
        if (existing is not None and len(existing) == 3
                and all(tex.format() == fmt and tex.pixelSize() == size
                        for tex, size in zip(existing, sizes))):
            return
        self._destroy_textures(existing)
        textures = []
        for size in sizes:
            texture = self._rhi.newTexture(fmt, size, 1)
            if not texture.create():
                raise RuntimeError(f"plane texture {size.width()}x{size.height()} ({fmt}) was refused by QRhi")
            textures.append(texture)
        setattr(self, attr, textures)
        self._build_plane_bindings()
        # The pipeline was compiled against the SRB objects that named the old textures: rebuild it, exactly as
        # `_ensure_texture` does for the RGBA path (a stale-layout bind samples black rather than raising).
        if self._yuv_pipeline is not None:
            self._yuv_pipeline.destroy()
            self._yuv_pipeline = None

    def _build_plane_bindings(self) -> None:
        """(Re)create the planes SRBs: a uniform buffer at 0 and the three plane samplers at 1, 2, 3.

        The binding indices must equal `binding = 1/2/3` in `yuv.frag` for the same reason the RGBA one must equal 1:
        QRhi assigns texture units by SRB binding index, and a mismatch reads an unbound unit - black, which reads
        exactly like "the planes were never uploaded".
        """
        rhi = self._rhi
        if rhi is None:
            return
        flags = (QRhiShaderResourceBinding.StageFlag.VertexStage
                 | QRhiShaderResourceBinding.StageFlag.FragmentStage)
        frag = QRhiShaderResourceBinding.StageFlag.FragmentStage
        specs = (
            ("_yuv_bindings", self._yuv_uniforms, self._plane_textures, self._sampler_yuv_linear),
            ("_yuv_bindings_nearest", self._yuv_uniforms, self._plane_textures, self._sampler_yuv_nearest),
            ("_original_yuv_bindings", self._original_yuv_uniforms, self._original_plane_textures,
             self._sampler_yuv_linear),
            ("_original_yuv_bindings_nearest", self._original_yuv_uniforms, self._original_plane_textures,
             self._sampler_yuv_nearest),
        )
        for attribute, uniforms, textures, sampler in specs:
            current = getattr(self, attribute, None)
            if current is not None:
                current.destroy()
                setattr(self, attribute, None)
            if textures is None or len(textures) != 3 or uniforms is None:
                continue
            bindings = rhi.newShaderResourceBindings()
            bindings.setBindings([
                QRhiShaderResourceBinding.uniformBuffer(0, flags, uniforms),
                QRhiShaderResourceBinding.sampledTexture(1, frag, textures[0], sampler),
                QRhiShaderResourceBinding.sampledTexture(2, frag, textures[1], sampler),
                QRhiShaderResourceBinding.sampledTexture(3, frag, textures[2], sampler),
            ])
            bindings.create()
            setattr(self, attribute, bindings)

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _pixel_bytes(frame: PreviewFrame) -> int:
        """The bytes one uploaded picture costs, for the upload telemetry.

        RGBA8888 is 4 per pixel; a 4:2:0 set is `width * height * 1.5` samples, each 1 byte (8-bit) or 2 (10-bit) -
        which is the whole point of the plane path (3 B/px against 4 for 10-bit, 1.5 against 4 for 8-bit).
        """
        if frame.format in YUV_FORMATS:
            (luma_w, luma_h), (chroma_w, chroma_h) = plane_geometry(frame.width, frame.height)
            samples = luma_w * luma_h + 2 * chroma_w * chroma_h
            return samples * SAMPLE_BYTES[frame.format]
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
