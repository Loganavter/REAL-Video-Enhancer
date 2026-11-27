"""Frame-level temporal reuse: may the previous high-resolution frame be re-emitted?

Two detectors live here:

* :class:`TemporalReuse` - the original one, comparing a frame against the *previous*
  frame on an 8x-downsampled luma (grain averages out, motion does not);
* :class:`ReferenceFrameGate` - TheAnimeScripter-style detection: compare against the
  **last kept** frame with a 32 px block-mean statistic, so a static shot can keep reusing
  one rendered output for arbitrarily long instead of being cut off by drift accumulated
  against the previous frame.

The decision layer of the gate (:func:`block_grid`, :func:`luma_weights`,
:func:`decide_reference`) is plain Python so it can be unit-tested in a cell without
torch; only :meth:`ReferenceFrameGate.blocks` needs tensors.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Sequence, Tuple

try:  # keep the decision helpers importable where torch is not installed
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - exercised only in torch-less test cells
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


def _inference_mode(fn):
    """``torch.inference_mode()`` decorator that survives a torch-less import.

    Used by :class:`TemporalReuse`, whose only stored tensors are its own low-passed
    grids, which are never handed to autograd-enabled code.
    """
    if torch is None:  # pragma: no cover - only in torch-less cells
        return fn
    return torch.inference_mode()(fn)


def _no_grad(fn):
    """``torch.no_grad()`` decorator that survives a torch-less import.

    The gate keeps the reference grid and compares later grids against it; ``no_grad``
    (rather than ``inference_mode``) keeps those tensors usable from ordinary code, the
    same reason ``TiledUpscale`` uses it.
    """
    if torch is None:  # pragma: no cover - only in torch-less cells
        return fn
    return torch.no_grad()(fn)


def _require_torch() -> None:
    if torch is None:  # pragma: no cover - only in torch-less cells
        raise RuntimeError(
            "ReferenceFrameGate.blocks needs torch; block_grid/decide_reference and "
            "ReferenceFrameGate.observe work without it")


class TemporalReuse:
    """Skip the upscaler when the frame is (structurally) the same as the previous one.

    Comparing raw pixels does not work on encoded video: on a clean 10-bit HEVC source
    there are 0% bit-identical consecutive frames, because dither/codec noise differs in
    every frame (grain is even worse when the studio adds it back). So the decision is
    taken on an 8x-downsampled luma - grain and dither average out, real motion does not -
    and the *worst* 512 px tile decides: if no tile moved, the SR result of this frame is
    a copy of the previous one.

    Measured on 40 s of 1440x1080 23.976 fps film (threshold 0.35/255, chain <= 2):
    42% of frames can be reused, while pans sit at ~3.1/255 and are never reused.

    Reuse is only a *skipping* optimisation: it never invents pixels, it re-emits the
    already rendered high-resolution frame, so a wrong decision shows up as a frozen
    frame, not as a blended/ghosted one. `max_chain` bounds how long that can last.
    """

    def __init__(self, threshold: float = 0.35, max_chain: int = 2,
                 downscale: int = 8, tile: int = 512):
        self.threshold = threshold      # in 8-bit luma levels (0..255)
        self.max_chain = max_chain
        self.downscale = downscale
        self.tile = tile
        self._prev_tiles = None
        self._chain = 0
        self._last_frame = None
        self.processed = 0
        self.reused = 0
        # reuses per chain depth, so the depth table can be read off the render's own population
        # instead of a replay that re-decides the gate
        self.depth_counts: Dict[int, int] = {}

    @_inference_mode
    def _tiles(self, tensor: torch.Tensor) -> torch.Tensor:
        # (1, 3, H, W) in [0, 1] -> (1, 1, h, w) tile means in 8-bit units
        luma = tensor.mean(dim=1, keepdim=True)
        small = F.avg_pool2d(luma.float(), self.downscale)
        k = max(1, self.tile // self.downscale)
        return F.avg_pool2d(small, k, ceil_mode=True) * 255.0

    def can_reuse(self) -> bool:
        return self._last_frame is not None

    @_inference_mode
    def should_reuse(self, frame) -> bool:
        """Feeds the low-resolution frame and returns True if the previous HR frame
        may be emitted instead of running the model."""
        self.processed += 1
        tiles = self._tiles(frame.get_frame_tensor())
        reuse = False
        worst_tile = None
        if (self._prev_tiles is not None
                and self._chain < self.max_chain
                and tiles.shape == self._prev_tiles.shape):
            worst_tile = (tiles - self._prev_tiles).abs().max().item()
            reuse = worst_tile <= self.threshold
        self._prev_tiles = tiles
        if reuse:
            self._chain += 1
            self.reused += 1
            self.depth_counts[self._chain] = self.depth_counts.get(self._chain, 0) + 1
            if os.environ.get("RVE_DEBUG_DEPTH"):
                import logging as _lg
                # This branch used to read a `delta` that does not exist in this scope: the debug flag that exists
                # to show WHY a frame was reused raised a NameError instead (pyflakes, 2026-11-27). Log the number the
                # decision was made on.
                _lg.getLogger("RVE").warning("reuse frame=%d depth=%d tile_delta=%.5f threshold=%.5f",
                                             self.processed - 1, self._chain,
                                             -1.0 if worst_tile is None else worst_tile, self.threshold)
        else:
            self._chain = 0
        return reuse

    def remember(self, hr_frame) -> None:
        self._last_frame = hr_frame

    def last_frame(self):
        return self._last_frame

    def stats(self) -> str:
        pct = 100.0 * self.reused / self.processed if self.processed else 0.0
        return (f"temporal reuse: {self.reused}/{self.processed} frames reused ({pct:.1f}%), "
                f"threshold={self.threshold}/255, max chain={self.max_chain}")


# ======================================================================================
# reference-frame gate (compare against the LAST KEPT frame, block-mean statistic)
# ======================================================================================
#
# Calibration is taken from TheAnimeScripter (nevermindNilas) `dedup.py`:
#
#   threshold   = `--dedup_sens` (0-100, default 35) / 7  ->  5.0 luma levels (0-255)
#   block side  = max(2, round(32 * height / 1080))       ->  32 px at 1080p
#   chain cap   = `--smooth_dedup_max_span` (default 4)
#   pixel veto  = 40 luma levels, full-resolution max |delta|, applied to a would-be drop
#
# Their published accuracy at those defaults (PARAMETERS.MD: lossless anime master against
# six re-encodes) is 2.0-2.2% of real changes dropped and <= 1.3% of duplicates missed.
REFERENCE_THRESHOLD = 5.0       # in 8-bit luma levels, = TheAnimeScripter sens 35 / 7
REFERENCE_MAX_CHAIN = 4         # TheAnimeScripter `smooth_dedup_max_span`
REFERENCE_PIXEL_VETO = 40.0     # full-res max |delta| that rescues a would-be drop
REFERENCE_SCENE_CUT = 32.0      # global level jump that drops the reference
REFERENCE_LUMA_TAPS = (1, 2, 1)  # R/G/B weights, normalized to sum 1
REFERENCE_BLOCK_BASE = 32       # block side at 1080p (scales with the frame height)
REFERENCE_BLOCK_REFERENCE_HEIGHT = 1080
# The coarse scale, in LOW-RESOLUTION pixels. Not a preference: a 1-2 px black line moved by half a pixel
# changes by tens of levels at full resolution, and a 32 px block mean is not enough to hide it. Measured on the
# denoised Pink Panther (1440x1080, calm window at t=120): a 0.35 px synthetic shift reads 7.3-8.8 in the shipped
# statistic and 1.2-1.8 after a 3 px low-pass with the translation compensated, on the same frame and the same
# block size (harness/scripts/reuse_structural_probe.py --sigma). The scale is where the drawing is stable against
# the scan's own registration error, which is what the statistic has to see.
COARSE_SIGMA = 3.0
COARSE_REGISTER_DIVISOR = 1     # registration grid: the statistic grid itself (half resolution). Measured:
                                # downscaling again is not faster (FFT at 270x360 vs 540x720 both ~0.33 ms) and loses
                                # accuracy on larger shifts (9.15 -> 4.10 levels against 9.15 -> 2.59)
COARSE_MIN_RESPONSE = 0.2       # below this the correlation has no peak (a flat frame) and the shift is refused
COARSE_MAX_SHIFT_FRAC = 0.06    # a shift beyond this fraction of the frame is a cut or a bad estimate


def scaled_block(height: int) -> int:
    """Block side for a frame ``height``: ``max(2, round(32 * height / 1080))``.

    The block has to cover the same fraction of the picture at every resolution, otherwise
    the number of blocks per frame - and with it the max-over-blocks statistic - changes
    with the source: 32 px at 1080p, 21 at 720p, 64 at 2160p.
    """
    if height <= 0:
        raise ValueError(f"height must be positive, got {height}")
    return max(2, int(round(REFERENCE_BLOCK_BASE * height / REFERENCE_BLOCK_REFERENCE_HEIGHT)))


def block_grid(height: int, width: int, block: int) -> Tuple[int, int]:
    """Shape of the block-mean grid: ceil on both axes (the tail blocks are partial)."""
    if block <= 0:
        raise ValueError(f"block must be positive, got {block}")
    if height <= 0 or width <= 0:
        raise ValueError(f"empty frame {height}x{width}")
    return -(-height // block), -(-width // block)


def luma_weights(taps: Sequence[float]) -> Tuple[float, ...]:
    """Channel weights normalized to sum 1 (``(1, 2, 1)`` -> ``(0.25, 0.5, 0.25)``)."""
    total = float(sum(taps))
    if not taps or total <= 0:
        raise ValueError(f"luma taps must be positive weights, got {tuple(taps)}")
    return tuple(float(t) / total for t in taps)


def decide_reference(delta: float, chain: int, threshold: float, max_chain: int,
                     scene_cut: bool = False) -> bool:
    """Is this frame a duplicate of the reference frame?

    ``delta`` is the worst block mean delta in 8-bit luma levels (``inf`` when there is no
    reference yet), ``chain`` how many consecutive frames have already been served from the
    reference. A frame is a duplicate when it moved by no more than ``threshold``, a scene
    cut did not drop the history, and the chain has not reached ``max_chain``
    (``max_chain == 0`` means unbounded: a static shot may reuse one output indefinitely).

    The full-resolution pixel veto and the state update live in
    :meth:`ReferenceFrameGate.observe`; this function is the cheap decision alone.
    """
    if scene_cut:
        return False
    if max_chain > 0 and chain >= max_chain:
        return False
    return delta <= threshold


class ReferenceFrameGate:
    """Duplicate detection against the last **kept** frame, not the previous one.

    Why not compare with the previous frame: SSIM/MSE over the whole frame average away
    exactly the motion anime cares about (a mouth flap changes a few hundred pixels), and a
    previous-frame comparison accumulates drift - a slow fade or a slightly noisy encode
    makes every frame "different from the previous one" after a while, so a held cel stops
    being reusable. Comparing against the frame whose output was actually rendered keeps a
    static shot reusable for as long as the cel is held.

    Three rules, weakest to strongest (TheAnimeScripter's, see the module-level note):

    1. **block mean** - luma (3-tap weighted), then ``block x block`` block means; the frame
       is a duplicate when the *worst* block mean moved by no more than ``threshold``. Taking
       the mean inside a block is what makes codec dither and grain average out, taking the
       max across blocks is what keeps a small moving object visible.
    2. **chain cap** - at most ``max_chain`` consecutive frames may be copied from one
       reference (``0`` = unbounded), which bounds both a freeze after a wrong decision and
       the drift a fade accumulates.
    3. **pixel veto** - a would-be duplicate whose full-resolution max |delta| exceeds
       ``pixel_veto`` is a miss: the block mean dilutes a mouth flap or a blink to a couple
       of levels, the per-pixel max does not. ``None`` disables the veto (and the reference
       luma is then not stored at all).

    Usage per frame::

        if gate.is_duplicate(lr_frame):
            emit(previous_hr_frame)          # nothing was rendered, reference unchanged
        else:
            emit(model(lr_frame))            # new frame becomes the reference

    ``max_chain = 0`` is unbounded reuse: a wrong decision can freeze the picture until the
    content moves. The default ``max_chain = 4`` matches TheAnimeScripter.
    """

    def __init__(self, block: Optional[int] = None, threshold: float = REFERENCE_THRESHOLD,
                 statistic: str = "mean_abs", share_level: float = 8.0,
                 max_block_mean: Optional[float] = None,
                 max_chain: int = REFERENCE_MAX_CHAIN,
                 luma_taps: Sequence[float] = REFERENCE_LUMA_TAPS,
                 scene_threshold: float = REFERENCE_SCENE_CUT,
                 pixel_veto: Optional[float] = REFERENCE_PIXEL_VETO,
                 coarse_sigma: float = COARSE_SIGMA) -> None:
        if block is not None and block <= 0:
            raise ValueError(f"block must be positive (None = scale with the frame), "
                             f"got {block}")
        if max_chain < 0:
            raise ValueError(f"max_chain must be >= 0 (0 = unbounded), got {max_chain}")
        if scene_threshold < 0:
            raise ValueError(f"scene_threshold must be >= 0, got {scene_threshold}")
        if pixel_veto is not None and pixel_veto < 0:
            raise ValueError(f"pixel_veto must be >= 0 or None, got {pixel_veto}")
        self.block = None if block is None else int(block)   # None = scaled per frame
        self.threshold = float(threshold)          # in 8-bit luma levels (0..255)
        self.max_chain = int(max_chain)
        self.luma_taps = tuple(float(t) for t in luma_taps)
        self.scene_threshold = float(scene_threshold)
        self.pixel_veto = None if pixel_veto is None else float(pixel_veto)
        self._weights: Dict[object, "torch.Tensor"] = {}
        self._ref = None                # block grid of the last KEPT frame, 8-bit levels
        self._ref_mean = 0.0            # its global level (mean of the block-mean grid)
        # "block_mean": max |box(I) - box(ref)| - invariant to a permutation of pixels inside a
        #   block, so a fist changing pose inside one 32 px block barely moves it.
        # "mean_abs": max box(|I - ref|) - the box filter applied to the absolute difference,
        #   which does see that change. Measured on the moving-fist segment at the same reuse
        #   share (70.2%): 0.004% of HR pixels above 8 levels for block_mean, 0.000% for
        #   mean_abs, and mean_abs stays at 0.004% up to threshold 8 where block_mean reaches
        #   0.460%.
        if statistic not in ("block_mean", "mean_abs", "channel_max_share", "coarse"):
            raise ValueError(f"statistic must be 'block_mean', 'mean_abs', 'channel_max_share' or "
                             f"'coarse', got {statistic!r}")
        self.statistic = statistic
        # The coarse statistic's scale, in LR pixels. Named here rather than only in the module constants so a
        # caller can measure another scale without editing the file.
        self.coarse_sigma = float(coarse_sigma)
        if self.coarse_sigma <= 0:
            raise ValueError(f"coarse_sigma must be > 0, got {coarse_sigma}")
        # The acceptance bar counts pixels whose worst channel is more than 8 levels off, so the
        # channel-max statistic is the same quantity the bar measures. `share_level` is that 8; the
        # threshold, in this mode, is a PERCENT of pixels (0.07 means 0.07%), not a level.
        self.share_level = float(share_level)
        # The share threshold accepts a small area with a large delta. This second condition bounds the
        # per-BLOCK MEAN of the channel-max field - it is `max of block means`, not `max within block`,
        # so it dilutes a few bright pixels exactly like the old luma statistic did (measured: no
        # firing at 4 or 8 levels on the calm cut). None = off.
        self.max_block_mean = None if max_block_mean is None else float(max_block_mean)
        self._ref_rgb = None
        self._ref_luma = None           # its full-res luma (needed by mean_abs and by the veto)
        self._ref_blur = None           # the coarse statistic's reference: half-res luma, low-passed
        self._chain = 0                 # consecutive frames served from the reference
        # Geometry of the last observed frame against the current reference, in FULL-RESOLUTION LR pixels,
        # and the increment against the previous frame's shift on that same reference. The increment is what a
        # reuse has to move the cached HR frame by: the gate's reference can be four frames old, so the shift
        # from the reference is not the shift from the frame the cached output came from.
        self.last_shift: Tuple[float, float] = (0.0, 0.0)
        self.last_shift_step: Tuple[float, float] = (0.0, 0.0)
        self._shift_prev: Tuple[float, float] = (0.0, 0.0)
        self.registration_refusals = 0
        # Whether the frame just observed dropped the reference as a scene cut. The gate resets
        # its own history there, and the auto tuner has to reset its HR history at the same
        # frames: otherwise its "two frames back" delta compares across the cut, which is
        # something the gate never does. Measured on Chainsaw Man: calm frames read 0.2 at depth
        # 1 and 16-35 at depths 2-4 for exactly that reason, and the chain was pinned to 1.
        self.last_scene_cut = False
        self.processed = 0
        self.reused = 0
        # reuses per chain depth, so the depth table can be read off the render's own population
        # instead of a replay that re-decides the gate
        self.depth_counts: Dict[int, int] = {}
        self.scene_cuts = 0
        self.vetoes = 0
        self.resets = 0

    # -- history ----------------------------------------------------------------------
    def reset(self) -> None:
        """Forget the reference (new video, resolution/model change, manual seek).

        Cumulative counters are kept; only the reuse state is dropped, so the next frame is
        guaranteed to be a miss.
        """
        self._ref = None
        self._ref_mean = 0.0
        self._ref_luma = None
        self._ref_rgb = None
        self._ref_blur = None
        self._chain = 0
        self.last_shift = (0.0, 0.0)
        self.last_shift_step = (0.0, 0.0)
        self._shift_prev = (0.0, 0.0)
        self.resets += 1

    @property
    def can_reuse(self) -> bool:
        return self._ref is not None

    @property
    def chain(self) -> int:
        return self._chain

    @property
    def reference_mean(self) -> float:
        """Global level of the reference frame (0..255); 0.0 while there is none."""
        return self._ref_mean

    def block_for(self, height: int) -> int:
        """Block side this gate uses for a frame of ``height`` pixels."""
        return scaled_block(height) if self.block is None else self.block

    def progress(self) -> str:
        """One line for a long render, so the reuse share is visible while it runs.

        The full :meth:`stats` is printed at the end, and a cancelled render prints nothing - a user watching a
        25-minute encode had no way to tell whether the gate was doing anything, which reads exactly like "it does
        not work".
        """
        pct = 100.0 * self.reused / self.processed if self.processed else 0.0
        deepest = max(self.depth_counts) if self.depth_counts else 0
        return (f"{pct:.1f}% of {self.processed} frames ({self.reused} reused), depth up to {deepest}, "
                f"{self.scene_cuts} scene cuts, threshold {self.threshold:g}/255")

    def stats(self) -> str:
        pct = 100.0 * self.reused / self.processed if self.processed else 0.0
        scale = (f" on a {self.coarse_sigma:g} px low-pass with the translation compensated"
                 if self.statistic == "coarse" else "")
        return (f"reference reuse: {self.reused}/{self.processed} frames reused ({pct:.1f}%), "
                f"{self.scene_cuts} scene cuts, {self.vetoes} pixel vetoes, "
                f"threshold={self.threshold}/255 over "
                f"{self.block or 'scaled'}px blocks ({self.statistic}{scale}), "
                f"max chain={self.max_chain or 'unbounded'}, "
                f"pixel veto={self.pixel_veto if self.pixel_veto is not None else 'off'}; "
                f"depth histogram {dict(sorted(self.depth_counts.items()))}")

    # -- detector ---------------------------------------------------------------------
    def _channel_weights(self, channels: int, device, dtype):
        key = (channels, str(device))
        weight = self._weights.get(key)
        if weight is None or weight.dtype != dtype:
            weight = torch.tensor(luma_weights(self.luma_taps[:channels]), dtype=dtype,
                                  device=device).view(1, channels, 1, 1)
            self._weights[key] = weight
        return weight

    @_no_grad
    def luma(self, frame_tensor) -> "torch.Tensor":
        """``(1, C, H, W)`` in [0, 1] -> ``(1, 1, H, W)`` fp32 luma, on the same device.

        Upcast to fp32 on the way in: a 32x32 fp16 average of near-1.0 values drifts by more
        than the 5-level threshold, and the pixel veto compares single pixels where fp16
        resolution is ~1 level at 0.5.
        """
        _require_torch()
        if frame_tensor.dim() != 4:
            raise ValueError(
                f"expected a (1, C, H, W) tensor, got shape {tuple(frame_tensor.shape)}")
        if int(frame_tensor.shape[0]) != 1:
            raise ValueError(f"expected batch size 1, got {int(frame_tensor.shape[0])}")
        channels = int(frame_tensor.shape[1])
        if channels == 1:
            return frame_tensor.float()
        if channels != len(self.luma_taps):
            raise ValueError(f"{channels} channels but {len(self.luma_taps)} luma taps")
        weight = self._channel_weights(channels, frame_tensor.device, torch.float32)
        return (frame_tensor.float() * weight).sum(dim=1, keepdim=True)

    def blocks_of(self, luma) -> "torch.Tensor":
        """``(1, 1, H, W)`` luma in [0, 1] -> block means in 8-bit levels.

        ``ceil_mode=True`` makes a tail block average only the pixels that exist, so odd
        frame sizes need no padding, and it matches the numpy reference in the tests.
        """
        _require_torch()
        block = self.block_for(int(luma.shape[-2]))
        return F.avg_pool2d(luma, block, ceil_mode=True) * 255.0

    def blocks(self, frame_tensor) -> "torch.Tensor":
        """``(1, C, H, W)`` in [0, 1] -> ``(1, 1, ceil(H/b), ceil(W/b))`` in 8-bit levels."""
        return self.blocks_of(self.luma(frame_tensor))

    # -- coarse statistic (compare the drawing, not the scan's registration) --------------
    def _gauss_kernel(self, sigma: float, device, dtype):
        """1-D Gaussian kernel of ``sigma`` pixels, cached per (sigma, device, dtype)."""
        key = ("gauss", round(sigma, 4), str(device), str(dtype))
        kernel = self._weights.get(key)
        if kernel is None:
            radius = max(1, int(3.0 * sigma + 0.5))
            x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
            k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
            kernel = (k / k.sum()).to(dtype).view(1, 1, 1, -1)
            self._weights[key] = kernel
        return kernel

    @_no_grad
    def _lowpass(self, x, sigma: float) -> "torch.Tensor":
        """Separable Gaussian on ``(1, 1, h, w)``, replicate borders (the frame edge is not a feature)."""
        _require_torch()
        k = self._gauss_kernel(sigma, x.device, x.dtype)
        radius = (k.shape[-1] - 1) // 2
        xp = F.pad(x, (radius, radius, 0, 0), mode="replicate")
        xp = F.conv2d(xp, k)
        xp = F.pad(xp, (0, 0, radius, radius), mode="replicate")
        return F.conv2d(xp, k.view(1, 1, -1, 1))

    def _hann(self, h: int, w: int, device, dtype) -> "torch.Tensor":
        """Separable Hann window for the correlation, cached per shape (a border step is not registration)."""
        key = ("hann", h, w, str(device), str(dtype))
        window = self._weights.get(key)
        if window is None:
            wy = torch.hann_window(h, periodic=False, dtype=torch.float32, device=device).view(h, 1)
            wx = torch.hann_window(w, periodic=False, dtype=torch.float32, device=device).view(1, w)
            window = (wy * wx).to(dtype).view(1, 1, h, w)
            self._weights[key] = window
        return window

    def _base_grid(self, h: int, w: int, device, dtype) -> "torch.Tensor":
        key = ("grid", h, w, str(device), str(dtype))
        grid = self._weights.get(key)
        if grid is None:
            ys = torch.arange(h, dtype=torch.float32, device=device)
            xs = torch.arange(w, dtype=torch.float32, device=device)
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            grid = torch.stack(((gx / max(1, w - 1)) * 2.0 - 1.0,
                                (gy / max(1, h - 1)) * 2.0 - 1.0), dim=-1).unsqueeze(0)
            grid = grid.to(dtype)
            self._weights[key] = grid
        return grid

    @_no_grad
    def _shift_tensor(self, x, dx: float, dy: float) -> "torch.Tensor":
        """``(1, C, h, w)`` -> the same picture with its content moved by ``(dx, dy)`` pixels.

        A single bilinear resample with replicated borders: no wraparound, so a frame edge cannot leak onto the
        opposite side and hand the max-over-blocks statistic a spike that is not in the picture.
        """
        _require_torch()
        if abs(dx) < 1e-4 and abs(dy) < 1e-4:
            return x
        h, w = int(x.shape[-2]), int(x.shape[-1])
        grid = self._base_grid(h, w, x.device, x.dtype).clone()
        # grid_sample reads INPUT at the grid's coordinates, so an offset of +d moves the CONTENT by -d.
        # Measured: without this sign the reference was warped away from the frame it was being aligned to.
        grid[..., 0] -= 2.0 * dx / max(1, w - 1)
        grid[..., 1] -= 2.0 * dy / max(1, h - 1)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)

    @_no_grad
    def _register(self, ref_small, cur_small) -> Tuple[float, float, float, bool]:
        """Translation putting ``cur`` on top of ``ref``: ``(dx, dy, response, trusted)``.

        Phase correlation on the low-passed luma, at 1/``COARSE_REGISTER_DIVISOR`` of the statistic grid. The
        shift is returned in FULL-RESOLUTION luma pixels, and the sub-pixel part comes from a parabolic fit on
        the correlation peak - the estimator is biased low on broad peaks (measured 0.37 px read as 0.21), which
        is why the caller never trusts it alone: :meth:`coarse_decision` keeps whichever comparison is cheaper,
        warped or not, so a bad estimate degrades to the un-warped number instead of inventing damage.

        Two refusals, both from a measured failure: a uniform frame has no peak and the estimator answers with
        half the picture (120 px on a flat field, response 0.02), and a shift beyond
        ``COARSE_MAX_SHIFT_FRAC`` of the frame is a cut, not registration.
        """
        _require_torch()
        ref = F.avg_pool2d(ref_small, COARSE_REGISTER_DIVISOR)
        cur = F.avg_pool2d(cur_small, COARSE_REGISTER_DIVISOR)
        hh, ww = int(ref.shape[-2]), int(ref.shape[-1])
        if hh < 4 or ww < 4:
            return 0.0, 0.0, 0.0, False
        window = self._hann(hh, ww, ref.device, ref.dtype)
        spectrum = torch.fft.rfft2(ref * window) * torch.conj(torch.fft.rfft2(cur * window))
        spectrum = spectrum / (spectrum.abs() + 1e-8)
        corr = torch.fft.irfft2(spectrum, s=(hh, ww))
        flat = int(corr.flatten().argmax())
        iy, ix = divmod(flat, ww)
        # 3x3 neighbourhood on the wrapped surface, then a parabola through each axis
        rows = torch.tensor([(iy - 1) % hh, iy, (iy + 1) % hh], device=corr.device)
        cols = torch.tensor([(ix - 1) % ww, ix, (ix + 1) % ww], device=corr.device)
        patch = corr[0, 0][rows][:, cols].float()
        centre, up, down = float(patch[1, 1]), float(patch[2, 1]), float(patch[0, 1])
        left, right = float(patch[1, 0]), float(patch[1, 2])
        den_y = up - 2.0 * centre + down
        den_x = right - 2.0 * centre + left
        # Parabola through (left, centre, right): peak at t = 0.5*(c[t-1] - c[t+1]) / (c[t+1] - 2c[t] + c[t-1]).
        # Written the other way round it returns -t, which measured as a sub-pixel shift read back with the
        # wrong sign (-0.20 for a +0.40 px move) while larger moves still read roughly right.
        frac_y = 0.0 if abs(den_y) < 1e-12 else 0.5 * (down - up) / den_y
        frac_x = 0.0 if abs(den_x) < 1e-12 else 0.5 * (left - right) / den_x
        frac_y = max(-0.5, min(0.5, frac_y))
        frac_x = max(-0.5, min(0.5, frac_x))
        sy, sx = iy + frac_y, ix + frac_x
        if sx > ww / 2.0:
            sx -= ww
        if sy > hh / 2.0:
            sy -= hh
        # grid units -> statistic-grid pixels -> full-resolution luma pixels
        scale = float(COARSE_REGISTER_DIVISOR * 2)
        dx, dy = -sx * scale, -sy * scale
        response = centre
        limit = COARSE_MAX_SHIFT_FRAC * max(ref_small.shape[-2], ref_small.shape[-1])
        trusted = (response >= COARSE_MIN_RESPONSE and abs(dx) <= limit and abs(dy) <= limit)
        if not trusted:
            return 0.0, 0.0, response, False
        return dx, dy, response, True

    def _blocks_half(self, grid) -> "torch.Tensor":
        """Block means of the half-resolution grid, with the block covering the same FRACTION of the picture.

        ``block_for(H)`` is defined on the full frame (32 px at 1080p); the half grid needs half of it, so the
        number of blocks per frame - and with it the max-over-blocks statistic - is unchanged by the scale.
        """
        _require_torch()
        block = max(1, self.block_for(int(grid.shape[-2]) * 2) // 2)
        return F.avg_pool2d(grid, block, ceil_mode=True) * 255.0

    @_no_grad
    def coarse_decision(self, luma) -> Tuple[Optional["torch.Tensor"], "torch.Tensor", "torch.Tensor"]:
        """The coarse error field against the reference: ``(decision, block grid, blurred grid)``.

        ``decision`` is ``None`` while there is no reference (the first frame after a reset), so the caller's
        state machine treats it exactly like the first frame of a ``mean_abs`` run.

        The decision keeps the MINIMUM of two comparisons - with the measured translation compensated and
        without it. The registration is a biased estimator at this scale, and an estimate that made the
        statistic bigger would refuse frames the un-warped comparison accepts; taking the minimum means the
        estimate can only ever help, and ``registration_refusals`` counts the frames where it was not used.
        """
        _require_torch()
        if luma.dim() != 4:
            raise ValueError(f"expected a (1, 1, H, W) luma, got {tuple(luma.shape)}")
        half = F.avg_pool2d(luma, 2)
        blurred = self._lowpass(half, self.coarse_sigma / 2.0)
        levels = self._blocks_half(blurred)
        self.last_shift = (0.0, 0.0)
        decision = None
        if self._ref_blur is not None and tuple(self._ref_blur.shape) == tuple(blurred.shape):
            dx, dy, _response, trusted = self._register(self._ref_blur, blurred)
            if trusted:
                self.last_shift = (dx, dy)
                aligned = self._shift_tensor(self._ref_blur, dx / 2.0, dy / 2.0)
                warped = self._blocks_half((aligned - blurred).abs())
                plain = self._blocks_half((self._ref_blur - blurred).abs())
                decision = warped if float(warped.max()) <= float(plain.max()) else plain
            else:
                self.registration_refusals += 1
                decision = self._blocks_half((self._ref_blur - blurred).abs())
        return decision, levels, blurred

    def observe(self, blocks, luma=None, decision=None, share=None, rgb=None, blur=None) -> bool:
        """The decision and the state update, given a block grid (torch or numpy).

        Kept separate from :meth:`blocks` so the state machine - reference keeping, chain
        limit, scene cut, pixel veto - can be tested with numpy grids where torch is not
        installed. Nothing but the chain counter changes on a duplicate; on a miss the
        current grid (and luma, when a veto is configured) becomes the reference and the
        chain restarts.

        The global level used for the scene cut is the mean of the block-mean grid: exact
        when the block size divides both frame axes, off by a fraction of a level on the
        partial tail blocks, which is irrelevant against a 32 level cut threshold.

        ``decision`` is the grid the threshold acts on. For ``statistic="mean_abs"`` it is
        already ``box(|luma - ref_luma|)``, i.e. a non-negative error field measured against
        the reference, so the delta is simply its maximum and no subtraction happens here.
        """
        self.processed += 1
        self.last_delta = float("inf")
        mean = float(blocks.mean().item())
        scene_cut = False
        delta = float("inf")
        if share is not None:
            # already a percentage of pixels above the acceptance level, measured against the
            # reference - the threshold is in the same unit
            delta = float(share)
            self.last_delta = delta
        elif decision is not None:
            # the error field is already relative to the reference
            delta = float(decision.max().item())
            self.last_delta = delta
        elif self._ref is not None:
            if tuple(blocks.shape) == tuple(self._ref.shape):
                delta = float(abs(blocks - self._ref).max().item())
                self.last_delta = delta
            else:                       # resolution change: history is meaningless
                scene_cut = True
        # A global brightness jump is a cut even when the block delta is small (a cut between two
        # dim scenes). This used to live inside the `elif` above, i.e. it only ran when the caller
        # supplied no decision grid - which stopped being true when the statistic became
        # `mean_abs`, so since then it has never executed.
        if self._ref is not None and abs(mean - self._ref_mean) > self.scene_threshold:
            scene_cut = True
        reuse = bool(decide_reference(delta, self._chain, self.threshold, self.max_chain,
                                      scene_cut))
        if reuse and self.pixel_veto is not None and luma is not None:
            if self._ref_luma is not None and tuple(luma.shape) == tuple(self._ref_luma.shape):
                # luma is in [0, 1]; the veto threshold is in 8-bit levels
                worst_pixel = float(abs(luma - self._ref_luma).max().item()) * 255.0
                if worst_pixel > self.pixel_veto:
                    reuse = False
                    self.vetoes += 1
        self.last_scene_cut = bool(scene_cut)
        if reuse:
            self._chain += 1
            self.reused += 1
            self.depth_counts[self._chain] = self.depth_counts.get(self._chain, 0) + 1
            # what a reuse has to move: the increment against the previous frame's shift on the SAME reference
            step = (self.last_shift[0] - self._shift_prev[0], self.last_shift[1] - self._shift_prev[1])
            self._shift_prev = self.last_shift
            if os.environ.get("RVE_DEBUG_DEPTH"):
                import logging as _lg
                _lg.getLogger("RVE").warning("reuse frame=%d depth=%d share=%.5f",
                                             self.processed - 1, self._chain, delta)
        else:
            if scene_cut:
                self.scene_cuts += 1
            self.last_scene_cut = bool(scene_cut)
            step = (0.0, 0.0)
            self._shift_prev = (0.0, 0.0)
        self.last_shift_step = step
        if not reuse:
            self._ref = blocks
            self._ref_mean = mean
            self._ref_luma = luma if (self.pixel_veto is not None
                                      or self.statistic == "mean_abs") else None
            if self.statistic == "coarse":
                self._ref_blur = blur
            if self.statistic == "channel_max_share":
                self._ref_rgb = rgb.detach().clone() if rgb is not None else None
            self._chain = 0
        return reuse

    def is_duplicate(self, frame_tensor) -> bool:
        """Feed the low-resolution frame: True means the reference's output may be emitted."""
        luma = self.luma(frame_tensor)
        levels = self.blocks_of(luma)
        decision = None
        share = None
        blur = None
        ref = self._ref_luma
        if self.statistic == "coarse":
            decision, levels, blur = self.coarse_decision(luma)
        elif self.statistic == "channel_max_share":
            ref_rgb = self._ref_rgb
            if ref_rgb is not None and tuple(ref_rgb.shape) == tuple(frame_tensor.shape):
                # |I - ref| per channel, then the worst channel per pixel: the quantity the bar
                # counts, on the LOW-resolution frame the gate actually sees
                cm = (frame_tensor.float() - ref_rgb).abs().amax(dim=1, keepdim=True)
                share = float((cm > self.share_level / 255.0).float().mean()) * 100.0
                if self.max_block_mean is not None:
                    # worst 32 px block mean of the same field, in 8-bit levels
                    pooled = torch.nn.functional.avg_pool2d(cm, 32, ceil_mode=True) * 255.0
                    if float(pooled.max()) > self.max_block_mean:
                        share = float("inf")     # a miss, whatever the area says

        elif (self.statistic == "mean_abs" and ref is not None
                and tuple(ref.shape) == tuple(luma.shape)):
            # box(|I - ref|): one subtraction, one abs, one pool over the frame
            decision = self.blocks_of((luma - ref).abs())
        return self.observe(levels, luma, decision, share=share,
                            rgb=frame_tensor if self.statistic == "channel_max_share" else None,
                            blur=blur)

    def delta(self, frame_tensor) -> float:
        """Worst block-mean delta of ``frame_tensor`` against the current reference.

        Diagnostics only - the decision is :meth:`is_duplicate`; this lets a caller log
        *how* close a miss was without duplicating the pooling. ``inf`` when there is no
        reference or the frame shape changed.
        """
        blocks = self.blocks(frame_tensor)
        if self._ref is None or tuple(blocks.shape) != tuple(self._ref.shape):
            return float("inf")
        return float((blocks - self._ref).abs().max().item())


def worst_chain_delta(grid, history, depth: int) -> float:
    """Worst block delta between ``grid`` and the last ``depth`` entries of ``history``.

    The gate serves a frame from a reference up to ``max_chain`` frames old, so a single-step
    delta understates the error a reuse can cause. Plain arithmetic on whatever the caller
    passes (torch tensor or numpy array), so the rule can be tested without tensors.
    """
    if not history or depth < 1:
        return float("inf")
    worst = 0.0
    for past in history[-depth:]:
        if tuple(getattr(past, "shape", ())) != tuple(getattr(grid, "shape", ())):
            continue
        worst = max(worst, float(abs(grid - past).max()))
    return worst


class AutoGateTuner:
    """Choosing the duplicate threshold from measurement instead of a constant.

    Every frame that has to be rendered anyway yields both sides of the trade: ``lr_delta``
    (the gate's own statistic, worst 32 px block mean) and ``hr_delta`` (how much the upscaled
    result actually changed, same statistic shape at the output). Frames whose ``lr_delta`` is
    under a candidate threshold *would* have been reused, and the error that reuse would have
    cost is exactly their measured ``hr_delta``.

    So after a handful of rendered frames we can score every candidate on real data:

      - reuse share = fraction of calibration frames with ``lr_delta <= candidate``
      - cost       = p90 of their ``hr_delta`` (levels of 255)

    and pick the largest candidate whose p90 stays inside ``budget``. That is content-agnostic:
    grainy film has a higher noise floor and lands on a larger threshold, clean anime on a
    smaller one, without anyone typing a number.

    The whole thing costs no extra inference (the frames are rendered regardless) and a few
    hundred microseconds per calibration frame for the HR statistic.
    """

    def __init__(self, budget: float = 2.0, min_pairs: int = 16, seed: float = 5.0,
                 candidates=(2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0),
                 clamp=(2.0, 4.0), noise_floor_quiet: float = 0.35,
                 fallback: str = "strictest", chain_budget: float = 12.0,
                 # None disables the guard. It exists because on grainy material a 32 px block mean cannot
                 # separate "the grain changed" from "something moved one pixel", and lowering the threshold
                 # only costs reuse. The coarse statistic removes grain by construction, so the guard's premise
                 # is gone there and it would refuse a floor that is a low-pass level, not a noise level
                 # (measured on the denoised Pink Panther: a floor of ~3 against the 1.5 limit). The picture is
                 # still bounded by the acceptance certifier, at the same scale as the decision.
                 noise_floor_refuse: float | None = 1.5,
                 chain_policy: str = "certified",
                 accept_bar: float = 0.1,
                 calibration_window: int = 0,
                 certify_on: str = "proxy",
                 min_informative: float | None = None,
                 patience: int = 0):
        self.budget = float(budget)
        # The clamp is load-bearing *once the tile path is out of the way*: with tile reuse
        # enabled, thresholds 12 and 4 measured the same (0.288% against 0.287% of pixels above
        # 8 levels in RGB) because the tile artifact dominated both; with tile reuse off the same
        # pair reads 0.517% against 0.045%, a tenfold difference in the same direction as the
        # owner's complaint. So the gate's threshold stays inside a range where a moving object
        # cannot hide inside a 32 px block mean.
        if clamp == (2.0, 12.0):
            clamp = (2.0, 4.0)
        # Two different units live next to each other and both were misread for a day:
        #  * `budget` is the one-step HR proxy, a max over the pooled HR grid, and 2 levels of it
        #    corresponds to about 2 levels of *actual* worst-block error (measured);
        #  * `chain_budget` is the same proxy at depth >= 2, where it grows with content motion:
        #    on the anime clip depth 4 reads 9.3 while the realised error of those same reuses is
        #    0.00% of pixels above 8 levels and a worst block of 2.0 levels.
        # So a strict one-step bar and a loose chain bar are not inconsistent; the chain number is
        # a motion measure, not an error measure.
        self.chain_budget = float(chain_budget)
        # "certified": the chain may only be as deep as the measurement proves safe.
        # "max": always use the full chain (the speed-first preset). Measured on the anime clip,
        # the difference is 0.39% of pixels above 4 levels at depth 1 against 1.13% with peaks
        # of 100 levels at depth 4, for 11.5 s against 7.65 s.
        if chain_policy not in ("certified", "max"):
            raise ValueError(f"chain_policy must be 'certified' or 'max', got {chain_policy!r}")
        self.chain_policy = chain_policy
        self.min_pairs = int(min_pairs)
        # the value the pipeline was configured with: kept when no candidate qualifies, which
        # happens on an unrepresentative opening (a busy first scene measured a "noise floor"
        # of 62.9 levels and drove the choice to the bottom of the range)
        self.seed = float(seed)
        self.candidates = tuple(float(c) for c in candidates)
        self.clamp = (float(clamp[0]), float(clamp[1]))
        self.noise_floor_quiet = float(noise_floor_quiet)
        # Chosen to sit between the last passing and the first failing measurement, NOT as a
        # calibrated boundary - the mapping is flat across the band the limit sits in, and two
        # measured cases a tenth of a level apart straddle it. Four points, floor against
        # worst-frame defect:
        #   0.95 -> 0.005%  (digital with added grain, Chainsaw Man)
        #   1.32 -> 0.097%  (clean digital, Violet Evergarden - inside, with 3% of margin)
        #   1.84 -> 0.105%  (clean anime clip + synthetic grain 4)
        #   2.27 -> 0.470%  (35 mm TV, Cowboy Bebop - the same at threshold 4 and at 2)
        # so the bar is crossed between 1.32 and 1.84 and the limit sits between them, not at the
        # 2.0 that shipped first (which let the 1.84 clip through at 0.105%). That separation is
        # 0.008% of defect wide, i.e. the two cases straddle the limit rather than being decided by
        # it, and a 48-frame Violet Evergarden run (floor 1.32) passed this guard AND the acceptance
        # sample and still landed at 0.121% on the clip - the two guards share a failure mode.
        # Known boundary, not a proof. Above it the gate refuses instead of tuning: on grainy
        # material a 32 px block mean cannot tell "the grain changed" from "something moved one
        # pixel", and lowering the threshold only costs reuse.
        self.noise_floor_refuse = None if noise_floor_refuse is None else float(noise_floor_refuse)
        # The number acceptance is actually judged by: per cent of pixels more than 8 levels off in
        # any RGB channel, on the worst frame of the sample. When the caller supplies it (the app
        # does), the depth and the threshold are certified against THIS, not against a block-mean
        # proxy - the proxy's 2-level budget and this bar were two different instruments, and every
        # false "price of quality" in the ledger came from certifying one and measuring the other.
        self.accept_bar = float(accept_bar)
        # How many rendered frames to observe before choosing, and how many of them to keep. Keeping
        # the first 16 in arrival order made the acceptance statistic read 0.00 on three of four
        # titles - the worst frame of a clip is not in the first sixteen - which left the depth
        # certified by the reachability filter and the window size rather than by the picture. Keeping
        # the `min_pairs` frames with the LARGEST LR delta instead is the adversarial choice: those are
        # the marginal reuses, which is what a worst-frame rule is about. The window is the price: the
        # first 64 frames render unconditionally (~3 s at 4K once per video).
        # 0 means "the same frames the proxy always used": the window experiment is recorded in the
        # ledger (E25) and did not survive its own verification.
        self.calibration_window = max(int(calibration_window) or int(min_pairs), int(min_pairs))
        # Which frames may DECIDE the threshold. A frame whose own delta is below the narrowest candidate is
        # reused by every candidate, so it cannot tell them apart - but it dominates the sample: a film that opens
        # on black fed this tuner 16 frames at delta 0.01 and the threshold for a 25-minute episode was chosen
        # there (measured: noise floor 0.01, would reuse 100% of the calibration frames, and the run then reused
        # 42.6% while the same material is 33-57% at thresholds 4-8). None keeps the old behaviour, which is what
        # the shipped `mean_abs` path still uses: its noise-floor guard was calibrated on the unfiltered sample.
        self.min_informative = None if min_informative is None else float(min_informative)
        # How many frames to observe before giving up and deciding on what arrived: bounded patience, because a
        # clip with no informative frame at all (a still, a black test) must still settle - an unsettled tuner
        # disables reuse for the whole render, which is worse than a threshold chosen from little evidence.
        self.patience = max(int(patience) or 4 * self.calibration_window, self.calibration_window)
        # Which instrument certifies the DEPTH. "proxy" is the block-mean p90; "acceptance" is the
        # worst per-pixel share. Measured on Violet Evergarden: the acceptance path refused chain 2
        # because one simulated depth-2 reuse read 99.69% of pixels, while the actual chain-2 render
        # differs from ungated by 0.011% on its worst frame in the same window - the simulation
        # over-predicts reuses the gate never makes, and the chain-1 configuration it picked instead
        # has a 0.129% frame, i.e. one over the bar. The proxy decided chain 2 on the same clip, and
        # that configuration measures 0.011%. So the statistic is a guard, not yet a certifier.
        if certify_on not in ("proxy", "acceptance"):
            raise ValueError(f"certify_on must be 'proxy' or 'acceptance', got {certify_on!r}")
        self.certify_on = certify_on
        self._window: list[tuple[float, tuple[float, ...], tuple[float, ...]]] = []   # informative only
        self._all: list[tuple[float, tuple[float, ...], tuple[float, ...]]] = []      # everything observed
        self._skipped = 0
        self._pairs: list[tuple[float, tuple[float, ...]]] = []
        if fallback not in ("strictest", "off"):
            raise ValueError(f"fallback must be 'strictest' or 'off', got {fallback!r}")
        # what to do when no candidate keeps the error inside the budget:
        #   "strictest" - reuse as little as the range allows (speed-first)
        #   "off"       - do not reuse at all (quality-first; the tuner just established that
        #                 nothing in its range is safe on this content)
        self.fallback = fallback
        self.chosen: float | None = None
        self.chosen_chain: int | None = None
        self.veto: float | None = None
        # calibration is finished even when the verdict is "do not reuse", so this cannot
        # double as `chosen is not None`
        self.settled = False

    def add(self, lr_delta: float, hr_deltas, accept_deltas=None) -> None:
        """Record one calibration frame.

        ``hr_deltas`` is the error that reusing this frame would cost at every chain depth the
        gate can reach: element 0 is "reuse it in place of the previous frame", element 1 is
        "two frames back", and so on. A scalar is accepted and treated as depth 1.

        The depth matters because the gate serves a frame from a reference up to ``max_chain``
        frames old, and that is where the visible error lives: on the clean anime clip the
        one-step delta is 1.6 levels while the four-step delta is 9.3, because four frames of
        real motion sit between them.
        """
        if self.settled or lr_delta <= 0:
            return
        if isinstance(hr_deltas, (int, float)):
            hr_deltas = (hr_deltas,)
        clean = [float(d) for d in hr_deltas
                 if d == d and d >= 0]          # drop NaN, keep inf as "usable nowhere"
        if accept_deltas is not None and not isinstance(accept_deltas, (int, float)):
            accept = [float(d) for d in accept_deltas if d == d and d >= 0]
        elif accept_deltas is not None:
            accept = [float(accept_deltas)]
        else:
            accept = []
        if clean:
            sample = (float(lr_delta), tuple(clean), tuple(accept))
            self._all.append(sample)
            if self.min_informative is None or sample[0] >= self.min_informative:
                self._window.append(sample)
            else:
                self._skipped += 1
        if len(self._window) >= self.calibration_window or len(self._all) >= self.patience:
            self.settle()

    def settle(self) -> None:
        """Decide on the frames seen so far.

        Needed for two callers: a clip shorter than the calibration window (which would otherwise
        never calibrate), and tests that feed a fixed number of frames. Keeps the same adversarial
        subset - the `min_pairs` frames with the largest LR delta.
        """
        if self.settled or not self._all:
            return
        # The whole observed window decides, not a subset of it. Selecting the top-k by LR delta -
        # the obvious "adversarial" move - starves the narrow candidates: the 16 most marginal frames
        # all sit near the widest threshold, so a candidate of 2 has almost no samples left and the
        # tuner refuses material it has been reusing safely for weeks (measured on Chainsaw Man: every
        # depth read a worst share of 0.35% and reuse went off, against 0.005% on the clip).
        #
        # The adversarial part is already inside `_decide`: for each candidate it scores only the
        # frames that candidate would actually reuse (`lr <= cand`) and takes the WORST of them. A
        # population that matches the decision is the whole point - a top-k of the marginal frames
        # measures the thresholds we never use.
        pool = (self._window if len(self._window) >= self.min_pairs else self._all)
        self._pairs = list(pool)
        self._decide()

    @property
    def ready(self) -> bool:
        """Calibration is over. ``chosen is None`` with ``ready`` means "do not reuse"."""
        return self.settled

    def _noise_floor(self) -> float:
        # median of the calmest third of the OBSERVED frames: robust to a couple of wild frames at
        # the start, while a mean over a fifth was effectively "the two calmest frames" and jumped
        # to 62.9. It has to come from the whole window, not from `_pairs`: `_pairs` holds the frames
        # with the LARGEST deltas (the adversarial subset a worst-frame rule needs), and their
        # calmest third is not calm at all - measured on Chainsaw Man the same clip read a floor of
        # 0.95 with arrival-order frames and 9.38 with the adversarial subset, which refused reuse on
        # material that had been reusing safely at 0.005%.
        source = self._pairs or self._window or self._all
        ordered = sorted(p[0] for p in source)
        k = max(1, len(ordered) // 3)
        return ordered[k // 2]

    def _decide(self) -> None:
        # Which chain depth can be certified? A deeper chain reuses more (down to
        # max_chain frames per render) but serves staler references, so the answer can be
        # "1 frame deep" on moving content where "4" was assumed.
        deepest = max((len(deltas) for _, deltas, _ in self._pairs), default=0)
        # Two instruments live here. The block-mean proxy is what the tuner has measured since the
        # beginning (2-level budget at depth 1, 12 deeper); the acceptance statistic is what the
        # pictures are actually judged by (per cent of pixels above 8 levels in RGB, worst frame).
        # When the caller feeds the second one - the app does, from the same forced-render
        # calibration frames - it decides, and the proxy is logged only. Every false "price of
        # quality" in the ledger came from certifying one instrument and measuring the other.
        has_accept = self._uses_acceptance() and self.certify_on == "acceptance"
        certified: list[tuple[int, float, float]] = []      # (depth, threshold, worst)
        self._certified: list[tuple[int, float, float]] = certified
        self._depth_p90: list[tuple[int, float, float, float | None]] = []
        # The best value each depth can reach over the candidate range, certified or not: the
        # rejected depths decide the chain, so they belong in the log too. The third element is
        # the limit that applied, so the line reads "d2 worst share 0.20 (limit 0.1)".
        for depth in range(1, deepest + 1):
            limit = self.accept_bar if has_accept else (
                self.budget if depth == 1 else self.chain_budget)
            best_cand = None
            best_value = None
            floor_value = None
            for cand in self.candidates:
                values = []
                for index, (lr, deltas, accept) in enumerate(self._pairs):
                    if lr > cand:
                        continue
                    series = accept if has_accept else deltas
                    if len(series) < depth:
                        continue
                    # Reachability: the reference is only `depth` frames old if the frames between
                    # it and this one were reused too, so a frame whose predecessor failed the
                    # threshold never costs a depth-2 reuse. Measured on Chainsaw Man ep01: calm
                    # frames read deltas of 0.2 at depth 1 and 16-35 at depths 2-4 because they
                    # sit next to a one-frame flash - a configuration the gate cannot build - and
                    # the statistic of those unreachable reuses pinned the chain to 1 on every
                    # title with that structure, while forcing chain 4 measured 0.005% at 1.7x.
                    if any(self._pairs[j][0] > cand
                           for j in range(max(0, index - depth + 1), index)):
                        continue
                    values.append(series[depth - 1])
                if len(values) < max(2, len(self._pairs) // 4):
                    continue
                if has_accept:
                    value = max(values)          # acceptance is a worst-frame rule
                else:
                    values.sort()
                    value = values[min(len(values) - 1, int(0.90 * (len(values) - 1)))]
                floor_value = value
                if value <= limit:
                    best_cand, best_value = cand, value
            if floor_value is not None:
                # record the value at the candidate that was actually certified, and name it: printing
                # the widest candidate's value under the word "certified" read as "0.35 (limit 0.1)"
                # while the certified threshold was 4.0, i.e. a line that contradicted itself.
                self._depth_p90.append((depth, best_value if best_value is not None else floor_value,
                                        limit, best_cand))
            if best_cand is not None:
                certified.append((depth, best_cand, best_value or 0.0))

        if os.environ.get("RVE_TUNER_DEBUG"):
            print(f"[tuner] budget {self.budget} fallback {self.fallback} "
                  f"pairs {len(self._pairs)}", file=sys.stderr)
            for lr, deltas, _accept in self._pairs:
                print(f"[tuner]   lr {lr:.2f} at depths "
                      + ", ".join(f"{d:.2f}" for d in deltas), file=sys.stderr)
            for depth, cand, p90 in certified:
                print(f"[tuner] depth {depth}: threshold {cand}, p90 {p90:.2f}", file=sys.stderr)

        if self.noise_floor_refuse is not None and self._noise_floor() > self.noise_floor_refuse:
            self.settled = True
            self.chosen = None
            self.veto = None
            return

        if certified:
            # Longest usable chain, then the widest threshold certified for it.
            depth, cand, _ = max(certified, key=lambda row: (row[0], row[1]))
            if self.chain_policy == "max":
                # speed-first: keep the threshold that depth 1 certified, but let the gate walk
                # the whole chain. The content is the owner's call, the numbers are in the docs.
                depth = deepest
            lo, hi = self.clamp
            self.chosen = max(lo, min(hi, cand))
            self.chosen_chain = depth
            self.veto = None
            self.settled = True
            return

        best = None
        if best is None:
            # Nothing qualified, i.e. no threshold in the range keeps the error inside the
            # budget. "strictest" reuses as little as the range allows; "off" refuses to reuse
            # at all. Keeping the configured seed (the old behaviour) meant a grainy clip ran
            # at whatever number happened to be set, and on the film clip that produced frames
            # at 34.7 dB against the ungated render.
            self.settled = True
            if self.fallback == "off":
                self.chosen = None
                self.veto = None
                return
            # nothing certified at any depth, so reuse only in place (depth 1)
            self.chosen_chain = 1
            best = self.candidates[0]
        lo, hi = self.clamp
        self.chosen = max(lo, min(hi, best))
        # The veto is NOT enabled automatically: on grainy film it destroys reuse (43 vetoes in
        # 94 frames, 1.1% left) and on clean anime it costs ~1 ms/frame for a difference we
        # measured as invisible (52.6 dB median, 48.1 dB worst frame without it). The noise floor
        # is still computed because it is the useful diagnostic when someone does want it.
        self.veto = None
        self.settled = True

    def _uses_acceptance(self) -> bool:
        # length, not truthiness: a measured zero is the common answer on clean material (0.00% of
        # pixels above 8 levels) and `any((0.0,))` is False, so a truthiness test silently threw away
        # every measured sample and fell back to the block-mean proxy.
        return any(len(accept) > 0 for _, _, accept in self._pairs)

    def stats(self) -> str:
        if not self.settled:
            if self.min_informative is not None:
                return (f"auto gate: calibrating ({len(self._window)}/{self.calibration_window} informative "
                        f"frames, {len(self._all)} observed, {self._skipped} skipped as trivially identical)")
            return (f"auto gate: calibrating ({len(self._window)}/{self.calibration_window} frames "
                    f"observed, keeping the {self.min_pairs} most marginal)")
        if self.chosen is None:
            if self.noise_floor_refuse is not None and self._noise_floor() > self.noise_floor_refuse:
                return (f"auto gate: noise floor {self._noise_floor():.2f} is above the "
                        f"{self.noise_floor_refuse:.1f} limit for this material, so reuse is OFF "
                        f"- at this noise the measured worst frame is 0.105-0.470% of pixels "
                        f"above 8 levels, past the 0.1% acceptance bar")
            instrument = ("the worst acceptance share inside %.2f%%" % self.accept_bar
                          if self._uses_acceptance()
                          else "the p90 error inside %.1f levels" % self.budget)
            best = ", ".join(f"d{depth} {value:.2f}" for depth, value, _, _ in self._depth_p90)
            return (f"auto gate: no candidate kept {instrument} on "
                    f"{len(self._pairs)} calibration frames (noise floor "
                    f"{self._noise_floor():.2f}); per depth: {best or 'no reachable samples'}, "
                    f"so reuse is OFF")
        reuse = 100.0 * sum(1 for pair in self._pairs if pair[0] <= self.chosen) / len(self._pairs)
        chosen_value = next((v for depth, cand, v in self._certified
                             if depth == self.chosen_chain and cand == self.chosen), None)
        # the label must follow the instrument that DECIDED, not the one that merely has data: with
        # certify_on="proxy" the acceptance shares are still measured and printed, and calling the
        # p90 numbers "worst share" reads as if the picture had certified them.
        label = ("worst share" if (self._uses_acceptance() and self.certify_on == "acceptance")
                 else "p90")
        depths = ", ".join(
            f"d{depth} {label} {value:.2f} at {cand:g} (limit {limit:g})" if cand is not None
            else f"d{depth} none ({value:.2f} at the widest, limit {limit:g})"
            for depth, value, limit, cand in self._depth_p90)
        return (f"auto gate: threshold {self.chosen:.1f}/255, chain "
                f"{self.chosen_chain or '?'}, veto {self.veto}, "
                f"noise floor {self._noise_floor():.2f}, would reuse {reuse:.0f}% of the "
                f"{len(self._pairs)} calibration frames"
                + (f"; at this threshold {chosen_value:.2f}" if chosen_value is not None else "")
                + (f"; certified {depths}" if depths else "")
                + (f"; calibration: {len(self._window)} informative of {len(self._all)} frames"
                   if self.min_informative is not None else ""))
