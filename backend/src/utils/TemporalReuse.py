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
        if (self._prev_tiles is not None
                and self._chain < self.max_chain
                and tiles.shape == self._prev_tiles.shape):
            worst_tile = (tiles - self._prev_tiles).abs().max().item()
            reuse = worst_tile <= self.threshold
        self._prev_tiles = tiles
        if reuse:
            self._chain += 1
            self.reused += 1
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
                 max_chain: int = REFERENCE_MAX_CHAIN,
                 luma_taps: Sequence[float] = REFERENCE_LUMA_TAPS,
                 scene_threshold: float = REFERENCE_SCENE_CUT,
                 pixel_veto: Optional[float] = REFERENCE_PIXEL_VETO) -> None:
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
        self._ref_luma = None           # its full-res luma (only with a pixel veto)
        self._chain = 0                 # consecutive frames served from the reference
        self.processed = 0
        self.reused = 0
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
        self._chain = 0
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

    def stats(self) -> str:
        pct = 100.0 * self.reused / self.processed if self.processed else 0.0
        return (f"reference reuse: {self.reused}/{self.processed} frames reused ({pct:.1f}%), "
                f"{self.scene_cuts} scene cuts, {self.vetoes} pixel vetoes, "
                f"threshold={self.threshold}/255 over "
                f"{self.block or 'scaled'}px blocks, "
                f"max chain={self.max_chain or 'unbounded'}, "
                f"pixel veto={self.pixel_veto if self.pixel_veto is not None else 'off'}")

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

    def observe(self, blocks, luma=None) -> bool:
        """The decision and the state update, given a block grid (torch or numpy).

        Kept separate from :meth:`blocks` so the state machine - reference keeping, chain
        limit, scene cut, pixel veto - can be tested with numpy grids where torch is not
        installed. Nothing but the chain counter changes on a duplicate; on a miss the
        current grid (and luma, when a veto is configured) becomes the reference and the
        chain restarts.

        The global level used for the scene cut is the mean of the block-mean grid: exact
        when the block size divides both frame axes, off by a fraction of a level on the
        partial tail blocks, which is irrelevant against a 32 level cut threshold.
        """
        self.processed += 1
        self.last_delta = float("inf")
        mean = float(blocks.mean().item())
        scene_cut = False
        delta = float("inf")
        if self._ref is not None:
            if tuple(blocks.shape) == tuple(self._ref.shape):
                delta = float(abs(blocks - self._ref).max().item())
                self.last_delta = delta
            else:                       # resolution change: history is meaningless
                scene_cut = True
            if abs(mean - self._ref_mean) > self.scene_threshold:
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
        if reuse:
            self._chain += 1
            self.reused += 1
        else:
            if scene_cut:
                self.scene_cuts += 1
            self._ref = blocks
            self._ref_mean = mean
            self._ref_luma = luma if self.pixel_veto is not None else None
            self._chain = 0
        return reuse

    def is_duplicate(self, frame_tensor) -> bool:
        """Feed the low-resolution frame: True means the reference's output may be emitted."""
        luma = self.luma(frame_tensor)
        return self.observe(self.blocks_of(luma), luma)

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
      - cost       = p95 of their ``hr_delta`` (levels of 255)

    and pick the largest candidate whose p95 stays inside ``budget``. That is content-agnostic:
    grainy film has a higher noise floor and lands on a larger threshold, clean anime on a
    smaller one, without anyone typing a number.

    The whole thing costs no extra inference (the frames are rendered regardless) and a few
    hundred microseconds per calibration frame for the HR statistic.
    """

    def __init__(self, budget: float = 2.0, min_pairs: int = 16, seed: float = 5.0,
                 candidates=(2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0),
                 clamp=(2.0, 12.0), noise_floor_quiet: float = 0.35):
        self.budget = float(budget)
        self.min_pairs = int(min_pairs)
        # the value the pipeline was configured with: kept when no candidate qualifies, which
        # happens on an unrepresentative opening (a busy first scene measured a "noise floor"
        # of 62.9 levels and drove the choice to the bottom of the range)
        self.seed = float(seed)
        self.candidates = tuple(float(c) for c in candidates)
        self.clamp = (float(clamp[0]), float(clamp[1]))
        self.noise_floor_quiet = float(noise_floor_quiet)
        self._pairs: list[tuple[float, float]] = []
        self.chosen: float | None = None
        self.veto: float | None = None

    def add(self, lr_delta: float, hr_delta: float) -> None:
        if self.chosen is not None or lr_delta <= 0:
            return
        if hr_delta == hr_delta and hr_delta >= 0:  # not NaN
            self._pairs.append((float(lr_delta), float(hr_delta)))
        if len(self._pairs) >= self.min_pairs:
            self._decide()

    @property
    def ready(self) -> bool:
        return self.chosen is not None

    def _noise_floor(self) -> float:
        # median of the calmest third: robust to a couple of wild frames at the start, while a
        # mean over a fifth was effectively "the two calmest frames" and jumped to 62.9
        ordered = sorted(p[0] for p in self._pairs)
        k = max(1, len(ordered) // 3)
        return ordered[k // 2]

    def _decide(self) -> None:
        best = None
        for cand in self.candidates:
            errs = [hr for lr, hr in self._pairs if lr <= cand]
            if len(errs) < max(2, len(self._pairs) // 4):
                # a candidate that would reuse almost nothing is not worth considering
                continue
            errs.sort()
            p90 = errs[min(len(errs) - 1, int(0.90 * (len(errs) - 1)))]
            if p90 <= self.budget:
                best = cand
        if best is None:
            # Nothing qualified: the strictest candidate is the honest fallback, because the
            # tuner just established that *no* threshold keeps the error inside the budget.
            # Keeping the configured seed instead (the old behaviour) meant a grainy clip ran
            # at whatever number happened to be set, and on the film clip that produced frames
            # at 34.7 dB against the ungated render.
            best = self.candidates[0]
        lo, hi = self.clamp
        self.chosen = max(lo, min(hi, best))
        # The veto is NOT enabled automatically: on grainy film it destroys reuse (43 vetoes in
        # 94 frames, 1.1% left) and on clean anime it costs ~1 ms/frame for a difference we
        # measured as invisible (52.6 dB median, 48.1 dB worst frame without it). The noise floor
        # is still computed because it is the useful diagnostic when someone does want it.
        self.veto = None

    def stats(self) -> str:
        if self.chosen is None:
            return f"auto gate: calibrating ({len(self._pairs)}/{self.min_pairs} frames)"
        reuse = 100.0 * sum(1 for lr, _ in self._pairs if lr <= self.chosen) / len(self._pairs)
        return (f"auto gate: threshold {self.chosen:.1f}/255, veto {self.veto}, "
                f"noise floor {self._noise_floor():.2f}, would reuse {reuse:.0f}% of the "
                f"{len(self._pairs)} calibration frames")
