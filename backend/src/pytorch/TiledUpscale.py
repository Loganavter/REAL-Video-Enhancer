"""Tile-level temporal reuse for the PyTorch / TensorRT upscalers.

Frame-level reuse (`backend/src/utils/TemporalReuse.py`) answers "may the whole frame be
re-emitted?"; this module answers the finer question: *which tiles* of the low-resolution
frame actually changed, so only those are run through the SR network and the rest of the
high-resolution frame is copied from the previous output.

Why the detector works (measured on 40 s of 1440x1080 23.976 fps film, 2x Compact model):

* a low-passed LR tile delta of 0.25-0.35 levels predicts an SR delta of 0.77 mean /
  0.95 max (8-bit levels), i.e. "LR tile unchanged" really does mean "SR tile unchanged"
  to sub-1-level error; LR delta >= 4 levels already means SR delta ~17.8 levels;
* at threshold 0.35 on 360 px LR tiles, 16.7% of tiles can be skipped; at 0.5, 33%;
* SRVGGNetCompact has a receptive radius of 18 LR px (1 + 16 + 1 convs), so every tile
  needs 18 px of halo on each side before the output is cropped back - otherwise seams
  appear at the tile borders.

Geometry contract
-----------------
The upscaler is handed a *fixed* input shape ``(1, 3, T, T)`` with ``T = tile_lr +
2 * halo_lr`` on every call, because a TensorRT engine has a static input shape. This class
does the framing itself:

* core tiles are a plain grid of ``tile_lr`` steps over the LR frame, end-exclusive, the
  last row/column clipped to the frame;
* a tile's input window is the core grown by ``halo_lr`` on each side and clipped to the
  frame, then extended to ``T`` by *replicate* padding (the row/column index vector is
  clamped to ``[0, H-1]`` - identical to ``F.pad(..., mode="replicate")``, but it works for
  any frame size, including frames smaller than one tile);
* the network output is cropped back to the core (offset ``top``/``left`` inside the
  ``T * scale`` window) and written into the high-resolution frame.

Nothing in this module copies a whole frame: ``compose`` writes the recomputed tiles
straight into ``prev_hr`` (the caller hands over ownership of that buffer).

The geometry and the mask decision are deliberately plain Python (integers, lists of
floats) so they can be unit-tested without torch - a TensorRT cell has no torch, and a
tile layout bug is a geometry bug long before it is a tensor bug.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

try:  # keep the geometry helpers importable where torch is not installed
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - exercised only in torch-less test cells
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

__all__ = [
    "Rect",
    "TileWindow",
    "tile_grid",
    "halo_window",
    "replicate_window",
    "tile_window",
    "hr_write_slice",
    "hr_read_slice",
    "block_bounds",
    "block_grid",
    "decide_tiles",
    "is_scene_cut",
    "TiledUpscaler",
]

# (y0, y1, x0, x1) in LR pixels, end-exclusive.
Rect = Tuple[int, int, int, int]


# --------------------------------------------------------------------------------------
# pure geometry (no torch - unit-tested directly)
# --------------------------------------------------------------------------------------
def tile_grid(height: int, width: int, tile_lr: int) -> List[Rect]:
    """Cores of the LR tile grid, row-major, end-exclusive.

    The last row/column is clipped to the frame, so the cores never overlap and their union
    covers the frame exactly.
    """
    if tile_lr <= 0:
        raise ValueError(f"tile_lr must be positive, got {tile_lr}")
    if height <= 0 or width <= 0:
        raise ValueError(f"empty frame {height}x{width}")
    cores: List[Rect] = []
    for y0 in range(0, height, tile_lr):
        y1 = min(y0 + tile_lr, height)
        for x0 in range(0, width, tile_lr):
            x1 = min(x0 + tile_lr, width)
            cores.append((y0, y1, x0, x1))
    return cores


def halo_window(y0: int, y1: int, x0: int, x1: int, halo: int, height: int,
                width: int) -> Rect:
    """Core grown by ``halo`` on every side and clipped to the frame."""
    if halo < 0:
        raise ValueError(f"halo must be >= 0, got {halo}")
    return (max(0, y0 - halo), min(height, y1 + halo),
            max(0, x0 - halo), min(width, x1 + halo))


def replicate_window(start: int, length: int, limit: int) -> List[int]:
    """``[start, start + length)`` clamped to ``[0, limit - 1]`` (replicate padding).

    Equivalent to padding a size-``limit`` axis with ``F.pad(mode="replicate")``: indices
    below 0 repeat the first element, indices past the end repeat the last one.
    """
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    last = limit - 1
    return [0 if i < 0 else (last if i > last else i) for i in range(start, start + length)]


@dataclass(frozen=True)
class TileWindow:
    """Everything needed to render one core tile into the high-resolution frame."""

    core: Rect                # (y0, y1, x0, x1) LR core, end-exclusive
    rows: Tuple[int, ...]     # T LR row indices (clamped => replicate padding)
    cols: Tuple[int, ...]     # T LR column indices
    top: int                  # core offset inside the T window (rows)
    left: int                 # core offset inside the T window (cols)

    @property
    def core_height(self) -> int:
        return self.core[1] - self.core[0]

    @property
    def core_width(self) -> int:
        return self.core[3] - self.core[2]

    @property
    def window_size(self) -> int:
        return len(self.rows)

    def __post_init__(self) -> None:
        if len(self.rows) != len(self.cols):
            raise ValueError("window must be square")
        if self.top < 0 or self.left < 0:
            raise ValueError("negative window offset")
        if self.top + self.core_height > len(self.rows):
            raise ValueError("core does not fit inside the window")
        if self.left + self.core_width > len(self.cols):
            raise ValueError("core does not fit inside the window")


def tile_window(core: Rect, halo: int, height: int, width: int,
                tile_lr: int) -> TileWindow:
    """Fixed ``T x T`` LR input window for ``core`` (``T = tile_lr + 2 * halo``)."""
    y0, y1, x0, x1 = core
    window = tile_lr + 2 * halo
    hy0, _, hx0, _ = halo_window(y0, y1, x0, x1, halo, height, width)
    return TileWindow(
        core=core,
        rows=tuple(replicate_window(hy0, window, height)),
        cols=tuple(replicate_window(hx0, window, width)),
        top=y0 - hy0,
        left=x0 - hx0,
    )


def hr_write_slice(core: Rect, scale: int) -> Tuple[slice, slice]:
    """HR destination of a core tile: ``out[:, :, write[0], write[1]]``."""
    y0, y1, x0, x1 = core
    return slice(y0 * scale, y1 * scale), slice(x0 * scale, x1 * scale)


def hr_read_slice(window: TileWindow, scale: int) -> Tuple[slice, slice]:
    """Where the core sits inside the ``T * scale`` network output."""
    top, left = window.top, window.left
    return (slice(top * scale, (top + window.core_height) * scale),
            slice(left * scale, (left + window.core_width) * scale))


def block_grid(height: int, width: int, downscale: int) -> Tuple[int, int]:
    """Shape of the low-passed luma grid (ceil, so the last block is partially padded)."""
    if downscale <= 0:
        raise ValueError(f"downscale must be positive, got {downscale}")
    return -(-height // downscale), -(-width // downscale)


def block_bounds(core: Rect, downscale: int, blocks_h: int,
                 blocks_w: int) -> Tuple[int, int, int, int]:
    """Low-pass block range ``(by0, by1, bx0, bx1)`` covered by a core tile.

    Adjacent tiles may share one block when ``tile_lr`` is not a multiple of ``downscale``
    (``floor`` start / ``ceil`` end); sharing is conservative - a change in the shared block
    marks both tiles - and coverage is complete.
    """
    y0, y1, x0, x1 = core
    by0 = min(max(0, y0 // downscale), blocks_h - 1)
    by1 = min(max(by0 + 1, -(-y1 // downscale)), blocks_h)
    bx0 = min(max(0, x0 // downscale), blocks_w - 1)
    bx1 = min(max(bx0 + 1, -(-x1 // downscale)), blocks_w)
    return by0, by1, bx0, bx1


def decide_tiles(tile_maxes: Sequence[float], threshold: float, chains: Sequence[int],
                 max_chain: int) -> List[bool]:
    """Which tiles must be recomputed.

    ``tile_maxes`` is the per-tile worst low-pass delta in 8-bit luma levels, ``chains`` how
    many consecutive frames each tile has already been served from the cache. A tile is
    recomputed when it changed above ``threshold`` *or* its chain reached ``max_chain``
    (a tile may be reused for at most ``max_chain`` consecutive frames, so a wrong decision
    can last at most that long).
    """
    if len(tile_maxes) != len(chains):
        raise ValueError(
            f"tile_maxes ({len(tile_maxes)}) and chains ({len(chains)}) disagree")
    mask: List[bool] = []
    for delta, chain in zip(tile_maxes, chains):
        if delta > threshold:
            mask.append(True)
        elif chain >= max_chain:
            mask.append(True)
        else:
            mask.append(False)
    return mask


def is_scene_cut(prev_mean: float, current_mean: float, threshold: float) -> bool:
    """Global low-pass brightness jump that should drop all history."""
    return abs(current_mean - prev_mean) >= threshold


def _require_torch() -> None:
    if torch is None:  # pragma: no cover - only in torch-less cells
        raise RuntimeError(
            "TiledUpscaler needs torch; the geometry helpers (tile_grid, tile_window, ...) "
            "work without it")


def _no_grad(fn):
    """``torch.no_grad()`` decorator that survives a torch-less import.

    ``no_grad`` rather than ``inference_mode``: the composed frame is returned to the
    caller and later read/written by the encoder path, and inference tensors cannot be
    handed to autograd-enabled code.
    """
    if torch is None:  # pragma: no cover - only in torch-less cells
        return fn
    return torch.no_grad()(fn)


# --------------------------------------------------------------------------------------
# the stateful upscaler
# --------------------------------------------------------------------------------------
class TiledUpscaler:
    """Recompute only the LR tiles that changed, copy the rest from the previous HR frame.

    Usage per frame::

        rects = upscaler.plan(lr)                    # None => re-emit the previous frame
        if rects is None:
            emit(prev_hr)
        else:
            hr = upscaler.compose(scale, upscale_fn, prev_hr, lr)

    ``plan`` is a *query*: it reads the stored history and arms ``compose`` with what it
    decided. ``compose`` consumes that decision and is the only place where the history
    (chain counters, previous luma grid) advances. Calling ``plan`` twice for one frame
    repeats the decision and double-counts the statistics, but cannot change reuse.

    Thresholds are in 8-bit luma levels (0..255), the unit the measurements above use.
    """

    def __init__(self, tile_lr: int = 512, halo_lr: int = 18, threshold: float = 0.35,
                 max_chain: int = 2, downscale: int = 8,
                 scene_threshold: float = 32.0, tile_stat: str = "mean") -> None:
        if tile_lr <= 0:
            raise ValueError(f"tile_lr must be positive, got {tile_lr}")
        if halo_lr < 0:
            raise ValueError(f"halo_lr must be >= 0, got {halo_lr}")
        if downscale <= 0:
            raise ValueError(f"downscale must be positive, got {downscale}")
        if max_chain < 0:
            raise ValueError(f"max_chain must be >= 0, got {max_chain}")
        self.tile_lr = int(tile_lr)
        self.halo_lr = int(halo_lr)
        self.threshold = float(threshold)
        self.max_chain = int(max_chain)
        self.downscale = int(downscale)
        self.scene_threshold = float(scene_threshold)
        if tile_stat not in ("mean", "max"):
            raise ValueError(f"tile_stat must be 'mean' or 'max', got {tile_stat!r}")
        self.tile_stat = tile_stat

        self._prev_blocks = None          # (1, 1, bh, bw) low-passed luma, 8-bit levels
        self._prev_shape: Optional[Tuple[int, int]] = None
        self._chains: List[int] = []
        self._pending = None              # decision armed by plan(), consumed by compose()
        self._index_cache: Dict[Tuple[int, int, int, int], "torch.Tensor"] = {}

        self.frames = 0
        self.frames_reused = 0
        self.tiles_planned = 0
        self.tiles_reused = 0
        self.tiles_rendered = 0
        self.scene_cuts = 0
        self.resets = 0

    # -- configuration ------------------------------------------------------------
    @property
    def tile(self) -> int:
        """Static network input size ``T = tile_lr + 2 * halo_lr``."""
        return self.tile_lr + 2 * self.halo_lr

    def configure(self, tile_lr: int, halo_lr: int, threshold: float, max_chain: int,
                  downscale: int = 8) -> None:
        """Apply a quality change: re-plan sizes and forget the history."""
        self.tile_lr = int(tile_lr)
        self.halo_lr = int(halo_lr)
        self.threshold = float(threshold)
        self.max_chain = int(max_chain)
        self.downscale = int(downscale)
        self._index_cache.clear()
        self.reset()

    # -- history ------------------------------------------------------------------
    def reset(self) -> None:
        """Forget the history (scene cut, new video, quality/model change).

        Cumulative statistics are kept; only the reuse state is dropped, so the next
        ``plan()`` recomputes everything.
        """
        self._prev_blocks = None
        self._prev_shape = None
        self._chains = []
        self._pending = None
        self.resets += 1

    @property
    def can_reuse(self) -> bool:
        return self._prev_blocks is not None

    def stats(self) -> str:
        tile_pct = (100.0 * self.tiles_reused / self.tiles_planned
                    if self.tiles_planned else 0.0)
        frame_pct = 100.0 * self.frames_reused / self.frames if self.frames else 0.0
        return (f"tiled reuse: {self.tiles_reused}/{self.tiles_planned} tiles reused "
                f"({tile_pct:.1f}%), {self.frames_reused}/{self.frames} frames fully reused "
                f"({frame_pct:.1f}%), {self.scene_cuts} scene cuts, "
                f"threshold={self.threshold}/255, tile={self.tile_lr}+{2 * self.halo_lr} halo, "
                f"max chain={self.max_chain}")

    def tile_windows(self, height: int, width: int) -> List[TileWindow]:
        """Windows of the whole grid (useful for warm-up / diagnostics)."""
        return [tile_window(core, self.halo_lr, height, width, self.tile_lr)
                for core in tile_grid(height, width, self.tile_lr)]

    # -- detector -----------------------------------------------------------------
    def _indices(self, start: int, length: int, limit: int,
                 axis: int, device) -> "torch.Tensor":
        key = (axis, start, length, limit)
        cached = self._index_cache.get(key)
        if cached is None or cached.device != device:
            cached = torch.tensor(replicate_window(start, length, limit), device=device,
                                  dtype=torch.long)
            self._index_cache[key] = cached
        return cached

    @staticmethod
    def _frame_shape(lr) -> Tuple[int, int, int]:
        if lr.dim() != 4:
            raise ValueError(f"expected a (1, C, H, W) tensor, got shape {tuple(lr.shape)}")
        if int(lr.shape[0]) != 1:
            raise ValueError(f"expected batch size 1, got {int(lr.shape[0])}")
        return int(lr.shape[1]), int(lr.shape[-2]), int(lr.shape[-1])

    @_no_grad
    def _luma_blocks(self, lr):
        """``(1, 3, H, W)`` in [0, 1] -> ``(1, 1, ceil(H/ds), ceil(W/ds))`` in 8-bit levels.

        Done entirely on the tensor's own device (no ``.cpu()``/``.numpy()``): repeat the
        last row/column so the pooling divides by a full window, then average.
        """
        ds = self.downscale
        luma = lr.mean(dim=1, keepdim=True).float()
        height, width = int(luma.shape[-2]), int(luma.shape[-1])
        blocks_h, blocks_w = block_grid(height, width, ds)
        rows = self._indices(0, blocks_h * ds, height, 0, luma.device)
        cols = self._indices(0, blocks_w * ds, width, 1, luma.device)
        padded = luma.index_select(2, rows).index_select(3, cols)
        return F.avg_pool2d(padded, ds) * 255.0

    @_no_grad
    def _tile_deltas(self, blocks, prev, cores: Sequence[Rect]) -> List[float]:
        delta = (blocks - prev).abs()
        blocks_h, blocks_w = int(blocks.shape[-2]), int(delta.shape[-1])
        per_tile = []
        for core in cores:
            by0, by1, bx0, bx1 = block_bounds(core, self.downscale, blocks_h, blocks_w)
            window = delta[..., by0:by1, bx0:bx1]
            # The per-tile statistic must be the MEAN of the low-passed deltas, not the max:
            # a 512px tile holds 64x64 8px blocks, so a max trips on a single dithered block
            # and the gate would never fire (measured: 9/9 tiles "changed" on every frame of
            # a moving film). The measured predictor is mean-based: LR mean 0.25-0.35 ->
            # HR mean 0.77 / max 0.95 levels, i.e. sub-1-level, invisible error.
            per_tile.append(window.mean() if self.tile_stat == "mean" else window.amax())
        # one host sync for the whole frame instead of one per tile
        return [float(v) for v in torch.stack(per_tile).tolist()]

    def _decide(self, blocks, cores: Sequence[Rect], height: int,
                width: int) -> List[bool]:
        prev = self._prev_blocks
        same_geometry = (
            prev is not None
            and self._prev_shape == (height, width)
            and len(self._chains) == len(cores)
            and tuple(prev.shape) == tuple(blocks.shape)
        )
        if not same_geometry:
            return [True] * len(cores)
        deltas = self._tile_deltas(blocks, prev, cores)
        if is_scene_cut(float(prev.mean().item()), float(blocks.mean().item()),
                        self.scene_threshold):
            self.scene_cuts += 1
            return [True] * len(cores)
        return decide_tiles(deltas, self.threshold, self._chains, self.max_chain)

    def _commit(self, blocks, height: int, width: int, mask: Sequence[bool]) -> None:
        """Advance the history: chains for reused tiles, reset for recomputed ones."""
        fresh = (self._prev_blocks is None
                 or len(self._chains) != len(mask)
                 or self._prev_shape != (height, width)
                 or tuple(self._prev_blocks.shape) != tuple(blocks.shape))
        if fresh:
            self._chains = [0] * len(mask)
        for i, recomputed in enumerate(mask):
            self._chains[i] = 0 if recomputed else self._chains[i] + 1
        self._prev_blocks = blocks
        self._prev_shape = (height, width)

    def _account(self, mask: Sequence[bool]) -> None:
        self.frames += 1
        self.tiles_planned += len(mask)
        self.tiles_reused += sum(1 for m in mask if not m)

    # -- public API ---------------------------------------------------------------
    def plan(self, lr_tensor) -> Optional[List[Rect]]:
        """LR cores (y0, y1, x0, x1) that must be recomputed, or ``None`` for full reuse.

        ``lr_tensor`` is ``(1, 3, H, W)``, values 0..1, on any device. An empty list is
        never returned: "nothing changed" is ``None`` (the frame-level shortcut), and a new
        output that still has to be produced every pixel of (because ``prev_hr`` is gone)
        arrives as the full grid through ``compose``.

        After a ``None`` the frame is settled: the chain counters were already advanced, so
        do *not* call ``compose`` for that frame - there is nothing to render.
        """
        _require_torch()
        _, height, width = self._frame_shape(lr_tensor)
        cores = tile_grid(height, width, self.tile_lr)
        blocks = self._luma_blocks(lr_tensor)
        mask = self._decide(blocks, cores, height, width)
        self._account(mask)
        changed = [core for core, must in zip(cores, mask) if must]
        if not changed:
            # the caller re-emits the previous frame: commit, compose() will not be called
            self._pending = None
            self.frames_reused += 1
            self._commit(blocks, height, width, mask)
            return None
        self._pending = (height, width, changed, blocks, mask)
        return changed

    @_no_grad
    def compose(self, scale: int, upscale_fn: Callable, prev_hr, lr_tensor,
                reuse_prev: bool = True):
        """Render the tiles ``plan`` asked for into ``prev_hr`` and return the HR frame.

        ``upscale_fn`` is always called with ``(1, 3, T, T)``; its output must be
        ``(1, 3, T * scale, T * scale)`` and is cropped back to the core tile.

        With ``reuse_prev=True`` (the default) the recomputed tiles are written into
        ``prev_hr`` and the same tensor is returned - no full-frame copy. The caller hands
        over ownership of that buffer: anything still expected to hold the *previous*
        frame's pixels must be emitted before ``compose`` (or pass ``reuse_prev=False`` and
        pay for one copy).
        """
        _require_torch()
        scale = int(scale)
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        channels, height, width = self._frame_shape(lr_tensor)
        out_shape = (1, channels, height * scale, width * scale)

        pending = self._pending
        self._pending = None
        consumed = pending is not None and pending[0] == height and pending[1] == width
        cores: List[Rect]
        if prev_hr is None or tuple(int(v) for v in prev_hr.shape) != out_shape:
            # everything has to be rendered, whatever the history says
            if prev_hr is not None:
                self.reset()
            blocks = self._luma_blocks(lr_tensor)
            cores = tile_grid(height, width, self.tile_lr)
            mask = [True] * len(cores)
            out = lr_tensor.new_zeros(out_shape)
            if not consumed:
                self._account(mask)
        elif consumed:
            cores, blocks, mask = pending[2], pending[3], pending[4]
            out = prev_hr if reuse_prev else prev_hr.clone()
        else:
            # compose() without plan(): decide here so the state stays consistent
            blocks = self._luma_blocks(lr_tensor)
            grid = tile_grid(height, width, self.tile_lr)
            mask = self._decide(blocks, grid, height, width)
            self._account(mask)
            cores = [core for core, must in zip(grid, mask) if must]
            out = prev_hr if reuse_prev else prev_hr.clone()

        tile_size = self.tile
        expected = (tile_size * scale, tile_size * scale)
        for core in cores:
            window = tile_window(core, self.halo_lr, height, width, self.tile_lr)
            rows = self._indices(window.rows[0], tile_size, height, 0, lr_tensor.device)
            cols = self._indices(window.cols[0], tile_size, width, 1, lr_tensor.device)
            crop = lr_tensor.index_select(2, rows).index_select(3, cols)
            rendered = upscale_fn(crop)
            if tuple(int(v) for v in rendered.shape[-2:]) != expected:
                raise ValueError(
                    f"upscale_fn returned {tuple(rendered.shape[-2:])} for a "
                    f"{tuple(crop.shape[-2:])} input, expected {expected} "
                    f"(fixed T x T contract, T={tile_size}, scale={scale})")
            read_y, read_x = hr_read_slice(window, scale)
            write_y, write_x = hr_write_slice(core, scale)
            out[:, :, write_y, write_x] = rendered[:, :, read_y, read_x]

        self.tiles_rendered += len(cores)
        self._commit(blocks, height, width, mask)
        return out
