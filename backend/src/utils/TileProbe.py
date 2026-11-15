"""Predict, before the first frame, whether tile-level reuse will pay for itself.

The tile path and the frame gate answer different questions, and which of them is worth
enabling depends on content that is not visible in the file name: on a static anime episode
the tile path adds 1.4x on top of the gate, on a moving film it costs 1.7% of wall clock and
0.8 dB of fidelity for nothing, and on grainy material it would cost ~9% because every tile
"changes" every frame.

Rather than asking the user, sample the source and answer it with the same statistics the
runtime uses:

* the frame gate compares 32 px block means of the low-resolution luma against the last frame
  it kept, and skips the whole model when the worst block moved less than ``gate_threshold``;
* the tile path compares the mean of the same block grid inside each core tile against the
  last frame it rendered, and re-runs the model only for tiles whose mean moved more than
  ``tile_threshold``.

Both are computable from a handful of decoded frames, so the probe decodes a few frames at a
few positions (a seek breaks temporal continuity, so each position is its own run), replays
the two decisions on them, and reports the share of tiles that would have been reused *inside
the frames the gate would have rendered*. That share is the only number the decision needs:
with an area tax ``T`` the tile path costs ``T x (1 - reuse)`` of a full-frame call, so it
wins exactly when ``reuse > 1 - 1/T``.

Deliberately numpy-only and subprocess-only: no torch, no CUDA, no app imports, so the whole
decision layer is unit-testable where torch is not installed.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - only in a torch-less, numpy-less cell
    np = None  # type: ignore[assignment]

__all__ = [
    "ProbeResult",
    "block_means",
    "area_tax",
    "predict",
    "sample_luma",
    "probe",
]

# The scene-cut rule the runtime gate uses, in 8-bit luma levels.
_SCENE_THRESHOLD = 32.0


def _require_numpy() -> None:
    if np is None:  # pragma: no cover
        raise RuntimeError("TileProbe needs numpy for the block statistics")


@dataclass(frozen=True)
class ProbeResult:
    """What the probe measured, and the verdict that follows from it."""

    frames_sampled: int
    frames_gated: int          # the gate would have skipped these
    tiles_planned: int
    tiles_reused: int
    area_tax: float
    tile_reuse: float          # reused / planned, inside the frames the gate renders
    break_even: float          # 1 - 1 / area_tax: the reuse at which the two paths cost the same
    predicted_cost: float      # tile path cost as a fraction of one full-frame call
    required_cost: float       # the bar it has to clear: 1 / area_tax
    use_tiles: bool
    reason: str = ""

    def stats(self) -> str:
        return (
            f"tile probe: sampled {self.frames_sampled} frames, gate would skip "
            f"{self.frames_gated} ({_pct(self.frames_gated, self.frames_sampled)}), "
            f"inside the rest {self.tiles_reused}/{self.tiles_planned} tiles would be reused "
            f"({100.0 * self.tile_reuse:.1f}%) -> tiles would cost "
            f"{self.predicted_cost:.2f} of a full-frame call against a bar of "
            f"{self.required_cost:.2f} at {self.area_tax:.3f}x area -> tiles "
            f"{'ON' if self.use_tiles else 'OFF'}"
        )


def _pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:.0f}%" if whole else "0%"


def block_means(luma, block: int):
    """``(H, W)`` luma in 0..255 -> ``(ceil(H/b), ceil(W/b))`` means, tails averaged as they are.

    Mirrors ``F.avg_pool2d(..., block, ceil_mode=True)``: a partially covered tail block
    averages only the pixels that exist, which matters for 1080 (1080 = 33.75 x 32).
    """
    _require_numpy()
    if block <= 0:
        raise ValueError(f"block must be positive, got {block}")
    luma = np.asarray(luma, dtype=np.float64)
    if luma.ndim != 2:
        raise ValueError(f"expected a 2-D luma frame, got shape {luma.shape}")
    height, width = luma.shape[-2], luma.shape[-1]
    rows = -(-height // block)
    cols = -(-width // block)
    out = np.empty((rows, cols), dtype=np.float64)
    for r in range(rows):
        r0, r1 = r * block, min((r + 1) * block, height)
        strip = luma[r0:r1]
        for c in range(cols):
            c0, c1 = c * block, min((c + 1) * block, width)
            out[r, c] = strip[:, c0:c1].mean()
    return out


def area_tax(tile_h: int, tile_w: int, frame_h: int, frame_w: int, halo: int) -> float:
    """Halo-inflated tile area over the frame's area."""
    if tile_h <= 0 or tile_w <= 0:
        raise ValueError(f"tile sizes must be positive, got {tile_h}x{tile_w}")
    if frame_h <= 0 or frame_w <= 0:
        raise ValueError(f"empty frame {frame_h}x{frame_w}")
    if halo < 0:
        raise ValueError(f"halo must be >= 0, got {halo}")
    tiles_h = -(-frame_h // tile_h)
    tiles_w = -(-frame_w // tile_w)
    covered = tiles_h * tiles_w * (tile_h + 2 * halo) * (tile_w + 2 * halo)
    return covered / float(frame_h * frame_w)


def _tile_windows(tile_h: int, tile_w: int, frame_h: int, frame_w: int,
                  block: int) -> List[Tuple[int, int, int, int]]:
    """Block ranges ``(by0, by1, bx0, bx1)`` of every core tile, on the block grid."""
    rows = -(-frame_h // block)
    cols = -(-frame_w // block)
    windows: List[Tuple[int, int, int, int]] = []
    for y0 in range(0, frame_h, tile_h):
        y1 = min(y0 + tile_h, frame_h)
        by0 = min(max(0, y0 // block), rows - 1)
        by1 = min(max(by0 + 1, -(-y1 // block)), rows)
        for x0 in range(0, frame_w, tile_w):
            x1 = min(x0 + tile_w, frame_w)
            bx0 = min(max(0, x0 // block), cols - 1)
            bx1 = min(max(bx0 + 1, -(-x1 // block)), cols)
            windows.append((by0, by1, bx0, bx1))
    return windows


def predict(runs: Sequence[Sequence], *, tile_h: int, tile_w: int, halo: int,
            gate_threshold: float, tile_threshold: float, block: int = 32,
            gate_max_chain: int = 4, tile_max_chain: int = 2) -> ProbeResult:
    """Replay both decisions over sampled frames and say whether tiles pay.

    ``runs`` is a sequence of *continuity runs*: each inner sequence holds consecutive luma
    frames (0..255) that can be compared with each other. A seek starts a new run, because
    the frame before it is not the frame the runtime would have compared against.

    A frame the gate renders is then scored tile by tile against the last frame the tile path
    rendered, exactly as ``TiledUpscaler`` does: if the mean of the tile's block grid moved
    more than ``tile_threshold``, or its chain reached ``tile_max_chain``, the tile is
    recomputed. The two chain caps are separate because the runtime uses different values for
    them (a default gate chain of 4 against a tile chain of 2), and using one number for both
    changes the predicted reuse by a third.
    """
    _require_numpy()
    if not runs or not runs[0]:
        raise ValueError("predict needs at least one continuity run with one frame")

    runs = [[np.asarray(f, dtype=np.float64) for f in run] for run in runs]
    first = runs[0][0]
    if first.ndim != 2:
        raise ValueError(f"expected 2-D luma frames, got shape {first.shape}")
    shape = (int(first.shape[-2]), int(first.shape[-1]))
    tax = area_tax(tile_h, tile_w, shape[0], shape[1], halo)
    windows = _tile_windows(tile_h, tile_w, shape[0], shape[1], block)

    sampled = 0
    gated = 0
    planned = 0
    reused = 0

    for run in runs:
        gate_ref = None          # block grid the gate compares against
        tile_ref = None          # block grid the tile path compares against
        gate_chain = 0
        tile_chains = [0] * len(windows)
        for luma in run:
            sampled += 1
            grid = block_means(luma, block)
            if gate_ref is not None and grid.shape == gate_ref.shape:
                delta = float(np.abs(grid - gate_ref).max())
                scene = abs(float(grid.mean()) - float(gate_ref.mean())) >= _SCENE_THRESHOLD
            else:
                delta, scene = float("inf"), True
            skip = (not scene) and delta <= gate_threshold and gate_chain < gate_max_chain
            if skip:
                gate_chain += 1
                gated += 1
                continue
            gate_ref, gate_chain = grid, 0

            if tile_ref is None or tile_ref.shape != grid.shape:
                tile_ref = grid
                tile_chains = [0] * len(windows)
                planned += len(windows)
                continue
            changed = 0
            for i, (by0, by1, bx0, bx1) in enumerate(windows):
                # mean of the ABSOLUTE block deltas, which is what TiledUpscaler._tile_deltas
                # computes. Using |mean(a) - mean(b)| instead lets local changes cancel
                # against each other and predicted 38% reuse where the runtime then achieved
                # 9.5% (measured on the moving film clip).
                moved = float(np.abs(grid[by0:by1, bx0:bx1]
                                     - tile_ref[by0:by1, bx0:bx1]).mean())
                if moved > tile_threshold or tile_chains[i] >= tile_max_chain:
                    changed += 1
                    tile_chains[i] = 0
                else:
                    tile_chains[i] += 1
            planned += len(windows)
            reused += len(windows) - changed
            tile_ref = grid

    share = (reused / planned) if planned else 0.0
    break_even = 1.0 - 1.0 / tax if tax > 0 else 1.0
    predicted = (1.0 - share) * tax
    # Equality is not a safe decision: the probe reads a 16-frame sample, and on the moving
    # film clip it predicted 15% reuse where the render then achieved 9.5%. Requiring the
    # saving to be at least as large as the halo tax (cost <= 1/tax) puts a one-tax margin
    # between "wins" and "loses"; with it, anime (33% predicted, 58% actual) stays on and
    # film (15% predicted, 9.5% actual) goes off, which is what measurement says is right.
    required = 1.0 / tax if tax > 0 else 1.0
    use = planned > 0 and predicted <= required
    if not planned:
        reason = "the gate rendered too few sampled frames to judge tile reuse"
    elif use:
        reason = (f"tiles would cost {predicted:.2f} against a bar of {required:.2f}: "
                  f"{share * 100:.0f}% of tiles reused")
    else:
        reason = (f"tiles would cost {predicted:.2f} against a bar of {required:.2f} "
                  f"({share * 100:.0f}% of tiles reused): the halo eats the saving, so the "
                  f"full-frame path is cheaper")
    return ProbeResult(
        frames_sampled=sampled, frames_gated=gated, tiles_planned=planned,
        tiles_reused=reused, area_tax=tax, tile_reuse=share, break_even=break_even,
        predicted_cost=predicted, required_cost=required, use_tiles=use, reason=reason)


def sample_luma(path: str, *, width: int, height: int, positions: int = 4,
                per_position: int = 4, ffmpeg_path: str = "ffmpeg",
                expected_frames: Optional[int] = None,
                runner: Optional[Callable[[List[str], int], bytes]] = None) -> List[List[object]]:
    """Decode ``per_position`` consecutive 8-bit luma frames at ``positions`` seek points.

    Returns one run per position. Decoding 8-bit luma rather than the full 10-bit frame is
    enough: the statistics are block means of luma compared against thresholds expressed in
    8-bit levels, and the extra two bits of a 10-bit source do not move a 32 px mean across a
    5-level threshold.

    ``runner`` takes ``(argv, expected_size)`` and returns stdout bytes; the default spawns
    ffmpeg. It exists so the seek/parse logic can be tested without an encoder.
    """
    _require_numpy()
    if positions < 1 or per_position < 1:
        raise ValueError("positions and per_position must be >= 1")
    frame_bytes = width * height

    def default_runner(argv: List[str], expected: int) -> bytes:
        proc = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if len(proc.stdout) < expected:
            return proc.stdout
        return proc.stdout[:expected]

    run_fn = runner or default_runner
    starts = _seek_points(expected_frames, positions, per_position)
    runs: List[List[object]] = []
    for start in starts:
        argv = [ffmpeg_path, "-v", "error", "-nostdin"]
        if start:
            argv += ["-ss", f"{start:.3f}"]
        argv += ["-i", path, "-frames:v", str(per_position),
                 "-pix_fmt", "gray", "-f", "rawvideo", "-"]
        raw = run_fn(argv, frame_bytes * per_position)
        run: List[object] = []
        for i in range(per_position):
            chunk = raw[i * frame_bytes:(i + 1) * frame_bytes]
            if len(chunk) < frame_bytes:
                break
            run.append(np.frombuffer(chunk, dtype=np.uint8)
                       .reshape(height, width).astype(np.float64))
        if run:
            runs.append(run)
    return runs


def _seek_points(expected_frames: Optional[int], positions: int,
                 per_position: int) -> List[float]:
    """Frame indices to seek to, spread so that no single scene decides the verdict.

    Sampling only the opening is exactly the mistake the auto gate tuner made (a busy first
    scene measured a "noise floor" of 74 levels against 0.2 for the rest of the clip). When
    the frame count is unknown the caller gets an even spread of relative positions instead,
    which ffmpeg accepts as percentages of the duration.
    """
    if expected_frames is None or expected_frames <= 0:
        return [0.0] + [(i + 1) / (positions + 1) for i in range(positions - 1)]
    points: List[float] = []
    for i in range(positions):
        index = int(expected_frames * (i + 0.5) / positions)
        index = max(0, min(index, max(0, expected_frames - per_position)))
        if not points or index > points[-1]:
            points.append(float(index))
    return points or [0.0]


def probe(path: str, *, width: int, height: int, tile_size: int, halo: int,
          ffmpeg_path: str = "ffmpeg", positions: int = 4, per_position: int = 4,
          gate_threshold: float = 5.0, tile_threshold: float = 0.35, block: int = 32,
          gate_max_chain: int = 4, tile_max_chain: int = 2,
          expected_frames: Optional[int] = None,
          runner: Optional[Callable[[List[str], int], bytes]] = None) -> ProbeResult:
    """Sample the source and decide. The tile size is fitted the way the runtime fits it."""
    # divisor_tile lives with the tile geometry. That module guards its torch import, so this
    # stays importable (and testable) where torch is not installed.
    from ..pytorch.TiledUpscale import divisor_tile

    tile_h = divisor_tile(height, tile_size)
    tile_w = divisor_tile(width, tile_size)
    runs = sample_luma(path, width=width, height=height, positions=positions,
                       per_position=per_position, ffmpeg_path=ffmpeg_path,
                       expected_frames=expected_frames, runner=runner)
    if not runs:
        tax = area_tax(tile_h, tile_w, height, width, halo)
        return ProbeResult(0, 0, 0, 0, tax, 0.0, 1.0 - 1.0 / tax, 1.0, 1.0 / tax,
                           False, "no frames could be decoded from the source")
    return predict(runs, tile_h=tile_h, tile_w=tile_w, halo=halo,
                   gate_threshold=gate_threshold, tile_threshold=tile_threshold,
                   block=block, gate_max_chain=gate_max_chain,
                   tile_max_chain=tile_max_chain)
