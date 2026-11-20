"""Where the error sits INSIDE a frame: periodicity, and the shape of the worst block.

Two questions the share statistic cannot answer, both asked by the owner in front of the images:

  1. Is the difference block-periodic? A pitch at 8/16/32/64/128 px names a mechanism (chroma layout,
     DCT-size block processing, kernel tile, upscaler tile) and a pitch at none of them rules all of
     those out. The test is the mean |delta| on the rows/columns that fall on the pitch against the
     mean on the rest - a ratio of 1.0 is no structure.
  2. What shape is the worst 64x64 block? A thin saturated line and a soft wash can have the same block
     mean; the block mean alone cannot tell them apart, and the owner's rejection was of a shape.

Usage:
  python harness/scripts/diff_structure.py A.mkv B.mkv FRAME [--block 64]
"""
from __future__ import annotations

import argparse
import subprocess

import numpy as np

FF = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2/bin/ffmpeg"
W, H = 3840, 2160
SIZE = W * H * 3
BAR = 8


def frame_at(path: str, idx: int) -> np.ndarray:
    raw = subprocess.run([FF, "-loglevel", "error", "-i", path, "-vf", f"select=eq(n\\,{idx})",
                          "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
                         capture_output=True).stdout
    if len(raw) < SIZE:
        raise SystemExit(f"no frame {idx} in {path}")
    return np.frombuffer(raw[:SIZE], dtype=np.uint8).reshape(H, W, 3)


def block_map(d: np.ndarray, by: int, bx: int, block: int, rows: int = 16, cols: int = 32) -> str:
    sub = d[by:by + block, bx:bx + block]
    ch, cw = block // rows, block // cols
    chars = " .:-=+*#%@"
    out = []
    for r in range(rows):
        line = ""
        for c in range(cols):
            v = sub[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw].mean()
            line += chars[min(len(chars) - 1, int(v / 140 * (len(chars) - 1)))]
        out.append("    |" + line + "|")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("frame", type=int)
    ap.add_argument("--block", type=int, default=64)
    args = ap.parse_args(argv)

    a, b = frame_at(args.a, args.frame), frame_at(args.b, args.frame)
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2).astype(np.float32)

    print(f"frame {args.frame}: share {(d > BAR).mean() * 100:.5f}%  peak {int(d.max())}  "
          f"mean |d| {d.mean():.3f}")

    print("\nperiodicity (mean |d| on the pitch against the rest):")
    for p in (4, 8, 16, 32, 64, 128, 256):
        rows = np.where(np.arange(d.shape[0]) % p == 0)[0]
        cols = np.where(np.arange(d.shape[1]) % p == 0)[0]
        on = d[rows, :].mean() if len(rows) else 0.0
        off = np.delete(d, rows, axis=0).mean() if len(rows) < d.shape[0] else 0.0
        onc = d[:, cols].mean() if len(cols) else 0.0
        offc = np.delete(d, cols, axis=1).mean() if len(cols) < d.shape[1] else 0.0
        print(f"  pitch {p:>3}: rows {on / max(off, 1e-9):.3f}x   cols {onc / max(offc, 1e-9):.3f}x")

    # Does the error track local contrast? A per-tensor activation scale clips where the signal is
    # large, so |d| should rise with the reference's gradient if that is the mechanism - and stay flat
    # if the error is a global quantisation floor.
    grey = b.astype(np.float32).mean(axis=2)
    gx = np.abs(np.diff(grey, axis=1))          # (H, W-1)
    gy = np.abs(np.diff(grey, axis=0))          # (H-1, W)
    g = gx[:-1, :] + gy[:, :-1]                # (H-1, W-1), both terms on the same interior pixels
    dg = d[:-1, :-1]
    edges = np.percentile(g, [50, 80, 90, 95, 99])
    print("\n|d| against the reference's local gradient:")
    prev = -1.0
    for i, hi in enumerate(list(edges) + [float("inf")]):
        m = (g > prev) & (g <= hi)
        if m.any():
            print(f"  gradient {prev:>6.1f}..{hi if hi != float('inf') else -1:>6.1f}: "
                  f"mean |d| {dg[m].mean():6.2f}  share>8 {(dg[m] > BAR).mean() * 100:6.3f}%  "
                  f"pixels {m.mean() * 100:5.1f}%")
        prev = hi

    print("\n|d| against the reference's brightness (clipping test):")
    for lo, hi in ((0, 32), (32, 64), (64, 128), (128, 192), (192, 234), (234, 256)):
        m = (grey >= lo) & (grey < hi)
        if m.any():
            print(f"  level {lo:>3}..{hi:>3}: mean |d| {d[m].mean():6.2f}  "
                  f"share>8 {(d[m] > BAR).mean() * 100:6.3f}%  pixels {m.mean() * 100:5.1f}%")

    print("\nper channel:")
    for ci, name in enumerate("RGB"):
        dc = np.abs(a[:, :, ci].astype(np.int16) - b[:, :, ci].astype(np.int16))
        print(f"  {name}: mean |d| {dc.mean():5.2f}  share>8 {(dc > BAR).mean() * 100:6.3f}%  "
              f"peak {int(dc.max())}")

    hh, ww = (H // args.block) * args.block, (W // args.block) * args.block
    s = d[:hh, :ww].reshape(hh // args.block, args.block, ww // args.block, args.block).mean(axis=(1, 3))
    by, bx = np.unravel_index(int(s.argmax()), s.shape)
    by, bx = by * args.block, bx * args.block
    sub = d[by:by + args.block, bx:bx + args.block]
    bad = (sub > BAR)
    print(f"\nworst {args.block}px block at ({bx},{by}): mean {sub.mean():.2f}  "
          f"pixels >8: {int(bad.sum())}/{bad.size} = {bad.mean() * 100:.1f}%  max {int(sub.max())}")
    if bad.any():
        ys, xs = np.nonzero(bad)
        print(f"  the flagged pixels span {xs.max() - xs.min() + 1} x {ys.max() - ys.min() + 1} px "
              f"inside the block, filling {bad.mean() * 100:.1f}% of it")
    print("  shape (rows x cols, ' '=0  '@'=140 levels):")
    print(block_map(d, by, bx, args.block))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
