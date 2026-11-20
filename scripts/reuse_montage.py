"""A montage of 1:1 crops from several renders of the same frame, stacked, for the eye.

The worst-damage 64 px block is located on the first two videos (the pair whose difference is the artifact),
and every video contributes the same 256x256 region around it, so the panels differ only by construction.

Usage:
  python3 scripts/reuse_montage.py --videos a.mkv,b.mkv,... --frame 122 --output m.png [--size 256]
Bands are drawn in the order given; the caller knows which is which.
"""
from __future__ import annotations

import argparse
import subprocess

import numpy as np

FF = "ffmpeg"


def frame_at(path: str, idx: int, w: int, h: int) -> np.ndarray:
    proc = subprocess.run([FF, "-v", "error", "-i", path, "-vf", f"select=eq(n\\,{idx})",
                           "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
                          capture_output=True)
    buf = proc.stdout
    if len(buf) < w * h * 3:
        raise SystemExit(f"{path}: no frame {idx}")
    return np.frombuffer(buf[:w * h * 3], dtype=np.uint8).reshape(h, w, 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--width", type=int, default=3840)
    ap.add_argument("--height", type=int, default=2160)
    ap.add_argument("--xy", default="", help="explicit X,Y centre instead of locating the worst block")
    ap.add_argument("--diff-x", type=float, default=0.0,
                    help="after each panel, also draw |panel - first panel| x this factor (the difference is "
                         "what the pixel statistic counts; the crop alone rarely shows it at chat scale)")
    a = ap.parse_args()

    paths = [p.strip() for p in a.videos.split(",")]
    frames = [frame_at(p, a.frame, a.width, a.height) for p in paths]
    if a.xy:
        cx, cy = (int(v) for v in a.xy.split(","))
    else:
        d = np.abs(frames[0].astype(np.int16) - frames[1].astype(np.int16)).max(axis=2)
        B = 64
        hb, wb = d.shape[0] // B, d.shape[1] // B
        blocks = d[:hb * B, :wb * B].reshape(hb, B, wb, B).max(axis=(1, 3))
        y, x = np.unravel_index(int(np.argmax(blocks)), blocks.shape)
        cy, cx = y * B + B // 2, x * B + B // 2
    half = a.size // 2
    x0 = min(max(cx - half, 0), a.width - a.size)
    y0 = min(max(cy - half, 0), a.height - a.size)
    crops = [f[y0:y0 + a.size, x0:x0 + a.size] for f in frames]
    gap = np.full((6, a.size, 3), 255, dtype=np.uint8)
    stack = []
    base = crops[0].astype(np.int16)
    for c in crops:
        stack.append(c)
        if a.diff_x:
            d = np.clip(np.abs(c.astype(np.int16) - base) * a.diff_x, 0, 255).astype(np.uint8)
            stack.append(d)
        stack.append(gap)
    img = np.vstack(stack[:-1])
    subprocess.run([FF, "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{img.shape[1]}x{img.shape[0]}", "-i", "-", "-frames:v", "1",
                    a.output, "-y"], input=img.tobytes(), check=True)
    print(f"{a.output}: {len(crops)} panels of {a.size}px around ({cx},{cy}) at frame {a.frame}, "
          f"order {', '.join(p.split('/')[-1] for p in paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
