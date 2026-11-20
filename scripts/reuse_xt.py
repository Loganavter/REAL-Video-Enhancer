"""An x-t strip: one column of pixels, stacked across frames, so texture motion becomes visible.

The reviewer's visual standard. A frozen texture draws as horizontal bands (the same pixels repeated), live
grain draws as sand. One image answers "does this look dead" without a metric.

Usage:
  python3 scripts/reuse_xt.py --videos a.mkv,b.mkv --rect X,Y,W,H --frames 110:150 --output strip.png
Writes one horizontal band per video, in the order given, each labelled-free (the caller knows the order).
"""
from __future__ import annotations

import argparse
import subprocess

import numpy as np

FF = "ffmpeg"


def band(path: str, rect, start: int, end: int) -> np.ndarray:
    x, y, w, h = rect
    # crop, then keep only the frames we want; the crop keeps the pixel geometry 1:1
    proc = subprocess.Popen([FF, "-v", "error", "-i", path,
                             "-vf", f"crop={w}:{h}:{x}:{y}", "-pix_fmt", "gray",
                             "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    size = w * h
    out, i = [], 0
    while True:
        buf = proc.stdout.read(size)
        if not buf or len(buf) < size:
            break
        if start <= i < end:
            out.append(np.frombuffer(buf, dtype=np.uint8).reshape(h, w).mean(axis=0))
        i += 1
    proc.stdout.close()
    proc.terminate()
    return np.stack(out, axis=1).astype(np.uint8)      # x across time: (h, n_frames)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="comma-separated, in the order the bands are drawn")
    ap.add_argument("--rect", required=True, help="X,Y,W,H in the video's own pixels")
    ap.add_argument("--frames", required=True, help="start:end")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    rect = tuple(int(v) for v in a.rect.split(","))
    s, e = (int(v) for v in a.frames.split(":"))
    bands = [band(p.strip(), rect, s, e) for p in a.videos.split(",")]
    gap = np.full((4, bands[0].shape[1]), 255, dtype=np.uint8)
    stack = []
    for b in bands:
        stack += [b, gap]
    img = np.vstack(stack[:-1])
    # write PNG through ffmpeg (no PIL dependency in this interpreter)
    subprocess.run([FF, "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray",
                    "-s", f"{img.shape[1]}x{img.shape[0]}", "-i", "-",
                    "-frames:v", "1", a.output, "-y"], input=img.tobytes(), check=True)
    print(f"{a.output}: {img.shape[0]} rows x {img.shape[1]} columns, {len(bands)} bands "
          f"({', '.join(p.strip().split('/')[-1] for p in a.videos.split(','))})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
