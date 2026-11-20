"""Prototype step 1: a motion-rejected temporal filter on the LR source, written back as a lossless clip.

Why: verbatim skip cannot be safe on noisy material because the reused frame carries the reference's noise
(docs/wsp/verified/reuse-decision-statistic.txt: 119 of 130 reuses frozen, 82 of them passing the 0.1% bar). The
review's architecture separates the plate from the grain before any reuse happens, so the gate decides on clean
content and the grain is re-added per frame afterwards.

The filter is a 3-tap bilateral in time: neighbour weight exp(-|I_t - I_neighbour| / tau), normalised against
the current frame. At grain the differences are small, so both neighbours count and the noise drops by ~sqrt(3);
at motion the differences are large, so the weights vanish and the current frame is kept - that is the
"motion-rejected" part, and it is why a still drawing is not smeared.

Usage:
  python3 scripts/reuse_denoise.py --input clip.mkv --output clean.mkv [--tau 4] [--limit N]

NOT checked: whether the filter's residual carries motion that the plate then freezes, and the cost against
the 26-30 ms per-frame body budget (this prints its own wall clock).
"""
from __future__ import annotations

import argparse
import subprocess
import time

import numpy as np

FF = "ffmpeg"


class RGB:
    def __init__(self, path: str, w: int, h: int):
        self.w, self.h, self.size = w, h, w * h * 3
        self.proc = subprocess.Popen([FF, "-v", "error", "-i", path, "-f", "rawvideo",
                                      "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)

    def read(self):
        buf = self.proc.stdout.read(self.size)
        if not buf or len(buf) < self.size:
            return None
        return np.frombuffer(buf, dtype=np.uint8).reshape(self.h, self.w, 3)

    def close(self):
        self.proc.stdout.close()
        self.proc.terminate()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tau", type=float, default=4.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    a = ap.parse_args()

    src = RGB(a.input, a.width, a.height)
    enc = subprocess.Popen([FF, "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{a.width}x{a.height}", "-r", "24000/1001", "-i", "-",
                            "-c:v", "ffv1", "-level", "3", "-pix_fmt", "yuv444p",
                            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
                            a.output, "-y"], stdin=subprocess.PIPE)

    prev = src.read()
    cur = src.read()
    nxt = src.read()
    frames, t0 = 0, time.time()
    while cur is not None:
        f_cur = cur.astype(np.float32)
        acc = f_cur.copy()
        wsum = np.ones(f_cur.shape[:2], dtype=np.float32)
        for nb in (prev, nxt):
            if nb is None:
                continue
            d = np.abs(f_cur - nb.astype(np.float32)).max(axis=2)
            w = np.exp(-d / a.tau).astype(np.float32)
            acc += w[..., None] * nb.astype(np.float32)
            wsum += w
        out = np.clip(acc / wsum[..., None], 0, 255).astype(np.uint8)
        enc.stdin.write(out.tobytes())
        frames += 1
        if a.limit and frames >= a.limit:
            break
        prev, cur, nxt = cur, nxt, src.read()

    src.close()
    enc.stdin.close()
    enc.wait()
    dt = time.time() - t0
    print(f"{a.output}: {frames} frames in {dt:.1f} s = {dt / max(frames, 1) * 1000:.1f} ms/frame "
          f"(tau={a.tau})")
    print("NOT checked: motion the filter keeps in the plate, and the same clip at 4K")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
