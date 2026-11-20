"""Prototype step 2: put the grain back on every frame, from the source, at the output resolution.

The architecture in docs/wsp/REVIEW-reuse-gate-2026-11-19-REPLY.md: the plate is what the gate and the network
see, and the grain is re-added afterwards on ALL frames - rendered and reused alike - otherwise the reused
frames carry dead grain while the rendered ones carry fresh grain and the difference pulses at the held-frame
rate.

grain_t = LR_t - LR_clean_t   (the filter's own residual, i.e. exactly what was removed)
out_t   = render_t + nearest-2x(grain_t)

Usage:
  python3 harness/scripts/reuse_regrain.py --render hr.mkv --lr clip.mkv --clean plate.mkv --output out.mkv
"""
from __future__ import annotations

import argparse
import subprocess
import time

import numpy as np

FF = "ffmpeg"


class Frames:
    def __init__(self, path: str, w: int, h: int, pix: str):
        self.w, self.h, self.ch = w, h, 1 if pix == "gray" else 3
        self.size = w * h * self.ch
        vf = f"scale={w}:{h}" + (",format=gray" if self.ch == 1 else "")
        self.proc = subprocess.Popen([FF, "-v", "error", "-i", path, "-vf", vf,
                                      "-pix_fmt", pix,
                                      "-f", "rawvideo", "-"], stdout=subprocess.PIPE)

    def read(self):
        buf = self.proc.stdout.read(self.size)
        if not buf or len(buf) < self.size:
            return None
        a = np.frombuffer(buf, dtype=np.uint8)
        return a.reshape(self.h, self.w) if self.ch == 1 else a.reshape(self.h, self.w, 3)

    def close(self):
        self.proc.stdout.close()
        self.proc.terminate()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", required=True)
    ap.add_argument("--lr", required=True)
    ap.add_argument("--clean", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    a = ap.parse_args()

    w, h = a.width, a.height
    lr = Frames(a.lr, w, h, "rgb24")
    cl = Frames(a.clean, w, h, "rgb24")
    rd = Frames(a.render, w * 2, h * 2, "rgb24")
    enc = subprocess.Popen([FF, "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{w * 2}x{h * 2}", "-r", "24000/1001", "-i", "-",
                            "-c:v", "ffv1", "-pix_fmt", "yuv444p", a.output, "-y"],
                           stdin=subprocess.PIPE)
    frames, t0, gain = 0, time.time(), []
    while True:
        f_lr, f_cl, f_rd = lr.read(), cl.read(), rd.read()
        if f_lr is None or f_cl is None or f_rd is None:
            break
        grain = f_lr.astype(np.float32) - f_cl.astype(np.float32)
        up = np.repeat(np.repeat(grain, 2, axis=0), 2, axis=1)
        out = np.clip(f_rd.astype(np.float32) + up, 0, 255).astype(np.uint8)
        enc.stdin.write(out.tobytes())
        gain.append(float(np.abs(grain).mean()))
        frames += 1
    for f in (lr, cl, rd):
        f.close()
    enc.stdin.close()
    enc.wait()
    dt = time.time() - t0
    print(f"{a.output}: {frames} frames in {dt:.1f} s = {dt / max(frames, 1) * 1000:.1f} ms/frame, "
          f"mean |grain| {np.mean(gain):.2f} levels")
    print("NOT checked: the grain's spectrum at 4K (it is LR-band nearest-2x), and the cost of an upsample "
          "in a real pipeline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
