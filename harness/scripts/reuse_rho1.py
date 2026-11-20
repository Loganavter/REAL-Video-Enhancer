"""Lag-1 temporal autocorrelation of the high-frequency residual: does the texture still move?

Why this and not the ratio in reuse_texture.py: the ratio needs an ungated render as the definition of
"fresh", so it can only be used for judging a measurement, never at runtime. Reviewer's suggestion: on flat
static blocks, the lag-1 autocorrelation rho1 of the high-frequency residual is ~0 for live grain and ->1 for
frozen grain, and it needs no reference at all.

Optionally measures the residual-reuse construction instead of a video: out = video + bilinear(LR_t - LR_ref),
which is what "reuse with a correction" would emit if the correction is the upscaled LR difference (the
algebra in harness/docs/wsp/REVIEW-reuse-gate-2026-11-19.md, reviewer's step 1).

Usage:
  python3 harness/scripts/reuse_rho1.py --video on.mkv --lr clip.mkv [--radd --depth-log gate.log]
      [--block 64] [--hf 5] [--emit out.json]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess

import numpy as np

FF = "ffmpeg"


class Gray:
    """Grayscale frames through ffmpeg at a fixed size."""

    def __init__(self, path: str, w: int, h: int):
        self.w, self.h, self.size = w, h, w * h
        self.proc = subprocess.Popen([FF, "-v", "error", "-i", path, "-vf",
                                      f"scale={w}:{h},format=gray", "-f", "rawvideo", "-"],
                                     stdout=subprocess.PIPE)

    def read(self):
        buf = self.proc.stdout.read(self.size)
        if not buf or len(buf) < self.size:
            return None
        return np.frombuffer(buf, dtype=np.uint8).reshape(self.h, self.w).astype(np.float32)

    def close(self):
        self.proc.stdout.close()
        self.proc.terminate()


def hf(img: np.ndarray, k: int) -> np.ndarray:
    """High-frequency residual: the frame minus a box-blurred copy of itself (grain and line detail)."""
    c = np.cumsum(np.cumsum(img, axis=0), axis=1)
    pad = np.pad(c, ((1, 0), (1, 0)))
    h, w = img.shape
    r = k // 2
    y0, y1 = np.clip(np.arange(h) - r, 0, h), np.clip(np.arange(h) + r + 1, 0, h)
    x0, x1 = np.clip(np.arange(w) - r, 0, w), np.clip(np.arange(w) + r + 1, 0, w)
    s = (pad[y1][:, x1] - pad[y0][:, x1] - pad[y1][:, x0] + pad[y0][:, x0])
    n = (y1 - y0)[:, None] * (x1 - x0)[None, :]
    return img - s / n


def blocks(img: np.ndarray, b: int) -> np.ndarray:
    h, w = img.shape
    hb, wb = h // b, w // b
    return img[:hb * b, :wb * b].reshape(hb, b, wb, b)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--lr", required=True)
    ap.add_argument("--depth-log", default="")
    ap.add_argument("--radd", action="store_true", help="measure video + bilinear(LR_t - LR_ref)")
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--hf", type=int, default=5)
    ap.add_argument("--flat-quantile", type=float, default=0.3)
    ap.add_argument("--remove-temporal-mean", action="store_true",
                    help="correlate only the FLUCTUATING part of the high-frequency residual: on this "
                         "material the raw residual is dominated by static detail and reads rho1~0.94 even "
                         "for fresh frames")
    ap.add_argument("--emit", default="")
    a = ap.parse_args()

    depth = {}
    if a.depth_log:
        for line in open(a.depth_log, errors="ignore"):
            m = re.search(r"reuse frame=(\d+) depth=(\d+)", line)
            if m:
                depth[int(m.group(1))] = int(m.group(2))

    mean_hf = None
    if a.remove_temporal_mean:
        m = Gray(a.video, 1920, 1080)
        n = 0
        while True:
            f = m.read()
            if f is None:
                break
            mean_hf = hf(f, a.hf) if mean_hf is None else mean_hf + hf(f, a.hf)
            n += 1
        m.close()
        mean_hf /= max(n, 1)

    v = Gray(a.video, 1920, 1080)
    l = Gray(a.lr, 1920, 1080)
    prev_v = v.read()
    lr_frames = [l.read()]
    rows, i = [], 0
    while True:
        cur_v = v.read()
        lr_t = l.read()
        if cur_v is None or lr_t is None:
            break
        i += 1
        lr_frames.append(lr_t)
        k = depth.get(i)          # None = rendered, so "reused" below is not a truthiness accident
        kk = k or 1
        ref = lr_frames[max(0, len(lr_frames) - 1 - kk)]
        out = cur_v
        if a.radd:
            # the correction an LR-residual reuse would add, at the output resolution
            out = cur_v + (lr_t - ref)
        h_now, h_prev = hf(out, a.hf), hf(prev_v if not a.radd else out, a.hf)
        if mean_hf is not None:
            h_now, h_prev = h_now - mean_hf, h_prev - mean_hf
        if a.radd:
            # previous output frame must be reconstructed the same way to compare like with like
            ref_prev = lr_frames[max(0, len(lr_frames) - 2 - kk)]
            prev_out = prev_v + (lr_frames[-2] - ref_prev)
            h_prev = hf(prev_out, a.hf)
            if mean_hf is not None:
                h_prev = h_prev - mean_hf
        # static + flat blocks, defined on the LR source so the criterion needs no reference render
        flat = blocks(np.abs(np.diff(lr_t, axis=1, append=lr_t[:, -1:])), a.block).mean(axis=(1, 3))
        motion = blocks(np.abs(lr_t - ref), a.block).mean(axis=(1, 3))
        sel = (flat <= np.quantile(flat, a.flat_quantile)) & (motion <= np.quantile(motion, a.flat_quantile))
        bn = blocks(h_now, a.block)
        bp = blocks(h_prev, a.block)
        vals = [pearson(bn[y][x], bp[y][x]) for y, x in zip(*np.where(sel))]
        rows.append({"frame": i, "depth": k, "blocks": len(vals),
                     "rho1": float(np.median(vals)) if len(vals) >= 8 else float("nan")})
        prev_v = cur_v
    v.close(); l.close()

    ok = [r["rho1"] for r in rows if r["rho1"] == r["rho1"]]
    reused = [r["rho1"] for r in rows if r["rho1"] == r["rho1"] and r["depth"]]
    fresh = [r["rho1"] for r in rows if r["rho1"] == r["rho1"] and not r["depth"]]
    out = {"frames": len(rows), "mode": "video+radd" if a.radd else "video",
           "rho1_all": round(float(np.median(ok)), 3) if ok else None,
           "rho1_reused": round(float(np.median(reused)), 3) if reused else None,
           "rho1_fresh": round(float(np.median(fresh)), 3) if fresh else None,
           "n_reused": len(reused), "n_fresh": len(fresh)}
    print(json.dumps(out, indent=1))
    if a.emit:
        json.dump({"summary": out, "rows": rows}, open(a.emit, "w"))
        print("wrote", a.emit)
    print("NOT checked: the perceptual threshold on rho1, transitions (a reuse starts on a fresh frame, so "
          "rho1 there is not frozen by construction), and other material")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
