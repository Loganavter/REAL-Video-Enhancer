"""The composed correction: warp where the flow is trustworthy, residual elsewhere, grain on top.

Two measured facts drive this composition. (1) A non-warped residual fixes the structural class on the calm clip
(0/130 over the bar) but cannot fix coherent motion (the held-out clip keeps 4/124). (2) The warped version fixes
the motion but inherits the flow's failures (6/130 on the calm clip). So gate the warp by forward-backward flow
consistency and fall back to the non-warped residual where the two flows disagree; then add the grain half at
alpha on flat static blocks, which the frequency split showed is free at the bar.

  confident: |u + warp(u_back, u)| <= fb_tol           (per LR pixel, blurred into a weight)
  geometry  = w * [warp(SR_ref, 2u) + up(r_warp)] + (1 - w) * [SR_ref + up(r)]
  out       = geometry + up(alpha_g * highpass(r))     alpha_g = alpha only where flat and |r| small

Usage:
  python3 scripts/reuse_compose.py --gated on.mkv --ungated off.mkv --lr clip.mkv --depth-log gate.log
      [--alpha 1.35] [--mask-tau 3] [--mask-grad 5] [--fb-tol 1.0] [--output out.mkv] [--emit f.json]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time

import cv2
import numpy as np

FF = "ffmpeg"


class Frames:
    def __init__(self, path, w, h, pix):
        self.w, self.h, self.pix = w, h, pix
        self.size = w * h * (1 if pix == "gray" else 3)
        self.proc = subprocess.Popen([FF, "-v", "error", "-i", path, "-vf", f"scale={w}:{h}",
                                      "-pix_fmt", pix, "-f", "rawvideo", "-"], stdout=subprocess.PIPE)

    def read(self):
        b = self.proc.stdout.read(self.size)
        if not b or len(b) < self.size:
            return None
        a = np.frombuffer(b, dtype=np.uint8)
        return a.reshape(self.h, self.w).astype(np.float32) if self.size == self.w * self.h \
            else a.reshape(self.h, self.w, 3).astype(np.float32)

    def close(self):
        self.proc.stdout.close()
        self.proc.terminate()


def box2(img: np.ndarray, k: int) -> np.ndarray:
    r = k // 2
    pad = np.pad(img, ((r, r), (r, r)) if img.ndim == 2 else ((r, r), (r, r), (0, 0)), mode="reflect")
    c = np.cumsum(np.cumsum(pad, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)) if img.ndim == 2 else ((1, 0), (1, 0), (0, 0)))
    s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return s / (k * k)


def blockmean(img, b):
    h, w = img.shape
    hb, wb = h // b, w // b
    return img[:hb * b, :wb * b].reshape(hb, b, wb, b).mean(axis=(1, 3))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gated", required=True)
    ap.add_argument("--ungated", required=True)
    ap.add_argument("--lr", required=True)
    ap.add_argument("--depth-log", required=True)
    ap.add_argument("--alpha", type=float, default=1.35)
    ap.add_argument("--mask-tau", type=float, default=3.0)
    ap.add_argument("--mask-grad", type=float, default=5.0)
    ap.add_argument("--fb-tol", type=float, default=1.0)
    ap.add_argument("--blur", type=int, default=5)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--output", default="")
    ap.add_argument("--emit", default="")
    a = ap.parse_args()

    depth = {}
    for line in open(a.depth_log, errors="ignore"):
        m = re.search(r"reuse frame=(\d+) depth=(\d+)", line)
        if m:
            depth[int(m.group(1))] = int(m.group(2))

    lw, lh = 1920, 1080
    lr = Frames(a.lr, lw, lh, "rgb24")
    g = Frames(a.gated, lw * 2, lh * 2, "rgb24")
    u = Frames(a.ungated, lw * 2, lh * 2, "rgb24")
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    gx2, gy2 = np.meshgrid(np.arange(lw * 2, dtype=np.float32), np.arange(lh * 2, dtype=np.float32))
    gx, gy = np.meshgrid(np.arange(lw, dtype=np.float32), np.arange(lh, dtype=np.float32))
    enc = None
    if a.output:
        enc = subprocess.Popen([FF, "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                "-s", f"{lw * 2}x{lh * 2}", "-r", "24000/1001", "-i", "-",
                                "-c:v", "ffv1", "-pix_fmt", "yuv444p", a.output, "-y"],
                               stdin=subprocess.PIPE)

    hist, rows, prev_out, prev_fresh, i = [], [], None, None, 0
    t0 = time.time()
    while True:
        f_lr, f_g, f_u = lr.read(), g.read(), u.read()
        if f_lr is None or f_g is None or f_u is None:
            break
        hist.append(f_lr)
        k = depth.get(i)
        if k:
            ref = hist[max(0, len(hist) - 1 - k)]
            ref8, cur8 = ref.astype(np.uint8), f_lr.astype(np.uint8)
            grey_ref = cv2.cvtColor(ref8, cv2.COLOR_RGB2GRAY)
            grey_cur = cv2.cvtColor(cur8, cv2.COLOR_RGB2GRAY)
            uf = flow.calc(grey_ref, grey_cur, None)
            ub = flow.calc(grey_cur, grey_ref, None)
            ub_w = cv2.remap(ub, gx + uf[..., 0], gy + uf[..., 1], interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
            fb = np.sqrt(((uf + ub_w) ** 2).sum(axis=2))
            w_conf = box2((fb <= a.fb_tol).astype(np.float32), 9)                 # soft, seam-free
            r = f_lr - ref
            # warped path
            warp_ref = cv2.remap(ref, gx + uf[..., 0], gy + uf[..., 1], interpolation=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
            r_warp = f_lr - warp_ref
            uf_hr = cv2.resize(uf, (lw * 2, lh * 2), interpolation=cv2.INTER_LINEAR) * 2.0
            warp_sr = cv2.remap(f_g, gx2 + uf_hr[..., 0], gy2 + uf_hr[..., 1],
                                interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            hi_w = np.repeat(np.repeat(r_warp, 2, axis=0), 2, axis=1)
            geo_w = warp_sr + hi_w
            # unwarped path
            geo_u = f_g + np.repeat(np.repeat(r, 2, axis=0), 2, axis=1)
            w_hr = cv2.resize(w_conf, (lw * 2, lh * 2), interpolation=cv2.INTER_LINEAR)
            wc = w_hr[..., None] if geo_w.ndim == 3 else w_hr
            geo = wc * geo_w + (1.0 - wc) * geo_u
            # grain half, flat and low-residual only
            lf = box2(r, a.blur)
            hf = r - lf
            gmag = np.abs(np.diff(f_lr.mean(axis=2), axis=0, append=f_lr.mean(axis=2)[-1:, :])) + \
                np.abs(np.diff(f_lr.mean(axis=2), axis=1, append=f_lr.mean(axis=2)[:, -1:]))
            keep = (box2(np.abs(r), 5).mean(axis=2) < a.mask_tau) & (gmag < a.mask_grad)
            # the geometry path already carries the full residual (r or r_warp), so only the EXTRA grain
            # beyond unit gain is added here - adding corr again would count r twice (measured: 2.9% error)
            gain = np.zeros(hf.shape[:2], dtype=np.float32)
            gain[keep] = a.alpha - 1.0
            corr = (gain[..., None] if hf.ndim == 3 else gain) * hf
            out = np.clip(geo + np.repeat(np.repeat(corr, 2, axis=0), 2, axis=1), 0, 255)
        else:
            out = f_g
        if enc is not None:
            enc.stdin.write(np.clip(out, 0, 255).astype(np.uint8).tobytes())
        if k:
            d = np.abs(out - f_u).max(axis=2)
            share = float((d > 8).mean()) * 100.0
        else:
            share = 0.0
        ratio = None
        if k and prev_out is not None:
            flat = blockmean(np.abs(np.diff(f_lr.mean(axis=2), axis=1, append=f_lr.mean(axis=2)[:, -1:])), a.block // 2)
            to = blockmean(np.abs(out - prev_out).max(axis=2), a.block)
            tf = blockmean(np.abs(f_u - prev_fresh).max(axis=2), a.block)
            sel = flat <= np.quantile(flat, 0.35)
            if sel.sum() >= 16:
                ratio = float(np.median(to[sel]) / max(float(np.median(tf[sel])), 1e-6))
        rows.append({"frame": i, "depth": k or 0, "share": share, "ratio": ratio})
        prev_out, prev_fresh = out, f_u
        i += 1
    for f in (lr, g, u):
        f.close()
    if enc is not None:
        enc.stdin.close()
        enc.wait()

    reused = [r for r in rows if r["depth"]]
    shares = [r["share"] for r in reused]
    ratios = [r["ratio"] for r in reused if r["ratio"] is not None]
    out = {"frames": len(rows), "reused": len(reused), "alpha": a.alpha, "fb_tol": a.fb_tol,
           "share_worst": round(max(shares), 5) if shares else None,
           "frames_over_bar": sum(1 for s in shares if s > 0.1),
           "share_median": round(float(np.median(shares)), 5) if shares else None,
           "ratio_median": round(float(np.median(ratios)), 3) if ratios else None,
           "ratio_share_below_0.5": round(float(np.mean([x < 0.5 for x in ratios])), 3) if ratios else None,
           "ms_per_frame_this_implementation": round((time.time() - t0) / max(len(rows), 1) * 1000, 1)}
    print(json.dumps(out))
    if a.emit:
        json.dump({"summary": out, "rows": rows}, open(a.emit, "w"))
    print("NOT checked: a GPU implementation (this is numpy/cv2), codec motion vectors, and the 4K grain spectrum")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
