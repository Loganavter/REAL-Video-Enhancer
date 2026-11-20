"""The published correction, done as published: warp the cached SR, then add the upscaled residual.

NEMO (MobiCom'20) reconstructs a non-anchor block as

    b.SR = b.ref.SR + interp(b.res, scale)

with the motion vector scaled by the SR ratio, where `b.res = b.LR - b.ref.LR` is the codec's residual after
motion compensation. Their ablation prices the two terms: 1.6 dB without the residual, 9.9 dB without the motion
vector. This lane's earlier "correction" added an LR residual to a NON-warped frame, i.e. it omitted the term
worth 9.9 dB - this script does it the published way, with DIS flow standing in for the codec's motion vectors.

  warped_sr  = remap(cached_SR, 2 * u)                    u = flow(ref -> cur) at LR, scaled to HR
  residual   = LR_t - warp(LR_ref, u)                     in LR
  out        = warped_sr + bilinear_up(residual)

Instruments: the acceptance share against the ungated render, and the temporal-texture ratio on flat static
blocks (0 = frozen, 1 = fresh).

Usage:
  python3 harness/scripts/reuse_warp.py --gated on.mkv --ungated off.mkv --lr clip.mkv --depth-log gate.log
      [--output out.mkv] [--emit out.json]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess

import cv2
import numpy as np

FF = "ffmpeg"


class Gray:
    def __init__(self, path, w, h):
        self.w, self.h, self.size = w, h, w * h
        self.proc = subprocess.Popen([FF, "-v", "error", "-i", path, "-vf", f"scale={w}:{h}",
                                      "-pix_fmt", "gray", "-f", "rawvideo", "-"], stdout=subprocess.PIPE)

    def read(self):
        b = self.proc.stdout.read(self.size)
        if not b or len(b) < self.size:
            return None
        return np.frombuffer(b, dtype=np.uint8).reshape(self.h, self.w)

    def close(self):
        self.proc.stdout.close()
        self.proc.terminate()


class RGB:
    def __init__(self, path, w, h):
        self.w, self.h, self.size = w, h, w * h * 3
        self.proc = subprocess.Popen([FF, "-v", "error", "-i", path, "-pix_fmt", "rgb24",
                                      "-f", "rawvideo", "-"], stdout=subprocess.PIPE)

    def read(self):
        b = self.proc.stdout.read(self.size)
        if not b or len(b) < self.size:
            return None
        return np.frombuffer(b, dtype=np.uint8).reshape(self.h, self.w, 3)

    def close(self):
        self.proc.stdout.close()
        self.proc.terminate()


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
    ap.add_argument("--gain", type=float, default=1.0, help="multiplier on the LR residual")
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
    lr = Gray(a.lr, lw, lh)
    g = Gray(a.gated, lw * 2, lh * 2)
    u = Gray(a.ungated, lw * 2, lh * 2)
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    enc = None
    if a.output:
        enc = subprocess.Popen([FF, "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray",
                                "-s", f"{lw * 2}x{lh * 2}", "-r", "24000/1001", "-i", "-",
                                "-c:v", "ffv1", "-pix_fmt", "yuv420p", a.output, "-y"],
                               stdin=subprocess.PIPE)

    gx, gy = np.meshgrid(np.arange(lw * 2, dtype=np.float32), np.arange(lh * 2, dtype=np.float32))
    hist, prev_out, prev_fresh, rows, i = [], None, None, [], 0
    while True:
        f_lr, f_g, f_u = lr.read(), g.read(), u.read()
        if f_lr is None or f_g is None or f_u is None:
            break
        hist.append(f_lr.astype(np.float32))
        k = depth.get(i, 0)
        if k:
            ref = hist[max(0, len(hist) - 1 - k)]
            cur_f = f_lr.astype(np.float32)
            uvw = flow.calc(ref.astype(np.uint8), f_lr, None)          # LR displacement ref -> cur
            warp_ref = cv2.remap(ref, np.arange(lw, dtype=np.float32)[None, :] + uvw[..., 0],
                                 np.arange(lh, dtype=np.float32)[:, None] + uvw[..., 1],
                                 interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            residual = cur_f - warp_ref
            up = a.gain * cv2.resize(residual, (lw * 2, lh * 2), interpolation=cv2.INTER_LINEAR)
            uvw_hr = cv2.resize(uvw, (lw * 2, lh * 2), interpolation=cv2.INTER_LINEAR) * 2.0
            warp_sr = cv2.remap(f_g.astype(np.float32), gx + uvw_hr[..., 0], gy + uvw_hr[..., 1],
                                interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            out = np.clip(warp_sr + up, 0, 255)
        else:
            out = f_g.astype(np.float32)
        if enc is not None:
            enc.stdin.write(np.clip(out, 0, 255).astype(np.uint8).tobytes())
        if k:
            d = np.abs(out - f_u.astype(np.float32))
            share = float((d > 8).mean()) * 100.0
        else:
            share = 0.0
        ratio = None
        if k and prev_out is not None:
            flat = blockmean(np.abs(np.diff(f_lr, axis=1, append=f_lr[:, -1:])).astype(np.float32), a.block // 2)
            to = blockmean(np.abs(out - prev_out), a.block)
            tf = blockmean(np.abs(f_u.astype(np.float32) - prev_fresh), a.block)
            sel = flat <= np.quantile(flat, 0.35)
            if sel.sum() >= 16:
                ratio = float(np.median(to[sel]) / max(float(np.median(tf[sel])), 1e-6))
        rows.append({"frame": i, "depth": k, "share": share, "ratio": ratio})
        prev_out, prev_fresh = out, f_u.astype(np.float32)
        i += 1
    for f in (lr, g, u):
        f.close()
    if enc is not None:
        enc.stdin.close()
        enc.wait()

    reused = [r for r in rows if r["depth"]]
    shares = [r["share"] for r in reused]
    ratios = [r["ratio"] for r in reused if r["ratio"] is not None]
    out = {"frames": len(rows), "reused": len(reused), "gain": a.gain,
           "share_worst": round(max(shares), 5) if shares else None,
           "share_median": round(float(np.median(shares)), 5) if shares else None,
           "frames_over_bar": sum(1 for s in shares if s > 0.1),
           "ratio_median": round(float(np.median(ratios)), 3) if ratios else None,
           "ratio_share_below_0.5": round(float(np.mean([x < 0.5 for x in ratios])), 3) if ratios else None}
    print(json.dumps(out, indent=1))
    if a.emit:
        json.dump({"summary": out, "rows": rows}, open(a.emit, "w"))
        print("wrote", a.emit)
    print("NOT checked: codec motion vectors (DIS stands in), a real decoder path, and the 4K grain spectrum")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
