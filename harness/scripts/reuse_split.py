"""Split the correction by frequency instead of by geometry: structure at 1x, grain at alpha.

The measured trade-off (docs/wsp/verified/reuse-decision-statistic.txt) says one scalar gain cannot satisfy both
instruments: unit gain fixes the structure (0/130 over the bar) and leaves the texture at 0.388 ~ 1/alpha; alpha on
the whole residual restores the texture and breaks the structure. The two components are not distinguished by a
gain, they are distinguished by their *content*: the structural part of `LR_t - LR_ref` is a low-frequency
displacement, the grain part is high-frequency and temporally independent, and the network amplifies only the
latter (alpha ~ 2.04 measured).

So split the residual with a spatial low-pass:

    r      = LR_t - LR_ref
    r_lf   = box(r, k)              the displacement / tone part, added at gain 1
    r_hf   = r - r_lf               the grain part, added at gain alpha
    out    = cached_SR + up(r_lf + alpha * r_hf)

Usage:
  python3 harness/scripts/reuse_split.py --gated on.mkv --ungated off.mkv --lr clip.mkv --depth-log gate.log
      --alpha 2.04 --blur 5 [--output out.mkv] [--emit f.json]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess

import numpy as np

FF = "ffmpeg"


class Frames:
    def __init__(self, path, w, h, pix):
        self.w, self.h = w, h
        self.size = w * h * (1 if pix == "gray" else 3)
        vf = f"scale={w}:{h}"
        self.proc = subprocess.Popen([FF, "-v", "error", "-i", path, "-vf", vf, "-pix_fmt", pix,
                                      "-f", "rawvideo", "-"], stdout=subprocess.PIPE)

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


def box(img: np.ndarray, k: int) -> np.ndarray:
    """Box blur with an integral image; the residual's low-frequency part."""
    r = k // 2
    pad = np.pad(img, ((r, r), (r, r), (0, 0)) if img.ndim == 3 else ((r, r), (r, r)),
                 mode="reflect")
    c = np.cumsum(np.cumsum(pad, axis=0), axis=1)
    if img.ndim == 3:
        c = np.pad(c, ((1, 0), (1, 0), (0, 0)))
        s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    else:
        c = np.pad(c, ((1, 0), (1, 0)))
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
    ap.add_argument("--alpha", type=float, default=2.04)
    ap.add_argument("--blur", type=int, default=5)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--mask-tau", type=float, default=0.0,
                    help="apply the alpha gain only where the local |residual| is under this many levels; 0 = "
                         "everywhere (which doubles line-art high frequency and breaks the share)")
    ap.add_argument("--recursive", action="store_true",
                    help="apply the correction to the PREVIOUS EMITTED frame with the consecutive-frame residual "
                         "(LR_t - LR_{t-1}), as a codec decodes in order, instead of jumping k frames back to the "
                         "cached reference; the correction then stays small at depth > 1")
    ap.add_argument("--mask-grad", type=float, default=999.0,
                    help="and where the local gradient is under this many levels, i.e. flat areas")
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
    enc = None
    if a.output:
        enc = subprocess.Popen([FF, "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                "-s", f"{lw * 2}x{lh * 2}", "-r", "24000/1001", "-i", "-",
                                "-c:v", "ffv1", "-pix_fmt", "yuv444p", a.output, "-y"],
                               stdin=subprocess.PIPE)

    hist, rows, prev_out, prev_fresh, i = [], [], None, None, 0
    while True:
        f_lr, f_g, f_u = lr.read(), g.read(), u.read()
        if f_lr is None or f_g is None or f_u is None:
            break
        hist.append(f_lr)
        k = depth.get(i)
        if k:
            if a.recursive and prev_out is not None:
                ref = hist[-2] if len(hist) >= 2 else f_lr
                base = prev_out
            else:
                ref = hist[max(0, len(hist) - 1 - k)]
                base = f_g
            r = f_lr - ref
            lf = box(r, a.blur)
            hf = r - lf
            gain = np.full(hf.shape[:2], a.alpha, dtype=np.float32)
            if a.mask_tau > 0 or a.mask_grad < 999:
                gmag = np.abs(np.diff(f_lr.mean(axis=2), axis=0, append=f_lr.mean(axis=2)[-1:, :])) + \
                       np.abs(np.diff(f_lr.mean(axis=2), axis=1, append=f_lr.mean(axis=2)[:, -1:]))
                loc = box(np.abs(r), 5).mean(axis=2)
                keep = ((loc < a.mask_tau) | (a.mask_tau <= 0)) & (gmag < a.mask_grad)
                gain[~keep] = 1.0
            corr = lf + (gain[..., None] if hf.ndim == 3 else gain) * hf
            up = np.repeat(np.repeat(corr, 2, axis=0), 2, axis=1)          # nearest, as the pipeline does
            out = np.clip(base + up, 0, 255)
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
    out = {"alpha": a.alpha, "blur": a.blur, "reused": len(reused),
           "share_worst": round(max(shares), 5) if shares else None,
           "frames_over_bar": sum(1 for s in shares if s > 0.1),
           "ratio_median": round(float(np.median(ratios)), 3) if ratios else None,
           "ratio_share_below_0.5": round(float(np.mean([x < 0.5 for x in ratios])), 3) if ratios else None}
    print(json.dumps(out))
    if a.emit:
        json.dump({"summary": out, "rows": rows}, open(a.emit, "w"))
    print("NOT checked: a per-channel blur kernel, the net's alpha per block, and the 4K spectrum of the added grain")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
