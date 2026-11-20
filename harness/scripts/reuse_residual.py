"""Residual reuse: does `SR_ref + bilinear(LR_t - LR_ref)` revive the texture, and at what pixel cost?

The review's algebra: with a net that multiplies noise by alpha (median 2.04 on this material), a skip that
adds only the upscaled LR difference leaves `(alpha - 1) * n_ref` frozen in the frame. This measures it instead
of arguing: on the frames the gate reused, build the construction and report

  * the acceptance share against the ungated render (pixels >8 levels off, any channel), worst frame; and
  * the temporal-texture ratio against the ungated render on flat static blocks (0 = frozen, 1 = fresh).

Both are reported next to the plain skip (the gated render) and the fresh render, so the comparison is on one
instrument. Reference for statickness is the LR source, so a render is not needed to define the population.

Usage:
  python3 harness/scripts/reuse_residual.py --gated on.mkv --ungated off.mkv --lr clip.mkv --depth-log gate.log
      [--block 64] [--limit N] [--emit out.json]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess

import numpy as np

FF = "ffmpeg"


class Frames:
    """Frames through ffmpeg: gray at the LR size for statistics, rgb24 at the HR size for the share."""

    def __init__(self, path: str, w: int, h: int, pix: str):
        self.w, self.h, self.pix = w, h, pix
        self.ch = 1 if pix == "gray" else 3
        self.size = w * h * self.ch
        vf = f"scale={w}:{h}" + (",format=gray" if pix == "gray" else "")
        self.proc = subprocess.Popen([FF, "-v", "error", "-i", path, "-vf", vf,
                                      "-pix_fmt", pix,          # NOT the input's pix_fmt: rawvideo would
                                      "-f", "rawvideo", "-"],  # otherwise emit yuv420p and every read()
                                      stdout=subprocess.PIPE)  # would consume two frames


    def read(self):
        buf = self.proc.stdout.read(self.size)
        if not buf or len(buf) < self.size:
            return None
        a = np.frombuffer(buf, dtype=np.uint8)
        return a.reshape(self.h, self.w) if self.ch == 1 else a.reshape(self.h, self.w, 3)

    def close(self):
        self.proc.stdout.close()
        self.proc.terminate()


def blockmean(img: np.ndarray, b: int) -> np.ndarray:
    h, w = img.shape
    hb, wb = h // b, w // b
    return img[:hb * b, :wb * b].reshape(hb, b, wb, b).mean(axis=(1, 3))


def up_double(x: np.ndarray) -> np.ndarray:
    """Nearest 2x, which is what the pipeline's own upsampler does for the residual path."""
    return np.repeat(np.repeat(x, 2, axis=0), 2, axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gated", required=True)
    ap.add_argument("--ungated", required=True)
    ap.add_argument("--lr", required=True)
    ap.add_argument("--depth-log", required=True)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gain", type=float, default=1.0,
                    help="multiplier on the LR difference. Unit gain restores the correction at 1x while the "
                         "net multiplies noise by alpha (median 2.04 here), so a unit-gain correction leaves "
                         "the texture at ~1/alpha of fresh; the gain is calibrated by one pass and reported")
    ap.add_argument("--emit", default="")
    ap.add_argument("--output", default="", help="write the constructed video (ffv1) for the eye")
    a = ap.parse_args()

    depth = {}
    for line in open(a.depth_log, errors="ignore"):
        m = re.search(r"reuse frame=(\d+) depth=(\d+)", line)
        if m:
            depth[int(m.group(1))] = int(m.group(2))

    lw, lh = 1920, 1080
    lr = Frames(a.lr, lw, lh, "gray")
    g = Frames(a.gated, lw * 2, lh * 2, "rgb24")
    u = Frames(a.ungated, lw * 2, lh * 2, "rgb24")

    enc = None
    if a.output:
        enc = subprocess.Popen([FF, "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                "-s", f"{lw * 2}x{lh * 2}", "-r", "24000/1001", "-i", "-",
                                "-c:v", "ffv1", "-pix_fmt", "yuv444p", a.output, "-y"],
                               stdin=subprocess.PIPE)
    lr_hist = []
    prev_pred = prev_u = None
    rows, i = [], 0
    while True:
        fr_l, fr_g, fr_u = lr.read(), g.read(), u.read()
        if fr_l is None or fr_g is None or fr_u is None:
            break
        lr_hist.append(fr_l.astype(np.float32))
        k = depth.get(i, 0)
        if k:
            ref = lr_hist[max(0, len(lr_hist) - 1 - k)]
            diff = a.gain * up_double(fr_l.astype(np.float32) - ref)  # HR-sized correction
            pred = np.clip(fr_g.astype(np.float32) + diff[..., None], 0, 255)
        else:
            pred = fr_g.astype(np.float32)                           # a rendered frame is its own output
        if enc is not None:
            enc.stdin.write(np.clip(pred, 0, 255).astype(np.uint8).tobytes())
        if k:
            d = np.abs(pred - fr_u.astype(np.float32)).max(axis=2)
            share = float((d > 8).mean()) * 100.0
        else:
            share = 0.0
        ratio = None
        if k and prev_pred is not None:
            flat = blockmean(np.abs(np.diff(fr_l, axis=1, append=fr_l[:, -1:])).astype(np.float32), a.block // 2)
            tp = blockmean(np.abs(pred - prev_pred).max(axis=2), a.block)
            tu = blockmean(np.abs(fr_u.astype(np.float32) - prev_u).max(axis=2), a.block)
            sel = flat <= np.quantile(flat, 0.35)
            if sel.sum() >= 16:
                ratio = float(np.median(tp[sel]) / max(float(np.median(tu[sel])), 1e-6))
        rows.append({"frame": i, "depth": k, "share": share, "ratio": ratio})
        prev_pred, prev_u = pred, fr_u.astype(np.float32)
        i += 1
        if a.limit and i >= a.limit:
            break
    for f in (lr, g, u):
        f.close()
    if enc is not None:
        enc.stdin.close()
        enc.wait()

    reused = [r for r in rows if r["depth"]]
    shares = [r["share"] for r in reused]
    ratios = [r["ratio"] for r in reused if r["ratio"] is not None]
    out = {
        "frames": len(rows), "reused": len(reused),
        "share_worst": round(max(shares), 5) if shares else None,
        "share_median": round(float(np.median(shares)), 5) if shares else None,
        "frames_over_bar": sum(1 for s in shares if s > 0.1),
        "ratio_median": round(float(np.median(ratios)), 3) if ratios else None,
        "ratio_share_below_0.5": round(float(np.mean([x < 0.5 for x in ratios])), 3) if ratios else None,
    }
    print(json.dumps(out, indent=1))
    if a.emit:
        json.dump({"summary": out, "rows": rows}, open(a.emit, "w"))
        print("wrote", a.emit)
    print("NOT checked: a denoised plate, the model's own noise gain per block, and whether the added LR-band "
          "grain looks right at 4K")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
