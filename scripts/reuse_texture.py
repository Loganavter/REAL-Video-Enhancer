"""Did temporal reuse freeze the grain? A temporal-texture criterion the 0.1% share cannot see.

The acceptance statistic (share of pixels >8 levels off, worst frame) judges a single frame against the
ungated render. It cannot see the artifact the owner rejected by eye: a reused frame carries the *reference's*
noise, so on flat, static areas the output stops changing from frame to frame - the grain freezes - even when
every pixel is within 8 levels of the ungated render.

This tool measures that directly, per frame, on a gated/ungated render pair:

  flat static blocks   blocks whose spatial gradient (in the ungated render) is low and whose temporal change
                       there is small - the population where a fresh render shows grain and a frozen one
                       shows nothing
  texture ratio        median over those blocks of  mean|cur - prev| (gated) / mean|cur - prev| (ungated)
                       ~1 means the gated output changes like a fresh render; ~0 means the texture is frozen

Frames are joined to the gate's own per-frame decisions (`RVE_DEBUG_DEPTH=1` log, "reuse frame=N depth=K"),
so the ratio can be reported for reused and rendered frames separately. This is a criterion, not yet a
threshold: what value is acceptable is a separate, perceptual question.

Usage:
  python3 scripts/reuse_texture.py --gated on.mkv --ungated off.mkv --depth-log gate.log [--emit out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re

import subprocess

import numpy as np


class Gray:
    """Grayscale frames of a video through ffmpeg, at half resolution.

    Why not cv2.VideoCapture: it refuses these ffv1 4K yuv444p files (opened False, 0 frames), and the whole
    point of this tool is to run on the pair the app actually wrote.
    """

    def __init__(self, path: str, width: int, height: int):
        self.w, self.h = width, height
        self.proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale={width}:{height},format=gray",
             "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
        self.size = width * height

    def read(self):
        buf = self.proc.stdout.read(self.size)
        if not buf or len(buf) < self.size:
            return None
        return np.frombuffer(buf, dtype=np.uint8).reshape(self.h, self.w)

    def close(self):
        self.proc.stdout.close()
        self.proc.terminate()


def block_view(img: np.ndarray, block: int) -> np.ndarray:
    h, w = img.shape
    hb, wb = h // block, w // block
    return img[:hb * block, :wb * block].reshape(hb, block, wb, block)


def temporal_map(prev: np.ndarray, cur: np.ndarray, block: int) -> np.ndarray:
    """Per-block mean absolute frame difference, in luma levels."""
    d = np.abs(cur.astype(np.float32) - prev.astype(np.float32))
    return block_view(d, block).mean(axis=(1, 3))


def spatial_energy(img: np.ndarray, block: int) -> np.ndarray:
    """Per-block mean gradient magnitude: low values mark flat areas, where grain is the only thing moving."""
    f = img.astype(np.float32)
    gx = np.abs(np.diff(f, axis=1, append=f[:, -1:]))
    gy = np.abs(np.diff(f, axis=0, append=f[-1:, :]))
    return block_view(gx + gy, block).mean(axis=(1, 3))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gated", required=True)
    ap.add_argument("--ungated", required=True)
    ap.add_argument("--depth-log", default="", help="a render log with RVE_DEBUG_DEPTH=1 lines")
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--flat-quantile", type=float, default=0.35,
                    help="blocks below this quantile of spatial energy are 'flat'")
    ap.add_argument("--static-quantile", type=float, default=0.35,
                    help="and fall in this quantile of the ungated temporal change (relative, not absolute: "
                         "a fresh render's grain already moves several levels per frame)")
    ap.add_argument("--emit", default="")
    a = ap.parse_args()

    reused = {}
    if a.depth_log and os.path.exists(a.depth_log):
        for line in open(a.depth_log, errors="ignore"):
            m = re.search(r"reuse frame=(\d+) depth=(\d+)", line)
            if m:
                reused[int(m.group(1))] = int(m.group(2))

    # the pair is 4K; statistics live at half resolution, which is where the grain lives too
    cg, cu = Gray(a.gated, 1920, 1080), Gray(a.ungated, 1920, 1080)
    pg, pu = cg.read(), cu.read()
    rows, i = [], 0
    while True:
        cg_i, cu_i = cg.read(), cu.read()
        if cg_i is None or cu_i is None:
            break
        i += 1
        tg = temporal_map(pg, cg_i, a.block)
        tu = temporal_map(pu, cu_i, a.block)
        flat = spatial_energy(cu_i, a.block)
        cut = np.quantile(flat, a.flat_quantile)
        tcut = np.quantile(tu, a.static_quantile)
        sel = (flat <= cut) & (tu <= tcut) & (tu > 0)
        ratio = float("nan")
        if sel.sum() >= 16:
            ratio = float(np.median(tg[sel]) / max(float(np.median(tu[sel])), 1e-6))
        rows.append({"frame": i, "blocks": int(sel.sum()), "ratio": ratio,
                     "temporal_gated": float(np.median(tg[sel])) if sel.any() else None,
                     "temporal_ungated": float(np.median(tu[sel])) if sel.any() else None,
                     "reused": i in reused})
        pg, pu = cg_i, cu_i
    cg.close(); cu.close()

    vals = [r["ratio"] for r in rows if r["ratio"] == r["ratio"]]
    if not vals:
        raise SystemExit("no frame had enough flat static blocks; loosen --flat-quantile/--static-max")
    yes = [r["ratio"] for r in rows if r["reused"] and r["ratio"] == r["ratio"]]
    no = [r["ratio"] for r in rows if not r["reused"] and r["ratio"] == r["ratio"]]
    out = {
        "frames": len(rows),
        "reused_frames": len(reused),
        "ratio_all": {"median": round(float(np.median(vals)), 4),
                      "p10": round(float(np.percentile(vals, 10)), 4),
                      "p90": round(float(np.percentile(vals, 90)), 4)},
        "ratio_reused": {"n": len(yes), "median": round(float(np.median(yes)), 4) if yes else None,
                         "share_below_0.5": round(float(np.mean([v < 0.5 for v in yes])), 3) if yes else None},
        "ratio_rendered": {"n": len(no), "median": round(float(np.median(no)), 4) if no else None,
                           "share_below_0.5": round(float(np.mean([v < 0.5 for v in no])), 3) if no else None},
    }
    print(json.dumps(out, indent=1))
    if a.emit:
        json.dump({"summary": out, "rows": rows}, open(a.emit, "w"))
        print("wrote", a.emit)
    print("NOT checked: what ratio a viewer accepts, the same measurement on a synthesised-grain output, "
          "and whether flat blocks on this clip represent the title")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
