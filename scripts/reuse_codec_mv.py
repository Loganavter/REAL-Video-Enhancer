"""Warp by the codec's own motion vectors instead of estimating a flow, and verify the field first.

The remaining reuse failures are concentrated at depth > 1, where a jump-k residual stops being small (RU033). The
codec already stores, per frame, the displacement each block was predicted with - in ffmpeg's AVMotionVector side
data, which PyAV surfaces only when the decoder option `flags2=+export_mvs` is set (without it the frames carry
timecode only; ffprobe prints the side data but not its contents).

Two stages, in this order:

  1. VERIFY: warp `LR_{t-1}` by frame t's motion vectors and compare the residual against the un-warped one on the
     blocks the codec actually coded with motion. If the field is not the one this picture was coded against, the
     residual does not shrink and nothing downstream is worth measuring.
  2. CORRECT: recursive, hop by hop - `out_t = out_{t-1} + up(split(LR_t - warp(LR_{t-1}, MV_t)))` with the
     frequency split (low-pass at gain 1, high-pass at alpha on flat static blocks).

Usage:
  python3 scripts/reuse_codec_mv.py --clip cut.mkv --gated on.mkv --ungated off.mkv --depth-log gate.log
      [--alpha 1.35] [--verify-only] [--output out.mkv] [--emit f.json]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess

import numpy as np

FF = "ffmpeg"


def motion_vectors(path: str):
    """Per frame: list of (src_x, src_y, w, h, mv_x, mv_y). Requires the decoder option; see the docstring."""
    import av

    out = []
    c = av.open(path)
    s = c.streams.video[0]
    s.codec_context.options = {"flags2": "+export_mvs"}
    for f in c.decode(s):
        mvs = []
        for key, sd in f.side_data.items():
            if "MOTION" in str(key):
                for m in sd:
                    # ffmpeg's AVMotionVector: dst = block position in this frame, src = its position in the
                    # reference, motion in quarter-pel units (motion_scale), source = reference index
                    if m.source != -1:         # only vectors that point exactly one frame back (source is the
                                               # reference offset: -1 = previous, +1 = a future frame in B coding)
                        continue
                    scale = float(m.motion_scale) or 1.0
                    mvs.append((int(m.dst_x), int(m.dst_y), int(m.w), int(m.h),
                                float(m.motion_x) / scale, float(m.motion_y) / scale))
        out.append(mvs)
    c.close()
    return out


class Gray:
    def __init__(self, path, w, h):
        self.w, self.h, self.size = w, h, w * h
        self.proc = subprocess.Popen([FF, "-v", "error", "-i", path, "-pix_fmt", "gray",
                                      "-f", "rawvideo", "-"], stdout=subprocess.PIPE)

    def read(self):
        b = self.proc.stdout.read(self.size)
        if not b or len(b) < self.size:
            return None
        return np.frombuffer(b, dtype=np.uint8).reshape(self.h, self.w).astype(np.float32)

    def close(self):
        self.proc.stdout.close()
        self.proc.terminate()


def displacement(mvs, w, h, down=1):
    """Per-pixel displacement from block motion vectors; blocks not coded with a vector stay at zero."""
    dx = np.zeros((h, w), dtype=np.float32)
    dy = np.zeros((h, w), dtype=np.float32)
    for sx, sy, bw, bh, mx, my in mvs:
        x0, y0 = sx // down, sy // down
        x1, y1 = min(x0 + max(bw // down, 1), w), min(y0 + max(bh // down, 1), h)
        if x1 > x0 and y1 > y0:
            dx[y0:y1, x0:x1] = mx / down
            dy[y0:y1, x0:x1] = my / down
    return dx, dy


def warp(img, dx, dy):
    import cv2
    h, w = img.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx + dx, gy + dy, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def box(img, k):
    r = k // 2
    pad = np.pad(img, ((r, r), (r, r)), mode="reflect")
    c = np.cumsum(np.cumsum(pad, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return s / (k * k)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--gated", required=True)
    ap.add_argument("--ungated", required=True)
    ap.add_argument("--depth-log", default="")
    ap.add_argument("--alpha", type=float, default=1.35)
    ap.add_argument("--blur", type=int, default=5)
    ap.add_argument("--mask-tau", type=float, default=3.0)
    ap.add_argument("--mask-grad", type=float, default=5.0)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--emit", default="")
    a = ap.parse_args()

    print("extracting motion vectors ...", flush=True)
    mv = motion_vectors(a.clip)
    with_mv = sum(1 for m in mv if m)
    print(f"  {len(mv)} frames, {with_mv} carrying motion vectors, "
          f"median {int(np.median([len(m) for m in mv if m])) if with_mv else 0} blocks each")

    lw, lh = 1920, 1080
    lr = Gray(a.clip, lw, lh)
    prev, rows = None, []
    i = 0
    while True:
        cur = lr.read()
        if cur is None:
            break
        if prev is not None and i < len(mv) and mv[i]:
            dx, dy = displacement(mv[i], lw, lh)
            pred = warp(prev, dx, dy)
            moving = (np.abs(dx) + np.abs(dy)) > 0.5
            if moving.sum() > 1000:
                r_warp = float(np.abs(cur - pred)[moving].mean())
                r_plain = float(np.abs(cur - prev)[moving].mean())
                rows.append({"frame": i, "blocks_moving_pct": round(float(moving.mean()) * 100, 2),
                             "residual_warped": round(r_warp, 3), "residual_plain": round(r_plain, 3),
                             "gain_pct": round(100 * (1 - r_warp / max(r_plain, 1e-6)), 1)})
        prev = cur
        i += 1
        if a.verify_only and i > 200:
            break
    lr.close()

    if rows:
        gains = [r["gain_pct"] for r in rows if r["residual_plain"] > 0.05]
        print(f"verification: {len(rows)} frames with a coded motion field, "
              f"median residual gain from warping {np.median(gains):.1f}%, "
              f"share of frames improved {100*np.mean([g > 0 for g in gains]):.0f}%")
        print("  worst five:", sorted(rows, key=lambda r: r["gain_pct"])[:5])
    if a.emit:
        json.dump({"frames": len(mv), "with_mv": with_mv, "rows": rows}, open(a.emit, "w"))
        print("wrote", a.emit)
    print("NOT checked: B-frame reference indices (the field is used against t-1), and the correction itself "
          "until the verification passes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
