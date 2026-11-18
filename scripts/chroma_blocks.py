"""Are there tint differences, or only feelings about them? Chroma measured per block, not by eye.

"Оттенки непонятки" is testable: compare each render's chroma against the SOURCE at block scale and read
the distribution, not the mean. A tint is a systematic signed shift on a region; dithering is symmetric.

Reports, per render: signed mean dU/dV (the tint), and the p99 / max of |dU|,|dV| per block, plus how many
blocks sit above 2 levels - 2 levels of chroma is roughly the smallest step a viewer reliably reports on
flat anime fills.
"""
from __future__ import annotations

import argparse
import subprocess

import numpy as np

FF = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2/bin/ffmpeg"
W, H = 3840, 2160


def planes(path: str, idx: int, size=(W, H)) -> tuple[np.ndarray, np.ndarray]:
    w, h = size
    raw = subprocess.run([FF, "-loglevel", "error", "-i", path, "-vf", f"select=eq(n\\,{idx})",
                          "-frames:v", "1", "-pix_fmt", "yuv420p", "-f", "rawvideo", "-"],
                         capture_output=True).stdout
    n = w * h + 2 * (h // 2) * (w // 2)
    a = np.frombuffer(raw[:n], dtype=np.uint8)
    u = a[w * h:w * h + (h // 2) * (w // 2)].reshape(h // 2, w // 2).astype(np.float32)
    v = a[w * h + (h // 2) * (w // 2):].reshape(h // 2, w // 2).astype(np.float32)
    return u, v


def blocks(x: np.ndarray, k: int) -> np.ndarray:
    h, w = (x.shape[0] // k) * k, (x.shape[1] // k) * k
    return x[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=169)
    ap.add_argument("--source", default="/tmp/opencode/bd_violet20s.mkv")
    ap.add_argument("--fp16", default="/var/tmp/rve-rem/ref_fp16_y8.mkv")
    ap.add_argument("--int8", dest="int8_path", default="/var/tmp/rve-rem/int8_pt.mkv")
    args = ap.parse_args(argv)

    su, sv = planes(args.source, args.frame, (1920, 1080))
    su, sv = blocks(su, 8), blocks(sv, 8)      # 8 px of source = 16 px of render

    print(f"frame {args.frame}: chroma against the source, per 16 px block")
    print(f"{'render':<12}{'dU':>7}{'dV':>7}{'|dU| p99':>10}{'|dV| p99':>10}{'|dU| max':>10}"
          f"{'|dV| max':>10}{'blocks >2':>11}")
    for name, path in (("fp16", args.fp16), ("int8", args.int8_path)):
        u, v = planes(path, args.frame)
        u, v = blocks(u, 16), blocks(v, 16)
        du, dv = u - su, v - sv
        over = float(((np.abs(du) > 2) | (np.abs(dv) > 2)).mean() * 100)
        print(f"{name:<12}{du.mean():+7.2f}{dv.mean():+7.2f}"
              f"{np.percentile(np.abs(du), 99):10.2f}{np.percentile(np.abs(dv), 99):10.2f}"
              f"{np.abs(du).max():10.2f}{np.abs(dv).max():10.2f}{over:10.1f}%")
    print("\nA tint is a signed shift; a p99 near the mean means it is uniform, a max far above it means a "
          "few regions differ (which is the thing the eye calls 'оттенки').")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
