"""Is the int8 render closer in COLOUR to the source than the fp16 one? Measurable, not a matter of taste.

The frame's detail is the model's job (it invents edges), so "closer to the source" only means something
for the low-frequency content: colour and large-scale level. This compares each render against the source
frame on a coarse grid (16x16 block means at 4K, i.e. 8x8 source pixels), in RGB and in YUV, and reports
mean signed and mean absolute differences per channel.

A colour difference of a couple of levels on a 16x16 block is a tint; on Y it is brightness. Both are what
an eye reads as "the colour is off".

Usage: python harness/scripts/colour_vs_source.py --frame 169
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import numpy as np

FF = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2/bin/ffmpeg"
W, H = 3840, 2160


def rgb_frame(path: str, idx: int, size=(W, H)) -> np.ndarray:
    w, h = size
    raw = subprocess.run([FF, "-loglevel", "error", "-i", path, "-vf",
                          f"select=eq(n\\,{idx}),scale={w}:{h}:flags=neighbor",
                          "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
                         capture_output=True).stdout
    if len(raw) < w * h * 3:
        raise SystemExit(f"no frame {idx} in {path}")
    return np.frombuffer(raw[:w * h * 3], dtype=np.uint8).reshape(h, w, 3).astype(np.float32)


def yuv_frame(path: str, idx: int, size=(W, H)) -> np.ndarray:
    w, h = size
    raw = subprocess.run([FF, "-loglevel", "error", "-i", path, "-vf", f"select=eq(n\\,{idx})",
                          "-frames:v", "1", "-pix_fmt", "yuv420p", "-f", "rawvideo", "-"],
                         capture_output=True).stdout
    n = w * h + 2 * (h // 2) * (w // 2)
    a = np.frombuffer(raw[:n], dtype=np.uint8)
    y = a[:w * h].reshape(h, w).astype(np.float32)
    u = a[w * h:w * h + (h // 2) * (w // 2)].reshape(h // 2, w // 2).astype(np.float32)
    v = a[w * h + (h // 2) * (w // 2):].reshape(h // 2, w // 2).astype(np.float32)
    return y, u, v


def blocks(x: np.ndarray, k: int = 16) -> np.ndarray:
    h, w = (x.shape[0] // k) * k, (x.shape[1] // k) * k
    return x[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=169)
    ap.add_argument("--source", default="/tmp/opencode/bd_violet20s.mkv")
    ap.add_argument("--fp16", default="/var/tmp/rve-rem/ref_fp16_y8.mkv")
    ap.add_argument("--int8", dest="int8_path", default="/var/tmp/rve-rem/int8_pt.mkv")
    args = ap.parse_args(argv)

    # the render is 2x the source, so a 16 px block there covers the same picture area as an 8 px block
    # in the source - compare on the same physical grid or the shapes do not even match
    def k_for(size):
        return 16 if size[0] >= W else 8

    def rgb_blocks(path, size, idx):
        f = rgb_frame(path, idx, size)
        k = k_for(size)
        return np.stack([blocks(f[:, :, c], k) for c in range(3)])

    def yuv_blocks(path, size, idx):
        y, u, v = yuv_frame(path, idx, size)
        k = k_for(size)
        return blocks(y, k), blocks(u, k), blocks(v, k)

    src_rgb = rgb_blocks(args.source, (1920, 1080), args.frame)
    sy, su, sv = yuv_blocks(args.source, (1920, 1080), args.frame)

    print(f"frame {args.frame}: 16x16 block means against the SOURCE (RGB nearest-upscaled)")
    print(f"{'render':<14}{'dR':>8}{'dG':>8}{'dB':>8}{'|dRGB|':>9}{'dY':>8}{'dU':>8}{'dV':>8}"
          f"{'|dY|':>8}{'|dUV|':>8}")
    for name, path in (("fp16 engine", args.fp16), ("int8 body", args.int8_path)):
        got_rgb = rgb_blocks(path, (W, H), args.frame)
        drgb = (got_rgb - src_rgb).mean(axis=(1, 2))
        ay, au, av = yuv_blocks(path, (W, H), args.frame)
        dY = float((ay - sy).mean())
        dU = float((au - su).mean())
        dV = float((av - sv).mean())
        print(f"{name:<14}{drgb[0]:+8.2f}{drgb[1]:+8.2f}{drgb[2]:+8.2f}"
              f"{np.abs(got_rgb - src_rgb).mean():9.2f}"
              f"{dY:+8.2f}{dU:+8.2f}{dV:+8.2f}"
              f"{np.abs(ay - sy).mean():8.2f}{0.5 * (np.abs(au - su).mean() + np.abs(av - sv).mean()):8.2f}")
    # per-block verdict: is one render closer than the other, and by how much
    got_a = rgb_blocks(args.fp16, (W, H), args.frame)
    got_b = rgb_blocks(args.int8_path, (W, H), args.frame)
    da = np.abs(got_a - src_rgb).mean(axis=0)
    db = np.abs(got_b - src_rgb).mean(axis=0)
    closer = float((db < da).mean() * 100)
    print("\nper block (16 px at 4K = 8 px of source), |dRGB| mean per block:")
    print(f"  fp16  median {np.median(da):.2f}  p90 {np.percentile(da, 90):.2f}  max {da.max():.2f}")
    print(f"  int8  median {np.median(db):.2f}  p90 {np.percentile(db, 90):.2f}  max {db.max():.2f}")
    print(f"  int8 is closer on {closer:.1f}% of blocks; the difference of the means is "
          f"{da.mean() - db.mean():+.3f} levels")
    print("\nSigned deltas are a tint (or a brightness shift on Y); the |...| columns are the size of it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
