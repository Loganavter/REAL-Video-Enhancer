"""Whole-frame panels for one frame: source | fp16 | int8, and the amplified difference, no crop.

The 1:1 crops answer "is the pixel the statistic counted visible"; the whole frame answers "where is it in
the picture, and does the picture read the same at viewing size". Both are needed, and the crop alone has
already caused a reviewer to miss that the damage is spread along the ink.

The difference panel is max-pooled (not area-averaged) when it is downscaled: a 2 px line at 4K averaged
over 3x3 becomes 1/9 of its amplitude and would disappear exactly where the defect is.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import numpy as np
from PIL import Image

FF = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2/bin/ffmpeg"
W, H = 3840, 2160
SW, SH = 1920, 1080


def frame_png(src: str, idx: int, out: str, size: tuple[int, int] | None = None) -> None:
    vf = f"select=eq(n\\,{idx})" + (f",scale={size[0]}:{size[1]}:flags=lanczos" if size else "")
    subprocess.run([FF, "-loglevel", "error", "-y", "-i", src, "-vf", vf, "-frames:v", "1", out],
                   check=True)


def load(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB")).astype(np.int16)


def max_pool(d: np.ndarray, factor: int = 3) -> np.ndarray:
    h, w = (d.shape[0] // factor) * factor, (d.shape[1] // factor) * factor
    return d[:h, :w].reshape(h // factor, factor, w // factor, factor).max(axis=(1, 3))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=169)
    ap.add_argument("--source", default="/tmp/opencode/bd_violet20s.mkv")
    ap.add_argument("--fp16", default="/var/tmp/rve-rem/ref_fp16_y8.mkv")
    ap.add_argument("--int8", dest="int8_path", default="/var/tmp/rve-rem/int8_pt.mkv")
    ap.add_argument("--outdir", default="/home/jorj/eye-check/2026-11-19-int8-per-channel")
    args = ap.parse_args(argv)

    f = args.frame
    src_png = f"{args.outdir}/_full_src.png"
    a_png = f"{args.outdir}/_full_fp16.png"
    b_png = f"{args.outdir}/_full_int8.png"
    frame_png(args.source, f, src_png, (W, H))          # source ×2 nearest-free? lanczos, labelled below
    frame_png(args.fp16, f, a_png)
    frame_png(args.int8_path, f, b_png)

    src, a, b = load(src_png), load(a_png), load(b_png)
    d = np.abs(a - b).max(axis=2)
    pooled = max_pool(d, 3)                              # 1280 x 720
    diff = np.clip(pooled * 16, 0, 255).astype(np.uint8)

    small = lambda x: np.asarray(Image.fromarray(x.astype(np.uint8)).resize((1280, 720), Image.LANCZOS))
    panels = [small(src), small(a), small(b)]
    Image.fromarray(np.hstack(panels)).save(f"{args.outdir}/full_f{f}_src_fp16_int8.png")
    Image.fromarray(diff).save(f"{args.outdir}/full_f{f}_diff_x16_maxpool.png")

    print(f"frame {f}: share {(d > 8).mean() * 100:.5f}%  peak {int(d.max())}")
    print(f"  {args.outdir}/full_f{f}_src_fp16_int8.png   (source | fp16 | int8, 1280 wide each)")
    print(f"  {args.outdir}/full_f{f}_diff_x16_maxpool.png (|fp16-int8| x16, max-pooled, 1280 wide)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
