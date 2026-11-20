"""Is the tint the model's, or the implementation's? Compare every candidate to what the weights compute.

"Задумано ли, что веса отдают микро-галло" has an operational answer: the model's output IS whatever its
weights compute in fp16, so run that (torch, fp16) and ask, per candidate:

  * model vs SOURCE        - does the model itself add a tint/halo on this frame?
  * engine vs MODEL        - does the TRT fp16 engine deviate from the weights, and in which direction?
  * int8 vs MODEL          - same for the quantised body.

Block means (16 px at 4K = 8 px of source) so the number describes a region, not a pixel: a tint is a
signed shift on a block, dithering is symmetric.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import numpy as np
import torch

import os as _bos, sys as _bsys
_HERE = _bos.path.dirname(_bos.path.abspath(__file__))
_ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in _bsys.path:
        _bsys.path.insert(0, _p)
from cut_channels_naive import load  # noqa: E402

FF = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2/bin/ffmpeg"
W, H = 3840, 2160


def decode(path: str, idx: int, size=(W, H)) -> np.ndarray:
    w, h = size
    raw = subprocess.run([FF, "-loglevel", "error", "-i", path, "-vf", f"select=eq(n\\,{idx})",
                          "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
                         capture_output=True).stdout
    if len(raw) < w * h * 3:
        raise SystemExit(f"no frame {idx} in {path}")
    return np.frombuffer(raw[:w * h * 3], dtype=np.uint8).reshape(h, w, 3).astype(np.float32)


def blocks(x: np.ndarray, k: int) -> np.ndarray:
    h, w = (x.shape[0] // k) * k, (x.shape[1] // k) * k
    return x[:h, :w].reshape(h // k, k, w // k, k, x.shape[2]).mean(axis=(1, 3))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=169)
    ap.add_argument("--source", default="/tmp/opencode/bd_violet20s.mkv")
    ap.add_argument("--fp16", default="/var/tmp/rve-rem/ref_fp16_y8.mkv")
    ap.add_argument("--int8", dest="int8_path", default="/var/tmp/rve-rem/int8_pt.mkv")
    ap.add_argument("--model", default="/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2/"
                                       "custom_models/2x-Futsuu-Anime(1).pth")
    args = ap.parse_args(argv)

    model, _, _ = load(args.model)
    model = model.eval().cuda().half()
    src = decode(args.source, args.frame, (1920, 1080))
    t = torch.from_numpy(src / 255.0).permute(2, 0, 1)[None].cuda().half()
    with torch.no_grad():
        out = model(t).float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255.0

    bs, bm = blocks(src, 8), blocks(out, 16)
    be = blocks(decode(args.fp16, args.frame), 16)
    bi = blocks(decode(args.int8_path, args.frame), 16)

    print(f"frame {args.frame}: block-mean RGB deltas (16 px at 4K), levels")
    print(f"{'comparison':<22}{'dR':>8}{'dG':>8}{'dB':>8}{'|dRGB|':>9}")
    for name, a, b in (("model - source", bm, bs), ("engine - model", be, bm), ("int8 - model", bi, bm)):
        d = (a - b).mean(axis=(0, 1))
        print(f"{name:<22}{d[0]:+8.2f}{d[1]:+8.2f}{d[2]:+8.2f}{np.abs(a - b).mean():9.2f}")
    # Halo test. lap = 4*centre - neighbours, so lap < 0 means the pixel is a DIP (the ink core of a line)
    # and lap > 0 the lighter side of it. A halo is the rim being brighter than the model's rim; losing the
    # halo is the rim flattening while the ink shifts. Both are signed, so they cannot hide in an absolute
    # difference.
    def luma(x):
        return 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]

    lm = luma(out)
    lap = (4 * lm[1:-1, 1:-1] - lm[:-2, 1:-1] - lm[2:, 1:-1] - lm[1:-1, :-2] - lm[1:-1, 2:])
    mask = np.abs(lap) > np.percentile(np.abs(lap), 98)
    ink, rim = mask & (lap < 0), mask & (lap > 0)
    print("\nedge probe on the top-2% laplacian pixels of the MODEL (luma, levels):")
    print(f"{'image':<12}{'ink core':>10}{'outer rim':>11}{'ink - model':>13}{'rim - model':>13}")
    base_ink, base_rim = lm[1:-1, 1:-1][ink].mean(), lm[1:-1, 1:-1][rim].mean()
    print(f"{'model':<12}{base_ink:10.2f}{base_rim:11.2f}{0.0:+13.2f}{0.0:+13.2f}")
    for name, img in (("engine", decode(args.fp16, args.frame)),
                      ("int8", decode(args.int8_path, args.frame))):
        lc = luma(img)[1:-1, 1:-1]
        print(f"{name:<12}{lc[ink].mean():10.2f}{lc[rim].mean():11.2f}"
              f"{lc[ink].mean() - base_ink:+13.2f}{lc[rim].mean() - base_rim:+13.2f}")
    print("\n'ink - model' positive = that render draws the line LIGHTER than the weights do;")
    print("'rim - model' positive = its rim next to the line is brighter (a halo the model has, kept).")
    print("A candidate that lifts the ink but not the rim flattens the halo; one that lifts both keeps it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
