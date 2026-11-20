"""Does the delivered video actually show the render? Encode fidelity for an A/B a human will judge.

The renders differ by a mean of ~0.1 levels and half a pixel of contour. An 8-bit HEVC at qp 18 can add
±1-2 levels of its own, i.e. the same size as the effect - so the delivered file can hide or invent what the
viewer is asked to judge. This measures both directions on a frame:

  * encode vs source render, per encode: what the codec added;
  * (int8 - fp16) computed on the ENCODES against the same difference on the RENDERS: whether the A/B the
    viewer sees is the A/B the renders contain.

Usage: python harness/scripts/encode_fidelity.py --frame 169
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import numpy as np

FF = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2/bin/ffmpeg"
W, H = 3840, 2160


def rgb(path: str, idx: int) -> np.ndarray:
    raw = subprocess.run([FF, "-loglevel", "error", "-i", path, "-vf", f"select=eq(n\\,{idx})",
                          "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
                         capture_output=True).stdout
    if len(raw) < W * H * 3:
        raise SystemExit(f"no frame {idx} in {path}")
    return np.frombuffer(raw[:W * H * 3], dtype=np.uint8).reshape(H, W, 3).astype(np.int16)


def stats(name: str, a: np.ndarray, b: np.ndarray) -> None:
    d = np.abs(a - b)
    per = d.max(axis=2)
    print(f"  {name:<34} mean |d| {d.mean():5.3f}  max {int(per.max()):3d}  "
          f"share >8 {(per > 8).mean() * 100:7.4f}%")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=169)
    ap.add_argument("--src-int8", default="/var/tmp/rve-rem/int8_pt.mkv")
    ap.add_argument("--src-fp16", default="/var/tmp/rve-rem/ref_fp16_y8.mkv")
    ap.add_argument("--enc-int8", default="/home/jorj/eye-check/2026-11-19-int8-per-channel/video/int8_4k_hevc10.mp4")
    ap.add_argument("--enc-fp16", default="/home/jorj/eye-check/2026-11-19-int8-per-channel/video/fp16_4k_hevc10.mp4")
    args = ap.parse_args(argv)

    si, sf = rgb(args.src_int8, args.frame), rgb(args.src_fp16, args.frame)
    ei, ef = rgb(args.enc_int8, args.frame), rgb(args.enc_fp16, args.frame)

    print(f"frame {args.frame}")
    print("codec against its own render (what the viewer is not supposed to see):")
    stats("int8 encode vs int8 render", ei, si)
    stats("fp16 encode vs fp16 render", ef, sf)

    print("the A/B itself, renders against encodes:")
    d_src = (si - sf)
    d_enc = (ei - ef)
    print(f"  render A/B   mean |d| {np.abs(d_src).mean():5.3f}  "
          f"share >8 {(np.abs(d_src).max(axis=2) > 8).mean() * 100:7.4f}%")
    print(f"  encode A/B   mean |d| {np.abs(d_enc).mean():5.3f}  "
          f"share >8 {(np.abs(d_enc).max(axis=2) > 8).mean() * 100:7.4f}%")
    corr = float(np.corrcoef(d_src.ravel(), d_enc.ravel())[0, 1])
    print(f"  correlation of the two difference fields: {corr:.3f} (1.0 = the encode preserves the A/B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
