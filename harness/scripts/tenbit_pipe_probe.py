"""Does ten-bit material keep its ten bits through the reader and the packer?

Every change in this lane sits between these two conversions: the reader turns the pipe's yuv420p10le into an
fp16 tensor, the packer turns the tensor back into yuv420p10le. Neither may collapse the material to 8 bits,
and "it looks fine" is not a test - an 8-bit collapse inside a 10-bit container is a legal file of a legal
size with a picture that is merely posterised.

Three instruments, because they answer three different questions:

  1. CODE RAMP (synthetic, in gamut). Y and then U/V swept over all 1024 codes with the other components
     neutral. This isolates depth: if it does not round-trip at +-1 code, something quantises. A full-range
     RGB round trip is exact algebra for in-gamut YUV, so any delta here is depth or rounding.
  2. CLIP FRAMES (real, 10-bit HEVC). Distinct Y codes in and out. An 8-bit collapse cannot exceed 256 codes;
     the material carries ~650-810.
  3. CLIPPING DIAGNOSTIC. On the real frames, the share of pixels whose RGB leaves [0, 1] in the reader's own
     conversion. The reader clamps there, so a saturated pixel cannot round-trip - that is gamut clipping in
     a pre-existing conversion, not a depth loss, and the two must not be reported as one number.

Usage: python harness/scripts/tenbit_pipe_probe.py [--frames 5] [--out-file x.json]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "backend"))

from src.pytorch.TorchUtils import TorchUtils  # noqa: E402

APP = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2"
CLIP = "/home/jorj/2Tb/remastering/native1440_20s.mkv"
W, H = 1440, 1080


def decode(path: str, frame: int, w: int, h: int) -> bytes:
    """One frame of the clip as packed yuv420p10le, exactly what the reader gets."""
    cmd = [f"{APP}/bin/ffmpeg", "-v", "error", "-ss", f"{frame / 25.0:.3f}", "-i", path,
           "-frames:v", "1", "-pix_fmt", "yuv420p10le", "-f", "rawvideo", "-s", f"{w}x{h}", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    want = w * h + 2 * (h // 2) * (w // 2)
    if len(out) != want * 2:
        raise SystemExit(f"decoded {len(out)} bytes, expected {want * 2} for {w}x{h}")
    return out


def roundtrip(tu, st, raw: bytes, n: int, c: int):
    """reader -> tensor -> packer, on the app's own two functions."""
    tensor = tu.yuv420p10_to_tensor(raw, st, torch.device("cuda", 0), torch.float16)
    packed = tu.tensor_to_yuv420p10(tensor, "yuv420p10le")
    return tensor, np.asarray(packed).reshape(-1).astype(np.int32)


def luma_ramp(tu, st, n: int, c: int) -> dict:
    """Every LEGAL luma code once (64..940), chroma neutral: depth, with no gamut and no clamping involved.

    Codes outside the limited range are excluded on purpose - the packer clamps to 64..940 by design, so
    including them would measure the clamp (83 codes at the top) and call it a depth loss.
    """
    codes = np.arange(64, 941, dtype=np.uint16)
    y = np.full(n, 512, np.uint16)
    y[: codes.size] = codes
    raw = np.concatenate([y, np.full(c, 512, np.uint16), np.full(c, 512, np.uint16)]).tobytes()
    _tensor, got = roundtrip(tu, st, raw, n, c)
    want = np.frombuffer(raw, dtype=np.uint16).astype(np.int32)
    d = np.abs(got[: codes.size] - want[: codes.size])
    return dict(plane="Y", max_delta=int(d.max()), codes=int(d.size),
                share_gt1_pct=float((d > 1).mean() * 100.0), first=int(codes[0]), last=int(codes[-1]))


def chroma_uniform(tu, st, n: int, c: int, plane: str, band: float = 0.01) -> dict:
    """Every in-gamut chroma code as a UNIFORM plane, one frame each: the cleanest depth instrument.

    Constant planes are the identity under the reader's bilinear upsample followed by the packer's box
    filter, so a delta here is depth or rounding and nothing else. Luma is co-varied (`B = 0.5` for U,
    `R = 0.5` for V) to keep RGB inside the gamut, and codes whose RGB lands within `band` of 0 or 1 are
    excluded - the reader clamps there, and a clipped channel cannot round-trip through RGB by construction.
    """
    bad, swept, worst, swept_codes = [], 0, (0, None), []
    for code in range(64, 961):
        u_norm = (code - 512.0) / 896.0 if plane == "U" else 0.0
        v_norm = (code - 512.0) / 896.0 if plane == "V" else 0.0
        k = 1.8556 if plane == "U" else 1.5748
        y_norm = 0.5 - k * (u_norm if plane == "U" else v_norm)
        g_norm = y_norm - 0.1873 * u_norm - 0.4681 * v_norm
        rgb = (y_norm + 1.5748 * v_norm, g_norm, y_norm + 1.8556 * u_norm)
        if not all(band <= v <= 1.0 - band for v in rgb):
            continue
        y_code = int(round(y_norm * 876.0 + 64.0))
        if not (64 <= y_code <= 940):
            continue
        if not (band <= (y_code - 64) / 876.0 <= 1.0 - band):
            continue
        u = np.full(c, code if plane == "U" else 512, np.uint16)
        v = np.full(c, code if plane == "V" else 512, np.uint16)
        raw = np.concatenate([np.full(n, y_code, np.uint16), u, v]).tobytes()
        _tensor, got = roundtrip(tu, st, raw, n, c)
        off = n if plane == "U" else n + c
        d = int(np.abs(got[off:off + c] - code).max())
        swept += 1
        swept_codes.append(code)
        if d > 1:
            bad.append(code)
        if d > worst[0]:
            worst = (d, code)
    legal = 961 - 64
    return dict(plane=plane, max_delta=worst[0], worst_code=worst[1], codes=swept,
                swept_codes=swept_codes,
                share_gt1_pct=float(100.0 * len(bad) / max(1, swept)),
                coverage_pct=float(100.0 * swept / legal))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=CLIP)
    ap.add_argument("--frames", type=int, default=5, help="frames, spread over the clip")
    ap.add_argument("--out-file", default="")
    args = ap.parse_args(argv)

    dev = torch.device("cuda", 0)
    tu = TorchUtils(width=W, height=H, device_type="cuda", hdr_mode=False, gpu_id=0,
                    high_bitdepth=True)
    st = torch.cuda.Stream(device=dev)
    n, c = W * H, (H // 2) * (W // 2)

    ramps = [luma_ramp(tu, st, n, c), chroma_uniform(tu, st, n, c, "U"),
             chroma_uniform(tu, st, n, c, "V")]
    for r in ramps:
        extra = (f", {r['coverage_pct']:.0f}% of the legal range in gamut, worst code "
                 f"{r['worst_code']}") if "coverage_pct" in r else ""
        print(f"{r['plane']}: max|delta| {r['max_delta']} over {r['codes']} codes{extra}, "
              f"{r['share_gt1_pct']:.2f}% off by more than one")

    rows = []
    for k in range(args.frames):
        idx = int(k * 494 / max(1, args.frames))
        raw = decode(args.clip, idx, W, H)
        src = np.frombuffer(raw, dtype=np.uint16)
        tensor, got = roundtrip(tu, st, raw, n, c)
        want = src.astype(np.int32)
        dy = np.abs(got[:n] - want[:n])
        # the reader's own RGB math, in numpy, to say how much of the residual is clamping: in-gamut YUV
        # maps into [0, 1] by construction, so anything outside it is a saturated pixel the reader clips
        yv = ((want[:n] - 64.0) / 876.0).reshape(H, W)
        uv = (want[n:n + 2 * c].reshape(2, H // 2, W // 2) - 512.0) / 896.0
        uu = np.repeat(np.repeat(uv[0], 2, 0), 2, 1)[:H, :W]
        vv = np.repeat(np.repeat(uv[1], 2, 0), 2, 1)[:H, :W]
        rgb = np.stack([yv + 1.5748 * vv, yv - 0.1873 * uu - 0.4681 * vv, yv + 1.8556 * uu])
        # a guard band, not just the [0,1] boundary: a channel within 1% of it can still clip after the
        # reader's fp16 rounding, so those pixels cannot be judged for depth by this instrument
        band = 0.01
        clipped = ((rgb < band) | (rgb > 1.0 - band)).any(axis=0)
        in_range = ~clipped.reshape(-1)
        rows.append(dict(
            frame=idx,
            y_max_delta=int(dy.max()),
            y_max_delta_in_gamut=int(dy[in_range].max()) if in_range.any() else 0,
            y_p99_delta=float(np.percentile(dy, 99)),
            share_le1_pct=float((dy <= 1).mean() * 100.0),
            clipped_pct=float(clipped.mean() * 100.0),
            in_levels=int(np.unique(want[:n]).size),
            out_levels=int(np.unique(got[:n]).size),
        ))
        r = rows[-1]
        print(f"frame {idx:4d}: Y max|d| {r['y_max_delta']} (in gamut {r['y_max_delta_in_gamut']}), "
              f"p99 {r['y_p99_delta']:.2f}, {r['share_le1_pct']:.3f}% within one level; "
              f"codes in/out {r['in_levels']} -> {r['out_levels']}; clipped {r['clipped_pct']:.2f}%")

    clip_summary = {
        "y_max_delta_in_gamut": max(r["y_max_delta_in_gamut"] for r in rows),
        "y_p99_delta_worst": max(r["y_p99_delta"] for r in rows),
        "share_le1_pct_min": min(r["share_le1_pct"] for r in rows),
        "clipped_pct_max": max(r["clipped_pct"] for r in rows),
        "in_levels_min": min(r["in_levels"] for r in rows),
        "out_levels_min": min(r["out_levels"] for r in rows),
    }
    print(f"\nworst over {len(rows)} frames: in-gamut Y max|d| {clip_summary['y_max_delta_in_gamut']}, "
          f"p99 {clip_summary['y_p99_delta_worst']:.2f}, {clip_summary['share_le1_pct_min']:.3f}% <=1 level; "
          f"codes in >= {clip_summary['in_levels_min']}, out >= {clip_summary['out_levels_min']}; "
          f"clipped {clip_summary['clipped_pct_max']:.2f}%")
    print("scope: reader+packer only (no model); the model's precision is a different question")

    if args.out_file:
        os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
        json.dump({"take": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "clip": os.path.basename(args.clip), "width": W, "height": H,
                   "ramps": ramps, "frames": rows, "summary": clip_summary},
                  open(args.out_file, "w"), indent=1)
        print(f"# -> {args.out_file}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
