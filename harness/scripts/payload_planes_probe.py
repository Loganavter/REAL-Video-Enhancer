"""Native chroma: compare the pipe payloads plane by plane, before the sink.

The 494-frame ffv1 pair compares two CONFIGURATIONS through a 4:4:4 sink, which means ffmpeg upsamples
4:2:0 to 4:4:4 on both arms first - a chroma defect shared by both arms (or introduced by that upsample)
is invisible there, and that limitation is written on the page. This probe removes the sink: both arms
produce the 12.44 MB yuv420p payload the writer would pipe, and the comparison happens on the Y, U and V
planes as they are.

Two comparisons, because they answer different questions:

  lane vs fp16   the int8 body's payload against the fp16 model's payload, both packed to yuv420p. This is
                 the lane's own error ON CHROMA, which the sinks cannot show (the RGB gate reports luma and
                 colour mixed in RGB).
  config vs config  the yuv tail against the RGB tail on the SAME codes - the arithmetic-order cost of
                 in-kernel colour, which the 494-frame pair already covered in RGB and this shows per plane.

Usage: python harness/scripts/payload_planes_probe.py [--frames 4] [--out-file x.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "backend"))
import os as _bos, sys as _bsys
_HERE = _bos.path.dirname(_bos.path.abspath(__file__))
_ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in _bsys.path:
        _bsys.path.insert(0, _p)

from cut_channels_naive import load  # noqa: E402
from src.pytorch.Int8Body import Int8Body  # noqa: E402
from src.pytorch.TorchUtils import TorchUtils  # noqa: E402
from tail_int8_emulation import CALIB, CLIP, H, W, decode_yuv10  # noqa: E402

APP = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2"
MODEL = f"{APP}/custom_models/2x-Futsuu-Anime(1).pth"
BAR = 8.0


def planes(payload: np.ndarray, n: int, c: int) -> dict:
    return {"Y": payload[:n], "U": payload[n:n + c], "V": payload[n + c:n + 2 * c]}


def compare(a: np.ndarray, b: np.ndarray, n: int, c: int) -> dict:
    """Per-plane |delta| in 8-bit levels, plus the share over the project's 8-level threshold."""
    pa, pb = planes(a, n, c), planes(b, n, c)
    out = {}
    for name in ("Y", "U", "V"):
        d = np.abs(pa[name].astype(np.int16) - pb[name].astype(np.int16))
        out[name] = {"max": int(d.max()), "mean": float(d.mean()),
                     "share_gt8_pct": float(100.0 * (d > BAR).mean()),
                     "share_gt0_pct": float(100.0 * (d > 0).mean())}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=4, help="held-out frames, evenly spread")
    ap.add_argument("--out-file", default="")
    args = ap.parse_args(argv)

    dev = torch.device("cuda", 0)
    tu = TorchUtils(width=W, height=H, device_type="cuda", hdr_mode=True, gpu_id=0, high_bitdepth=True)
    st = torch.cuda.Stream(device=dev)

    def frame(i):
        return tu.yuv420p10_to_tensor(decode_yuv10(i), st, dev, torch.float16)

    model, _, _ = load(MODEL)
    model = model.eval().cuda().half()
    body = Int8Body(model)
    body.calibrate([frame(i) for i in CALIB])
    body._build()
    n, c = H * W, (H // 2) * (W // 2)

    rows = []
    for k in range(args.frames):
        idx = 100 + int(k * 300 / max(1, args.frames))     # held out: the calibration frames are CALIB
        x = frame(idx)
        with torch.no_grad():
            ref = model(x.clamp(0.0, 1.0))                  # the fp16 engine's own output
            got = body(x)                                   # yuv payload by default (RVE_TAIL_YUV=1)
            if got.dim() != 1:                              # gate refused: pack the RGB tensor the old way
                got = tu.tensor_to_yuv420p(got)
        fp16_payload = np.asarray(tu.tensor_to_yuv420p(ref)).reshape(-1)
        yuv_payload = np.asarray(got).reshape(-1)
        # config vs config: the same codes through the RGB tail, converted in the pack
        body.out_yuv = False
        with torch.no_grad():
            rgb_out = body(x)
        body.out_yuv = True
        rgb_payload = np.asarray(tu.tensor_to_yuv420p(rgb_out)).reshape(-1)

        row = {"frame": idx,
               "lane_vs_fp16": compare(yuv_payload, fp16_payload, n, c),
               "config_vs_config": compare(yuv_payload, rgb_payload, n, c)}
        rows.append(row)
        lv = row["lane_vs_fp16"]
        print(f"frame {idx:4d}: lane-vs-fp16  Y {lv['Y']['share_gt8_pct']:.4f}% "
              f"/ U {lv['U']['share_gt8_pct']:.4f}% / V {lv['V']['share_gt8_pct']:.4f}% of samples >8 levels "
              f"(max {lv['Y']['max']}/{lv['U']['max']}/{lv['V']['max']})")
        cv = row["config_vs_config"]
        print(f"            config-vs-config  Y {cv['Y']['max']} U {cv['U']['max']} V {cv['V']['max']} max levels")

    summary = {}
    for key in ("lane_vs_fp16", "config_vs_config"):
        summary[key] = {p: {"share_gt8_pct_worst": max(r[key][p]["share_gt8_pct"] for r in rows),
                            "max_worst": max(r[key][p]["max"] for r in rows),
                            "mean_worst": max(r[key][p]["mean"] for r in rows)}
                        for p in ("Y", "U", "V")}
    print(f"\nworst over {len(rows)} frames: lane-vs-fp16 share>8 levels "
          + ", ".join(f"{p} {summary['lane_vs_fp16'][p]['share_gt8_pct_worst']:.4f}%" for p in "YUV"))
    print("scope: payloads compared BEFORE any sink; no encoder, no ffv1, no chroma upsampling on either side")

    if args.out_file:
        os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
        json.dump({"take": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "clip": os.path.basename(CLIP), "height": H, "width": W,
                   "frames": rows, "summary": summary},
                  open(args.out_file, "w"), indent=1)
        print(f"# -> {args.out_file}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
