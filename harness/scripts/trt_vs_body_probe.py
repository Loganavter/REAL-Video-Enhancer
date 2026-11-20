"""The Triton int8 body against TensorRT's own calibrated int8, in ONE process, on the same frame.

`trtexec` cannot time the body: it loads TensorRT plans, and the body is a set of Triton kernels inside the
pipeline. So the only way to get a same-instrument, same-scope ratio is to run both arms here - one CUDA event
bracket, one input tensor, interleaved rounds - and take the median of rounds 2+, exactly as `wall_ab.sh` does for
the app-level arms.

Scope is the model: input frame -> output frame. Both arms cover the same graph (head, hops, tail, shuffle,
residual, and the LR upsample that the tail emulates by adding one value to four subpixels). The app's pack, queue
and copies are outside both.

The body's calibration here is eight raw frames, not the app's pipeline frames, so the ACCURACY numbers are a
probe's numbers; the shipped pair stays the 494-frame one in `int8-tail-csc.txt`. The timing is the point.

Usage:
  RVE_TAIL_YUV=0 python3 harness/scripts/trt_vs_body_probe.py --fp16 /tmp/opencode/futsuu_fp16.trt \
      --int8 /tmp/opencode/futsuu_int8_entropy.trt --calib /tmp/opencode/calib-frames \
      --frames /tmp/opencode/eval-frames --rounds 4 --iters 10 --out-file /tmp/opencode/trt-vs-body.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys

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
from trt_int8_accuracy import Engine  # noqa: E402

MODEL = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2/custom_models/2x-Futsuu-Anime(1).pth"


def frame_tensor(path: str, h: int, w: int) -> torch.Tensor:
    raw = np.fromfile(path, dtype=np.uint8).reshape(h, w, 3)
    return (torch.from_numpy(raw).permute(2, 0, 1).float().div_(255.0)[None].contiguous().cuda().half())


def time_arm(fn, iters: int) -> float:
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp16", required=True)
    ap.add_argument("--int8", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out-file", default="/tmp/opencode/trt-vs-body.json")
    a = ap.parse_args(argv)

    calib = sorted(glob.glob(os.path.join(a.calib, "*.rgb")))
    evals = sorted(glob.glob(os.path.join(a.frames, "*.rgb")))
    if not calib or not evals:
        raise SystemExit("need .rgb frames in --calib and --frames")

    model, scale, _ = load(a.model)
    model = model.eval().cuda().half()
    body = Int8Body(model)
    with torch.no_grad():
        body.calibrate([frame_tensor(p, a.height, a.width) for p in calib])
        body._build()

    fp16, int8 = Engine(a.fp16), Engine(a.int8)
    x = frame_tensor(evals[0], a.height, a.width)

    with torch.no_grad():
        for _ in range(3):
            fp16(x), int8(x), body(x)
    torch.cuda.synchronize()

    arms = {"fp16-engine": lambda: fp16(x), "trt-int8": lambda: int8(x), "triton-body": lambda: body(x)}
    rounds: dict[str, list[float]] = {k: [] for k in arms}
    for r in range(a.rounds):
        for name, fn in arms.items():          # interleaved: drift is common to all arms
            rounds[name].append(round(time_arm(fn, a.iters), 4))
        print(f"round {r + 1}: " + "  ".join(f"{k} {rounds[k][-1]:.3f}" for k in arms), flush=True)

    med = {k: round(statistics.median(v[1:]), 4) for k, v in rounds.items()}   # rounds 2+
    print("\nmedian of rounds 2+: " + "  ".join(f"{k} {med[k]:.3f} ms" for k in med))

    # accuracy on EVERY held-out frame, not just the one that was timed: the two frames disagree about
    # TensorRT's int8 (5.85 codes on the bright one, 91 on the dark one), and the body has to be read the same way
    acc = {}
    for path in evals:
        xf = frame_tensor(path, a.height, a.width)
        with torch.no_grad():
            ref, out_int8, out_body = fp16(xf).float(), int8(xf).float(), body(xf).float()
        acc[os.path.basename(path)] = {}
        for name, out in (("trt-int8", out_int8), ("triton-body", out_body)):
            d = (out - ref).abs() * 255.0
            mse = float(((out - ref) ** 2).mean())
            row = {"max_codes": round(float(d.max()), 3), "mean_codes": round(float(d.mean()), 5),
                   "share_gt8_pct": round(float((d > 8).float().mean() * 100), 6),
                   "psnr_db": round(float(10 * np.log10(1.0 / max(mse, 1e-12))), 3),
                   "shape": list(out.shape)}
            acc[os.path.basename(path)][name] = row
            print(f"accuracy {os.path.basename(path):18s} {name:14s} max {row['max_codes']:7.3f} codes  "
                  f"mean {row['mean_codes']:.5f}  >8 {row['share_gt8_pct']:.4f}%  PSNR {row['psnr_db']:.2f} dB")

    res = {"model": a.model, "shape": [1, 3, a.height, a.width], "scale": scale,
           "rounds": rounds, "median_ms": med,
           "ratio_body_over_trt_int8": round(med["trt-int8"] / med["triton-body"], 4),
           "ratio_body_over_fp16": round(med["fp16-engine"] / med["triton-body"], 4),
           "ratio_trt_int8_over_fp16": round(med["fp16-engine"] / med["trt-int8"], 4),
           "accuracy": acc, "frames": [os.path.basename(p) for p in evals],
           "calibration_frames": [os.path.basename(p) for p in calib]}
    with open(a.out_file, "w") as f:
        json.dump(res, f, indent=1)
    print(f"# -> {a.out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
