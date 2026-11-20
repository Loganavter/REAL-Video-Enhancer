"""Calibration on crops against calibration on full frames: cost and picture, same instrument.

`int8-calibration-crops.txt` measured this on 2026-11-18 (0.45 s against 7.09 s, worst held-out share 0.01881%
against 0.02256%) but the path was never wired. This probe measures the wired version: same model, same frames,
same held-out set, only `RVE_INT8_CALIB_CROPS` differing.

Reported: calibration wall time (the block that costs 12.62 s in the app), build time, and the picture against the
fp16 model on frames that are in neither calibration arm.

Usage:
  RVE_TAIL_YUV=0 RVE_INT8_NO_EVENTS=1 python3 harness/scripts/calib_crops_probe.py \
      --calib /tmp/opencode/calib-frames --frames /tmp/opencode/eval3 --out-file /tmp/opencode/calib-crops.json
"""
from __future__ import annotations

import argparse
import glob
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

MODEL = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2/custom_models/2x-Futsuu-Anime(1).pth"


def frame_tensor(path: str, h: int, w: int) -> torch.Tensor:
    raw = np.fromfile(path, dtype=np.uint8).reshape(h, w, 3)
    return torch.from_numpy(raw).permute(2, 0, 1).float().div_(255.0)[None].contiguous().cuda().half()


def run(model, calib, evals, crops: int, k: str | None = None) -> dict:
    os.environ["RVE_INT8_CALIB_CROPS"] = str(crops)
    os.environ["RVE_INT8_CALIB_CACHE"] = "0"          # measure, do not read someone else's grids
    if k is not None:
        os.environ["RVE_INT8_GRID_K"] = k
    body = Int8Body(model)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        body.calibrate(calib)
    torch.cuda.synchronize()
    t_cal = time.time() - t0
    t1 = time.time()
    with torch.no_grad():
        body._build()
    torch.cuda.synchronize()
    t_build = time.time() - t1
    rows = {}
    for name, x in evals:
        with torch.no_grad():
            ref = model(x).float()
            out = body(x).float()
        d = (out - ref).abs() * 255.0
        rows[name] = {"share_gt8_pct": round(float((d > 8).float().mean() * 100), 6),
                      "max_codes": round(float(d.max()), 3), "mean_codes": round(float(d.mean()), 5)}
    del body
    torch.cuda.empty_cache()
    return {"crops": crops, "calibrate_s": round(t_cal, 3), "build_s": round(t_build, 3), "frames": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--calib", default="/tmp/opencode/calib-frames")
    ap.add_argument("--frames", default="/tmp/opencode/eval3")
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--crops", default="0,4")
    ap.add_argument("--k", default="2")
    ap.add_argument("--out-file", default="/tmp/opencode/calib-crops.json")
    a = ap.parse_args(argv)

    model, scale, _ = load(a.model)
    model = model.eval().cuda().half()
    calib = [frame_tensor(p, a.height, a.width) for p in sorted(glob.glob(os.path.join(a.calib, "*.rgb")))]
    evals = [(os.path.basename(p), frame_tensor(p, a.height, a.width))
             for p in sorted(glob.glob(os.path.join(a.frames, "*.rgb")))]

    res = {"model": os.path.basename(a.model), "scale": scale, "k": a.k, "arms": []}
    for crops in [int(c) for c in a.crops.split(",")]:
        arm = run(model, calib, evals, crops, k=a.k)
        res["arms"].append(arm)
        print(f"crops={crops:2d}  calibrate {arm['calibrate_s']:6.2f} s  build {arm['build_s']:5.2f} s   " +
              "  ".join(f"{n.split('.')[0]} {r['share_gt8_pct']:.5f}%" for n, r in arm["frames"].items()),
              flush=True)

    base = res["arms"][0]
    for arm in res["arms"][1:]:
        arm["speedup_vs_full"] = round(base["calibrate_s"] / max(arm["calibrate_s"], 1e-6), 2)
        print(f"crops={arm['crops']}: {arm['speedup_vs_full']}x cheaper to calibrate")
    with open(a.out_file, "w") as f:
        json.dump(res, f, indent=1)
    print(f"# -> {a.out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
