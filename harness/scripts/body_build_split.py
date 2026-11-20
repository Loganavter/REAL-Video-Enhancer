"""Split the body's first-frame block: calibration, build, Triton's first call, and the frame itself.

The start-up split left one line unsplit - "Triton JIT + 24 calibration frames + first body call" - and a line
that is not split is the question "where do the seconds go" coming back. This probe times each part separately,
and runs the whole thing twice: once with an empty Triton cache directory (what a first-ever run pays) and once
with the normal cache (what a re-render pays).

Usage:
  RVE_TAIL_YUV=0 RVE_INT8_NO_EVENTS=1 python3 harness/scripts/body_build_split.py --cold-dir /tmp/opencode/triton-cold
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--calib", default="/tmp/opencode/calib-frames")
    ap.add_argument("--frame", default="/tmp/opencode/eval3/violet_5.rgb")
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--cold-dir", default="/tmp/opencode/triton-cold")
    ap.add_argument("--out-file", default="/tmp/opencode/body-build-split.json")
    a = ap.parse_args(argv)

    calib = [frame_tensor(p, a.height, a.width) for p in sorted(glob.glob(os.path.join(a.calib, "*.rgb")))]
    x = frame_tensor(a.frame, a.height, a.width)
    model, _, _ = load(a.model)
    model = model.eval().cuda().half()

    res = {"model": os.path.basename(a.model), "calibration_frames": len(calib), "arms": {}}
    for arm in ("warm", "cold"):
        if arm == "cold":
            import shutil
            shutil.rmtree(a.cold_dir, ignore_errors=True)
            os.makedirs(a.cold_dir, exist_ok=True)
            os.environ["TRITON_CACHE_DIR"] = a.cold_dir
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
        t2 = time.time()
        with torch.no_grad():
            body(x)
        torch.cuda.synchronize()
        t_first = time.time() - t2
        later = []
        for _ in range(5):
            t3 = time.time()
            with torch.no_grad():
                body(x)
            torch.cuda.synchronize()
            later.append(time.time() - t3)
        res["arms"][arm] = {"calibrate_s": round(t_cal, 3), "build_s": round(t_build, 3),
                            "first_call_s": round(t_first, 3),
                            "steady_call_s": round(statistics.median(later), 4)}
        print(f"{arm:5s} calibrate {t_cal:6.3f}  build {t_build:6.3f}  first call {t_first:6.3f}  "
              f"steady {res['arms'][arm]['steady_call_s']:6.4f} s", flush=True)
        del body
        torch.cuda.empty_cache()
    with open(a.out_file, "w") as f:
        json.dump(res, f, indent=1)
    print(f"# -> {a.out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
