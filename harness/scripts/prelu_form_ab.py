"""Acceptance + interleaved timing for a set of TensorRT plans, one instrument, one session.

This is the instrument behind harness/docs/wsp/verified/prelu-form-int8.txt: the picture statistic is the lane's own
(share of pixels more than 8 levels off in any channel, worst channel, worst frame of a spread sample, bar
0.1%), the reference is the fp16 engine of the unmodified ONNX, and the timing alternates the arms inside one
session so the drift is common to both.

Usage:
  python3 harness/scripts/prelu_form_ab.py --ref <fp16.plan> --arm name=<plan> [--arm name=<plan> ...] \
      --frames <dir of raw rgb24 frames> [--width W --height H] [--rounds 2] [--iters 20]

Prints per-arm: worst frame, frames over the 0.1% bar, median share, max peak; then the interleaved timing
rounds and their medians. The uncovered scope is printed last, because a green table is a floor.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import tensorrt as trt

_DTYPES = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16}


def load(path: str):
    rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    eng = rt.deserialize_cuda_engine(open(path, "rb").read())
    if eng is None:
        raise SystemExit(f"{path}: the runtime refused the plan")
    ctx = eng.create_execution_context()
    names = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
    inp = next(n for n in names if eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
    out = next(n for n in names if eng.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
    return {"ctx": ctx, "in": inp, "out": out,
            "dtype": _DTYPES[eng.get_tensor_dtype(inp)],
            "shape": tuple(ctx.get_tensor_shape(out))}


def run(a, x):
    y = torch.empty(a["shape"], device="cuda", dtype=a["dtype"])
    a["ctx"].set_tensor_address(a["in"], int(x.data_ptr()))
    a["ctx"].set_tensor_address(a["out"], int(y.data_ptr()))
    a["ctx"].execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return y


def codes(t):
    return t.clamp(0, 1).mul(255.0).round()


def frame(path, h, w, dtype):
    raw = np.fromfile(path, dtype=np.uint8).reshape(h, w, 3).astype(np.float32) / 255.0
    return torch.from_numpy(raw).permute(2, 0, 1)[None].contiguous().cuda().to(dtype)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--arm", action="append", required=True, help="name=path")
    ap.add_argument("--frames", required=True)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()

    arms = dict(s.split("=", 1) for s in a.arm)
    frames = sorted(glob.glob(os.path.join(a.frames, "*.rgb")))
    if not frames:
        raise SystemExit(f"{a.frames}: no .rgb frames")
    torch.zeros(1, device="cuda")
    ref = load(a.ref)
    arms = {k: load(v) for k, v in arms.items()}

    rows = []
    for path in frames:
        x = frame(path, a.height, a.width, ref["dtype"])
        r = codes(run(ref, x))
        for name, arm in arms.items():
            d = (codes(run(arm, x)) - r).abs().amax(dim=1)
            rows.append({"frame": os.path.basename(path), "arm": name,
                         "share": float((d > 8).float().mean()) * 100.0,
                         "peak": int(d.max()), "mean": float(d.mean())})
        del x, r

    summary = {}
    for name in arms:
        r = [z for z in rows if z["arm"] == name]
        w = max(r, key=lambda z: z["share"])
        summary[name] = {"worst %": round(w["share"], 5), "at": w["frame"], "peak@worst": w["peak"],
                         "frames>0.1%": sum(1 for z in r if z["share"] > 0.1), "n": len(r),
                         "median %": round(float(np.median([z["share"] for z in r])), 5),
                         "max peak": max(z["peak"] for z in r)}
    print(json.dumps(summary, indent=1))

    x = frame(frames[0], a.height, a.width, ref["dtype"])
    ms = {k: [] for k in arms}
    for k in arms:
        run(arms[k], x)
    for _ in range(2):
        for k, arm in arms.items():
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record()
            for _ in range(a.iters):
                arm["ctx"].execute_async_v3(torch.cuda.current_stream().cuda_stream)
            e1.record()
            torch.cuda.synchronize()
            ms[k].append(e0.elapsed_time(e1) / a.iters)
    med = {k: round(float(np.median(v)), 2) for k, v in ms.items()}
    print("rounds ms:", {k: [round(z, 2) for z in v] for k, v in ms.items()})
    print("medians ms:", med)
    print("NOT checked: 4K unless built there, the app render path, NVENC, a second card, "
          "and whether the frame sample is representative of the title")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
