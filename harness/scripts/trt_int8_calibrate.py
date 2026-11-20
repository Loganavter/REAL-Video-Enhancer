"""Build the compact net in TensorRT with implicit-quantization int8, calibrated on our own frames.

Why: the lane closed the *explicit* Q/DQ route (lossless, slower) but never built TensorRT's own int8
with a calibrator - the one comparison that says whether the hand-written Triton body was needed at
all. This builds three engines from the same ONNX with the same builder settings:

  fp16            reference arm, same builder path (not trtexec's, so the pair is same-instrument)
  int8 entropy    IInt8EntropyCalibrator2 over the named frames
  int8 minmax     IInt8MinMaxCalibrator over the same frames

and writes the calibration cache, so the run can be reproduced without re-calibrating. Timing and
per-layer precision are read by `trtexec --loadEngine --dumpProfile --dumpLayerInfo`, i.e. by the
same instrument that produced the fp16 budget in `harness/docs/wsp/verified/fp16-engine-layers.txt`.

The frames are the population: their sha256 goes into the manifest, because "calibrated on 8 frames"
without naming them is the defect this project keeps re-learning.

Usage:
  python3 harness/scripts/trt_int8_calibrate.py --onnx /tmp/opencode/futsuu.onnx \
      --frames /tmp/opencode/calib-frames --height 1080 --width 1920 --out /tmp/opencode
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import time

import numpy as np
import tensorrt as trt
import torch


class _Frames:
    """Calibration data: raw RGB24 frames, one per batch, device buffer owned by torch.

    A mixin, not a base: TensorRT's calibrator classes refuse an `__init__` that does not call their own,
    so each algorithm gets a thin subclass that calls `IInt8*.__init__(self)` first.
    """

    def __init__(self, paths: list[str], h: int, w: int, cache: str) -> None:
        super().__init__()
        self.paths, self.h, self.w, self.cache = paths, h, w, cache
        self.i = 0
        self.buf = torch.empty(3 * h * w, dtype=torch.float32, device="cuda")

    def get_batch_size(self) -> int:  # noqa: D102
        return 1

    def get_batch(self, names):  # noqa: D102
        if self.i >= len(self.paths):
            return None
        raw = np.fromfile(self.paths[self.i], dtype=np.uint8).reshape(self.h, self.w, 3)
        x = torch.from_numpy(raw).permute(2, 0, 1).float().div_(255.0).contiguous().cuda()
        self.buf.copy_(x.reshape(-1))
        self.i += 1
        return [int(self.buf.data_ptr())]

    def read_calibration_cache(self):  # noqa: D102
        return open(self.cache, "rb").read() if os.path.exists(self.cache) else None

    def write_calibration_cache(self, cache):  # noqa: D102
        with open(self.cache, "wb") as f:
            f.write(bytes(cache))


class Entropy(_Frames, trt.IInt8EntropyCalibrator2):  # type: ignore[misc]
    """Entropy calibration (the default choice for this kind of model)."""

    def __init__(self, paths, h, w, cache) -> None:
        trt.IInt8EntropyCalibrator2.__init__(self)
        _Frames.__init__(self, paths, h, w, cache)


class MinMax(_Frames, trt.IInt8MinMaxCalibrator):  # type: ignore[misc]
    """The same data through the min-max algorithm, for the algorithm's own effect."""

    def __init__(self, paths, h, w, cache) -> None:
        trt.IInt8MinMaxCalibrator.__init__(self)
        _Frames.__init__(self, paths, h, w, cache)


def build(onnx: str, out: str, *, int8: str | None, frames: list[str], h: int, w: int,
          opt: int, workspace_gib: int = 6) -> dict:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    with open(onnx, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("parse:", parser.get_error(i))
            raise SystemExit("ONNX parse failed")
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.builder_optimization_level = opt
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gib << 30)
    cal = None
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        cache = os.path.join(os.path.dirname(out), f"calib-{int8}.cache")
        cal = (Entropy if int8 == "entropy" else MinMax)(frames, h, w, cache)
        config.int8_calibrator = cal
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    dt = time.time() - t0
    if serialized is None:
        raise SystemExit(f"build failed for {out}")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)   # --out is a directory nobody had created yet
    with open(out, "wb") as f:
        f.write(bytes(serialized))
    return {"engine": out, "build_s": round(dt, 3), "bytes": os.path.getsize(out),
            "int8": int8 or "", "opt": opt,
            "calibration_cache": getattr(cal, "cache", "")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--opt", type=int, default=5)
    ap.add_argument("--out", default="/tmp/opencode")
    ap.add_argument("--arms", default="fp16,int8-entropy,int8-minmax")
    a = ap.parse_args(argv)

    frames = sorted(glob.glob(os.path.join(a.frames, "*.rgb")))
    if not frames:
        raise SystemExit(f"{a.frames}: no .rgb frames")
    need = a.height * a.width * 3
    for p in frames:
        if os.path.getsize(p) != need:
            raise SystemExit(f"{p}: {os.path.getsize(p)} bytes is not {a.width}x{a.height}x3")
    digests = {os.path.basename(p): hashlib.sha256(open(p, "rb").read()).hexdigest()[:16] for p in frames}
    print(f"frames: {len(frames)} at {a.width}x{a.height}, {json.dumps(digests)}", flush=True)

    manifest_path = os.path.join(a.out, "trt-int8-manifest.json")
    if os.path.exists(manifest_path):          # a later --arms run must not erase the earlier arms
        manifest = json.load(open(manifest_path))
        manifest["arms"] = [r for r in manifest["arms"] if r.get("int8", "") not in
                            {("" if x == "fp16" else x.split("-", 1)[1]) for x in a.arms.split(",")}]
    else:
        manifest = {"onnx": a.onnx,
                    "onnx_sha256": hashlib.sha256(open(a.onnx, "rb").read()).hexdigest()[:16],
                    "frames": digests, "shape": [1, 3, a.height, a.width], "opt": a.opt, "arms": []}
    for arm in a.arms.split(","):
        kind = None if arm == "fp16" else arm.split("-", 1)[1]
        out = os.path.join(a.out, f"futsuu_{arm.replace('-', '_')}.trt")
        print(f"=== building {arm} -> {out}", flush=True)
        rec = build(a.onnx, out, int8=kind, frames=frames, h=a.height, w=a.width, opt=a.opt)
        print(f"    {rec['build_s']} s, {rec['bytes'] / 2**20:.1f} MiB", flush=True)
        manifest["arms"].append(rec)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=1)
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
