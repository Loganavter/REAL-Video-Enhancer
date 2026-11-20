"""Compare a TensorRT int8 engine against the fp16 engine of the same model, on held-out frames.

The comparator question is narrow: TensorRT's own int8 (calibrated, implicit quantization) against the fp16
engine on the same card and shape. A speed number alone would say "int8 is faster" without saying what it did
to the picture, and this project's int8 body is judged on picture pairs - so the same discipline here:

  max |delta| in 8-bit codes, mean |delta|, the share of samples off by more than 1, 2 and 8 codes (the last
is the threshold `engine-int8.txt` uses, so the two pages are comparable), and PSNR.

Frames must be held out from calibration: a delta measured on the calibration set is a fit, not a result.

Usage:
  python3 harness/scripts/trt_int8_accuracy.py --fp16 /tmp/opencode/futsuu_fp16.trt \
      --int8 /tmp/opencode/futsuu_int8_entropy.trt --int8 /tmp/opencode/futsuu_int8_minmax.trt \
      --frames /tmp/opencode/eval-frames --height 1080 --width 1920
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import tensorrt as trt
import torch


class Engine:
    def __init__(self, path: str) -> None:
        self.path = path
        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)
        with open(path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise SystemExit(f"{path}: deserialization failed")
        self.ctx = self.engine.create_execution_context()
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in self.names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in self.names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

    _DT = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16,
           trt.DataType.INT8: torch.int8, trt.DataType.INT32: torch.int32}

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        want = self._DT[self.engine.get_tensor_dtype(self.inputs[0])]
        x = x.to(want).contiguous()
        self.ctx.set_input_shape(self.inputs[0], tuple(x.shape))
        for n in self.inputs:
            self.ctx.set_tensor_address(n, int(x.data_ptr()))
        outs = []
        for n in self.outputs:
            shape = tuple(self.ctx.get_tensor_shape(n))
            t = torch.empty(shape, dtype=self._DT[self.engine.get_tensor_dtype(n)], device="cuda")
            self.ctx.set_tensor_address(n, int(t.data_ptr()))
            outs.append(t)
        if not self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise SystemExit(f"{self.path}: execute failed")
        torch.cuda.synchronize()
        return outs[0]


def load_frame(path: str, h: int, w: int) -> torch.Tensor:
    raw = np.fromfile(path, dtype=np.uint8).reshape(h, w, 3)
    x = torch.from_numpy(raw).permute(2, 0, 1).float().div_(255.0)[None].contiguous().cuda()
    return x


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp16", required=True)
    ap.add_argument("--int8", action="append", default=[])
    ap.add_argument("--frames", required=True)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--out", default="/tmp/opencode/trt-int8-accuracy.json")
    a = ap.parse_args(argv)

    frames = sorted(glob.glob(os.path.join(a.frames, "*.rgb")))
    if not frames:
        raise SystemExit(f"{a.frames}: no .rgb frames")

    ref_engine = Engine(a.fp16)
    arms = {os.path.basename(p): Engine(p) for p in a.int8}
    result = {"fp16_engine": a.fp16, "frames": {}, "arms": {}}

    for path in frames:
        x = load_frame(path, a.height, a.width)
        with torch.no_grad():
            ref = ref_engine(x)
        name = os.path.basename(path)
        result["frames"][name] = {}
        for arm, eng in arms.items():
            with torch.no_grad():
                out = eng(x)
            d = (out - ref).abs() * 255.0
            mse = float(((out - ref) ** 2).mean())
            row = {
                "max_codes": round(float(d.max()), 4),
                "mean_codes": round(float(d.mean()), 5),
                "share_gt1_pct": round(float((d > 1.0).float().mean() * 100), 6),
                "share_gt2_pct": round(float((d > 2.0).float().mean() * 100), 6),
                "share_gt8_pct": round(float((d > 8.0).float().mean() * 100), 6),
                "psnr_db": round(float(10 * np.log10(1.0 / max(mse, 1e-12))), 4),
                "out_range": [round(float(out.min()), 5), round(float(out.max()), 5)],
                "ref_range": [round(float(ref.min()), 5), round(float(ref.max()), 5)],
            }
            result["frames"][name][arm] = row
            print(f"{name:16s} {arm:28s} max {row['max_codes']:7.3f} codes  mean {row['mean_codes']:.5f}  "
                  f">1 code {row['share_gt1_pct']:.4f}%  PSNR {row['psnr_db']:.2f} dB", flush=True)

    # one summary per arm over the held-out set
    for arm in arms:
        rows = [result["frames"][f][arm] for f in result["frames"]]
        result["arms"][arm] = {
            "max_codes": max(r["max_codes"] for r in rows),
            "mean_codes": round(float(np.mean([r["mean_codes"] for r in rows])), 5),
            "share_gt1_pct": round(float(np.mean([r["share_gt1_pct"] for r in rows])), 6),
            "psnr_db": round(float(np.mean([r["psnr_db"] for r in rows])), 4),
        }
        print(f"== {arm}: worst max {result['arms'][arm]['max_codes']:.3f} codes, "
              f"mean {result['arms'][arm]['mean_codes']:.5f}, "
              f"PSNR {result['arms'][arm]['psnr_db']:.2f} dB")
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"# -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
