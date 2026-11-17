"""Calibrate a static Q/DQ model of the compact net on real frames.

The ledger's E9' build is "per-tensor activation scales calibrated on 8 of our own frames, per-channel
int8 weights, 16 body convs quantised, head and tail in FP16". This reproduces that pipeline with
onnxruntime's static quantisation: QDQ format, per-channel weights, QInt8 activations, and the first
and last convolution excluded so the head and tail stay in fp16.

The frames are the population: their sha256 goes into the manifest this writes, because "calibrated
on 8 frames" without naming them is the defect this project keeps re-learning.

Usage:
  python3 scripts/calibrate_qdq.py --onnx model.onnx --out model.qdq.onnx \
      --frames frames/ --width 1920 --height 1080
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np
import onnx
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static


class Frames(CalibrationDataReader):
    def __init__(self, paths: list[str], width: int, height: int) -> None:
        self.paths = paths
        self.width, self.height = width, height
        self.i = 0

    def _one(self, path: str) -> dict:
        raw = np.fromfile(path, dtype=np.uint8)
        if raw.size != self.width * self.height * 3:
            raise SystemExit(f"{path}: {raw.size} bytes is not {self.width}x{self.height}x3")
        x = raw.reshape(self.height, self.width, 3).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return {"input": x}

    def get_next(self):
        if self.i >= len(self.paths):
            return None
        path = self.paths[self.i]
        self.i += 1
        return self._one(path)

    def rewind(self) -> None:
        self.i = 0


def conv_nodes(model) -> list[str]:
    order = []
    for node in model.graph.node:
        if node.op_type == "Conv":
            order.append(node.name or f"conv_{len(order)}")
    return order


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--keep-fp16-head-tail", action="store_true", default=True)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.frames, "*.raw")))
    if len(paths) < 2:
        raise SystemExit(f"{args.frames}: need at least two .raw frames")
    model = onnx.load(args.onnx)
    convs = conv_nodes(model)
    exclude = []
    if args.keep_fp16_head_tail:
        # the first and the last convolution stay fp16: the ledger's build quantises the 16 body ones
        exclude = [c for c in (convs[0], convs[-1]) if c]
    print(f"{len(paths)} calibration frames, {len(convs)} conv nodes, excluding {exclude}")

    quantize_static(
        model_input=args.onnx, model_output=args.out,
        calibration_data_reader=Frames(paths, args.width, args.height),
        quant_format=QuantFormat.QDQ, per_channel=True,
        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
        nodes_to_exclude=exclude,
    )

    q = onnx.load(args.out)
    ops = [n.op_type for n in q.graph.node]
    manifest = {
        "source": args.onnx, "qdq": args.out,
        "frames": [{"file": os.path.basename(p), "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()}
                   for p in paths],
        "convs": convs, "excluded": exclude,
        "quantize_linear_nodes": ops.count("QuantizeLinear"),
        "dequantize_linear_nodes": ops.count("DequantizeLinear"),
        "per_channel": True, "activation": "QInt8", "weight": "QInt8",
    }
    with open(args.out + ".manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"qdq -> {args.out}: {manifest['quantize_linear_nodes']} Q nodes, "
          f"{manifest['dequantize_linear_nodes']} DQ nodes; manifest beside it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
