"""Synthesise an SRVGG-shaped network at a chosen width and export it, for width/fusion probes.

Tactics depend on shapes, not on values (the ledger says so and used the same trick for E30a), so a
random-weight network of the right geometry is enough to answer "does the body fuse at this width".
Nothing here touches the shipped weights.

Usage:
  python3 scripts/make_width_probe.py <out.onnx> --width-ch 32 --height 1080 --frame-width 1920
"""
from __future__ import annotations

import argparse
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class Probe(nn.Module):
    """head conv -> n body convs (C->C, PReLU) -> tail conv -> pixelshuffle -> + nearest."""

    def __init__(self, ch: int, body: int, scale: int) -> None:
        super().__init__()
        self.head = nn.Conv2d(3, ch, 3, 1, 1)
        self.body = nn.ModuleList()
        for _ in range(body):
            self.body.append(nn.Conv2d(ch, ch, 3, 1, 1))
            self.body.append(nn.PReLU(ch))
        self.tail = nn.Conv2d(ch, 3 * scale * scale, 3, 1, 1)
        self.upsampler = nn.PixelShuffle(scale)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.clamp(0.0, 1.0)
        out = self.head(out)
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(self.tail(out))
        return out + F.interpolate(x, scale_factor=self.scale, mode="nearest")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--width-ch", type=int, default=64, help="channel count of the body")
    ap.add_argument("--body", type=int, default=16)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--frame-width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    args = ap.parse_args()

    torch.manual_seed(0)
    m = Probe(args.width_ch, args.body, args.scale).eval()
    dummy = torch.zeros(1, 3, args.height, args.frame_width)
    with torch.no_grad():
        torch.onnx.export(m, dummy, args.out, opset_version=17,
                          input_names=["input"], output_names=["output"])
    n = sum(p.numel() for p in m.parameters())
    print(f"{args.out}: C={args.width_ch} body={args.body} scale={args.scale}x "
          f"{args.frame_width}x{args.height}, {n} params (random weights)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
