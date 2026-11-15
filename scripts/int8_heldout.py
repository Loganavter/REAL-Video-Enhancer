"""Calibrate on one frame, evaluate on another: the leak the last measurement had, closed.

The 0.000% / peak 7 was measured with scales calibrated on two frames, one of them the test frame. A
max-based scale calibrated on the test frame guarantees zero clipping on it, and clipping is exactly what
produces large peaks - so 7 is a lower bound for unseen content, not an upper one.

This calibrates on frame A only and evaluates on frame B, reporting the picture statistic and, per layer,
how much of B's activation exceeds the maximum A taught us. A non-zero clip rate names the layer and the
size, which turns "known hole" into a number.

Usage: int8_heldout.py <calib.raw> <test.raw> <W> <H>
"""
import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, "scripts")
from cut_channels_naive import load, statistic  # noqa: E402
from int8_accuracy_sim import calibrate, quantised_forward  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("calib")
    ap.add_argument("test")
    ap.add_argument("width", type=int)
    ap.add_argument("height", type=int)
    ap.add_argument("--model", default="/tmp/opencode/models2/futsuu.pth")
    args = ap.parse_args()

    def frame(path):
        raw = np.fromfile(path, dtype=np.uint8)
        img = torch.from_numpy(raw.reshape(args.height, args.width, 3).astype(np.float32) / 255.0)
        return img.permute(2, 0, 1)[None].cuda().half()

    model, _, _ = load(args.model)
    model = model.eval().cuda().half()
    a, b = frame(args.calib), frame(args.test)
    with torch.no_grad():
        ref = model(b).float()

    inputs, outputs, _ = calibrate(model, [a])

    # per-layer clip rate on the held-out frame
    clipped = {}
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            def pre(_m, inp, key=name):
                lim = inputs[key]
                clipped[key] = float((inp[0].abs() > lim).float().mean()) * 100
            handles.append(module.register_forward_pre_hook(pre))
        if isinstance(module, torch.nn.PReLU):
            def post(_m, _i, out, key=name):
                lim = outputs[key].view(1, -1, 1, 1)
                clipped[key] = float((out.abs() > lim).float().mean()) * 100
            handles.append(module.register_forward_hook(post))
    with torch.no_grad():
        model(b)
    for h in handles:
        h.remove()

    with torch.no_grad():
        got = quantised_forward(model, b, inputs, outputs, {}).float()
    share, peak, median = statistic(ref, got)

    worst = sorted(clipped.items(), key=lambda kv: -kv[1])[:5]
    print(f"held-out: >8 {share:.3f}% (bar 0.1)  peak {peak:.0f}  median {median:.2f} levels")
    print("clip rate (share of activations above the calibrated max), worst layers:")
    for name, value in worst:
        print(f"   {name:<12} {value:5.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
