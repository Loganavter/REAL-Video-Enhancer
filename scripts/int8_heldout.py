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
    ap.add_argument("test")
    ap.add_argument("--calib", nargs="+", default=None)
    ap.add_argument("width", type=int)
    ap.add_argument("height", type=int)
    ap.add_argument("--model", default="/tmp/opencode/models2/futsuu.pth")
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--margin-all", type=float, default=1.0)
    ap.add_argument("--localize", action="store_true")
    ap.add_argument("--sym", action="store_true", help="force symmetric scales for A/B")
    ap.add_argument("--fp16-tail", type=int, default=0,
                    help="leave the last N body pairs on the fp16 boundary")
    args = ap.parse_args()

    def frame(path):
        raw = np.fromfile(path, dtype=np.uint8)
        img = torch.from_numpy(raw.reshape(args.height, args.width, 3).astype(np.float32) / 255.0)
        return img.permute(2, 0, 1)[None].cuda().half()

    model, _, _ = load(args.model)
    model = model.eval().cuda().half()
    calib_paths = args.calib or [args.test]
    b = frame(args.test)
    with torch.no_grad():
        ref = model(b).float()

    inputs, outputs, _ = calibrate(model, [frame(p) for p in calib_paths])
    # the head conv's input is the image, clamped to [0,1] by the model itself: its range is a domain
    # constant and calibrating it on one frame made a dark frame clip a fifth of a different title
    inputs["body.0"] = (0.0, 1.0)
    # margin on the layer whose output still clips: the decisive knob between "calibration bug" and
    # "the int8 hop itself". 1.0 = the calibrated maximum, 2.0 = twice it.
    if args.margin != 1.0:
        lo, hi = outputs["body.1"]
        outputs["body.1"] = (lo * args.margin, hi * args.margin)
    if args.margin_all != 1.0:
        # every scale, both sides: if the peak falls now, distributed clipping was the cause; if it
        # stays, the int8 grid itself is
        inputs = {k: (v[0] * args.margin_all, v[1] * args.margin_all) for k, v in inputs.items()}
        outputs = {k: (v[0] * args.margin_all, v[1] * args.margin_all) for k, v in outputs.items()}

    # per-layer clip rate on the held-out frame
    clipped = {}
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            def pre(_m, inp, key=name):
                lim = max(abs(inputs[key][0]), abs(inputs[key][1]))
                clipped[key] = float((inp[0].abs() > lim).float().mean()) * 100
            handles.append(module.register_forward_pre_hook(pre))
        if isinstance(module, torch.nn.PReLU):
            def post(_m, _i, out, key=name):
                lim = torch.maximum(outputs[key][0].abs(), outputs[key][1].abs()).view(1, -1, 1, 1)
                clipped[key] = float((out.abs() > lim).float().mean()) * 100
            handles.append(module.register_forward_hook(post))
    with torch.no_grad():
        model(b)
    for h in handles:
        h.remove()

    with torch.no_grad():
        got = quantised_forward(model, b, inputs, outputs, {}, fp16_tail=args.fp16_tail,
                                asym=not args.sym).float()
    share, peak, median = statistic(ref, got)

    if args.localize:
        err = (ref - got).abs() * 255.0          # levels, the statistic's own unit
        per_px = err.amax(dim=1)[0]
        flat = int(per_px.argmax())
        y, x = divmod(flat, per_px.shape[1])
        ch = int(err[0, :, y, x].argmax())
        patch = per_px[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
        over = (per_px > 20).float().sum()
        over8 = (per_px > 8).float().sum()
        print(f"peak {float(per_px.max()):.0f} at (y={y}, x={x}) channel {ch}; "
              f"ref {float(ref[0, ch, y, x]):.4f} got {float(got[0, ch, y, x]):.4f}")
        print(f"3x3 error patch max {float(patch.max()):.0f}, pixels >20: {int(over)}, >8: {int(over8)}")
        print(f"neighbourhood: ref range {float(ref[0, :, max(0,y-1):y+2, max(0,x-1):x+2].min()):.3f}"
              f"..{float(ref[0, :, max(0,y-1):y+2, max(0,x-1):x+2].max()):.3f}, "
              f"saturated refs (0 or 1): {int(((ref < 0.002) | (ref > 0.998)).float().sum())} of {ref.numel()}")

    worst = sorted(clipped.items(), key=lambda kv: -kv[1])[:5]
    print(f"held-out: >8 {share:.3f}% (bar 0.1)  peak {peak:.0f}  median {median:.2f} levels")
    print("clip rate (share of activations above the calibrated max), worst layers:")
    for name, value in worst:
        print(f"   {name:<12} {value:5.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
