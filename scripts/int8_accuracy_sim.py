"""Would the int8 body hold the bar? Simulated with real scales, before any kernel is wired in.

The kernel probe proved the time; it said nothing about accuracy, and everything this investigation has
learned says a speed number without a picture number does not survive. The design being tested:

  activations  per-tensor symmetric int8 on the INPUT of each conv (weights are per-channel)
  weights      per-channel symmetric int8, calibrated as max|w| / 127 per output channel
  epilogue     per-channel PReLU, then per-channel symmetric int8 on the OUTPUT

E9' used a per-TENSOR output scale and measured peak 4 levels against a bar of 8. Per-channel output
scales should be better, not worse - the slopes span -1.39..1.70, so the post-PReLU range differs by
factors across channels and a per-tensor scale spends its bits on the union.

The simulation rounds weights and activations onto the int8 grid but computes the convolutions in
float, which isolates the quantisation error from the kernel's arithmetic (exact) and from the
accumulator width (int32, plenty).

Usage: int8_accuracy_sim.py <frame.raw> <W> <H> [--calib frames.raw ...] [--model ...]
"""
import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as TF

sys.path.insert(0, "scripts")
from cut_channels_naive import load, statistic  # noqa: E402


def quantise_weights(state):
    """Per-output-channel symmetric int8, which is what the kernel multiplies with."""
    out = {}
    scales = {}
    for name, tensor in state.items():
        if tensor.ndim == 4:
            scale = tensor.abs().amax(dim=(1, 2, 3), keepdim=True) / 127.0
            scale = scale.clamp_min(1e-8)
            q = torch.round(tensor / scale).clamp(-127, 127)
            out[name] = q * scale
            scales[name] = scale.flatten()
        else:
            out[name] = tensor
    return out, scales


def calibrate(model, frames, device="cuda"):
    """Per-tensor input scales and per-channel output scales, from the quantised model's own activations."""
    state = model.state_dict()
    qstate, wscales = quantise_weights(state)
    model.load_state_dict(qstate)

    inputs, outputs = {}, {}
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            def pre_hook(_m, inp, key=name):
                lo, hi = float(inp[0].min()), float(inp[0].max())
                prev = inputs.get(key)
                inputs[key] = (min(prev[0], lo), max(prev[1], hi)) if prev else (lo, hi)
            handles.append(module.register_forward_pre_hook(pre_hook))
        if isinstance(module, torch.nn.PReLU):
            def post_hook(_m, _inp, out, key=name):
                pair = (out.amin(dim=(0, 2, 3)), out.amax(dim=(0, 2, 3)))
                prev = outputs.get(key)
                outputs[key] = pair if prev is None else (torch.minimum(prev[0], pair[0]),
                                                          torch.maximum(prev[1], pair[1]))
            handles.append(module.register_forward_hook(post_hook))
    with torch.no_grad():
        for frame in frames:
            model(frame)
    for h in handles:
        h.remove()
    return inputs, outputs, wscales


def quantised_forward(model, x, inputs, outputs, wscales, fp16_tail=0, asym=True):
    """The same stack with every conv input and PReLU output snapped to an int8 grid.

    Mirrors SRVGGNetCompact.forward exactly: clamp, the body stack, pixel shuffle, residual add. The
    first version stopped at the body and compared a 12-channel map against a 4x picture.
    """
    x = x.clamp(0.0, 1.0)
    out = x
    convs = [n for n, m in model.body.named_children() if isinstance(m, torch.nn.Conv2d)]
    # the last `fp16_tail` pairs keep the half-precision boundary: their error lands in the tail almost
    # directly, while the early layers' error is mixed by everything downstream
    stop_quantising = set(convs[len(convs) - fp16_tail:]) if fp16_tail else set()
    for name, module in model.body.named_children():
        if isinstance(module, torch.nn.Conv2d) and name in stop_quantising:
            out = module(out)
        elif isinstance(module, torch.nn.Conv2d):
            key = f"body.{name}"
            lo, hi = inputs[key]
            if asym:                      # 256 codes over the observed span instead of 127 symmetric
                step = (hi - lo) / 255.0
                zp = lo / step
                out = (torch.round(out / step - zp).clamp(0, 255) + zp) * step
            else:
                s_in = max(abs(lo), abs(hi)) / 127.0
                out = torch.round(out / s_in).clamp(-127, 127) * s_in
            out = module(out)
        elif isinstance(module, torch.nn.PReLU) and str(int(name) - 1) in stop_quantising:
            out = module(out)
        elif isinstance(module, torch.nn.PReLU):
            key = f"body.{name}"
            out = module(out)
            lo, hi = outputs[key]
            if asym:
                step = ((hi - lo) / 255.0).view(1, -1, 1, 1).clamp_min(1e-8)
                zp = (lo / step.squeeze()).view(1, -1, 1, 1)
                out = (torch.round(out / step - zp).clamp(0, 255) + zp) * step
            else:
                s_out = (torch.maximum(lo.abs(), hi.abs()) / 127.0).view(1, -1, 1, 1).clamp_min(1e-8)
                out = torch.round(out / s_out).clamp(-127, 127) * s_out
        else:
            out = module(out)
    out = model.upsampler(out)
    return out + TF.interpolate(x, scale_factor=model.upscale, mode="nearest")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("frame")
    ap.add_argument("width", type=int)
    ap.add_argument("height", type=int)
    ap.add_argument("--calib", nargs="*", default=[])
    ap.add_argument("--model", default="/tmp/opencode/models2/futsuu.pth")
    args = ap.parse_args()

    def load_frame(path):
        raw = np.fromfile(path, dtype=np.uint8)
        img = torch.from_numpy(raw.reshape(args.height, args.width, 3).astype(np.float32) / 255.0)
        return img.permute(2, 0, 1)[None].cuda().half()

    model, scale, _ = load(args.model)
    model = model.eval().cuda().half()
    test = load_frame(args.frame)
    with torch.no_grad():
        ref = model(test).float()

    calib = []
    for path in args.calib or [args.frame]:
        calib.append(load_frame(path))
    calib.append(test)

    inputs, outputs, wscales = calibrate(model, calib)
    with torch.no_grad():
        got = quantised_forward(model, test, inputs, outputs, wscales).float()

    share, peak, median = statistic(ref, got)
    print(f"int8 simulation (per-channel output scales): >8 {share:.3f}% (bar 0.1)  "
          f"peak {peak:.0f}  median {median:.2f} levels")
    print(f"activation ranges: {len(inputs)} convs, input scale "
          f"{min(v for v in inputs.values()) / 127:.4f}..{max(inputs.values()) / 127:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
