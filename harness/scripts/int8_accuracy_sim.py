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

import os as _bos, sys as _bsys
_HARNESS_ROOT = _bos.path.dirname(_bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
if _HARNESS_ROOT not in _bsys.path:
    _bsys.path.insert(0, _HARNESS_ROOT)
from cut_channels_naive import load, statistic  # noqa: E402


def quantise_weights(state, per_tap=False):
    """Symmetric int8 weights: one scale per output channel, or per (output channel, tap).

    The kernel can do per-tap exactly - it dequantises inside the tap loop and accumulates each tap in
    int32 - so the tap dimension is a free axis that TensorRT does not have.
    """
    out = {}
    scales = {}
    for name, tensor in state.items():
        if tensor.ndim == 4:
            dims = (1,) if per_tap else (1, 2, 3)
            scale = tensor.abs().amax(dim=dims, keepdim=True) / 127.0
            scale = scale.clamp_min(1e-8)
            q = torch.round(tensor / scale).clamp(-127, 127)
            out[name] = q * scale
            scales[name] = scale.expand_as(tensor)
        else:
            out[name] = tensor
    return out, scales


def calibrate(model, frames, device="cuda", per_tap=False, act_per_channel=False, clip_pct=None):
    """Conv-input ranges (per tensor, or per channel) and per-channel PReLU output scales.

    `act_per_channel` and `clip_pct` exist because the BD re-measurement put the int8 error at the ends
    of the range (234-255: 100% of those pixels over the bar) and on thin line art, and those are the two
    quantiser knobs that could move the first class. They default off, so the recorded recipe is what
    runs when nobody asks.
    """
    state = model.state_dict()
    qstate, wscales = quantise_weights(state, per_tap=per_tap)
    model.load_state_dict(qstate)

    inputs, outputs = {}, {}
    samples: dict[str, list[torch.Tensor]] = {}
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            def pre_hook(_m, inp, key=name):
                x = inp[0]
                if act_per_channel or clip_pct:
                    lo_c, hi_c = x.amin(dim=(0, 2, 3)), x.amax(dim=(0, 2, 3))
                    prev = inputs.get(key)
                    inputs[key] = ((torch.minimum(prev[0], lo_c), torch.maximum(prev[1], hi_c))
                                   if prev else (lo_c, hi_c))
                    if clip_pct:
                        samples.setdefault(key, []).append(
                            x[:, :, ::8, ::8].flatten(2).float())
                else:
                    lo, hi = float(x.min()), float(x.max())
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
    if clip_pct:
        # percentile span: a handful of clipped highlights no longer sets the step for the whole tensor
        for key, sample in samples.items():
            vals = torch.cat(sample, dim=2)
            lo_c = torch.quantile(vals, (100.0 - float(clip_pct)) / 100.0, dim=2).squeeze(0)
            hi_c = torch.quantile(vals, float(clip_pct) / 100.0, dim=2).squeeze(0)
            bad = hi_c <= lo_c
            hi_c = torch.where(bad, lo_c + 1e-3, hi_c)
            if act_per_channel:
                inputs[key] = (lo_c, hi_c)
            else:
                inputs[key] = (float(lo_c.min()), float(hi_c.max()))
    elif act_per_channel:
        inputs = {k: (v[0].view(1, -1, 1, 1), v[1].view(1, -1, 1, 1)) for k, v in inputs.items()}
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
    ap.add_argument("--fp16-tail", type=int, default=0,
                    help="keep the LAST k conv pairs in fp16: the error of a late hop is not diluted by "
                         "anything downstream, so this is the cheapest precision lever in the chain")
    ap.add_argument("--act-scale", choices=("tensor", "channel"), default="tensor",
                    help="granularity of the int8 scale on each conv INPUT (default tensor = recorded)")
    ap.add_argument("--clip-pct", dest="clip", type=float, default=None,
                    help="percentile for the activation span instead of min/max (e.g. 99.9)")
    ap.add_argument("--emit", default="", help="append this result as a harness record (jsonl)")
    ap.add_argument("--frame-index", type=int, default=-1, help="frame index for the population label")
    ap.add_argument("--clip", dest="clip_name", default="bd_violet20s.mkv")
    ap.add_argument("--id", dest="rec_id", default="")
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

    inputs, outputs, wscales = calibrate(model, calib, act_per_channel=(args.act_scale == "channel"),
                                         clip_pct=args.clip)
    with torch.no_grad():
        got = quantised_forward(model, test, inputs, outputs, wscales,
                                fp16_tail=args.fp16_tail).float()

    share, peak, median = statistic(ref, got)
    print(f"int8 simulation (fp16 tail {args.fp16_tail}, act scale {args.act_scale}"
          f"{', clip p' + str(args.clip) if args.clip else ''}): >8 {share:.5f}% (bar 0.1)  "
          f"peak {peak:.0f}  median {median:.2f} levels")
    # The stroke probe: a signed effect sits below the 8-level bar, so the share cannot see it.
    def _luma(t):
        return 0.2126 * t[0] + 0.7152 * t[1] + 0.0722 * t[2]

    lm = _luma(ref[0].clamp(0, 1) * 255.0)
    lap = (4 * lm[1:-1, 1:-1] - lm[:-2, 1:-1] - lm[2:, 1:-1] - lm[1:-1, :-2] - lm[1:-1, 2:])
    mask = lap.abs() > torch.quantile(lap.abs().flatten().float(), 0.98)
    ink, rim = mask & (lap < 0), mask & (lap > 0)
    lg = _luma(got[0].clamp(0, 1) * 255.0)[1:-1, 1:-1]
    print(f"  stroke: ink {lg[ink].mean():.2f} ({(lg[ink].mean() - lm[1:-1, 1:-1][ink].mean()):+.2f})  "
          f"rim {lg[rim].mean():.2f} ({(lg[rim].mean() - lm[1:-1, 1:-1][rim].mean()):+.2f})  "
          f"against the model")

    if args.emit:
        import json as _json
        import os as _bos, sys as _bsys
        _HARNESS_ROOT = _bos.path.dirname(_bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
        if _HARNESS_ROOT not in _bsys.path:
            _bsys.path.insert(0, _HARNESS_ROOT)
        from harness.core import Population, Record
        pop = Population.of("frames", [f"{args.clip_name}#frame{args.frame_index}"],
                            f"{args.clip_name}, int8 simulation against the fp16 reference")
        rec = Record(value=float(share), unit="%", population=pop,
                     instrument="int8 simulation: weights per-output-channel, PReLU outputs per-channel",
                     config=f"conv-input scale {args.act_scale}"
                            + (f", clip p{args.clip}" if args.clip else "")
                            + ", calibration on 8 disjoint frames",
                     session="s-2026-11-18T09:00:00Z", statistic="worst_of", tier="measured",
                     run="harness/scripts/int8_accuracy_sim.py", id=args.rec_id)
        rec.validate()
        with open(args.emit, "a") as fh:
            fh.write(_json.dumps(rec.to_dict()) + "\n")
        print(rec.claims_line())
    return 0


if __name__ == "__main__":
    sys.exit(main())
