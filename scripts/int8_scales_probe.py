"""Is int8's accuracy on real material fixable by the activation SCALES? One probe, five policies.

The measured decomposition (docs/wsp/verified/logs/bd/diff-int8-violet-f*.log) says the error sits at the
ends of the range (234-255: 100% of those pixels over the bar; 0-32: 25-89%) and on thin line art (top 1%
gradient: 21-29%), while flat mid-tones are clean (0.001-0.003%). Only the range-end class is
scale-shaped; the edge class is 8-bit rounding and no scale policy touches it.

Policies, all with per-output-channel int8 weights (the kernel's policy) and per-channel asym on the
PReLU outputs (what the shipped sim does), varying ONLY the conv-input quantiser:

  A  per-tensor,  asym, max     <- the shipped recipe; must reproduce the recorded number
  B  per-tensor,  asym, p99.9   <- percentile clipping
  C  per-channel, asym, max     <- finer granularity
  D  per-channel, asym, p99.9   <- both
  E  per-tensor,  sym,  max     <- what a symmetric kernel would do

Asymmetric = 256 codes over [lo, hi]; decoders reconstruct q*step + lo.

The probe is anchored: A has to agree with scripts/int8_accuracy_sim.py on the same frame before any
other row is read. Frames are 1080p RGB dumps of the BD clip; calibration and test sets are disjoint.

Usage:
  python scripts/int8_scales_probe.py --calib 20,60,100,140,180,220,260,300 --test 29,44,274,460
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as TF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cut_channels_naive import load, statistic  # noqa: E402

W, H = 1920, 1080


def load_frame(path: str) -> torch.Tensor:
    raw = np.fromfile(path, dtype=np.uint8)
    img = torch.from_numpy(raw.reshape(H, W, 3).astype(np.float32) / 255.0)
    return img.permute(2, 0, 1)[None].cuda().half()


def quantise_weights(state: dict) -> dict:
    out = {}
    for name, t in state.items():
        if t.ndim == 4:
            s = (t.abs().amax(dim=(1, 2, 3), keepdim=True) / 127.0).clamp_min(1e-8)
            out[name] = torch.round(t / s).clamp(-127, 127) * s
        else:
            out[name] = t
    return out


@torch.no_grad()
def calibrate(model, frames, stride: int = 8):
    """Per-channel (lo, hi) of every conv input and PReLU output, plus a sample for the percentiles."""
    model.load_state_dict(quantise_weights(model.state_dict()))
    ins_range: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    ins_samp: dict[str, list[torch.Tensor]] = {}
    out_range: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    out_samp: dict[str, list[torch.Tensor]] = {}
    handles = []

    def pre(_m, inp, key):
        x = inp[0].detach()
        lo, hi = x.amin(dim=(0, 2, 3)), x.amax(dim=(0, 2, 3))
        prev = ins_range.get(key)
        ins_range[key] = (lo, hi) if prev is None else (torch.minimum(prev[0], lo), torch.maximum(prev[1], hi))
        ins_samp.setdefault(key, []).append(x[:, :, ::stride, ::stride].flatten(2))

    def post(_m, _i, out, key):
        o = out.detach()
        lo, hi = o.amin(dim=(0, 2, 3)), o.amax(dim=(0, 2, 3))
        prev = out_range.get(key)
        out_range[key] = (lo, hi) if prev is None else (torch.minimum(prev[0], lo), torch.maximum(prev[1], hi))
        out_samp.setdefault(key, []).append(o[:, :, ::stride, ::stride].flatten(2))

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            handles.append(module.register_forward_pre_hook(lambda m, i, k=name: pre(m, i, k)))
        if isinstance(module, torch.nn.PReLU):
            handles.append(module.register_forward_hook(lambda m, i, o, k=name: post(m, i, o, k)))
    for f in frames:
        model(f)
    for h in handles:
        h.remove()
    return ins_range, ins_samp, out_range, out_samp


def _span(rng, samp, chans: bool, pct: float):
    """(lo, hi) per channel, or one pair for the whole tensor."""
    lo, hi = rng
    if pct < 100.0:
        vals = torch.cat(samp, dim=2).float()
        lo = torch.quantile(vals, (100.0 - pct) / 100.0, dim=2).squeeze(0)
        hi = torch.quantile(vals, pct / 100.0, dim=2).squeeze(0)
        # a percentile can invert the ends on a skewed channel; keep a sane span
        bad = hi <= lo
        if bad.any():
            hi = torch.where(bad, lo + 1e-3, hi)
    if not chans:
        lo, hi = lo.min().expand_as(lo), hi.max().expand_as(hi)
    return lo, hi


def _quant(x, lo, hi, dtype, sym: bool = False):
    if sym:
        s = (torch.maximum(lo.abs(), hi.abs()) / 127.0).clamp_min(1e-8)
        s = s.to(dtype).view(1, -1, 1, 1)
        return torch.round(x / s).clamp(-127, 127) * s
    lo = lo.to(dtype).view(1, -1, 1, 1)
    hi = hi.to(dtype).view(1, -1, 1, 1)
    step = ((hi - lo) / 255.0).clamp_min(1e-8)
    return (torch.round((x - lo) / step).clamp(0, 255) * step + lo)


@torch.no_grad()
def forward(model, x, cal, chans: bool, pct: float, sym: bool = False):
    ins_range, ins_samp, out_range, out_samp = cal
    x = x.clamp(0.0, 1.0)
    out = x
    for name, module in model.body.named_children():
        if isinstance(module, torch.nn.Conv2d):
            lo, hi = _span(ins_range[f"body.{name}"], ins_samp[f"body.{name}"], chans, pct)
            out = _quant(out, lo, hi, out.dtype, sym=sym)
            out = module(out)
        elif isinstance(module, torch.nn.PReLU):
            out = module(out)
            # ALWAYS per channel here: that is what scripts/int8_accuracy_sim.py does, and that sim is
            # the instrument that produced the recorded numbers. Varying two axes at once would make
            # this probe incomparable to the anchor.
            lo, hi = _span(out_range[f"body.{name}"], out_samp[f"body.{name}"], True, pct)
            out = _quant(out, lo, hi, out.dtype, sym=sym)
        else:
            out = module(out)
    out = model.upsampler(out)
    return out + TF.interpolate(x, scale_factor=model.upscale, mode="nearest")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="/var/tmp/rve-rem/frames")
    ap.add_argument("--calib", default="20,60,100,140,180,220,260,300")
    ap.add_argument("--test", default="29,44,274,460")
    ap.add_argument("--model", default="/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2/"
                                       "custom_models/2x-Futsuu-Anime(1).pth")
    args = ap.parse_args(argv)

    def path(i):
        p = os.path.join(args.frames, f"v{i}.raw")
        if not os.path.exists(p):
            raise SystemExit(f"missing frame {p}")
        return p

    model, _, _ = load(args.model)
    model = model.eval().cuda().half()

    test = [(int(i), load_frame(path(int(i)))) for i in args.test.split(",")]
    # The reference is the UNQUANTISED model: calibrate() loads int8 weights in place, so computing the
    # reference afterwards measures the int8 model against itself. That mistake read 0.0026% where this
    # probe's anchor (scripts/int8_accuracy_sim.py) reads 0.065% on the same frame.
    refs = {}
    for i, f in test:
        refs[i] = model(f).float()
        torch.cuda.empty_cache()

    calib_frames = [load_frame(path(int(i))) for i in args.calib.split(",")]
    cal = calibrate(model, calib_frames)
    del calib_frames
    torch.cuda.empty_cache()

    policies = [
        ("A per-tensor  asym max", False, 100.0, False),
        ("B per-tensor  asym p99.9", False, 99.9, False),
        ("C per-channel asym max", True, 100.0, False),
        ("D per-channel asym p99.9", True, 99.9, False),
        ("E per-tensor  sym  max", False, 100.0, True),
    ]
    print(f"{'policy':<26}{'worst share':>13}{'peak':>7}   per frame")
    for label, chans, pct, sym in policies:
        shares, peaks = [], []
        for i, f in test:
            got = forward(model, f, cal, chans, pct, sym=sym).float()
            share, peak, _ = statistic(refs[i], got)
            shares.append(share)
            peaks.append(peak)
            del got
            torch.cuda.empty_cache()
        k = max(range(len(shares)), key=lambda j: shares[j])
        print(f"{label:<26}{shares[k]:>12.5f}%{max(peaks):>7.0f}   "
              + ", ".join(f"f{test[j][0]} {shares[j]:.4f}%" for j in range(len(shares))))
    print("\nNot tested: the edge class (8-bit rounding on 2-3 px lines) - no scale policy moves it; "
          "and the kernel's arithmetic (this rounds in float, the kernel accumulates in int32).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
