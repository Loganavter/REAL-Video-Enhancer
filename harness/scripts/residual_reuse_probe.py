"""Does out(x_t) equal out(x_{t-1}) + out(x_t - x_{t-1}) closely enough to reuse by residual?

The reuse lane is blocked on one measurement: the material is a held cel whose edges a
per-frame denoiser re-renders, so whole-frame reuse has nothing stable to freeze, and the gate
correctly refuses. Skip-Convolutions (Habibian et al., CVPR'21) computes `z_t = z_{t-1} +
w * (x_t - x_{t-1})` per layer - no tile seams, because the output is "the previous output plus
the change", not a paste. Its algebra is exact for a linear map; this model is sixteen convs
with PReLU between them, so the question is empirical: how far is

    out(x_{t-1}) + out(x_t - x_{t-1})        (residual path)
    from
    out(x_t)                                  (a full pass)

on the failing material? Damage is quoted with the project's own statistic: share of pixels
more than 8 levels off in the worst channel, on the worst frame, bar 0.1%.

Usage: python3 harness/scripts/residual_reuse_probe.py CLIP [--frames 6] [--start 0]
"""
from __future__ import annotations
import os as _bos, sys as _bsys
_HERE = _bos.path.dirname(_bos.path.abspath(__file__))
_ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
for _p in (_ROOT, _bos.path.join(_ROOT, "backend"), _HERE):
    if _p not in _bsys.path:
        _bsys.path.insert(0, _p)

import argparse
import subprocess

import numpy as np
import torch

MODEL = "custom_models/2x-Futsuu-Anime(1).pth"
FFMPEG = "bin/ffmpeg"


def read_frames(path, width, height, count, start=0):
    """Decode `count` frames as float32 (0..1) RGB, the way the renderer sees them."""
    cmd = [FFMPEG, "-v", "error", "-ss", str(start), "-i", path, "-frames:v", str(count),
           "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    arr = np.frombuffer(raw, np.uint8)
    n = arr.size // (width * height * 3)
    return arr[: n * width * height * 3].reshape(n, height, width, 3).astype(np.float32) / 255.0


def load_model():
    from src.pytorch.spandrel.architectures.Compact.__arch.SRVGG import SRVGGNetCompact
    state = torch.load(MODEL, map_location="cpu", weights_only=False)
    sd = state.get("params", state.get("state_dict", state)) if isinstance(state, dict) else state
    num_feat = int(sd["body.0.weight"].shape[0])
    # the body's 64->64 convs: the head (3->64) and the tail (64->12, 3 channels x 2x2)
    # are built separately by the class
    num_conv = sum(1 for k, v in sd.items()
                   if k.endswith(".weight") and tuple(v.shape) == (num_feat, num_feat, 3, 3))
    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=num_feat,
                            num_conv=num_conv, upscale=2)
    model.load_state_dict({k: v for k, v in sd.items()}, strict=False)
    return model.eval().half().cuda()


def to_tensor(rgb):
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).half().cuda()


def run(model, t):
    with torch.inference_mode():
        return model(t).clamp(0, 1)


def forward_recording(model, x):
    """A normal forward pass that also returns every module's output (the cache)."""
    acts = []
    out = x.clamp(0, 1)
    with torch.inference_mode():
        for module in model.body:
            out = module(out)
            acts.append(out)
        out = model.upsampler(out)
        out = out + torch.nn.functional.interpolate(x, scale_factor=model.upscale, mode="nearest")
    # exactly what the renderer emits: the picture is clamped, and both sides of every
    # comparison must be the same object or the overshoot at edges reads as damage
    return out.clamp(0, 1), acts


def gate_residual(r, eps):
    """Skip-Conv's input-norm gate: skip a position when the residual's l1 norm over the
    kernel support (channels x 3x3) is below `eps`. Returns the gated residual and the share
    of skipped positions - the stand-in for the speedup, since this probe computes densely."""
    if eps <= 0:
        return r, 0.0
    with torch.no_grad():
        support = torch.nn.functional.conv2d(
            r.abs(), torch.ones(r.shape[1], 1, 3, 3, device=r.device, dtype=r.dtype),
            padding=1, groups=r.shape[1]).sum(dim=1, keepdim=True)
        mask = (support > eps).to(r.dtype)
    return r * mask, float(1.0 - mask.mean())


def run_residual(model, x_prev, acts_prev, x_cur, eps=0.0):
    """Skip-Convolutions' recursion at full density: `z_k = z_k(prev) + conv(feature residual)`.

    Every skip is disabled (the gate would only trade this accuracy for speed), so this is the
    best case the mechanism can offer on this model - if the identity does not hold here, a
    threshold cannot rescue it. The residual is taken in feature space, exactly as the paper
    defines it, and the current frame's activations are the running approximation.
    """
    prev_in = x_prev.clamp(0, 1)
    cur_in = x_cur.clamp(0, 1)
    out = cur_in
    skipped_share = []
    with torch.inference_mode():
        for module, prev_out in zip(model.body, acts_prev):
            if isinstance(module, torch.nn.Conv2d):
                # w*x + b = (w*x_prev + b) + w*(x - x_prev): the bias is already inside
                # prev_out, so the residual contribution is conv(r) MINUS the bias. Without
                # this the recursion adds b once per layer even when the frames are identical.
                gated, skipped = gate_residual(cur_in - prev_in, eps)
                skipped_share.append(skipped)
                contribution = module(gated)
                if module.bias is not None:
                    contribution = contribution - module.bias.view(1, -1, 1, 1)
                out = prev_out + contribution
            else:
                out = module(out)
                cur_in, prev_in = out, prev_out
        out = model.upsampler(out)
        out = out + torch.nn.functional.interpolate(x_cur, scale_factor=model.upscale, mode="nearest")
    return out.clamp(0, 1), (sum(skipped_share) / len(skipped_share) if skipped_share else 0.0)


def damage(a, b):
    """Share of pixels more than 8 levels off in any channel, as the bar counts it."""
    d = (a - b).abs().max(dim=1).values * 255.0
    return float((d > 8).float().mean()) * 100.0, float(d.max())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("clip")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="decode smaller (0.5 = half each side); the 2x comparison is the same")
    args = ap.parse_args()

    width = int(args.width * args.scale) // 2 * 2
    height = int(args.height * args.scale) // 2 * 2
    frames = read_frames(args.clip, args.width, args.height, args.frames + 1, args.start)
    if args.scale != 1.0:
        import torch.nn.functional as F
        t = torch.from_numpy(frames).permute(0, 3, 1, 2)
        t = F.interpolate(t, size=(height, width), mode="area")
        frames = t.permute(0, 2, 3, 1).numpy()
    model = load_model()
    print(f"{len(frames)} frames {width}x{height}; model 2x on {torch.cuda.get_device_name(0)}")
    print("Skip-Conv recursion with the input-norm gate, damage against a full pass:")

    for eps_levels in (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
        eps = eps_levels / 255.0
        worst, worst_peak, skip, pairs = 0.0, 0.0, 0.0, 0
        prev_in = to_tensor(frames[0])
        prev_out, prev_acts = forward_recording(model, prev_in)
        for i in range(1, len(frames)):
            cur_in = to_tensor(frames[i])
            full, cur_acts = forward_recording(model, cur_in)
            layer_res, skipped = run_residual(model, prev_in, prev_acts, cur_in, eps=eps)
            d, peak = damage(layer_res, full)
            worst, worst_peak = max(worst, d), max(worst_peak, peak)
            skip += skipped
            pairs += 1
            prev_in, prev_out, prev_acts = cur_in, full, cur_acts
            torch.cuda.empty_cache()
        print(f"gate eps {eps_levels:>5.1f} levels: skipped {100*skip/max(1,pairs):>5.1f}% of positions"
              f" (mean over {pairs} pairs), worst share {worst:>8.5f}%  peak {worst_peak:>5.0f}")
    print("\nbar: 0.1% of pixels more than 8 levels off, worst channel, worst frame")
    print("uncovered: one clip, one model, fp16 torch (not the int8 body, not TRT); the "
          "residual path costs a full pass here - the speedup would come from sparse kernels, "
          "not measured by this probe")


if __name__ == "__main__":
    _bsys.exit(main())
