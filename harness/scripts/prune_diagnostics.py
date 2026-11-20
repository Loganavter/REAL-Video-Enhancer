"""Is the pruning error a DC shift or a structural loss? Three diagnostics in one pass.

The naive cut reports the same ~1.58% of pixels above 8 levels whether 8 channels are removed or 32,
with peak 59 in every case. A plateau like that has two candidate explanations and they lead to
different plans:

  (a) the stack's DC/scale balance breaks at once, so an affine refit should collapse the error;
  (b) the representation is broken structurally, so no cheap refit will help and ASVD/SNOWS is not
      worth a day.

Also a control the identity test cannot give: zero out K channels in place (width unchanged) and
compare with removing them. If zeroing costs the same as removing, the plateau is the model's answer;
if it costs less, the slicing/wiring is at fault.

Usage: prune_diagnostics.py <frame.raw> <W> <H> [--channels 56,48,32]
"""
import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F

import os as _bos, sys as _bsys
_HERE = _bos.path.dirname(_bos.path.abspath(__file__))
_ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in _bsys.path:
        _bsys.path.insert(0, _p)
from cut_channels_naive import cut_model, load, statistic  # noqa: E402


def flat_mask(img, levels=2.0, k=3):
    """Where the reference is locally flat, in levels - the region the DC hypothesis is about."""
    gray = img.mean(dim=1, keepdim=True) * 255.0
    mean = F.avg_pool2d(gray, k, 1, k // 2)
    mean_sq = F.avg_pool2d(gray * gray, k, 1, k // 2)
    std = (mean_sq - mean * mean).clamp_min(0).sqrt()
    return std < levels


def report(tag, ref, got, mask):
    share, peak, median = statistic(ref, got)
    diff = (got - ref) * 255.0
    signed = diff.mean().item()
    structural = diff - diff.mean()
    per_ch = diff.mean(dim=(0, 2, 3))
    over = diff.abs().amax(dim=1) > 8                      # [1, H, W]
    flat = mask.squeeze(1)                                 # [1, H, W]
    flat_share = over[flat].float().mean().item() * 100 if flat.any() else float("nan")
    edge_share = over[~flat].float().mean().item() * 100 if (~flat).any() else float("nan")
    print(f"{tag:<22} >8: {share:6.3f}%  peak {peak:5.0f}  median {median:5.2f}  "
          f"signed mean {signed:+7.2f}  |structural| rms {structural.pow(2).mean().sqrt():6.2f}  "
          f"per-channel mean {[round(v, 1) for v in per_ch.tolist()]}  flat {flat_share:6.3f}%  edge {edge_share:6.3f}%")
    return share


def affine_fit(ref, got):
    """Per output channel, the best a and b: out ~ a * got + b, fitted on this frame (an upper bound)."""
    out = got.clone()
    for c in range(got.shape[1]):
        x = got[:, c].flatten()
        y = ref[:, c].flatten()
        xm, ym = x.mean(), y.mean()
        a = ((x - xm) * (y - ym)).mean() / ((x - xm).pow(2).mean() + 1e-12)
        out[:, c] = a * got[:, c] + (ym - a * xm)
    return out


def zero_channels(model, k):
    """Same width, K channels silenced in place: head outputs, body in/out, tail inputs."""
    sd = {key: value.clone() for key, value in model.state_dict().items()}
    feat = sd["body.0.weight"].shape[0]
    kill = torch.arange(feat - k, feat)
    for name, tensor in sd.items():
        if tensor.ndim == 4 and tensor.shape[0] == feat:        # head or body output
            tensor[kill] = 0.0
        if tensor.ndim == 4 and tensor.shape[1] == feat:        # body or tail input
            tensor[:, kill] = 0.0
        if tensor.ndim == 1 and tensor.numel() == feat:         # PReLU slope or conv bias
            tensor[kill] = 0.0
    model.load_state_dict(sd)
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("frame")
    ap.add_argument("width", type=int)
    ap.add_argument("height", type=int)
    ap.add_argument("--channels", default="56,48,32")
    ap.add_argument("--model", default="/tmp/opencode/models2/futsuu.pth")
    args = ap.parse_args()

    raw = np.fromfile(args.frame, dtype=np.uint8)
    img = torch.from_numpy(raw.reshape(args.height, args.width, 3).astype(np.float32) / 255.0)
    x = img.permute(2, 0, 1)[None].cuda().half()

    model, scale, _ = load(args.model)
    model = model.cuda().half()
    with torch.no_grad():
        ref = model(x).float()
    mask = flat_mask(ref)

    print(f"flat pixels: {float(mask.float().mean()) * 100:.1f}% of the frame")
    for keep in [int(v) for v in args.channels.split(",")]:
        pruned, _ = cut_model(model, keep)
        pruned = pruned.cuda().half()
        with torch.no_grad():
            got = pruned(x).float()
        report(f"pruned 64->{keep}", ref, got, mask)
        report(f"  + affine refit", ref, affine_fit(ref, got), mask)

        zeroed, _, _ = load(args.model)
        zeroed = zero_channels(zeroed.cuda().half(), 64 - keep)
        with torch.no_grad():
            zgot = zeroed(x).float()
        report(f"zeroed {64 - keep} channels", ref, zgot, mask)

        zeroed2, _, _ = load(args.model)
        zeroed2 = zero_channels(zeroed2.cuda().half(), 64 - keep)
        with torch.no_grad():
            z2 = zeroed2(x).float()
        report(f"  + affine refit", ref, affine_fit(ref, z2), mask)
    return 0


if __name__ == "__main__":
    sys.exit(main())
