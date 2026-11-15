"""E30b, first look: what does a 64 -> 32 channel cut do to the picture, before any reconstruction?

The time half is settled (C=32 fuses, 22.4-24.9 ms, 2.07-2.36x). The picture half has never been
measured, and the cheapest reading of it is the worst case: cut by static importance, no calibration,
no reconstruction, and see how far the output moves. If a naive cut already lands near the bar, a
reconstruction has room; if it lands at tens of percent, the reconstruction has to be a miracle and the
width is not usable.

Importance is propagated through the stack (a channel matters if it feeds large weights downstream),
which is the standard first-order proxy and deliberately weaker than ASVD/SNOWS - the point is a floor,
not a candidate.

Usage: cut_channels_naive.py <frame.raw> <W> <H> [--keep 32] [--model /tmp/opencode/models2/futsuu.pth]
"""
import argparse
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F

BACKEND = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2/backend"
sys.path.insert(0, f"{BACKEND}/src/pytorch")
from spandrel.architectures.Compact.__arch.SRVGG import SRVGGNetCompact  # noqa: E402


def load(path, feat=64):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = sd.get("params", sd)
    tail = [k for k in sd if k.endswith("34.weight")][0]
    scale = int(round((sd[tail].shape[0] / 3) ** 0.5))
    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=feat, num_conv=16, upscale=scale)
    model.load_state_dict(sd)
    return model.eval(), scale, sd


def cut(sd, keep, feat=64):
    """Slice every body conv and PReLU to `keep` channels, ranked by propagated weight importance."""
    out = dict(sd)
    importance = torch.ones(feat)
    index = 0
    for k in sorted(sd, key=lambda s: int(s.split(".")[1]) if s.split(".")[1].isdigit() else 0):
        if not k.endswith(".weight") or sd[k].ndim != 4:
            continue
        w = sd[k]
        index += 1
        if index == 1:                       # head: 3 -> feat, keep output channels
            score = w.abs().sum(dim=(1, 2, 3))
            keep_out = torch.topk(score, keep).indices.sort().values
            out[k] = w[keep_out]
            out[k.replace("weight", "bias")] = sd[k.replace("weight", "bias")][keep_out]
            out[k.replace(".weight", ".weight")] = out[k]
            prelu = k.replace("weight", "weight")
            importance = torch.ones(keep)
            prev_keep = keep_out
            continue
        in_ch, out_ch = w.shape[1], w.shape[0]
        if out_ch == feat and in_ch == feat:  # body: feat -> feat
            score = (w.abs().sum(dim=(2, 3)) * importance[None, :]).sum(dim=1)
            keep_out = torch.topk(score, keep).indices.sort().values
            out[k] = w[keep_out][:, prev_keep]
            out[k.replace("weight", "bias")] = sd[k.replace("weight", "bias")][keep_out]
            importance = importance[keep_out] * (sd[k][keep_out].abs().sum(dim=(1, 2, 3)) + 1e-6)
            prev_keep = keep_out
            continue
        if in_ch == feat and out_ch != feat:  # tail: feat -> 3*scale^2, keep input channels
            out[k] = w[:, prev_keep]
            continue
        # anything else (head already handled) stays
    # PReLU weights follow their conv's selection: rebuild by walking the body again
    return out


def cut_model(model, keep):
    """Rebuild the tree at `keep` channels. PReLU slopes follow the conv selection, not a slice."""
    sd = model.state_dict()
    feat = sd["body.0.weight"].shape[0]
    conv_keys = sorted((k for k in sd if re.fullmatch(r"body\.\d+\.weight", k) and sd[k].ndim == 4),
                       key=lambda s: int(s.split(".")[1]))
    scale = int(round((sd[conv_keys[-1]].shape[0] / 3) ** 0.5))
    new = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=keep, num_conv=16, upscale=scale)
    nsd = new.state_dict()

    sel = {}                 # conv key -> output-channel selection (None = keep all)
    prev = None
    for k in conv_keys:
        w = sd[k]
        if w.shape[1] == 3:                                  # head: 3 -> feat
            out = torch.topk(w.abs().sum(dim=(1, 2, 3)), keep).indices.sort().values
            nsd[k] = w[out]
            nsd[k.replace(".weight", ".bias")] = sd[k.replace(".weight", ".bias")][out]
            prev = out
            sel[k] = out
        elif w.shape[0] == feat and w.shape[1] == feat:      # body: feat -> feat
            score = w.abs().sum(dim=(2, 3))[:, prev].sum(dim=1)
            out = torch.topk(score, keep).indices.sort().values
            nsd[k] = w[out][:, prev]
            nsd[k.replace(".weight", ".bias")] = sd[k.replace(".weight", ".bias")][out]
            prev = out
            sel[k] = out
        else:                                                # tail: feat -> 3*scale^2
            nsd[k] = w[:, prev]
            nsd[k.replace(".weight", ".bias")] = sd[k.replace(".weight", ".bias")]
            sel[k] = None

    for name, tensor in sd.items():
        if tensor.ndim == 1 and name.endswith(".weight"):    # a PReLU slope vector
            before = f"body.{int(name.split('.')[1]) - 1}.weight"
            out = sel.get(before)
            nsd[name] = tensor[out] if out is not None else tensor

    new.load_state_dict(nsd)                                 # strict: a mismatch is a bug, not a detail
    return new.eval(), sel


def statistic(a, b, bar=8.0):
    """The acceptance rule: share of pixels whose worst channel differs by more than `bar` levels."""
    diff = (a.float() - b.float()).abs() * 255.0
    per_px = diff.amax(dim=1)
    return float((per_px > bar).float().mean()) * 100.0, float(per_px.max()), float(per_px.median())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("frame")
    ap.add_argument("width", type=int)
    ap.add_argument("height", type=int)
    ap.add_argument("--keep", type=int, default=32)
    ap.add_argument("--model", default="/tmp/opencode/models2/futsuu.pth")
    args = ap.parse_args()

    raw = np.fromfile(args.frame, dtype=np.uint8)
    img = raw.reshape(args.height, args.width, 3).astype(np.float32) / 255.0
    x = torch.from_numpy(img).permute(2, 0, 1)[None]

    model, scale, sd = load(args.model)
    model = model.cuda().half()
    with torch.no_grad():
        ref = model(x.cuda().half()).float()

    small, _ = cut_model(model, args.keep)
    small = small.cuda().half()
    with torch.no_grad():
        got = small(x.cuda().half()).float()

    share, peak, median = statistic(ref, got)
    print(f"naive cut 64 -> {args.keep}: share above 8 levels {share:.3f}% (bar 0.1), "
          f"peak {peak:.0f}, median {median:.2f} levels")
    print(f"reference output range {ref.min():.3f}..{ref.max():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
