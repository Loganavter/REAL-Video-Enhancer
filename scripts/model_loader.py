"""Load any SRVGGNetCompact from a state dict, whatever its width and depth.

Why this exists: `cut_channels_naive.load()` (the probe loader this lane grew up with) hard-codes
`num_feat=64, num_conv=16` and finds the tail by the key `34.weight`. That is the production model's shape, not
the family's - so it cannot open a nine-hop SuperUltraCompact or a 48-channel Archivist denoiser, and a sweep
that uses it dies on the first such file with a KeyError that looks like a broken model.

Everything here is inferred from the state dict: the head's output width, the number of 3x3 body convs, and the
upscale factor from the tail's output channels (3 * r^2).
"""
from __future__ import annotations

import os
import sys

import torch

APP = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2"
if f"{APP}/backend/src/pytorch" not in sys.path:
    sys.path.insert(0, f"{APP}/backend/src/pytorch")

from spandrel.architectures.Compact.__arch.SRVGG import SRVGGNetCompact  # noqa: E402


def read_shape(path: str) -> dict:
    """(feat, inner convs, scale) from the weights alone - no model built, no GPU."""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = sd.get("params", sd)
    convs = [k for k in sd if k.endswith(".weight") and getattr(sd[k], "ndim", 0) == 4]
    convs.sort(key=lambda s: int(s.split(".")[1]))
    if len(convs) < 3:
        raise ValueError(f"{os.path.basename(path)}: {len(convs)} convolutions - not a Compact body")
    feat = int(sd[convs[0]].shape[0])
    tail_out = int(sd[convs[-1]].shape[0])
    scale = int(round((tail_out / 3) ** 0.5))
    return {"feat": feat, "inner": len(convs) - 2, "scale": scale, "tail_out": tail_out}


def load_compact(path: str):
    """(model.eval(), scale) for any Compact/SRVGG state dict, at its own width and depth."""
    shape = read_shape(path)
    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=shape["feat"],
                            num_conv=shape["inner"], upscale=shape["scale"])
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = sd.get("params", sd)
    model.load_state_dict(sd)
    return model.eval(), shape["scale"]
