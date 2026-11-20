"""Emulate the KERNEL's own quantiser in torch, prove the emulation, then let it explain the 6-14x gap.

The lane has a disagreement it cannot explain: `harness/scripts/int8_accuracy_sim.py` reads 0.06510% on frame 29
while the app's Triton body reads 0.42-0.91% over the same clip. Two earlier attempts to compare them hop by
hop failed because they compared two different recipes (the body quantises SYMMETRICALLY on +-127, the sim
ASYMMETRICALLY over 0..255), so the diff was a unit error.

This does it the other way round: write the body's arithmetic in torch, check it reproduces the body (which
proves both), and then swap one term at a time and read the picture against the model's own fp16 output. The
first term is the quantiser scheme, because that is the one the two instruments disagree about by design.

Integer accumulation is exact in float here: int8 codes times int8 weights sum over 64*9 terms, at most
64*9*127*127 = 9.3e6, far below 2^24, so the emulation cannot drift through its own arithmetic.

Usage: python harness/scripts/int8_kernel_emulation.py --frame 169
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as TF

REPO = "/home/jorj/Загрузки/projects/REAL-Video-Enhancer"
sys.path.insert(0, os.path.join(REPO, "backend"))
import os as _bos, sys as _bsys
_HERE = _bos.path.dirname(_bos.path.abspath(__file__))
_ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in _bsys.path:
        _bsys.path.insert(0, _p)

from cut_channels_naive import load  # noqa: E402
from src.pytorch.Int8Body import Int8Body, SHIFT  # noqa: E402

W, H = 1920, 1080


def load_frame(frames_dir: str, idx: int) -> torch.Tensor:
    raw = np.fromfile(os.path.join(frames_dir, f"v{idx}.raw"), dtype=np.uint8)
    img = torch.from_numpy(raw.reshape(H, W, 3).astype(np.float32) / 255.0)
    return img.permute(2, 0, 1)[None].cuda()


@torch.no_grad()
def emulate(body: Int8Body, x: torch.Tensor, scheme: str = "sym127") -> torch.Tensor:
    """The body's arithmetic: head -> 16 int8 hops -> dequantised (1,64,H,W).

    scheme: 'sym127' = what the kernel does (symmetric, +-127, requant into the next per-tensor grid);
            'asym255' = what the simulation does (256 codes over [lo, hi]); kept for the comparison.
    """
    h, w = x.shape[2], x.shape[3]
    s_act, outs = body._grids
    head = body.head_conv(x.clamp(0.0, 1.0).half())
    act = TF.prelu(head, body.head_prelu.weight)

    if scheme == "sym127":
        s_t1 = body._s_t1.view(1, -1, 1, 1)
        codes = torch.round(act / s_t1).clamp(-127, 127)
        s_in = body._s_t1
    else:
        lo, hi = act.amin(), act.amax()
        step = ((hi - lo) / 255.0).clamp_min(1e-8)
        codes = torch.round((act - lo) / step).clamp(0, 255)
        s_in = None

    for idx, conv in enumerate(body.inner):
        weight = conv.weight.detach().float()
        if scheme == "sym127":
            s_t = float(s_in.max())
            w_sc = weight.abs().amax(dim=(1, 2, 3)) / 127.0
            w_q = torch.round(weight / w_sc.view(-1, 1, 1, 1)).clamp(-127, 127)
            acc = TF.conv2d(codes.float(), w_q, None, padding=1)          # exact integers in float
            v = acc * (w_sc * s_t).view(1, -1, 1, 1) + conv.bias.detach().float().view(1, -1, 1, 1)
        else:
            raise SystemExit("asym255 emulation is not written yet - the first question is whether sym127 "
                             "reproduces the body at all")
        v = TF.prelu(v, body.inner_prelus[idx].weight.detach().float())
        s_out = outs[idx].view(1, -1, 1, 1)
        q = torch.floor(v / s_out + 0.5).clamp(-127, 127)
        if scheme == "sym127" and idx + 1 < len(body.inner):
            # the epilogue requants into the NEXT hop's per-tensor grid
            s_next = outs[idx].max()
            m = torch.round(outs[idx] / s_next * (1 << SHIFT)).to(torch.float32)
            q = torch.clamp(torch.floor((q * m.view(1, -1, 1, 1) + (1 << (SHIFT - 1))) / (1 << SHIFT)),
                            -127, 127)
            codes, s_in = q, outs[idx]
        else:
            codes, s_in = q, outs[idx]
    return (codes * outs[-1].view(1, -1, 1, 1)).half()


def image(model, body, act: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """The rest of the model, as the caller runs it: tail conv + upsampler + residual."""
    with torch.no_grad():
        out = model.upsampler(body.tail_conv(act.half()))
        return (out + TF.interpolate(x.clamp(0.0, 1.0), scale_factor=model.upscale, mode="nearest")).float()


def stat(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    d = (a - b).abs().amax(dim=1) * 255.0
    return float((d > 8).float().mean() * 100), float(d.max())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=169)
    ap.add_argument("--all-frames", action="store_true",
                    help="run the frames that are dumped, not just one")
    ap.add_argument("--frames", default="/var/tmp/rve-rem/frames")
    ap.add_argument("--calib", default="20,60,100,140,180,220,260,300")
    ap.add_argument("--model", default="/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2/"
                                       "custom_models/2x-Futsuu-Anime(1).pth")
    args = ap.parse_args(argv)

    model, _, _ = load(args.model)
    model = model.eval().cuda().half()
    kframes = [load_frame(args.frames, int(i)).half() for i in args.calib.split(",")]
    todo = ([29, 44, 100, 169, 274, 400, 460] if args.all_frames else [args.frame])
    xs = {f: load_frame(args.frames, f).half() for f in todo}

    body = Int8Body(model)
    body.calibrate(kframes)
    body._build()

    import subprocess
    FF = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2/bin/ffmpeg"
    REF = "/var/tmp/rve-rem/ref_fp16_y8.mkv"

    def engine_frame(f: int) -> torch.Tensor:
        raw = subprocess.run([FF, "-loglevel", "error", "-i", REF, "-vf", f"select=eq(n\\,{f})",
                              "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
                             capture_output=True).stdout
        t = torch.from_numpy(np.frombuffer(raw[:3840 * 2160 * 3], dtype=np.uint8).reshape(2160, 3840, 3))
        return (t.permute(2, 0, 1)[None].cuda().float() / 255.0)

    print("frame  emul-vs-kernel   body-vs-MODEL   engine-vs-MODEL")
    for f in todo:
        x = xs[f]
        with torch.no_grad():
            body_img = body(x).float()
            emul_img = image(model, body, emulate(body, x), x)
            model_out = model(x).float()
        ek, pk = stat(emul_img, body_img)
        sb, pb = stat(body_img, model_out)
        se, pe = stat(engine_frame(f), model_out)
        print(f"{f:>5}  {ek:8.5f}%       {sb:8.5f}%       {se:8.5f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
