"""Why does the per-channel int8 grid help some models and hurt others? Two candidate mechanisms, measured.

The mode changes two things at once, and only one of them is usually discussed:

  (a) WEIGHT FOLDING. Per-channel folds the per-input-channel scale into the weights (`w * s_in`) so the
      codes can use the full +-127 range; the quantisation error is then measured on the FOLDED weights,
      whose dynamic range across inputs is wider. Per-tensor quantises the original weights and restores the
      per-channel scale in the epilogue instead. So the "better input grid" is paid for in the weights.
  (b) HEADROOM. A per-tensor grid is one wide box, a per-channel grid is 64 tight ones. A held-out frame whose
      activation exceeds the calibration maximum of a low-range channel CLIPS in per-channel mode where the
      wide box would have absorbed it. The grid is calibrated on 24 frames and judged on others, so this is a
      generalisation question, not a bug.

This probe measures both, per layer, per mode, plus the resulting gross-error share:

  weight_rel_err   ||w - dequant(q(w))|| / ||w||, on the folded weights for per-channel and the raw ones for
                   per-tensor - the error the model actually sees, in the domain it multiplies the codes in
  clip_share       share of activations whose code would exceed +-127 against THAT mode's grid, on held-out
                   frames (the frames the calibration never saw)
  share            the same gross-error gate the app applies at start-up

Usage: python scripts/int8_grid_modes_probe.py [--models a.pth,b.pth] [--frames 3] [--out-file x.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

APP = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(APP, "backend", "src", "pytorch"))
sys.path.insert(0, os.path.join(REPO, "backend"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from cut_channels_naive import load  # noqa: E402
from src.pytorch.Int8Body import Int8Body  # noqa: E402
from src.pytorch.TorchUtils import TorchUtils  # noqa: E402
from tail_int8_emulation import CALIB, H, W, decode_yuv10  # noqa: E402

ROOT = "/home/jorj/Загрузки/Upscaling"
DEFAULT = ("2x-Futsuu-Anime(1).pth,2xken-v1-evA-01.pth,2xHFA2k_LUDVAE_compact(2).pth,"
           "2x_GarfieldJr_net_g_51000.pth,2x_AniSD_AC_G6i2a_Compact_72500.pth,2x-AniScale.pth")


def find(name: str) -> str:
    for dirpath, _dirs, files in os.walk(ROOT):
        if name in files:
            return os.path.join(dirpath, name)
    raise SystemExit(f"not found: {name}")


def weight_error(body, per_channel: bool) -> float:
    """Relative L2 of the quantised weights, in the domain the kernel multiplies codes in."""
    num = den = 0.0
    s_in = body._grids[0]
    for idx, conv in enumerate(body.inner):
        w = conv.weight.detach().float()
        if per_channel:
            w = w * s_in.view(1, -1, 1, 1)
        w_sc = w.abs().amax(dim=(1, 2, 3)) / 127.0
        w_q = torch.round(w / w_sc.view(-1, 1, 1, 1)).clamp(-127, 127)
        num += float(((w - w_q * w_sc.view(-1, 1, 1, 1)) ** 2).sum())
        den += float((w ** 2).sum())
        s_in = body._grids[1][idx]
    return (num / den) ** 0.5


def activation_stats(body, per_channel: bool, frame) -> tuple[float, float]:
    """(clip share, relative quantisation error) of the activations against THIS mode's grid.

    The other half of the trade: per-channel grids should quantise the activations better, because each
    channel uses its own +-127 range instead of a fraction of a shared one. Both errors are measured on
    frames the calibration never saw.
    """
    s_act, outs = body._grids
    total = clipped = 0
    err = ref = 0.0
    with torch.no_grad():
        act = torch.nn.functional.prelu(body.head_conv(frame.clamp(0.0, 1.0)), body.head_prelu.weight)
        grids = [s_act] + list(outs)
        for idx, (conv, prelu) in enumerate(zip(body.inner, body.inner_prelus)):
            grid = grids[idx]
            g = grid if per_channel else grid.max().expand_as(grid)
            # fp32 for the statistics: in fp16 the squared residual of a well-quantised activation is ~1e-6,
            # which underflows, and the first version of this probe read exactly 0.0 for every model
            a = act.float()
            total += a.numel()
            codes = a / g.view(1, -1, 1, 1).float()
            clipped += int((codes.abs() > 127.0).sum())
            dq = torch.round(codes).clamp(-127, 127) * g.view(1, -1, 1, 1).float()
            err += float(((a - dq) ** 2).sum())
            ref += float((a ** 2).sum())
            act = torch.nn.functional.prelu(conv(act), prelu.weight)
    return 100.0 * clipped / max(1, total), (err / max(1e-30, ref)) ** 0.5


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=DEFAULT)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--out-file", default="")
    args = ap.parse_args(argv)

    dev = torch.device("cuda", 0)
    tu = TorchUtils(width=W, height=H, device_type="cuda", hdr_mode=True, gpu_id=0, high_bitdepth=True)
    st = torch.cuda.Stream(device=dev)

    def frame(i):
        return tu.yuv420p10_to_tensor(decode_yuv10(i), st, dev, torch.float16)

    calib = [frame(i) for i in CALIB]
    held = [frame(i) for i in (100, 169, 250)]

    rows = []
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        path = find(name)
        model, _, _ = load(path)
        model = model.eval().cuda().half()
        row = {"file": name}
        for mode, pc in (("per_tensor", False), ("per_channel", True)):
            # the mode is read from the environment when the body builds; set it here so both builds happen
            # in one process on the same frames, which is what makes the two rows comparable
            os.environ["RVE_INT8_PER_CHANNEL"] = "1" if pc else "0"
            body = Int8Body(model)
            body.out_yuv = False
            body.calibrate(calib)
            body._build()
            stats = [activation_stats(body, pc, f) for f in held]
            row[mode] = {
                "weight_rel_err": round(weight_error(body, pc), 5),
                "clip_share_pct": [round(c, 5) for c, _e in stats],
                "input_rel_err": [round(e, 5) for _c, e in stats],
            }
            del body
            torch.cuda.empty_cache()
        rows.append(row)
        print(json.dumps(row))
    if args.out_file:
        os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
        json.dump({"take": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rows": rows},
                  open(args.out_file, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
