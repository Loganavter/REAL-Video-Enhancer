"""Native widths: does the int8 body work, and how fast, when the channel count is not 64?

The kernel takes the width as a `tl.constexpr`, so a 48-channel body is a recompile rather than a rewrite - but
the tile tuning (BLOCK_M=128, warps=4) was measured at 64 channels, and the previous verdict that closed widths
32/48 measured CUTTING a trained 64-channel model (14-37% per-layer residual: a model that cannot express its
teacher). A model TRAINED at 48 has no teacher to fail; what is unknown is the quantiser and the tile at that
width, which is what this measures.

The int8 dot's K must be a multiple of 16, so a 24-channel Compact cannot run as-is; `Int8Body` refuses it with
that reason rather than failing inside Triton.

Usage:
  python scripts/int8_width_probe.py --models "Archivist_AntiLines.pth,1x-RGB-max-Denoise.pth"
  python scripts/int8_width_probe.py --models "Archivist_AntiLines.pth" --sweep
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

APP = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(APP, "backend", "src", "pytorch"))
sys.path.insert(0, os.path.join(REPO, "backend"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from model_loader import load_compact, read_shape  # noqa: E402
from src.pytorch.Int8Body import Int8Body  # noqa: E402
from src.pytorch.TorchUtils import TorchUtils  # noqa: E402
from tail_int8_emulation import CALIB, H, W, decode_yuv10  # noqa: E402

ROOT = "/home/jorj/Загрузки/Upscaling"
FRAMES = (100, 169, 250)


def find(name: str) -> str:
    for dirpath, _dirs, files in os.walk(ROOT):
        if name in files:
            return os.path.join(dirpath, name)
    raise SystemExit(f"not found: {name}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--sweep", action="store_true", help="also sweep BLOCK_M and warps on the first model")
    ap.add_argument("--out-file", default="")
    args = ap.parse_args(argv)

    dev = torch.device("cuda", 0)
    tu = TorchUtils(width=W, height=H, device_type="cuda", hdr_mode=True, gpu_id=0, high_bitdepth=True)
    st = torch.cuda.Stream(device=dev)

    def frame(i):
        return tu.yuv420p10_to_tensor(decode_yuv10(i), st, dev, torch.float16)

    calib = [frame(i) for i in CALIB]
    held = [frame(i) for i in FRAMES]

    def timeit(fn):
        for _ in range(3):
            fn()
            torch.cuda.synchronize()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        for _ in range(args.iters):
            fn()
            torch.cuda.synchronize()
        b.record()
        torch.cuda.synchronize()
        return a.elapsed_time(b) / args.iters

    rows = []
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        path = find(name)
        shape = read_shape(path)
        row = {"file": name, **shape}
        try:
            model, _scale = load_compact(path)
            model = model.cuda().half()
            body = Int8Body(model)                     # width inferred from the model
            body.out_yuv = False
            body.calibrate(calib)
            body._build()
            shares = [body.certify(f)[0] for f in held]
            draws = [timeit(lambda: body(held[1])) for _ in range(args.rounds)]
            ms = statistics.median(draws[1:]) if len(draws) > 1 else draws[0]
            row.update(ok=True, feat=body.feat, share=round(max(shares), 5),
                       shares=[round(v, 5) for v in shares], body_ms=round(ms, 2),
                       draws=[round(d, 2) for d in draws])
            print(f"{name[:44]:44s} feat {body.feat:3d} x{shape['scale']}  share {max(shares):7.4f}%  "
                  f"body {ms:6.2f} ms", file=sys.stderr)
            del body
            torch.cuda.empty_cache()
        except Exception as error:                     # noqa: BLE001
            row.update(ok=False, error=f"{type(error).__name__}: {str(error)[:120]}")
            print(f"{name[:44]:44s} REFUSED {row['error']}", file=sys.stderr)
        rows.append(row)

    if args.sweep:
        name = [m.strip() for m in args.models.split(",") if m.strip()][0]
        model, _scale = load_compact(find(name))
        model = model.cuda().half()
        print(f"\n# tile sweep at feat={read_shape(find(name))['feat']} ({name})", file=sys.stderr)
        grid = []
        for block_m in (64, 128, 256):
            for warps in (2, 4, 8):
                try:
                    body = Int8Body(model, block_m=block_m, warps=warps)
                    body.out_yuv = False
                    body.calibrate(calib)
                    body._build()
                    draws = [timeit(lambda: body(held[1])) for _ in range(args.rounds)]
                    ms = statistics.median(draws[1:]) if len(draws) > 1 else draws[0]
                    grid.append({"block_m": block_m, "warps": warps, "body_ms": round(ms, 2)})
                    print(f"  BLOCK_M {block_m:3d} warps {warps}  {ms:6.2f} ms", file=sys.stderr)
                    del body
                    torch.cuda.empty_cache()
                except Exception as error:             # noqa: BLE001
                    print(f"  BLOCK_M {block_m:3d} warps {warps}  FAILED {str(error)[:70]}", file=sys.stderr)
        rows[0]["tile_sweep"] = grid

    if args.out_file:
        os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
        json.dump({"take": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "frames": list(FRAMES), "rows": rows}, open(args.out_file, "w"), indent=1)
        print(f"# -> {args.out_file}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
