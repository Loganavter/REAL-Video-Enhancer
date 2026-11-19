"""For every model the inventory says fits: does the int8 body actually certify on it, and is it faster?

The inventory (`model_inventory.py`) answers "can the pipeline run this arch at all" from the model's own
structure. This answers the two questions that follow, per model, on one held-out frame:

  picture   the body's gross-error share against its OWN fp16 model - percent of pixels more than 8 levels
            off in any channel. The app refuses a body above 1% on its first frame (FIRST_FRAME_SANITY), so
            this is the same gate the render applies, not a new one. It is n=1 and NOT an acceptance number:
            the picture is judged on 494-frame pairs.
  speed     body vs the model's own fp16 forward, interleaved, medians of rounds 2-3. The lane's bar is a
            ratio, and this is the ratio a model would get with the same body.

A model that certifies but does not get faster is a finding too - the int8 win is a property of 64 channels
and 16 hops, and a model with the same shape but a much heavier tail would dilute it.

Usage: python scripts/model_fit_probe.py [--limit N] [--out-file x.json]
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

from model_loader import load_compact  # noqa: E402  any Compact, not only the 16x64 production shape
from src.pytorch.Int8Body import FIRST_FRAME_SANITY, Int8Body  # noqa: E402
from src.pytorch.TorchUtils import TorchUtils  # noqa: E402
from tail_int8_emulation import CALIB, H, W, decode_yuv10  # noqa: E402

ROOT = "/home/jorj/Загрузки/Upscaling"
FRAME = 169


def candidates(inventory: str) -> list[dict]:
    rows = json.load(open(inventory))["rows"]
    return [r for r in rows if r["verdict"] in ("FULL", "BODY+TAIL", "BODY")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", default="/tmp/opencode/models_inventory.json")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="comma-separated substrings: measure only these files")
    ap.add_argument("--out-file", default="")
    args = ap.parse_args(argv)

    dev = torch.device("cuda", 0)
    tu = TorchUtils(width=W, height=H, device_type="cuda", hdr_mode=True, gpu_id=0, high_bitdepth=True)
    st = torch.cuda.Stream(device=dev)

    def frame(i):
        return tu.yuv420p10_to_tensor(decode_yuv10(i), st, dev, torch.float16)

    calib = [frame(i) for i in CALIB]
    x = frame(FRAME)

    rows = candidates(args.inventory)
    if args.only:
        wanted = [w.strip().lower() for w in args.only.split(",") if w.strip()]
        rows = [r for r in rows if any(w in r["file"].lower() for w in wanted)]
    if args.limit:
        rows = rows[: args.limit]

    results = []
    for row in rows:
        path = os.path.join(row["dir"], row["file"])
        out = {"file": row["file"], "verdict": row["verdict"], "scale": row["scale"], "ok": False}
        try:
            model, _scale = load_compact(path)
            model = model.cuda().half()
            body = Int8Body(model)
            body.out_yuv = False                 # this probe measures the body, not the payload path
            body.calibrate(calib)
            body._build()
            out.update(flags={k: bool(getattr(body, k, False)) for k in
                              ("head_fused", "tail_int8", "tail_shuffle", "tail_yuv")})
            # the calibration grid's DYNAMIC RANGE: a per-tensor grid is one wide box, so a model whose
            # channel maxima span an order of magnitude loses resolution for most of its channels. That
            # ratio is the predictor for "would per-channel grids rescue this model".
            s_act = body._grids[0].float()
            out.update(grid_spread=round(float(s_act.max() / s_act.median().clamp_min(1e-9)), 2))
            # three held-out frames, worst share: one frame is the app's start-up gate, not a picture claim
            shares = []
            for f in (100, FRAME, 250):
                sh, _ok = body.certify(frame(f))
                shares.append(sh)
            share = max(shares)
            out.update(share=round(share, 5), shares=[round(v, 5) for v in shares],
                       ok=bool(share <= FIRST_FRAME_SANITY))

            def timeit(fn):
                # one synchronise per call on purpose: the body reads its OWN event pair at the start of the
                # next frame (that is how the app attributes phases without syncing), so a back-to-back loop
                # asks elapsed_time of events that have not completed yet. Both arms pay the same sync.
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

            draws = {"fp16": [], "int8": []}
            for r in range(args.rounds):
                with torch.no_grad():
                    draws["fp16"].append(timeit(lambda: model(x.clamp(0.0, 1.0))))
                    draws["int8"].append(timeit(lambda: body(x)))
            kept = slice(1, None) if args.rounds > 1 else slice(None)
            fp16_ms = statistics.median(draws["fp16"][kept])
            int8_ms = statistics.median(draws["int8"][kept])
            out.update(fp16_ms=round(fp16_ms, 3), int8_ms=round(int8_ms, 3),
                       ratio=round(fp16_ms / int8_ms, 4), draws=draws)
            print(f"{out['verdict']:9s} {row['file'][:40]:40s} share {share:7.4f}%  "
                  f"fp16 {fp16_ms:6.2f}  int8 {int8_ms:6.2f}  ratio {out['ratio']:.3f}", file=sys.stderr)
        except Exception as error:                      # noqa: BLE001
            out["error"] = f"{type(error).__name__}: {str(error)[:140]}"
            print(f"{row['verdict']:9s} {row['file'][:40]:40s} ERROR {out['error']}", file=sys.stderr)
        results.append(out)
        torch.cuda.empty_cache()

    if args.out_file:
        os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
        json.dump({"take": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "frame": FRAME, "rounds": args.rounds, "iters": args.iters,
                   "rows": results}, open(args.out_file, "w"), indent=1)
        print(f"# -> {args.out_file}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
