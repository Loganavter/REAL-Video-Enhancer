"""A: can a cheap model serve a whole shot under the bar? Two models are enough to answer that.

The queue item is written around the 78-model collection, but its refutes clause is narrower than the
collection: "on every shot the cheap model exceeds the bar against the expensive one by more than 0.1%
of pixels at 8 levels, or the visible texture changes at shot boundaries". That is a comparison between
TWO models on the same frames - the collection is only needed for the routing half ("which of 78").

So this measures the half that decides whether the routing half is worth building at all: for each
frame, the expensive model's output is the reference and the cheap model's is the candidate; the
statistic is the project's usual one (share of pixels with any channel more than 8 levels off, plus the
peak), computed on the 4K output.

Usage (one GPU job at a time; ~0.3 s per frame per model):
  F=~/.cache/opencode/venvs/trt310/bin/python
  $F harness/harness/scripts/probe_shot_cascade.py --frames '/tmp/opencode/vin8/v*.raw' \
      --expensive "$B/custom_models/2x-Futsuu-Anime(1).pth" \
      --cheap "$B/custom_models/2x_ModernSpanimationV1.pth" --emit /tmp/opencode/cascade.jsonl
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch

import os as _bos, sys as _bsys
_HERE = _bos.path.dirname(_bos.path.abspath(__file__))
_ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in _bsys.path:
        _bsys.path.insert(0, _p)
from cut_channels_naive import load, statistic  # noqa: E402

BUNDLE = "/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2"


def load_any(path: str):
    """Any spandrel-format model: the compact SRVGG loader first, then the vendored spandrel.

    `cut_channels_naive.load` is built for the 34-conv compact net and raises on anything else (SPAN
    has no `34.weight`), so a two-model comparison needs both paths.
    """
    try:
        return load(path)
    except Exception:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "backend", "src", "pytorch"))
        from spandrel import ModelLoader  # type: ignore
        desc = ModelLoader().load_from_file(path)
        return desc.model, int(desc.scale), None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="glob of raw RGB frames")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--expensive", default=f"{BUNDLE}/custom_models/2x-Futsuu-Anime(1).pth")
    ap.add_argument("--cheap", default=f"{BUNDLE}/custom_models/2x_ModernSpanimationV1.pth")
    ap.add_argument("--emit", default="")
    args = ap.parse_args()

    ref_model, ref_scale, _ = load_any(args.expensive)
    chp_model, chp_scale, _ = load_any(args.cheap)
    print(f"expensive: {os.path.basename(args.expensive)} x{ref_scale}")
    print(f"cheap:     {os.path.basename(args.cheap)} x{chp_scale}")
    if ref_scale != chp_scale:
        print(f"REFUSED: scales differ ({ref_scale} vs {chp_scale}) - the outputs are not comparable")
        return 1
    ref_model = ref_model.eval().cuda().half()
    chp_model = chp_model.eval().cuda().half()

    files = sorted(glob.glob(args.frames))
    if not files:
        raise SystemExit(f"{args.frames}: no frames")

    rows = []
    worst_share = worst_peak = 0.0
    with torch.no_grad():
        for path in files:
            raw = np.fromfile(path, dtype=np.uint8)
            img = torch.from_numpy(raw.reshape(args.height, args.width, 3).astype(np.float32) / 255.0)
            x = img.permute(2, 0, 1)[None].cuda().half()
            ref = ref_model(x).float()
            got = chp_model(x).float()
            share, peak, median = statistic(ref, got)
            rows.append((os.path.basename(path), float(share), float(peak), float(median)))
            worst_share = max(worst_share, float(share))
            worst_peak = max(worst_peak, float(peak))
            print(f"  {os.path.basename(path):16s} >8 {float(share):8.4f}%  peak {float(peak):5.0f}  "
                  f"median {float(median):5.2f}")

    verdict = "interchangeable" if worst_share <= 0.1 else "refused"
    print(f"\nWORST of {len(files)}: >8 {worst_share:.4f}% (bar 0.1)  peak {worst_peak:.0f}  "
          f"-> shot-level cascade {verdict.upper()}")
    print("NOT checked: the routing half (which of the 78), the shot boundary itself, and the wall "
          "clock - this is a picture question only")

    if args.emit:
        from harness import Population, Record, Store
        store = Store()
        pop = Population.of("frames", [r[0] for r in rows],
                            label=f"{os.path.basename(args.cheap)} against "
                                  f"{os.path.basename(args.expensive)}, {args.width}x{args.height}")
        store.emit(Record(value=round(worst_share, 4), unit="%", statistic="worst_of", population=pop,
                          instrument="cut_channels_naive.statistic(): worst channel >8 levels",
                          config=f"fp16, {args.width}x{args.height} -> 4K, two models",
                          session=store.session, run="shot-cascade",
                          extra={"of": "share", "quantity": "accuracy",
                                 "peak_levels": round(worst_peak, 1),
                                 "per_frame": rows}))
        store.dump(args.emit)
        print(f"records -> {args.emit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
