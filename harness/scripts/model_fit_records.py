"""Records for the model-fit sweep: per model, the picture gate and the body's speed against its own fp16.

Two populations per model, deliberately separate - the gross-error share is a worst-of over three frames, the
timings are a median over rounds, and a ratio between them would be nonsense.

The fp16 column is EAGER PyTorch fp16, not the app's TensorRT engine: the probe calls the model directly, while
the app runs a per-model engine that is ~4x faster. So the ratios here compare the BODY ACROSS MODELS and say
nothing about the product's ratio - that one is measured in the app (`int8-tail-csc.jsonl` JC004). The
distinction is written into the instrument string, because it is exactly the "instrument swap" trap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import os as _bos, sys as _bsys
_HERE = _bos.path.dirname(_bos.path.abspath(__file__))
_ROOT = _bos.path.dirname(_bos.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in _bsys.path:
        _bsys.path.insert(0, _p)

from harness.core import Population, Record, Store  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="docs/wsp/verified/logs/model-fit/all.json")
    ap.add_argument("--out", default="docs/wsp/verified/records/model-fit.jsonl")
    ap.add_argument("--probe-pc", default="docs/wsp/verified/logs/model-fit/all-perchannel.json",
                    help="the same sweep with per-channel grids: its shares are records too, because the page "
                         "quotes them")
    args = ap.parse_args(argv)

    pr = json.load(open(args.probe))
    dig = hashlib.sha256(open(args.probe, "rb").read()).hexdigest()
    store = Store(session=f"s-{pr['take']}")
    out: list[Record] = []

    frames = [100, pr["frame"], 250]
    for row in pr["rows"]:
        if "error" in row or "share" not in row:
            continue
        name = row["file"]
        popf = Population.of("frames", [f"{name}#frame{i}" for i in frames],
                             f"{name}, three held-out frames, worst of the three")
        instr = "the int8 body's own first-frame gate: pixels more than 8 levels off in any channel, vs the " \
                "model's own fp16 forward"
        share = Record(value=row["share"], unit="%", population=popf, instrument=instr,
                       config=f"{name}, {row['verdict']}, calibrated on the same spread frames",
                       session=store.session, statistic="worst_of", tier="measured",
                       run="harness/scripts/model_fit_probe.py", artifact=args.probe,
                       extra={"artifact_sha256": dig, "of": "gross_error_share",
                              "shares": row.get("shares")})
        share.id = f"MF{len(out) + 1:03d}"
        store.emit(share)
        out.append(share)

        popr = Population.of("rounds", [f"{name}#round-{r}" for r in range(1, pr["rounds"] + 1)],
                             f"{name}, interleaved rounds, medians of rounds 2+")
        tinstr = ("torch.cuda.Event around the forward, one synchronise per call (the body reads its own "
                  "event pair on the next frame); EAGER fp16 for the model, not the app's TensorRT engine")
        ids = {}
        for arm, key in (("fp16", "fp16_ms"), ("int8", "int8_ms")):
            rec = Record(value=row[key], unit="ms", population=popr, instrument=tinstr,
                         config=f"{name}, {row['verdict']}, 1080p frame {pr['frame']}",
                         session=store.session, statistic="median", tier="measured",
                         run="harness/scripts/model_fit_probe.py", artifact=args.probe,
                         extra={"artifact_sha256": dig, "of": "body_time", "arm": arm})
            rec.id = f"MF{len(out) + 1:03d}"
            store.emit(rec)
            out.append(rec)
            ids[arm] = rec.id
        ratio = store.ratio(ids["fp16"], ids["int8"], instrument=tinstr,
                           note=f"{name}: eager fp16 / int8 body - a comparison ACROSS MODELS, not the "
                                f"product's ratio (that one is in int8-tail-csc.jsonl)")
        store.records.pop(ratio.id, None)
        ratio.id = f"MF{len(out) + 1:03d}"
        store.records[ratio.id] = ratio
        out.append(ratio)

    # the per-channel shares: same instrument, different grid - a separate record set, never merged with the
    # per-tensor ones, because the grid is the thing under test
    if args.probe_pc and os.path.exists(args.probe_pc):
        pcpr = json.load(open(args.probe_pc))
        pcdig = hashlib.sha256(open(args.probe_pc, "rb").read()).hexdigest()
        for row in pcpr["rows"]:
            if "error" in row or "share" not in row:
                continue
            popf = Population.of("frames", [f"{row['file']}#frame{i}" for i in frames],
                                 f"{row['file']}, three held-out frames, per-channel grid, worst of three")
            rec = Record(value=row["share"], unit="%", population=popf, instrument=instr,
                         config=f"{row['file']}, {row['verdict']}, RVE_INT8_PER_CHANNEL=1",
                         session=store.session, statistic="worst_of", tier="measured",
                         run="harness/scripts/model_fit_probe.py", artifact=args.probe_pc,
                         extra={"artifact_sha256": pcdig, "of": "gross_error_share",
                                "grid": "per_channel", "grid_spread": row.get("grid_spread")})
            rec.id = f"MG{len(out) + 1:03d}"
            store.emit(rec)
            out.append(rec)

    store.dump(args.out)
    for rec in out:
        print(rec.claims_line())
    print(f"# {len(out)} records -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
