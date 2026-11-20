"""Records for the peak look: what the shipped default's worst pixel actually is.

Usage: python harness/harness/scripts/peak_look_records.py [--frames docs/wsp/verified/logs/peak-look/frames.json]
                                           [--out docs/wsp/verified/records/peak-look.jsonl]
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
    ap.add_argument("--frames", default="docs/wsp/verified/logs/peak-look/frames.json")
    ap.add_argument("--out", default="docs/wsp/verified/records/peak-look.jsonl")
    args = ap.parse_args(argv)

    d = json.load(open(args.frames))
    dig = hashlib.sha256(open(args.frames, "rb").read()).hexdigest()
    clip = d["clip"]
    per_frame = [r for r in d["rows"] if "peak" in r]
    shares = [r for r in d["rows"] if r.get("threshold") == 8]
    pop = Population.of("frames", [f"{clip}#frame{r['frame']}" for r in per_frame],
                        f"{clip}, {len(per_frame)} spread frames (held out of the {len(d['calib'])} "
                        f"calibration frames), the SHIPPED int8 default against the model's fp16 output, "
                        f"the pre-encode 4K tensor")
    inst = "channel-max |difference| in RGB levels between the model and the body, per pixel, on the 4K tensor"
    cfg = "shipped default: fused head kernel, 16 int8 hops, int8 tail kernel (RVE_HEAD_FUSED=1, RVE_TAIL_INT8=1)"
    sess = "s-2026-11-20T04:30:00Z"
    st = Store(session=sess)

    def emit(rid, value, unit, statistic, why, **extra):
        st.emit(Record(value=value, unit=unit, population=pop, instrument=inst, config=cfg, session=sess,
                       statistic=statistic, tier="measured", id=rid, run="harness/scripts/peak_look.py",
                       artifact=args.frames, extra={"artifact_sha256": dig, **extra, "note": why}))

    emit("P001", float(max(r["peak"] for r in per_frame)), "levels", "max", "the worst pixel on any frame",
         of="peak", bar="bar_int8_peak")
    emit("P002", round(max(r["share_pct"] for r in shares), 5), "%", "worst_of",
         "the area statistic on the worst frame", of="share", bar="bar_int8_share")
    emit("P003", float(max(r["largest_component_8"] for r in per_frame)), "count", "max",
         "the largest connected run of >8-level pixels on any frame - specks, not patches", of="blob")
    emit("P004", float(min(r["largest_component_8"] for r in per_frame)), "count", "min",
         "the smallest largest-run across the frames", of="blob")
    st.dump(args.out)
    print(st.claims_text().strip())
    print(f"# -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
