"""Records for the 6-bit-codes falsifier: two variants against the 8-bit body, and the control.

Usage: python harness/scripts/sub8_records.py [--probe docs/wsp/verified/logs/sub8/probe.json]
                                      [--out docs/wsp/verified/records/int8-sub8-codes.jsonl]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import os as _bos, sys as _bsys
_HARNESS_ROOT = _bos.path.dirname(_bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
if _HARNESS_ROOT not in _bsys.path:
    _bsys.path.insert(0, _HARNESS_ROOT)

from harness.core import Population, Record, Store  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="docs/wsp/verified/logs/sub8/probe.json")
    ap.add_argument("--out", default="docs/wsp/verified/records/int8-sub8-codes.jsonl")
    args = ap.parse_args(argv)

    pr = json.load(open(args.probe))
    dig, sess = hashlib.sha256(open(args.probe, "rb").read()).hexdigest(), f"s-{pr['take']}"
    clip = pr["clip"]
    frames = [f"{clip}#frame{f}" for f in pr["held_out"]]
    pop = Population.of("frames", frames,
                        f"{clip} frames {pr['held_out']} (held out: calibrated on {pr['calib']}), "
                        f"sixteen hops with 6-bit codes against the 8-bit body")
    inst = "the body's arithmetic in torch, integer accumulation exact in float32 (9.3e6 < 2^24)"
    st = Store(session=sess)
    for i, r in enumerate(pr["rows"], start=1):
        st.emit(Record(value=round(r["rescale_share"], 5), unit="%", population=pop, instrument=inst,
                       config=f"6-bit codes, rescale variant (s' = s*127/31, same range), frame {r['frame']}",
                       session=sess, statistic="single", tier="measured", id=f"S{i:03d}",
                       run="scripts/sub8_codes_probe.py", artifact=args.probe,
                       extra={"artifact_sha256": dig, "of": "share", "frame": r["frame"],
                              "peak": round(r["rescale_peak"], 1),
                              "eight_bit_share": round(r["eight_share"], 5)}))
    st.emit(Record(value=round(pr["worst"]["rescale_vs_8"], 5), unit="%", population=pop,
                   instrument=inst, config="6-bit codes, rescale variant (s' = s*127/31, same range)",
                   session=sess, statistic="worst_of", tier="measured", id="S005",
                   run="scripts/sub8_codes_probe.py", artifact=args.probe,
                   extra={"artifact_sha256": dig, "of": "share", "bar": "bar_int8_share"}))
    st.emit(Record(value=round(pr["worst"]["rescale_peak"], 1), unit="levels", population=pop,
                   instrument="channel-max |delta| in RGB levels", 
                   config="6-bit codes, rescale variant (s' = s*127/31, same range)",
                   session=sess, statistic="max", tier="measured", id="S006",
                   run="scripts/sub8_codes_probe.py", artifact=args.probe,
                   extra={"artifact_sha256": dig, "of": "peak", "bar": "bar_int8_peak"}))
    st.emit(Record(value=round(pr["worst"]["clip_vs_8"], 5), unit="%", population=pop, instrument=inst,
                   config="6-bit codes, clip variant (same step, range clamped to +-31)",
                   session=sess, statistic="worst_of", tier="measured", id="S007",
                   run="scripts/sub8_codes_probe.py", artifact=args.probe,
                   extra={"artifact_sha256": dig, "of": "share", "bar": "bar_int8_share",
                          "peak": round(pr["worst"]["clip_peak"], 1)}))
    # the control: the same emulation at 8 bits, against the model - if this is not the recorded
    # 0.006-0.015%, the chain is not the body and the 6-bit numbers mean nothing
    st.emit(Record(value=round(max(r["eight_share"] for r in pr["rows"]), 5), unit="%", population=pop,
                   instrument=inst, config="8-bit body (the shipped recipe), the same chain - CONTROL",
                   session=sess, statistic="worst_of", tier="measured", id="S008",
                   run="scripts/sub8_codes_probe.py", artifact=args.probe,
                   extra={"artifact_sha256": dig, "of": "share",
                          "note": "must land at 0.006-0.016% - the corpus's recorded body-vs-model "
                                  "range - or the chain is not the body"}))
    st.dump(args.out)
    print(st.claims_text().strip())
    print(f"# -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
