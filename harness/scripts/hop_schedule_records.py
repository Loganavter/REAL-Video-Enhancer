"""Records for the hop's schedule: three arms, the codes they produce, and the achieved TOPS.

Usage: python harness/scripts/hop_schedule_records.py [--probe docs/wsp/verified/logs/hop-schedule/probe.json]
                                              [--out docs/wsp/verified/records/int8-hop-loads.jsonl]
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
    ap.add_argument("--probe", default="docs/wsp/verified/logs/hop-schedule/probe.json")
    ap.add_argument("--out", default="docs/wsp/verified/records/int8-hop-loads.jsonl")
    args = ap.parse_args(argv)

    pr = json.load(open(args.probe))
    dig, sess = hashlib.sha256(open(args.probe, "rb").read()).hexdigest(), f"s-{pr['take']}"
    clip = pr["clip"]
    pop = Population.of("frames", (f"{clip}#frame{pr['frame']}",),
                        f"{clip} frame {pr['frame']}, ONE hop of the int8 body on its own codes")
    inst = (f"torch.cuda.Event around {pr['iters']} launches behind 3 warm-ups, {pr['rounds']} "
            f"interleaved rounds, medians of rounds 2+")
    cfg = "int8 body hop, 64 -> 64 3x3, 1920x1080, 16200 CTAs' worth of pixels"
    st = Store(session=sess)
    labels = {"today": "shipped kernel, BLOCK_M 128, 4 warps",
              "hints": "shipped kernel + max_contiguous/multiple_of on the channel lanes",
              "hoist2": "BLOCK_M 64 x two row-tiles per CTA: each tap's weights loaded once for both dots"}
    for rid, arm in (("L001", "today"), ("L002", "hints"), ("L003", "hoist2")):
        st.emit(Record(value=round(pr["median"][arm], 3), unit="ms",
                       population=Population(kind=pop.kind, items=pop.items,
                                             label=f"{pop.label}, arm {arm}: {labels[arm]}"),
                       instrument=inst, config=cfg, session=sess, statistic="median", tier="measured",
                       id=rid, run="scripts/hop_schedule_probe.py", artifact=args.probe,
                       extra={"artifact_sha256": dig, "arm": arm, "of": "hop_time",
                              "tops": round(pr["tops"][arm], 1), "gbps": round(pr["gbps"][arm], 1)}))
    st.delta("L002", "L001", note="vectorisation hints against the shipped kernel")
    st.delta("L003", "L001", note="weight hoist (two tiles per CTA) against the shipped kernel")
    st.dump(args.out)
    print(st.claims_text().strip())
    print(f"# -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
