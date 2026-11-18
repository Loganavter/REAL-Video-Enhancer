"""Turn the input-path probe's draws into records, and the arm differences into derived records.

Four measured records (one per arm) plus three deltas. The arm is the variable: it lives in the
population label and in `run`, while config/instrument/session/statistic are held equal - the same
convention `wall_ab_records.py` uses, and what `Store._relation` demands of a delta.

Usage: python scripts/input_path_records.py --probe docs/wsp/verified/logs/input-path/probe.json \
           --out docs/wsp/verified/records/input-path.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness.core import Population, Record, Store  # noqa: E402

ARM_LABEL = {
    "conv": "shipped yuv420p10_to_tensor, trailing sync_stream included",
    "conv+clone": "shipped get_frame_tensor: conversion (sync included) plus .clone()",
    "conv-nosync": "conversion with sync_stream patched to a no-op",
    "nosync+clone": "conversion without sync, plus .clone() - both removals",
    "clone-only": ".clone() of a frame already converted (the clone alone)",
}


def sha256(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="docs/wsp/verified/logs/input-path/probe.json")
    ap.add_argument("--out", default="docs/wsp/verified/records/input-path.jsonl")
    args = ap.parse_args(argv)

    probe = json.load(open(args.probe))
    digest = sha256(args.probe)
    session = f"s-{probe['take']}"
    w, h = probe["width"], probe["height"]
    pop_items = (f"{w}x{h}#yuv420p10le-{probe['payload_mb']:.2f}MB",)
    instrument = (f"torch.cuda.Event around {probe['iters']} calls behind 3 warm-ups, wall clock "
                  f"beside it, round-robin, one process, one GPU job")
    config = (f"input path {w}x{h} yuv420p10le -> (1,3,{h},{w}) fp16, "
              f"frame {probe['frame_mb']:.2f} MB")

    st = Store(session=session)
    arm_ids: dict[str, str] = {}
    for i, (arm, label) in enumerate(ARM_LABEL.items(), start=1):
        rid = f"I{i:03d}"
        st.emit(Record(
            value=round(probe["median"][arm]["event"], 3), unit="ms",
            population=Population.of("frames", pop_items, f"{w}x{h} yuv420p10le payload, arm {arm}"),
            instrument=instrument, config=config, session=session, statistic="median",
            tier="measured", id=rid, run="scripts/input_path_probe.py", artifact=args.probe,
            extra={"artifact_sha256": digest, "arm": arm, "arm_meaning": label,
                   "rounds": probe["rounds"], "iters": probe["iters"],
                   "statistic_note": "median of rounds 2+, event time",
                   "wall_median": round(probe["median"][arm]["wall"], 3),
                   "min_draw": round(probe["median"][arm]["event_min"], 3)}))
        arm_ids[arm] = rid

    st.delta(arm_ids["conv"], arm_ids["conv-nosync"], note="cost of the trailing sync_stream")
    st.delta(arm_ids["conv+clone"], arm_ids["conv"], note="cost of the clone on the shipped arm")
    st.delta(arm_ids["nosync+clone"], arm_ids["conv"],
             note="both removals on the shipped arm - the row's decision number")

    st.dump(args.out)
    print(st.claims_text().strip())
    print(f"# -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
