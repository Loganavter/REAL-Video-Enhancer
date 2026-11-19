"""Records for the persistent-CTA falsifier: three arms, the codes, and the achieved TOPS.

Usage: python scripts/hop_persistent_records.py [--probe docs/wsp/verified/logs/hop-persistent/probe.json]
                                                [--out docs/wsp/verified/records/int8-hop-persistent.jsonl]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness.core import Population, Record, Store  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="docs/wsp/verified/logs/hop-persistent/probe.json")
    ap.add_argument("--out", default="docs/wsp/verified/records/int8-hop-persistent.jsonl")
    args = ap.parse_args(argv)
    pr = json.load(open(args.probe))
    dig, sess = hashlib.sha256(open(args.probe, "rb").read()).hexdigest(), f"s-{pr['take']}"
    clip = pr["clip"]
    pop = Population.of("frames", (f"{clip}#frame{pr['frame']}",),
                        f"{clip} frame {pr['frame']}, ONE hop of the int8 body on its own codes")
    inst = (f"torch.cuda.Event around {pr['iters']} launches behind 3 warm-ups, {pr['rounds']} "
            f"interleaved rounds, medians of rounds 2+")
    cfg = f"int8 body hop, 64 -> 64 3x3, 1920x1080, BLOCK_M 128, {pr['n_tiles']} tiles"
    st = Store(session=sess)
    labels = {"today": f"shipped kernel, one CTA per tile ({pr['n_tiles']} CTAs)",
              f"persist{pr['sms']}": f"persistent grid, {pr['sms']} CTAs (one per SM), each loops over tiles",
              f"persist{2 * pr['sms']}": f"persistent grid, {2 * pr['sms']} CTAs"}
    for rid, arm in (("N001", "today"), ("N002", f"persist{pr['sms']}"), ("N003", f"persist{2 * pr['sms']}")):
        st.emit(Record(value=round(pr["median"][arm], 3), unit="ms",
                       population=Population(kind=pop.kind, items=pop.items,
                                             label=f"{pop.label}, arm {arm}: {labels[arm]}"),
                       instrument=inst, config=cfg, session=sess, statistic="median", tier="measured",
                       id=rid, run="scripts/hop_persistent_probe.py", artifact=args.probe,
                       extra={"artifact_sha256": dig, "arm": arm, "of": "hop_time",
                              "tops": round(pr["tops"][arm], 1), "gbps": round(pr["gbps"][arm], 1)}))
    st.delta("N002", "N001", note="persistent grid at SM count against the shipped grid")
    st.delta("N003", "N001", note="persistent grid at 2x SM count against the shipped grid")
    st.dump(args.out)
    print(st.claims_text().strip())
    print(f"# -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
