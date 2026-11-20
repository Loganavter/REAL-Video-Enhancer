"""Assert that a probe's fp16 control never exceeds the full sweep it is supposed to sample.

A maximum over a subset cannot exceed the maximum over the set. If a probe's control (fp16 against
fp16 on the pairs the probe picked) reads above the worst of the full fp16 sweep, then its pairs are
not a subset of the render's population - the probe re-decided which frames were reused, and its
composition number is about a different population than the one the bar judges.

    python3 harness/scripts/check_control_ceiling.py <control_record> <sweep_record> [--records DIR]

Exit 0 when the ceiling holds, 1 when it does not, 2 on a usage error.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def load(spec: str, records_dir: str):
    """`<file>.jsonl:<ID>` when the id is not unique across the corpus, `<ID>` otherwise."""
    if ":" in spec:
        path, record_id = spec.rsplit(":", 1)
    else:
        path, record_id = None, spec
    paths = [path] if path else sorted(glob.glob(os.path.join(records_dir, "*.jsonl")))
    for p in paths:
        for line in open(p):
            rec = json.loads(line)
            if rec.get("id") == record_id:
                return rec, p
    raise SystemExit(f"record {spec} not found")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("control")
    ap.add_argument("sweep")
    ap.add_argument("--records", default="harness/docs/wsp/verified/records")
    args = ap.parse_args()

    control, cpath = load(args.control, args.records)
    sweep, spath = load(args.sweep, args.records)
    for rec, path in ((control, cpath), (sweep, spath)):
        if rec["unit"] != "%":
            raise SystemExit(f"{path}:{rec['id']} is in {rec['unit']}, not % - not comparable")

    print(f"control {control['id']} = {control['value']:.5f}%  ({os.path.basename(cpath)})")
    print(f"sweep   {sweep['id']} = {sweep['value']:.5f}%  ({os.path.basename(spath)})")
    if control["value"] > sweep["value"]:
        print(f"VIOLATION: a max over the probe's pairs ({control['value']:.5f}%) exceeds the max over "
              f"the render's population ({sweep['value']:.5f}%). The probe's pairs are not a subset - "
              f"its composition number is about a different population than the bar judges.")
        return 1
    print("ok: the control is at or below the sweep")
    return 0


if __name__ == "__main__":
    sys.exit(main())
