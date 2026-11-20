"""Records for the native-width probe: share and body time at 48 and 32 channels.

The population is frames for the share and rounds for the time, as everywhere else; the config carries the width
and the tile, because the tile was tuned at 64 and a native width is only trusted after it has been re-tuned.

Usage: python harness/harness/scripts/int8_width_records.py [--probe ...] [--out ...]
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
    ap.add_argument("--probe", default="docs/wsp/verified/logs/int8-width/probe.json")
    ap.add_argument("--out", default="docs/wsp/verified/records/int8-width.jsonl")
    args = ap.parse_args(argv)

    pr = json.load(open(args.probe))
    dig = hashlib.sha256(open(args.probe, "rb").read()).hexdigest()
    store = Store(session=f"s-{pr['take']}")
    frames = pr.get("frames", [100, 169, 250])
    out: list[Record] = []

    for row in pr["rows"]:
        if not row.get("ok"):
            continue
        popf = Population.of("frames", [f"{row['file']}#frame{i}" for i in frames],
                             f"{row['file']}, three held-out frames, worst of the three")
        rec = Record(value=row["share"], unit="%", population=popf,
                     instrument="the body's gross-error gate: pixels more than 8 levels off in any channel, "
                                "against the model's own fp16 forward",
                     config=f"{row['file']}, native width {row['feat']}, {row['inner']} inner convs, "
                            f"x{row['scale']}, grid clamp K=2, BLOCK_M 128 / warps 4",
                     session=store.session, statistic="worst_of", tier="measured",
                     id=f"WX{len(out) + 1:03d}", run="harness/scripts/int8_width_probe.py", artifact=args.probe,
                     extra={"artifact_sha256": dig, "of": "gross_error_share", "feat": row["feat"]})
        store.emit(rec)
        out.append(rec)

        popr = Population.of("rounds", [f"{row['file']}#round-{i}" for i in range(1, len(row["draws"]) + 1)],
                             f"{row['file']}, interleaved rounds, medians of rounds 2+")
        rec = Record(value=row["body_ms"], unit="ms", population=popr,
                     instrument="torch.cuda.Event around the body, one synchronise per call",
                     config=f"{row['file']}, native width {row['feat']}, grid clamp K=2",
                     session=store.session, statistic="median", tier="measured",
                     id=f"WX{len(out) + 1:03d}", run="harness/scripts/int8_width_probe.py", artifact=args.probe,
                     extra={"artifact_sha256": dig, "of": "body_time", "feat": row["feat"]})
        store.emit(rec)
        out.append(rec)

    store.dump(args.out)
    for rec in out:
        print(rec.claims_line())
    print(f"# {len(out)} records -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
