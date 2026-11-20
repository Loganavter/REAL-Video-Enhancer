"""Records for the grid-clamp ladder: one share per (model, K), plus the table the page quotes.

The ladder is the evidence for the lever this row turned into - `s[c] = max(s_in[c], s_t/K)` - and the reason it
is a ladder and not a choice between the two shipped modes. Every K costs the same arithmetic (the fold cancels),
so the population is frames and the only quantity is the gross-error share.

Usage: python harness/harness/scripts/int8_grid_k_records.py [--ladder ...] [--out ...] [--table]
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

FRAMES = (100, 169, 250)
ORDER = ("per_tensor", "K=1", "K=2", "K=3", "K=4", "per_channel")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", default="docs/wsp/verified/logs/grid-k/ladder.jsonl")
    ap.add_argument("--out", default="docs/wsp/verified/records/int8-grid-k-ladder.jsonl")
    ap.add_argument("--table", action="store_true", help="print the markdown table the page quotes")
    args = ap.parse_args(argv)

    rows = [json.loads(line) for line in open(args.ladder) if line.strip()]
    dig = hashlib.sha256(open(args.ladder, "rb").read()).hexdigest()
    store = Store(session="s-grid-k-2026-11-20")
    out: list[Record] = []
    for row in rows:
        pop = Population.of("frames", [f"{row['file']}#frame{i}" for i in FRAMES],
                            f"{row['file']}, three held-out frames, worst of the three")
        for mode, value in row["shares"].items():
            rec = Record(value=value, unit="%", population=pop,
                         instrument="the body's gross-error gate: pixels more than 8 levels off in any channel, "
                                    "against the model's own fp16 forward",
                         config=f"{row['file']}, grid {mode} (s[c] = max(s_in[c], s_t/K))",
                         session=store.session, statistic="worst_of", tier="measured",
                         id=f"GK{len(out) + 1:03d}", run="harness/scripts/int8_grid_k_probe.py",
                         artifact=args.ladder,
                         extra={"artifact_sha256": dig, "of": "gross_error_share", "grid": mode,
                                                      "scale": row.get("scale")})
            store.emit(rec)
            out.append(rec)
    store.dump(args.out)
    for rec in out:
        print(rec.claims_line())

    if args.table:
        print("\n| model | " + " | ".join(ORDER) + " | best |")
        print("|---" * (len(ORDER) + 2) + "|")
        better = 0
        for row in rows:
            s = row["shares"]
            cand = {k: v for k, v in s.items() if k.startswith("K=") and k != "K=1"}
            best = min(cand, key=cand.get) if cand else "-"
            base = min(s.get("per_tensor", 9e9), s.get("per_channel", 9e9))
            if cand and cand[best] < base:
                better += 1
            cells = [f"{s.get(k, float('nan')):.4f}" for k in ORDER]
            cells[ORDER.index(best)] = f"**{s[best]:.4f}**" if best in ORDER else cells[ORDER.index(best)]
            print(f"| `{row['file']}` | " + " | ".join(cells) + f" | {best} |")
        print(f"\nAn intermediate K beats both shipped modes on {better} of {len(rows)} models.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
