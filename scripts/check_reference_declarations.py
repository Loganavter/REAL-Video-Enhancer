"""Every render-pair record must say which side is the reference. List the ones that do not.

Why this is a checker and not a paragraph: the mistake it prevents is invisible from the number. A
comparison against an FP16 ENGINE RENDER reads like a measurement of the quantiser, and it is not - the
engine deviates from the model by 0.055-0.087% and the colour pipe alone moves 0.21% of pixels past the
8-level bar, while the quantiser's own error against the MODEL is 0.004-0.019%. Nothing in the resulting
figure says which one you are looking at, so the declaration has to be part of the record.

Exit 1 lists the undeclared records (a debt, like `harness.migrate` being red); exit 0 when every
render-pair record carries `extra.reference_kind`.

Usage:
  python scripts/check_reference_declarations.py                 # human list
  python scripts/check_reference_declarations.py --quiet         # exit code only
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

# gate_sweep is the only emitter that writes this suffix into `instrument`; it is the marker that a record
# came from a decoded render pair rather than from a kernel, a log line or a plan.
MARKER = "px blocks"
VALID = ("model", "engine-render", "config")


def undeclared(paths: list[str]) -> list[tuple[str, str, str]]:
    """[(file, id, instrument)] for render-pair records with no (or an invalid) reference_kind."""
    out = []
    for path in paths:
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if MARKER not in str(rec.get("instrument", "")):
                continue
            kind = (rec.get("extra") or {}).get("reference_kind")
            if kind not in VALID:
                out.append((os.path.basename(path), rec.get("id", "?"),
                            str(rec.get("instrument", ""))[:70]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="docs/wsp/verified/records/*.jsonl")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    paths = sorted(glob.glob(args.records))
    bad = undeclared(paths)
    total = sum(1 for p in paths for line in open(p) if MARKER in line)
    if not args.quiet:
        print(f"render-pair records: {total} in {len(paths)} files; undeclared: {len(bad)}")
        for name, rid, instr in bad:
            print(f"  {name} {rid}: {instr}")
        if bad:
            print("  -> these cannot say what they measure. Re-emit with "
                  "`gate_sweep.py --reference-kind {model|engine-render|config}`.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
