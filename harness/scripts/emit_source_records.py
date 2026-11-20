"""Emit the source facts as records, so `sources-bd.txt` cites them instead of restating them.

Reads docs/wsp/sources.yaml, probes each clip that is reachable, and writes one record per clip for
the bit rate and the packet count, plus the stream-copy proof when the manifest carries it. Prints the
`@claim` lines the prose file quotes.

Usage: python harness/harness/scripts/emit_source_records.py docs/wsp/sources.yaml --clip-root /tmp/opencode \
           --out docs/wsp/verified/records/sources-bd.jsonl --artifact docs/wsp/verified/logs/sources-bd.log
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

import yaml  # noqa: E402

from harness.core import Population, Record  # noqa: E402
from harness.source import probe  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--clip-root", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--artifact", default="")
    ap.add_argument("--session", default="s-2026-11-18T08:00:00Z")
    args = ap.parse_args(argv)

    manifest = yaml.safe_load(open(args.manifest)) or {}
    digest = (hashlib.sha256(open(args.artifact, "rb").read()).hexdigest()
              if args.artifact and os.path.exists(args.artifact) else "")
    recs, seq = [], 0
    for clip in manifest.get("clips", []):
        name = clip["file"]
        path = os.path.join(args.clip_root, name) if args.clip_root else name
        if not os.path.exists(path):
            continue
        f = probe(path)
        label = clip.get("label", "")
        pop = Population.of("clips", [name], label)
        for value, unit, what in ((f.bit_rate_mbps, "Mbps", "container bit rate"),
                                  (float(f.packets), "count", "video packets")):
            seq += 1
            recs.append(Record(
                value=value, unit=unit, population=pop,
                instrument=f"ffprobe ({what})", config=label, session=args.session,
                statistic="single", tier="measured",
                run="python3 -m harness.cli source-check docs/wsp/sources.yaml --clip-root /tmp/opencode",
                artifact=args.artifact, extra={"artifact_sha256": digest}, id=f"S{seq:03d}"))
        cut = clip.get("cut") or {}
        if cut.get("mode") == "stream_copy" and cut.get("compared"):
            seq += 1
            recs.append(Record(
                value=float(cut["compared"]), unit="count", population=pop,
                instrument="video packet-size subsequence match against the source",
                config=f"-c copy proof: source packet offset {cut.get('offset')} of "
                       f"{cut.get('packets')} clip packets", session=args.session,
                statistic="single", tier="measured",
                run="python -c 'stream_copy_proof(clip, source)'", artifact=args.artifact,
                extra={"artifact_sha256": digest}, id=f"S{seq:03d}"))

    with open(args.out, "w") as fh:
        for r in recs:
            r.validate()
            fh.write(json.dumps(r.to_dict()) + "\n")
    for r in recs:
        print(r.claims_line())
    print(f"# {len(recs)} records -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
