"""Mark every verified file that cites a contaminated input, once, mechanically.

`docs/wsp/sources.yaml` says which inputs are contaminated; a record's population items name them. This
walks the verified corpus, and for each file with at least one such record inserts a header right after
the H1. Idempotent (the marker line is the guard), and it never deletes or rewrites a number: the point
is that a reader meets the contamination before the table, not after the conclusion.

Usage:
  python harness/scripts/mark_contaminated.py docs/wsp/sources.yaml --records 'docs/wsp/verified/records/*.jsonl' \
      --dir docs/wsp/verified            # insert
  python harness/scripts/mark_contaminated.py ... --check   # report only, exit 1 if any file is unmarked
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import yaml

MARK = "<!-- population: contaminated input -->"
BLOCK = """\
{MARK}
> **Population flagged 2026-11-18.** {n} record(s) in this file ({ids}) name
> `{named}` - a {label}. Every content number below was measured on that material, so none of them is
> current. The BD replacement and the check that finds this mechanically are in
> `docs/wsp/verified/sources-bd.txt`; the verdict that replaces the conclusion is named there.
> Instrument-only rows (cost, timing, counters) are unaffected as measurements but their population
> label is still the wrong file.
"""


def contaminated(manifest: dict, records_glob: str) -> dict[str, tuple[list[str], set[str]]]:
    bad = {c["file"]: c for c in manifest.get("clips", []) if c.get("contaminated")}
    if not bad:
        return {}
    pat = re.compile(r"(?<![\w.-])(" + "|".join(re.escape(b) for b in bad) + r")(?![\w.-])")
    out: dict[str, tuple[list[str], set[str]]] = {}
    for p in sorted(glob.glob(records_glob)):
        ids, names = [], set()
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            items = (r.get("population") or {}).get("items") or []
            hit = {m.group(1) for i in items for m in [pat.search(i)] if m}
            if hit:
                ids.append(r.get("id", "?"))
                names |= hit
        if ids:
            out[os.path.basename(p).replace(".jsonl", ".txt")] = (ids, names)
    return out


def _labels(manifest: dict, names: set[str]) -> str:
    """One human phrase for the contaminated files, deduped: both re-encodes carry the same label."""
    seen, labels = set(), []
    for n in sorted(names):
        lab = manifest_lookup(manifest, n) or "re-encode"
        if lab not in seen:
            seen.add(lab)
            labels.append(lab)
    return " and ".join(labels)


def _strip_block(text: str) -> str:
    if MARK not in text:
        return text
    lines = text.split("\n")
    start = next(i for i, ln in enumerate(lines) if ln.strip() == MARK)
    end = start + 1
    while end < len(lines) and (lines[end].startswith(">") or lines[end] == ""):
        end += 1
    # keep one blank separator
    while start > 0 and lines[start - 1] == "":
        start -= 1
    return "\n".join(lines[:start] + lines[end:])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--records", required=True)
    ap.add_argument("--dir", default="docs/wsp/verified")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--refresh", action="store_true", help="rewrite an existing block")
    ap.add_argument("--skip", default="sources-bd.txt",
                    help="files that DOCUMENT the contamination, not measure it")
    args = ap.parse_args(argv)

    manifest = yaml.safe_load(open(args.manifest)) or {}
    hits = contaminated(manifest, args.records)
    skip = {s for s in args.skip.split(",") if s}
    todo = {k: v for k, v in hits.items() if k not in skip}

    unmarked = []
    for name, (ids, names) in sorted(todo.items()):
        path = os.path.join(args.dir, name)
        if not os.path.exists(path):
            print(f"MISSING: {name} has contaminated records but no verified file")
            continue
        text = open(path).read()
        if MARK in text and not args.refresh:
            continue
        unmarked.append(name)
        if args.check:
            continue
        text = _strip_block(text)
        label = _labels(manifest, names)
        block = BLOCK.format(MARK=MARK, n=len(ids), ids=", ".join(ids),
                             named=", ".join(sorted(names)), label=label)
        lines = text.split("\n")
        at = 1 if lines and lines[0].startswith("# ") else 0
        lines[at:at] = [""] + block.rstrip("\n").split("\n") + [""]
        open(path, "w").write("\n".join(lines))

    if args.check:
        for n in unmarked:
            print(f"UNMARKED: {n}")
        return 1 if unmarked else 0
    print(f"marked {len(unmarked)} file(s); {len(todo) - len(unmarked)} already carried the marker")
    return 0


def manifest_lookup(manifest: dict, name: str) -> str:
    for c in manifest.get("clips", []):
        if c.get("file") == name:
            return c.get("label", "")
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
