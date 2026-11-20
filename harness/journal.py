"""A dated journal of measurements and decisions, so a number can be placed in time.

The records answer "what was measured, on what population, with what instrument". They did not answer "when,
on which commit, against which build" - and that is how a wall ratio of 1.09-1.14x stayed quotable for a day
after the body it described had doubled in speed. The journal is the chronology: one line per event, stamped
with the commit, the card and the library versions, pointing at the record ids it concerns and at the ids it
supersedes.

    python3 -m harness.journal add "int8 wall re-measured: 1.65x end to end" --refs RU053 --supersedes W009
    python3 -m harness.journal render          # docs/wsp/JOURNAL.md from docs/wsp/journal.jsonl
    python3 -m harness.journal list            # the last ten entries

The file is append-only and the rendered page is a view, like the ledger.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

from .provenance import stamp

JOURNAL = Path("docs/wsp/journal.jsonl")
PAGE = Path("docs/wsp/JOURNAL.md")


def _read(path: Path = JOURNAL) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def add(text: str, refs: tuple[str, ...] = (), supersedes: tuple[str, ...] = (),
        root: str = ".", model: str | None = None, path: Path = JOURNAL) -> dict:
    entry = {"text": text, "refs": list(refs), "supersedes": list(supersedes),
             **stamp(root, model)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(entry) + "\n")
    return entry


def render(path: Path = JOURNAL, page: Path = PAGE) -> str:
    entries = _read(path)
    lines = ["# Journal", "",
             "Every measurement and every decision that changes a quoted number, newest first. Each entry carries",
             "the commit, the card and the library versions it was taken on: a number without those describes a",
             "build that may no longer exist (`harness/provenance.py`).", "",
             f"Entries: {len(entries)}", ""]
    for entry in reversed(entries):
        head = entry.get("measured_at", "?")
        commit = (entry.get("commit") or "unknown")[:12]
        dirty = " (dirty tree)" if entry.get("dirty") else ""
        card = entry.get("device") or "unknown card"
        cap = entry.get("capability") or "?"
        lines.append(f"## {head} - {commit}{dirty} - {card} (cc {cap})")
        versions = ", ".join(f"{k} {entry.get(k)}" for k in ("torch", "triton", "tensorrt") if entry.get(k))
        if versions:
            lines.append(f"*{versions}*")
        lines.append("")
        lines.extend(textwrap.wrap(entry.get("text", ""), 110) or [""])
        if entry.get("refs"):
            lines.append("")
            lines.append("Records: " + ", ".join(entry["refs"]))
        if entry.get("supersedes"):
            lines.append("")
            lines.append("Supersedes: " + ", ".join(entry["supersedes"]))
        lines.append("")
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text("\n".join(lines))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m harness.journal")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ap = sub.add_parser("add", help="append a dated entry, stamped with commit/device/versions")
    ap.add_argument("text")
    ap.add_argument("--refs", default="")
    ap.add_argument("--supersedes", default="")
    ap.add_argument("--model", default="")
    sub.add_parser("render", help="write docs/wsp/JOURNAL.md from the jsonl")
    sub.add_parser("list", help="print the last ten entries")
    args = parser.parse_args(argv)

    if args.cmd == "add":
        refs = tuple(x.strip() for x in args.refs.split(",") if x.strip())
        sup = tuple(x.strip() for x in args.supersedes.split(",") if x.strip())
        entry = add(args.text, refs, sup, model=args.model or None)
        print(f"journal: {entry['measured_at']} {entry.get('commit', '')[:12]} -> {JOURNAL}")
        render()
        return 0
    if args.cmd == "render":
        render()
        print(f"journal: {PAGE} written")
        return 0
    for entry in _read()[-10:]:
        print(f"{entry.get('measured_at','?')}  {(entry.get('commit') or '')[:10]:10s}  {entry.get('text','')[:100]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
