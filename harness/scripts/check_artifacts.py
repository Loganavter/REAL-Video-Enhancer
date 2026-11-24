#!/usr/bin/env python3
"""Every evidence record's `artifact` must be a file in THIS repository, and its digest must still match.

Why a gate (2026-11-24): the records used to cite `~/rve-scratch/...` - a 21 GB directory on one machine. The owner
rejected that arrangement, and he was right: a record that points outside the repository stops being evidence the
moment the directory is cleaned, and nobody can be asked to hand over the renders to check a number. The records now
carry repo-relative paths under `harness/docs/wsp/verified/artifacts/`, and this checker makes the next outside path, the
next missing file and the next changed-file-since-the-record loud instead of silent.

A digest that no longer matches is NOT repaired by recomputing it: the digest is what was read, and a file that changed
afterwards is a fact about the file, not licence to redate the record. The base of known-stale entries lives in
`artifacts-baseline.json`; a new one fails, an entry that no longer reproduces must be deleted from the baseline (the
ratchet shrinks, like the other gates in this tree).

Usage:
    python3 harness/scripts/check_artifacts.py               # check, exit 1 on a new violation
    python3 harness/scripts/check_artifacts.py --write-baseline
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
VERIFIED = ROOT / "harness" / "docs" / "wsp" / "verified"
RECORDS = VERIFIED / "records"
BASELINE = VERIFIED / "artifacts-baseline.json"

RULES = ("artifact-not-a-path", "artifact-outside-repo", "artifact-missing",
         "artifact-no-digest", "artifact-stale")


def _digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def findings() -> list[dict]:
    """One entry per violated rule; `[]` means every artifact is a file in this tree whose digest still matches."""
    out: list[dict] = []
    for record_file in sorted(RECORDS.glob("*.jsonl")):
        for line_no, line in enumerate(record_file.read_text(encoding="utf-8").splitlines(), 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            artifact = record.get("artifact") or ""
            if not artifact:
                continue
            entry = {"record": record_file.stem, "line": line_no, "id": record.get("id") or "",
                     "artifact": artifact}
            path = pathlib.Path(artifact)
            if len(artifact) == 64 and "/" not in artifact:
                out.append({**entry, "rule": "artifact-not-a-path"})
                continue
            if path.is_absolute() or ".." in path.parts:
                out.append({**entry, "rule": "artifact-outside-repo"})
                continue
            resolved = ROOT / path
            if not resolved.is_file():
                out.append({**entry, "rule": "artifact-missing"})
                continue
            recorded = (record.get("extra") or {}).get("artifact_sha256")
            if not recorded:
                out.append({**entry, "rule": "artifact-no-digest"})
                continue
            if _digest(resolved) != recorded:
                out.append({**entry, "rule": "artifact-stale", "recorded": recorded,
                            "actual": _digest(resolved)})
    return out


def _key(entry: dict) -> str:
    """A stable identity: the record's id (or its lane+line when it has none) plus the file it cites."""
    who = entry.get("id") or f"{entry['record']}:{entry['line']}"
    return f"{who}|{entry['artifact']}"


def load_baseline() -> dict[str, dict]:
    if not BASELINE.exists():
        return {}
    data = json.loads(BASELINE.read_text(encoding="utf-8"))
    return {item["key"]: item for item in data.get("known", [])}


def write_baseline(entries: list[dict]) -> None:
    payload = {"why": "Known-stale artifacts: the file changed after the record was written. The digest is what was "
                      "read; a new mismatch fails the gate.",
               "known": sorted(({"key": _key(e), "rule": e["rule"], "artifact": e["artifact"]} for e in entries),
                               key=lambda item: item["key"])}
    BASELINE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-baseline", action="store_true", help="record today's findings as known")
    args = parser.parse_args(argv)
    if not RECORDS.is_dir():
        print(f"check_artifacts: no records at {RECORDS} - refusing to pass an empty corpus", file=sys.stderr)
        return 2
    current = findings()
    if args.write_baseline:
        write_baseline(current)
        print(f"check_artifacts: baseline written with {len(current)} known artifact(s)")
        return 0
    known = load_baseline()
    new = [entry for entry in current if _key(entry) not in known]
    fixed = sorted(set(known) - {_key(entry) for entry in current})
    for entry in new:
        print(f"{entry['rule']}: {_key(entry)}" + (f" (recorded {entry.get('recorded', '')[:12]}, file now "
                                                   f"{entry.get('actual', '')[:12]})" if entry.get("recorded") else ""))
    if fixed:
        print(f"{len(fixed)} baseline entr(y/ies) no longer reproduce - delete them so the ratchet shrinks:")
        for key in fixed[:10]:
            print(f"  {key}")
    if new:
        print(f"check_artifacts: {len(new)} new artifact violation(s)", file=sys.stderr)
        return 1
    print(f"check_artifacts OK ({len(current)} known, {len(known)} in the baseline)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
