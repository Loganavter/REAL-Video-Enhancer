"""Migrate the existing findings into the harness, and fail while they are unprotected.

The 473 lines of `harness/docs/claims-*.txt` are 213 `@claim`, 46 free-text `@derived`, 16 `@bar`, and a handful
of relations. Every one of them predates the tuple, so the honest question is not "are they consistent"
(the legacy auditor answers that) but **what is each one missing before it can be evidence again**.

This tool answers it mechanically:

* a claim is ``protected`` only when a record with the same value, unit and population item-set exists;
* otherwise it is ``unverifiable`` and the missing fields are named, not summarised;
* the report ranks by **decision-bearing**: a finding cited by a ``@ratio``, ``@below``, ``@verdict``,
  ``@cause`` or ``@sum`` is load-bearing, and the gate fails while any of those is unverifiable;
* ``--emit-records`` writes every legacy claim into a store as ``assumed``, so the new harness can see it
  and refuse to use it as evidence instead of silently not knowing it exists.

Nothing is "fixed" here. A number becomes protected by being re-derived from its artifact, which is work
with a cost, and this tool exists to make that list finite and visible.
"""
from __future__ import annotations

import argparse
import glob
import pathlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .core import Population, Record, Store

CLAIM = re.compile(r'@claim\s+(\S+)\s+([-+]?[\d.]+)\s+(\S+)\s+"([^"]*)"(?:\s+(measured|derived|assumed))?')
DERIVED = re.compile(r'@derived\s+(\S+)\s+"([^"]*)"')
BAR = re.compile(r'@bar\s+(\S+)\s+(-?[\d.]+)\s+(\S+)')
SESSION = re.compile(r'@session=(\S+)')
CITES = {
    "ratio": re.compile(r'@ratio\s+(\S+)\s+(\S+)'),
    "below": re.compile(r'@below\s+(\S+)\s+(\S+)'),
    "cause": re.compile(r'@cause\s+(\S+)\s+"[^"]*"\s+(\S+)\s+(\S+)'),
    "sum": re.compile(r'@sum\s+[\d.]+%\s+@of\s+(.*)'),
    "verdict": re.compile(r'@verdict\s+(\S+)\s+(.+)'),
}

REQUIRED = ("items_sha", "instrument", "config", "session", "statistic", "n", "artifact")
# What a population string can honestly give us: the label is all we have, and it is a label.
CLIP = re.compile(r'\b(a\d+_[a-z0-9_]+|adv_[a-z0-9_]+|c\d+_[a-z0-9_]+|anime|film|Violet|Chainsaw)\b', re.I)
N_HINT = re.compile(r'\b(\d+)\s+(?:gated pairs|pairs|frames|problems|titles|samples|launches)\b')
STAT = re.compile(r'\b(worst|median|mean|min|max|p99|sum)\b', re.I)


@dataclass
class Finding:
    name: str
    value: float
    unit: str
    pop_text: str
    tier: str
    file: str
    line: int
    status: str = "unverifiable"
    missing: list[str] = field(default_factory=list)
    n_hint: int | None = None
    stat_hint: str = ""
    clip_hint: str = ""
    session: str = ""
    cited_by: list[str] = field(default_factory=list)

    def key(self) -> str:
        return f"{self.file}:{self.line}"


def _parse(path: str) -> tuple[list[Finding], list[tuple[str, str, str, int, str]], dict[str, list[str]],
                               list[tuple[str, int, str]]]:
    findings: list[Finding] = []
    unparsed: list[tuple[str, int, str]] = []
    bars: list[tuple[str, str, str, int, str]] = []
    cited: dict[str, list[str]] = defaultdict(list)
    for lineno, raw in enumerate(open(path), 1):
        line = raw.split("#")[0].strip() if not raw.strip().startswith('"') else raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("@") and not any(rx.match(line) for rx in (CLAIM, BAR, DERIVED)
                                           + tuple(CITES.values())):
            unparsed.append((path, lineno, line[:100]))
            continue
        if line.startswith("@claim") and not CLAIM.match(line):
            unparsed.append((path, lineno, line[:100]))
            continue
        if m := CLAIM.match(line):
            name, value, unit, pop, tier = m.groups()
            f = Finding(name=name, value=float(value), unit=unit, pop_text=pop,
                        tier=tier or "measured", file=path, line=lineno)
            f.n_hint = int(m2.group(1)) if (m2 := N_HINT.search(pop)) else None
            f.stat_hint = (m2.group(1).lower() if (m2 := STAT.search(pop)) else "")
            f.clip_hint = (m2.group(1) if (m2 := CLIP.search(pop)) else "")
            f.session = (m2.group(1) if (m2 := SESSION.search(line)) else "")
            findings.append(f)
        elif m := BAR.match(line):
            bars.append((m.group(1), m.group(2), m.group(3), lineno, path))
        for kind, rx in CITES.items():
            m = rx.match(line)
            if not m:
                continue
            if kind in ("ratio", "below"):
                names = [m.group(1), m.group(2)]
            elif kind == "cause":
                names = [m.group(2), m.group(3)]
            else:
                names = re.split(r"[+\s]+", m.groups()[-1])
            for n in names:
                n = n.strip()
                if n and not n.startswith("@"):
                    cited[n].append(f"{kind}@{path.split('/')[-1]}:{lineno}")
        if m := DERIVED.match(line):
            cited[m.group(1)].append(f"derived-text@{path.split('/')[-1]}:{lineno}")
    return findings, bars, cited, unparsed


def _missing(f: Finding, have_record: bool) -> list[str]:
    if have_record:
        return []
    miss = list(REQUIRED)
    if f.n_hint is not None:
        miss.remove("n")
    return miss


def migrate(paths: list[str], store: Store | None = None) -> tuple[list[Finding], list[tuple], dict]:
    all_f: list[Finding] = []
    all_bars: list[tuple] = []
    cited: dict[str, list[str]] = defaultdict(list)
    audit: list[tuple[str, int, str]] = []
    unparsed: list[tuple[str, int, str]] = []
    for p in paths:
        f, b, c, u = _parse(p)
        unparsed += u
        all_f += f
        all_bars += b
        for k, v in c.items():
            cited[k] += v
        audit.append(_legacy_audit(p))

    protected = 0
    for f in all_f:                             # statuses first: conflicts() filters on them
        if pathlib.Path(f.file).name == "claims-example.txt":
            f.status = "withdrawn"
            f.missing = ["withdrawn by the source file"]
    name_reuse, pop_conf, _, cross_run = conflicts(all_f)
    # Only name reuse is a contradiction. Three values on one population LABEL (fp16 / int8 /
    # int8-fused) are three arms the label does not distinguish - counted, not failed.
    conflicted = {e["name"] for e in name_reuse}
    for f in all_f:
        f.cited_by = sorted(set(cited.get(f.name, [])))
        if f.status == "withdrawn":         # the file's own header: published, then withdrawn
            continue
        if f.name in conflicted and f.status != "protected":
            f.status = "conflicted"
            f.missing = ["a second value under the same name or population"]
        if store is not None:
            # Only a measured, non-legacy record protects a claim. Without this, feeding back the
            # `assumed` records this tool writes itself flips GATE: FAIL to GATE: PASS with zero
            # re-derivation - a green gate manufactured by the gate.
            rec = next((r for r in store.records.values()
                        if r.value == f.value and r.unit == f.unit
                        and r.population.label == f.pop_text
                        and r.tier == "measured" and not r.extra.get("legacy")), None)
            if rec is not None:
                f.status = "protected"
                protected += 1
        if f.status == "unverifiable":
            f.missing = _missing(f, False)
    return all_f, all_bars, {"cited": dict(cited), "protected": protected,
                             "name_reuse": name_reuse, "pop_conflicts": pop_conf,
                             "legacy_audit": audit, "unparsed": unparsed,
                             "cross_run": cross_run}


def _legacy_audit(path: str) -> tuple[str, int, str]:
    """Run the project's own auditor on the file and report its verdict - the only falsification that
    needs no artifact, because it is arithmetic and units."""
    import subprocess
    auditor = str(pathlib.Path(__file__).resolve().parents[1] / "claim_check.py")
    try:
        r = subprocess.run([sys.executable, auditor, path, "--quiet"], capture_output=True, text=True,
                           timeout=60)
    except Exception as exc:                                    # noqa: BLE001
        return path, -1, f"auditor did not run: {exc}"
    fails = [ln.strip() for ln in r.stdout.splitlines() if ln.strip().startswith(tuple("0123456789"))]
    first = fails[0][:90] if fails else ("PASS" if r.returncode == 0 else "unknown")
    return path, len(fails), first


STAT_SUFFIX = re.compile(r"_(peak|median|mean|worst|min|max|p99|sum)$")


def stat_of(f: Finding) -> str:
    """The statistic lives in the claim's *name* in this corpus (`pipeline_peak`, `sim_median`).
    That is the debt, not a detail: two numbers on one population are not a contradiction when the
    statistic differs, and the only place that information exists is a string nobody checks."""
    if m := STAT_SUFFIX.search(f.name):
        return m.group(1)
    if f.stat_hint:
        return f.stat_hint
    return "single"


def conflicts(findings: list[Finding]) -> tuple[list[dict], list[dict], list[dict]]:
    """Three mechanical verdicts that need no artifact, only what is in the files.

    ``name_reuse``   one name, two values: either a re-measurement across sessions (the numbers
                     cannot be distinguished) or a contradiction. Both are findings.
    ``pop_stat``     one population and one statistic, two values in the same unit.
    ``stat_hidden``  one population, two values, different statistics that exist only in the name -
                     recorded so the debt is counted rather than mistaken for a contradiction.
    """
    by_name: dict[str, list[Finding]] = defaultdict(list)
    for f in findings:
        if f.status == "withdrawn":
            continue
        by_name[(f.name, f.session)].append(f)

    name_reuse = [{"name": k[0], "session": k[1],
                   "values": sorted({(x.value, x.unit, x.file.split('/')[-1], x.line) for x in v})}
                  for k, v in by_name.items()
                  if len({(x.value, x.unit) for x in v}) > 1]
    # The same name in two runs is not a contradiction - it is a re-measurement, and the corpus
    # carried no session at all until 2026-11-16, so nothing said which one is current.
    cross_run = []
    byname_any: dict[str, list[Finding]] = defaultdict(list)
    for f in findings:
        if f.status != "withdrawn":
            byname_any[f.name].append(f)
    for name, group in byname_any.items():
        sessions = {x.session for x in group if x.session}
        if len(sessions) > 1 and len({(x.value, x.unit) for x in group}) > 1:
            cross_run.append({"name": name, "sessions": sorted(sessions),
                              "values": sorted({(x.value, x.unit, x.session) for x in group})})

    pop_stat, stat_hidden = [], []
    loose: dict[tuple, list[Finding]] = defaultdict(list)
    for f in findings:
        if f.status == "withdrawn":
            continue
        loose[(f.unit, f.pop_text.strip().lower())].append(f)
    for key, group in loose.items():
        if len({(x.value, x.unit) for x in group}) <= 1:
            continue
        stats = {stat_of(x) for x in group}
        entry = {"unit": key[0], "population": key[1],
                 "values": sorted({(x.value, x.name) for x in group})}
        (pop_stat if len(stats) == 1 else stat_hidden).append(entry)
    return name_reuse, pop_stat, stat_hidden, cross_run


def emit_records(findings: list[Finding], store: Store) -> Store:
    """Every legacy claim enters the store as ``assumed``, with its missing fields named. It can be
    read; it cannot be used as evidence, which is the point."""
    for f in findings:
        if f.status == "protected":
            continue
        stat = {"worst": "worst_of"}.get(f.stat_hint, f.stat_hint or "single")
        rec = Record(value=f.value, unit=f.unit,
                     population=Population(kind=f.clip_hint or "legacy claim", label=f.pop_text),
                     instrument="unrecorded", config="unrecorded", session="unrecorded",
                     statistic=stat, tier="assumed",
                     id=f.name, note=f"{f.file}:{f.line}; missing: {', '.join(f.missing)}",
                     extra={"legacy": True, "missing": f.missing, "cited_by": f.cited_by,
                            "n_hint": f.n_hint})
        if rec.id not in store.records:
            store.records[rec.id] = rec
    return store


def report(findings: list[Finding], bars: list[tuple], meta: dict) -> tuple[str, bool]:
    withdrawn = [f for f in findings if f.status == "withdrawn"]
    live = [f for f in findings if f.status != "withdrawn"]
    protected = [f for f in live if f.status == "protected"]
    conflicted = [f for f in live if f.status == "conflicted"]
    unver = [f for f in live if f.status == "unverifiable"]
    def _rel(f: Finding) -> list[str]:
        return [c for c in f.cited_by if not c.startswith("derived-text")]

    bearing = [f for f in live if _rel(f)]                        # cited by a decision relation
    deriv_only = [f for f in live if f.cited_by and not _rel(f)]  # only a free-text @derived
    bearing_bad = [f for f in bearing if f.status != "protected"]

    lines = [f"{len(findings)} claim lines ({len({f.name for f in findings})} names), "
             f"{len(bars)} bars, {len(meta['cited'])} names cited"]
    lines.append(f"  withdrawn by the source file: {len(withdrawn)}   (claims-example.txt)")
    lines.append(f"  protected:                    {len(protected)}")
    lines.append(f"  CONFLICTED (two values, one name within one run): {len(conflicted)}")
    lines.append(f"  unverifiable (no tuple):      {len(unver)}")
    lines.append(f"  decision-bearing (cited by a relation): {len(bearing)}, "
                 f"not protected: {len(bearing_bad)}")
    lines.append(f"  derivation-only (nothing cites it; it has a free-text @derived): {len(deriv_only)}")

    if meta.get("name_reuse"):
        lines.append("")
        lines.append("One name, two values - a re-measurement or a contradiction, and nothing in the file says which:")
        for e in sorted(meta["name_reuse"], key=lambda e: e["name"]):
            vals = "; ".join(f"{v:g} {u} ({f}:{l})" for v, u, f, l in e["values"])
            lines.append(f"  {e['name']:<28} {vals}")

    if meta.get("cross_run"):
        lines.append("")
        lines.append("Same name in two runs - a re-measurement, not a contradiction, and nothing in the "
                     "file says which run is current:")
        for e in sorted(meta["cross_run"], key=lambda e: e["name"]):
            vals = ", ".join(f"{v:g} {u} ({s})" for v, u, s in e["values"])
            lines.append(f"  {e['name']:<28} {vals}")

    if meta.get("pop_conflicts"):
        lines.append("")
        lines.append("Three or more values on one population LABEL, and the label does not say which arm "
                     "each belongs to (fp16 / int8 / int8-fused). Not a contradiction - a population "
                     "that is not a population:")
        for e in meta["pop_conflicts"]:
            vals = ", ".join(f"{v:g} ({n})" for v, n in e["values"])
            lines.append(f"  [{e['unit']}] {e['population'][:58]:<58} {vals}")

    if meta.get("legacy_audit"):
        lines.append("")
        lines.append("What the legacy auditor says about these files today:")
        for path, failures, first in meta["legacy_audit"]:
            lines.append(f"  {path.split('/')[-1]:<28} {failures:>3} failure(s)  {first}")

    if meta.get("unparsed"):
        lines.append("")
        lines.append("DIRECTIVES NO PARSER READS (invisible to this tool and to claim_check.py - the "
                     "legacy value pattern has no leading '+', so these two were silently skipped by "
                     "both while the file reported PASS):")
        for path, lineno, text in meta["unparsed"]:
            lines.append(f"  {path.split('/')[-1]}:{lineno}  {text}")
        lines.append("  -> counted as a gate failure, not as zero findings")

    counts = Counter(m for f in unver for m in f.missing)
    lines.append("")
    lines.append("  missing field -> findings:")
    for name, n in counts.most_common():
        lines.append(f"    {name:<12} {n}")

    if bearing_bad:
        lines.append("")
        lines.append("DECISION-BEARING AND UNPROTECTED (re-derive before using again):")
        for f in sorted(bearing_bad, key=lambda f: -len(f.cited_by)):
            lines.append(f"  {f.name:<28} {f.value:>9g} {f.unit:<6} n={f.n_hint or '?'} "
                         f"[{f.status}; {', '.join(f.cited_by)}]  {f.file.split('/')[-1]}:{f.line}")

    lines.append("")
    lines.append(f"{len(bars)} bars: none names what it is a bar of (bar.of is absent from the legacy "
                 f"format), so gates.below refuses every one of them by design")
    lines.append("NOT checked: whether a label is true; whether the item set is the one the decision "
                 "needs; whether the artifact still exists; free-text @derived arithmetic")
    gate_failed = bool(conflicted or bearing_bad or meta.get("unparsed"))
    return "\n".join(lines), gate_failed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m harness.migrate")
    ap.add_argument("paths", nargs="+", help="claims files (globs allowed)")
    ap.add_argument("--records", help="records.jsonl to protect against")
    ap.add_argument("--emit-records", help="write legacy claims into this records.jsonl as assumed")
    ap.add_argument("--tickets", help="write per-finding JSON tickets here")
    ap.add_argument("--strict", action="store_true", help="fail on ANY unverifiable claim")
    args = ap.parse_args(argv)

    paths: list[str] = []
    for p in args.paths:
        paths += sorted(glob.glob(p)) or [p]
    store = Store.load(args.records) if args.records else Store()
    findings, bars, meta = migrate(paths, store)
    if args.emit_records:
        emit_records(findings, store)
        store.dump(args.emit_records)
    if args.tickets:
        with open(args.tickets, "w") as fh:
            for f in findings:
                fh.write(json.dumps({**f.__dict__, "key": f.key()}) + "\n")
    text, gate_failed = report(findings, bars, meta)
    print(text)
    if args.strict:
        gate_failed = any(f.status != "protected" for f in findings)
    print("\nGATE:", "FAIL" if gate_failed else "PASS")
    return 1 if gate_failed else 0


if __name__ == "__main__":
    sys.exit(main())
