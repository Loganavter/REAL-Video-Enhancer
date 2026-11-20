"""CLI: verify a claims file against the records it cites, and print what was not verified.

    python -m harness.cli check docs/claims-x.txt --records /tmp/records.jsonl [--bars bars.json]

Difference from ``harness/scripts/claim_check.py``: that auditor compares *typed labels* in a text file, so
a derived number is free text and its arithmetic is read only for ``@ratio``. Here every ``@claim``
must match a record field for field (value, unit, population item-set hash), every ``@ratio`` is
rebuilt through ``Store.ratio`` (instrument, config, session, statistic, item set), every derived
*record* is recomputed from its inputs, and the run ends with the list of things it did not check.

The legacy file format is unchanged on purpose: ``Store.claims_text()`` emits ``@claim`` lines whose
population string is ``kind [items_sha] n=N``, so the old auditor also stops treating two equal
labels over different frames as the same population.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

from .core import RecordError, Store, unit_family
from .gates import Bar, Claim, Scope, above, below, check_claims

CLAIM = re.compile(r'@claim\s+(\S+)\s+([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s+(\S+)\s+"([^"]*)"(?:\s+(measured|derived|assumed))?')
RATIO = re.compile(r'@ratio\s+(\S+)\s+(\S+)\s+([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)')
BELOW = re.compile(r'@below\s+(\S+)\s+(\S+)')
ATLEAST = re.compile(r'@atleast\s+(\S+)\s+(\S+)')
CAUSE = re.compile(r'@cause\s+(\S+)\s+"([^"]*)"\s+(\S+)\s+(\S+)')
VERDICT = re.compile(r'@verdict\s+(\S+)\s+(.+)')
SUM = re.compile(r'@sum\s+([\d.]+)%\s+@of\s+(.*)')
DERIVED_TEXT = re.compile(r'@derived\s+(\S+)\s+"([^"]*)"')
BARLINE = re.compile(r'@bar\s+(\S+)\s+([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s+(\S+)')

NOT_CHECKED = """\
NOT checked here (a green file is a floor, not a ceiling):
  - whether a population LABEL is true (the item-set hash is checked, the human label is not)
  - whether the item set is the one the DECISION needs
  - whether a cost model is COMPLETE
  - whether a cited pair DISCRIMINATES the mechanism (@cause checks that exactly one field
    differs, not that the mechanism is the only candidate)
  - free-text @derived lines: their arithmetic is not computable from text; only derived RECORDS
    are recomputed (see --records)"""


def _bars_from_file(path: str | None) -> dict[str, Bar]:
    if not path:
        return {}
    raw = json.load(open(path))
    return {k: Bar(name=k, **v) for k, v in raw.items()}


def check(path: str, records_path: str | None, bars_path: str | None = None,
          quiet: bool = False) -> int:
    store = Store.load(records_path) if records_path else Store()
    bars = _bars_from_file(bars_path)
    sc = Scope(expected=set())
    problems: list[str] = []
    textual: list[tuple[int, str, str]] = []       # (lineno, kind, raw) for the report
    structural: list[Claim] = []
    n_lines = 0

    for lineno, line in enumerate(open(path), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        n_lines += 1

        if m := BARLINE.match(line):
            sc.expected.add("bar")
            name, value, unit = m.group(1), float(m.group(2)), m.group(3)
            bars.setdefault(name, Bar(name=name, value=value, unit=unit, of=""))
            sc.miss("bar", f"line {lineno}: a text @bar carries no 'of' field; bars are only "
                           f"authoritative from --bars")

        elif m := CLAIM.match(line):
            sc.expected.add("claim")
            rid, value, unit, pop, tag = m.groups()
            value = float(value)
            if not records_path:
                sc.miss("claim", f"line {lineno}: no --records, so a claim cannot be matched to a record")
                continue
            sc.cover("claim")
            try:
                rec = store.get(rid)
                if abs(rec.value - value) > 1e-9:
                    problems.append(f"{lineno}: {rid} recorded {rec.value:g}, claimed {value:g}")
                if rec.unit != unit:
                    problems.append(f"{lineno}: {rid} is {rec.unit}, claimed {unit}")
                if rec.population.claims_text() != pop:
                    problems.append(f"{lineno}: {rid} population is '{rec.population.claims_text()}', "
                                    f"claimed '{pop}'")
                if (tag or "measured") != rec.tier:
                    problems.append(f"{lineno}: {rid} is tier {rec.tier}, claimed {tag or 'measured'}")
                sc.cover("claim")
            except RecordError as exc:
                problems.append(f"{lineno}: {exc}")

        elif m := DERIVED_TEXT.match(line):
            textual.append((lineno, "derived", line))

        elif m := RATIO.match(line):
            sc.expected.add("ratio")
            num, den, claimed = m.groups()
            if not records_path:
                sc.miss("ratio", f"line {lineno}: no --records")
                continue
            sc.cover("ratio")
            try:
                store.ratio(num, den, float(claimed))
            except RecordError as exc:
                problems.append(f"{lineno}: {exc}")

        elif m := BELOW.match(line):
            sc.expected.add("below")
            rid, bar = m.groups()
            if not records_path:
                sc.miss("below", f"line {lineno}: no --records")
                continue
            if bar not in bars:
                problems.append(f"{lineno}: @below cites unknown bar '{bar}' - a bar is a record too")
                continue
            sc.cover("below")
            try:
                below(store.get(rid), bars[bar])
            except RecordError as exc:
                problems.append(f"{lineno}: {exc}")

        elif m := ATLEAST.match(line):
            sc.expected.add("atleast")
            rid, bar = m.groups()
            if not records_path:
                sc.miss("atleast", f"line {lineno}: no --records")
                continue
            sc.cover("atleast")
            try:
                above(store.get(rid), bars[bar])
            except RecordError as exc:
                problems.append(f"{lineno}: {exc}")

        elif m := SUM.match(line):
            sc.expected.add("sum")
            total, names = float(m.group(1)), [n.strip() for n in m.group(2).split("+")]
            if not records_path:
                sc.miss("sum", f"line {lineno}: no --records")
                continue
            sc.cover("sum")
            try:
                vals = [store.get(n) for n in names]
                for r in vals:
                    if r.tier != "measured":
                        raise RecordError(f"{r.id} is {r.tier} - an unmeasured share cannot enter a partition")
                    if unit_family(r.unit) != "share":
                        raise RecordError(f"{r.id} is {r.unit}, not a share - cannot enter a partition")
                keys = {(r.instrument, r.config, r.session, r.statistic) for r in vals}
                if len(keys) > 1:
                    raise RecordError(f"summands disagree on (instrument, config, session, statistic): "
                                      f"{sorted(keys)} - they are not parts of one partition")
                s = sum(r.value for r in vals)
                if abs(s - total) > 0.02:
                    problems.append(f"{lineno}: shares sum to {s:.3f}%, claimed {total:g}%")
                sc.cover("sum")
            except RecordError as exc:
                problems.append(f"{lineno}: {exc}")

        elif m := CAUSE.match(line):
            sc.expected.add("cause")
            cid, mech, a, b = m.groups()
            structural.append(Claim(id=cid, kind="cause", args=(a, b), mechanism=mech))

        elif m := VERDICT.match(line):
            sc.expected.add("verdict")
            structural.append(Claim(id=m.group(1), kind="verdict", args=tuple(m.group(2).split())))

        elif line.startswith("@"):
            problems.append(f"{lineno}: directive not read by this checker: {line[:60]}")
        else:
            sc.miss("prose", f"line {lineno}: not a directive: {line[:40]}")

    if structural:
        probs, sc2 = check_claims(store, structural, scope=sc, expected=set())
        problems += probs
        sc = sc2

    problems += [f"derived: {p}" for p in store.verify_derived()]
    if store.records:
        sc.cover("derived_record_arithmetic")

    for lineno, kind, _ in textual:
        sc.miss(kind, f"line {lineno}: free-text derivation has no computable arithmetic")

    try:
        sc.finalize()
    except RecordError as exc:
        problems.append(str(exc))

    if not quiet:
        print(f"{path}: {n_lines} directives, {len(store.records)} records")
    if problems:
        print(f"FAIL ({len(problems)}):")
        for p in problems:
            print("  " + p)
    else:
        print("PASS")
    print(sc.report())
    print(NOT_CHECKED)
    return 1 if problems else 0


def below_report(paths: list[str], bars_path: str, quiet: bool = False) -> int:
    """Every `@below` assertion in the corpus, split into arithmetic and eligibility.

    Arithmetic: is the value actually under the bar, same unit, same statistic? Eligibility: does the
    claim carry enough to be judged at all (n, statistic, instrument)? The corpus has 29 of these and
    no records behind them, so the honest output is two columns, not one verdict.
    """
    import glob as _glob
    bars = _bars_from_file(bars_path)
    rows: list[tuple[str, str, str, str]] = []
    counts = {"under": 0, "over": 0, "literal": 0, "unknown": 0, "ineligible": 0}
    for pat in paths:
        for path in sorted(_glob.glob(pat)) or [pat]:
            claims = {}
            for line in open(path):
                s_ = line.split(" # ")[0].strip()
                if m := CLAIM.match(s_):
                    claims[m.group(1)] = (float(m.group(2)), m.group(3))
            for line in open(path):
                s_ = line.split(" # ")[0].strip()
                if not (m := BELOW.match(s_)):
                    continue
                name, bar_name = m.group(1), m.group(2)
                if name not in claims:
                    rows.append((name, "?", bar_name, "NO SUCH CLAIM"))
                    counts["unknown"] += 1
                    continue
                value, unit = claims[name]
                if bar_name in bars:
                    bar = bars[bar_name]
                else:
                    try:
                        bar = Bar(name=bar_name, value=float(bar_name), unit=unit, of="share",
                                  statistic="worst_of")
                        literal = True
                    except ValueError:
                        rows.append((name, f"{value:g} {unit}", bar_name, "UNKNOWN BAR"))
                        counts["unknown"] += 1
                        continue
                if bar_name not in bars:
                    rows.append((name, f"{value:g} {unit}", bar_name, "LITERAL BAR (not a record)"))
                    counts["literal"] += 1
                elif unit != bar.unit:
                    rows.append((name, f"{value:g} {unit}", bar_name, f"UNIT {unit} vs bar {bar.unit}"))
                    counts["unknown"] += 1
                elif bar.min_n > 1:
                    rows.append((name, f"{value:g} {unit}", bar_name,
                                 ("under" if value < bar.value else "OVER") + " but n is not recorded"))
                    counts["under" if value < bar.value else "over"] += 1
                    counts["ineligible"] += 1
                else:
                    under = value < bar.value
                    rows.append((name, f"{value:g} {unit}", bar_name, "under" if under else "OVER"))
                    counts["under" if under else "over"] += 1
    if not quiet:
        for r in rows:
            print(f"  {r[0]:<32} {r[1]:>12}  vs {r[2]:<12} {r[3]}")
    print(f"\n{len(rows)} @below assertions: {counts['under']} under the bar, {counts['over']} OVER, "
          f"{counts['literal']} against a literal, {counts['unknown']} unknown, "
          f"{counts['ineligible']} not evaluable as a decision (no n recorded)")
    print("bars: docs/claims-bars.json; a bar that is not a record is not a bar")
    return 1 if (counts["over"] or counts["literal"] or counts["unknown"]) else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m harness.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="verify a claims file against records")
    c.add_argument("claims")
    c.add_argument("--records")
    c.add_argument("--bars")
    c.add_argument("--quiet", action="store_true")

    b = sub.add_parser("below-report", help="evaluate every @below assertion against docs/claims-bars.json")
    b.add_argument("paths", nargs="+")
    b.add_argument("--bars", default="docs/claims-bars.json")
    b.add_argument("--quiet", action="store_true")

    sub.add_parser("scope", help="print what this harness does and does not check")

    d = sub.add_parser("db", help="query the evidence index (records, claims, verdicts)")
    d.add_argument("query", choices=("build", "current", "lane", "lineage", "orphans", "stale",
                                     "summary", "current-block", "write-block", "export-md"))
    d.add_argument("arg", nargs="?")
    d.add_argument("--db", default="docs/wsp/ledger.db")
    d.add_argument("--no-evidence", action="store_true")
    d.add_argument("--with-logs", action="store_true")

    v = sub.add_parser("verdicts", help="validate the verdict registry and list what is current")
    v.add_argument("--path", default="docs/wsp/verdicts.yaml")
    v.add_argument("--cite", nargs="*", default=[], help="verdict ids a claim wants to cite")

    e = sub.add_parser("env-check", help="check the docs' claims about this machine against it")
    e.add_argument("--root", default=".")

    s = sub.add_parser("source-check", help="read what a measurement's input file IS, and refuse a "
                                            "false population label (bit rate, colour tags, "
                                            "-c copy proof, contaminated inputs)")
    s.add_argument("manifest", nargs="?", default="docs/wsp/sources.yaml")
    s.add_argument("--clip-root", default="")
    s.add_argument("--records", default="", help="records dir/glob whose populations name an input")
    s.add_argument("--no-probe", action="store_true")

    args = ap.parse_args(argv)
    if args.cmd == "db":
        from .db import main as db_main
        extra = ["--db", args.db]
        if args.no_evidence:
            extra.append("--no-evidence")
        if args.with_logs:
            extra.append("--with-logs")
        return db_main([args.query] + ([args.arg] if args.arg else []) + extra)
    if args.cmd == "verdicts":
        from .verdicts import check as check_verdicts, load, refuses
        import yaml
        verdicts = load(yaml.safe_load(open(args.path)) or [])
        problems = check_verdicts(verdicts)
        for p in problems:
            print(f"REFUSED: {p}")
        if args.cite:
            for p in refuses(verdicts, args.cite):
                print(f"REFUSED: {p}")
                problems.append(p)
        if problems:
            return 1
        cur = [v for v in verdicts if v.is_current]
        print(f"{args.path}: {len(verdicts)} verdicts, {len(cur)} current, "
              f"{len(verdicts) - len(cur)} superseded/refuted - registry readable")
        return 0
    if args.cmd == "env-check":
        from .envcheck import main as env_main
        return env_main(["--root", args.root]) if args.root != "." else env_main([])
    if args.cmd == "source-check":
        from .source import main as source_main
        extra = [args.manifest]
        if args.clip_root:
            extra += ["--clip-root", args.clip_root]
        if args.records:
            extra += ["--records", args.records]
        if args.no_probe:
            extra.append("--no-probe")
        return source_main(extra)
    if args.cmd == "below-report":
        return below_report(args.paths, args.bars, args.quiet)
    if args.cmd == "check":
        return check(args.claims, args.records, args.bars, args.quiet)
    print(__doc__)
    print(NOT_CHECKED)
    print("Eight classes and their gate:")
    for cls, gate in [
        ("1 number without the tuple", "core.Record + Store.ratio key comparison"),
        ("2 mechanism before a discriminating pair", "gates.check_claims 'cause' (exactly one field differs)"),
        ("3 leftover from a previous edit", "gates.aa_gate + refuse (A/A before A/B)"),
        ("4 intent as completion", "run.run returns a tuple; run.probe is three-state; ResultsTable refuses error"),
        ("5 model of an artifact", "run.parse_* schemas with require(); 'smem' must say static or dynamic"),
        ("6 cost model as a measurement", "Record.tier measured/derived/assumed; derived carries formula+inputs"),
        ("7 one sample as a measurement", "Record.statistic/n + gates.below(bar.of, bar.statistic)"),
        ("8 green check as validation", "gates.Scope + finalize(); this CLI prints its own uncovered scope"),
    ]:
        print(f"  {cls:<44} {gate}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
