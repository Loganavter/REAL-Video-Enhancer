"""SQLite index over the evidence: records, claims, verdicts. Files are truth; this is a cache.

The problem this solves is not storage, it is **entry point**. The research ledger grew into a
chronological notebook of 2,400 lines: verdicts from the 15th sit above measurements from the 17th, and
a reader who goes top-down stops at the first `closed` it meets - which is how "int8 is blocked for
weeks" was assembled out of four verdicts that our own measurements had replaced. Markdown cannot fix
that, because markdown has no notion of a record's lifetime and no way to refuse a stale one.

So: everything stays a file (records `*.jsonl`, claims `*.txt`, verdicts `*.yaml`, all in git and all
diffable), and this module builds a queryable index over them:

    python -m harness.cli db build            # (re)build docs/wsp/ledger.db from the files
    python -m harness.cli db current          # what is true now, grouped by lane
    python -m harness.cli db lane int8        # one lane: verdict, evidence, records
    python -m harness.cli db lineage <vid>    # what replaced what, in order
    python -m harness.cli db orphans          # records no verdict rests on
    python -m harness.cli db stale            # verdicts whose evidence file is gone

Why not MLflow/Weights&Biases: those are the right shape (runs, params, metrics, artifacts, a queryable
store, generated views) but they assume a server and a database as the source of truth. Here the source
of truth has to survive in git next to the code, and the index has to be rebuildable from a clone with
no network. This is that shape, with the files kept as truth: the DB is a cache you can delete.

The lifetime rule the schema encodes (and `harness.verdicts` enforces on the YAML before it lands here):
a verdict has `status`, `scope`, `date`, and - if it is `superseded` or `refuted` - a `superseded_by`
that resolves to a `current` verdict. A claim may cite a verdict; citing a superseded one is refused.
"""
from __future__ import annotations

import argparse
import glob
import re
import json
import os
import re
import sqlite3
import sys
from dataclasses import dataclass

SCHEMA = """
CREATE TABLE IF NOT EXISTS record (
  rid TEXT, source TEXT, value REAL, unit TEXT, statistic TEXT, tier TEXT,
  pop_kind TEXT, items_sha TEXT, n INTEGER, pop_label TEXT,
  instrument TEXT, config TEXT, session TEXT, run TEXT, artifact TEXT,
  formula TEXT, inputs TEXT, note TEXT, extra TEXT,
  PRIMARY KEY (source, rid)
);
CREATE TABLE IF NOT EXISTS verdict (
  vid TEXT PRIMARY KEY, lane TEXT, status TEXT, scope TEXT, date TEXT,
  statement TEXT, evidence TEXT, superseded_by TEXT, note TEXT, open_question TEXT
);
CREATE TABLE IF NOT EXISTS claim (
  name TEXT, value REAL, unit TEXT, population TEXT, tier TEXT, session TEXT,
  source TEXT, line INTEGER
);
CREATE INDEX IF NOT EXISTS claim_name ON claim(name);
CREATE INDEX IF NOT EXISTS record_name ON record(rid);
"""

CLAIM = re.compile(r'@claim\s+(\S+)\s+([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s+(\S+)\s+"([^"]*)"'
                   r'(?:\s+(measured|derived|assumed))?')
SESSION = re.compile(r'@session=(\S+)')

DEFAULT_RECORDS = "docs/wsp/verified/records/*.jsonl"
DEFAULT_CLAIMS = "docs/claims-*.txt"
DEFAULT_VERDICTS = "docs/wsp/verdicts.yaml"


def load_verdicts(path: str) -> list[dict]:
    import yaml
    with open(path) as fh:
        data = yaml.safe_load(fh) or []
    if not isinstance(data, list):
        raise SystemExit(f"{path}: expected a list of verdicts")
    return data


def build(db_path: str = "docs/wsp/ledger.db", records: str = DEFAULT_RECORDS,
          claims: str = DEFAULT_CLAIMS, verdicts: str = DEFAULT_VERDICTS,
          extra_records: tuple[str, ...] = ()) -> dict:
    """Rebuild the index from the files. Idempotent: the DB is dropped and recreated."""
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)

    counts = {"records": 0, "claims": 0, "verdicts": 0}
    files = sorted(glob.glob(records)) + list(extra_records)
    for path in files:
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pop = r.get("population", {})
            con.execute("INSERT OR REPLACE INTO record VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                r.get("id", ""), path, r.get("value"), r.get("unit"), r.get("statistic"),
                r.get("tier"), pop.get("kind"), pop.get("items_sha"), pop.get("n"),
                pop.get("label", ""), r.get("instrument"), r.get("config"), r.get("session"),
                r.get("run"), r.get("artifact"), r.get("formula"),
                ",".join(r.get("inputs") or []), r.get("note"), json.dumps(r.get("extra", {}))))
            counts["records"] += 1

    for path in sorted(glob.glob(claims)):
        for lineno, line in enumerate(open(path), 1):
            for m in CLAIM.finditer(line):
                name, value, unit, pop, tier = m.groups()
                sm = SESSION.search(line)
                con.execute("INSERT INTO claim VALUES (?,?,?,?,?,?,?,?)",
                            (name, float(value), unit, pop, tier or "measured",
                             sm.group(1) if sm else "", path, lineno))
                counts["claims"] += 1

    for v in load_verdicts(verdicts):
        con.execute("INSERT OR REPLACE INTO verdict VALUES (?,?,?,?,?,?,?,?,?,?)", (
            v["id"], v.get("lane", ""), v.get("status", ""), v.get("scope", ""), str(v.get("date", "")),
            " ".join((v.get("statement") or "").split()),
            (v.get("evidence") or "").strip(),
            v.get("superseded_by", ""), " ".join((v.get("note") or "").split()),
            " ".join((v.get("open_question") or "").split())))
        counts["verdicts"] += 1

    con.commit()
    con.close()
    return counts


def connect(db_path: str = "docs/wsp/ledger.db") -> sqlite3.Connection:
    if not os.path.exists(db_path):
        raise SystemExit(f"{db_path}: no index yet - run `python -m harness.cli db build`")
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


# --------------------------------------------------------------------------------------------------
# queries
# --------------------------------------------------------------------------------------------------
def current(con) -> list[sqlite3.Row]:
    return list(con.execute(
        "SELECT * FROM verdict WHERE status = 'current' ORDER BY scope, vid"))


def lanes(con) -> dict[str, list[sqlite3.Row]]:
    out: dict[str, list[sqlite3.Row]] = {}
    for row in current(con):
        lane = (row["lane"] or "other").strip().lower()
        out.setdefault(lane, []).append(row)
    return out


def lineage(con, vid: str) -> list[sqlite3.Row]:
    """Follow `superseded_by` from the oldest replacement down to the current verdict."""
    chain: list[sqlite3.Row] = []
    row = con.execute("SELECT * FROM verdict WHERE vid = ?", (vid,)).fetchone()
    seen = set()
    while row is not None and row["vid"] not in seen:
        seen.add(row["vid"])
        chain.append(row)
        if not row["superseded_by"]:
            break
        row = con.execute("SELECT * FROM verdict WHERE vid = ?", (row["superseded_by"],)).fetchone()
    return chain


def records_for(con, evidence: str) -> list[sqlite3.Row]:
    return list(con.execute("SELECT * FROM record WHERE source = ? ORDER BY rid", (evidence,)))


def orphans(con) -> list[sqlite3.Row]:
    """Record files no verdict names as its evidence - the notebook's unread tail.

    This is the number that motivated the index: a lane can have hundreds of records and three
    verdicts, and nothing in a markdown file says which of the rest are still load-bearing.
    """
    cited = {r["evidence"] for r in con.execute("SELECT DISTINCT evidence FROM verdict")
             if r["evidence"]}
    return [r for r in con.execute(
        "SELECT source, COUNT(*) n FROM record GROUP BY source ORDER BY n DESC") if r["source"] not in cited]


def stale(con) -> list[sqlite3.Row]:
    return [r for r in con.execute("SELECT vid, evidence FROM verdict WHERE evidence != ''")
            if not os.path.exists(r["evidence"])]


def summary(con) -> dict:
    one = lambda q: con.execute(q).fetchone()[0]          # noqa: E731
    return {
        "records": one("SELECT COUNT(*) FROM record"),
        "claims": one("SELECT COUNT(*) FROM claim"),
        "verdicts": one("SELECT COUNT(*) FROM verdict"),
        "current": one("SELECT COUNT(*) FROM verdict WHERE status='current'"),
        "superseded": one("SELECT COUNT(*) FROM verdict WHERE status='superseded'"),
        "refuted": one("SELECT COUNT(*) FROM verdict WHERE status='refuted'"),
        "claim_names": one("SELECT COUNT(DISTINCT name) FROM claim"),
    }


def render_current_state(con) -> str:
    s = summary(con)
    out = [
        "## CURRENT STATE",
        "",
        f"Index: {s['records']} records, {s['claims']} claims, {s['verdicts']} verdicts "
        f"({s['current']} current, {s['superseded']} superseded, {s['refuted']} refuted). "
        f"The files are truth (`docs/wsp/verdicts.yaml`, `docs/wsp/verified/records/*.jsonl`); this "
        f"section is a view. Everything below it is history and may be superseded - follow "
        f"`superseded_by` in the verdict registry, never the position in this file.",
        "",
    ]
    for lane, rows in sorted(lanes(con).items()):
        out.append(f"### {lane}")
        for r in rows:
            out.append(f"- **{r['vid']}** ({r['scope']}) - {r['statement']}")
            if r["open_question"]:
                out.append(f"  - OPEN: {r['open_question']}")
        out.append("")
    sup = con.execute("SELECT vid, superseded_by, scope FROM verdict "
                      "WHERE status != 'current' ORDER BY vid").fetchall()
    if sup:
        out.append("### superseded / refuted - do not cite")
        for r in sup:
            out.append(f"- {r['vid']} -> {r['superseded_by']}  ({r['scope']})")
        out.append("")
    return "\n".join(out)


BEGIN = "<!-- BEGIN CURRENT STATE (generated by `python -m harness.cli db write-block`) -->"
END = "<!-- END CURRENT STATE -->"


def splice(path: str, block: str) -> bool:
    """Replace the marked region of a markdown file with `block`. Idempotent.

    The ledger keeps its history (it is cited by line number from the provenance tables) and gets a
    generated head. Nobody has to trust the top of a notebook again: the head is a view over the
    registry, and re-running the command is what keeps it true.
    """
    text = open(path).read() if os.path.exists(path) else ""
    region = f"{BEGIN}\n{block.rstrip()}\n{END}"
    if BEGIN in text and END in text:
        head, rest = text.split(BEGIN, 1)
        _, tail = rest.split(END, 1)
        new = f"{head}{region}{tail}"
    else:
        # insert after the preamble: the first blank line that ends the opening paragraph block
        lines = text.splitlines()
        cut = 0
        for i, line in enumerate(lines):
            if i > 0 and not line.strip():
                cut = i + 1
                break
        new = "\n".join(lines[:cut]) + "\n\n" + region + "\n" + "\n".join(lines[cut:])
    if new != text:
        with open(path, "w") as fh:
            fh.write(new)
        return True
    return False


def _anchor(path: str) -> str:
    return "evidence-" + re.sub(r"[^a-z0-9]+", "-", path.lower()).strip("-")


def export_markdown(con, embed_evidence: bool = True, embed_logs: bool = False) -> str:
    """The whole index as one markdown document, for sending to someone.

    Not a replacement for the registry: it is a snapshot with no query behind it, generated so a reader
    who has neither the repo nor the harness can still see what is current, what replaced what, and
    which record each number came from.
    """
    s = summary(con)
    out = ["# RVE evidence index - export", "",
           f"Generated from `docs/wsp/ledger.db` ({s['records']} records, {s['claims']} claims, "
           f"{s['verdicts']} verdicts: {s['current']} current, {s['superseded']} superseded, "
           f"{s['refuted']} refuted).",
           "",
           "This is a snapshot. The source of truth is the files - `docs/wsp/verdicts.yaml`, "
           "`docs/wsp/verified/records/*.jsonl`, `docs/claims-*.txt` - and the index is rebuilt from "
           "them with `python -m harness.cli db build`. A verdict marked *superseded* is kept only so "
           "that the replacement can be followed; do not cite one.", ""]
    out += ["## Current state, by lane", ""]
    for lane, rows in sorted(lanes(con).items()):
        out += [f"### {lane}", ""]
        for r in rows:
            out.append(f"**{r['vid']}** ({r['date']}) - *{r['scope']}*")
            out.append("")
            out.append(f"> {r['statement']}")
            if r["evidence"]:
                out.append("")
                if embed_evidence and os.path.exists(r["evidence"]):
                    out.append(f"Evidence: [`{r['evidence']}`](#{_anchor(r['evidence'])}) "
                               f"- full text below")
                else:
                    out.append(f"Evidence: `{r['evidence']}`")
            if r["open_question"]:
                out.append("")
                out.append(f"**OPEN:** {r['open_question']}")
            out.append("")
    sup = con.execute("SELECT vid, superseded_by, scope, statement, date FROM verdict "
                      "WHERE status != 'current' ORDER BY vid").fetchall()
    if sup:
        out += ["## Superseded and refuted - do not cite", "",
                "| verdict | status | replaced by | scope |", "|---|---|---|---|"]
        for r in sup:
            st = con.execute("SELECT status FROM verdict WHERE vid=?", (r["vid"],)).fetchone()
            out.append(f"| `{r['vid']}` | {st['status']} | `{r['superseded_by']}` | {r['scope']} |")
        out.append("")
    out += ["## Records", "",
            "Each row is a measurement tuple: value, unit, statistic, and the population it was taken "
            "on. `of` is what the number is *of*, which is the field the old prose was missing.", ""]
    rows = con.execute("SELECT * FROM record ORDER BY source, rid").fetchall()
    by_source: dict[str, list] = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)
    for source, recs in sorted(by_source.items()):
        out += [f"### `{source}`", "",
                "| id | value | unit | statistic | of | tier | population |", "|---|---|---|---|---|---|---|"]
        for r in recs:
            try:
                extra = json.loads(r["extra"] or "{}")
            except json.JSONDecodeError:
                extra = {}
            of = extra.get("of", "")
            pop = (r["pop_label"] or r["pop_kind"] or "")[:70]
            out.append(f"| {r['rid']} | {r['value']:g} | {r['unit']} | {r['statistic']} | {of} | "
                       f"{r['tier']} | {pop} |")
        out.append("")
    out += ["## Claims", "",
            "The legacy corpus (`docs/claims-*.txt`): names, values and the session that measured "
            "them. These are the lines the migrate gate reports on, not record-backed claims.", "",
            "| name | value | unit | session | source:line |", "|---|---|---|---|---|"]
    for r in con.execute("SELECT * FROM claim ORDER BY name, session"):
        out.append(f"| `{r['name']}` | {r['value']:g} | {r['unit']} | {r['session']} | "
                   f"`{r['source']}:{r['line']}` |")
    cited: list[str] = []
    if embed_evidence or embed_logs:
        for r in con.execute("SELECT DISTINCT evidence FROM verdict WHERE evidence != ''"):
            cited.append(r["evidence"])
        for r in con.execute("SELECT DISTINCT artifact FROM record WHERE artifact != ''"):
            cited.append(r["artifact"])
        seen, ordered = set(), []
        for path in cited:
            if path in seen:
                continue
            seen.add(path)
            is_log = path.endswith(".log")
            if is_log and not embed_logs:
                continue
            if not is_log and not embed_evidence:
                continue
            ordered.append(path)
        out += ["## Evidence, in full", "",
                "Every file the verdicts and records rest on, embedded so this document stands alone. "
                "The claims lines (`@claim ...`) at the end of each one are the machine-checkable form; "
                "the prose above them is what the checks cannot verify.", ""]
        missing = []
        for path in ordered:
            if not os.path.exists(path):
                missing.append(path)
                continue
            with open(path, errors="replace") as fh:
                body = fh.read().rstrip()
            # demote the embedded file's own headings by one level, so only this document's sections
            # appear at `##` in a reader's outline
            def _demote(ln: str) -> str:
                m = re.match(r"^(#{1,6}) ", ln)
                if not m:
                    return ln
                return "#" * min(6, len(m.group(1)) + 3) + ln[len(m.group(1)):]
            body = "\n".join(_demote(ln) for ln in body.splitlines())
            out += [f'<a id="{_anchor(path)}"></a>', "", f"### `{path}`", "", body, ""]
        if missing:
            out += ["### Cited but missing on disk", ""]
            out += [f"- `{m}`" for m in missing]
            out.append("")

    out += ["", "## What this document does not tell you", "",
            "- whether a population label is true (the item-set hash is checked, the human label is not)",
            "- whether the item set is the one the decision needs",
            "- whether a cost model is complete",
            "- free-text arithmetic in the legacy claims files",
            "", "A green check is a floor, not a ceiling."]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m harness.cli db")
    ap.add_argument("query", choices=("build", "current", "lane", "lineage", "orphans", "stale",
                                      "summary", "current-block", "write-block", "export-md"))
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--db", default="docs/wsp/ledger.db")
    ap.add_argument("--no-evidence", action="store_true",
                    help="export-md: keep evidence as paths instead of embedding the files")
    ap.add_argument("--with-logs", action="store_true",
                    help="export-md: also embed the raw .log artifacts (bigger, noisier)")
    args = ap.parse_args(argv)

    if args.query == "build":
        counts = build(args.db)
        print(f"{args.db}: {counts['records']} records, {counts['claims']} claims, "
              f"{counts['verdicts']} verdicts")
        return 0

    con = connect(args.db)
    if args.query == "current":
        print(render_current_state(con))
    elif args.query == "current-block":
        print(render_current_state(con))
    elif args.query == "export-md":
        target = args.arg or "docs/wsp/ledger-export.md"
        text = export_markdown(con, embed_evidence=not args.no_evidence,
                               embed_logs=args.with_logs)
        with open(target, "w") as fh:
            fh.write(text)
        print(f"{target}: {len(text.splitlines())} lines, {os.path.getsize(target)} bytes")
    elif args.query == "write-block":
        target = args.arg or "docs/RESEARCH-LEDGER.md"
        changed = splice(target, render_current_state(con))
        print(f"{target}: {'written' if changed else 'unchanged'}")
    elif args.query == "summary":
        print(json.dumps(summary(con), indent=2))
    elif args.query == "lineage":
        if not args.arg:
            raise SystemExit("lineage needs a verdict id")
        for r in lineage(con, args.arg):
            print(f"{r['vid']:28s} {r['status']:10s} {r['date']}  -> {r['superseded_by'] or '-'}")
    elif args.query == "lane":
        needle = (args.arg or "").lower()
        for row in current(con):
            if needle and needle not in row["scope"].lower() and needle not in row["vid"]:
                continue
            print(f"{row['vid']}  ({row['scope']})\n    {row['statement']}")
            if row["open_question"]:
                print(f"    OPEN: {row['open_question']}")
            for rec in records_for(con, row["evidence"]):
                print(f"      {rec['rid']:5s} {rec['value']:>12} {rec['unit']:5s} "
                      f"{rec['statistic']:9s} {rec['pop_kind']}")
    elif args.query == "orphans":
        for r in orphans(con):
            print(f"{r['n']:5d}  {r['source']}")
    elif args.query == "stale":
        rows = stale(con)
        for r in rows:
            print(f"{r['vid']:28s} {r['evidence']}")
        print(f"\n{len(rows)} verdict(s) with a missing evidence file")
    return 0


if __name__ == "__main__":
    sys.exit(main())
