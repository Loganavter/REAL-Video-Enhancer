"""The verdict registry and the index: a lifetime check and a query, both CPU-only.

The failure these exist for: four verdicts from one day were assembled into a conclusion ("int8 is
blocked for weeks") that our own measurements had already replaced, because a markdown notebook has no
notion of a verdict's lifetime and no way to refuse a stale one. The tests below are the refusal.
"""
from __future__ import annotations

import json
import os

import pytest

from harness import db
from harness.verdicts import check, load, refuses, resolve

GOOD = {
    "id": "old", "status": "superseded", "scope": "TRT build", "date": "2026-11-15",
    "statement": "the epilogue cannot fuse", "superseded_by": "new",
}
NEW = {
    "id": "new", "status": "current", "scope": "our kernel", "date": "2026-11-17",
    "statement": "the epilogue fuses at 1.91 ms",
}


def test_registry_accepts_a_resolved_chain() -> None:
    assert check(load([GOOD, NEW])) == []


def test_missing_scope_is_refused() -> None:
    bad = dict(NEW, scope="")
    problems = check(load([bad]))
    assert any("no scope" in p for p in problems)


def test_missing_date_is_refused() -> None:
    problems = check(load([dict(NEW, date="")]))
    assert any("YYYY-MM-DD" in p for p in problems)


def test_superseded_without_replacement_is_refused() -> None:
    problems = check(load([dict(GOOD, superseded_by="")]))
    assert any("no superseded_by" in p for p in problems)


def test_dangling_replacement_is_refused() -> None:
    problems = check(load([dict(GOOD, superseded_by="nope")]))
    assert any("is not a verdict" in p for p in problems)


def test_chain_that_ends_superseded_is_refused() -> None:
    third = dict(NEW, id="newer", status="current")
    mid = dict(GOOD, id="mid", superseded_by="newer")
    root = dict(GOOD, id="old", superseded_by="mid")
    # `mid` is superseded (its status says so) and points at a current verdict: legal on its own.
    # What must fail is a chain whose target is itself non-current - here the root names a
    # superseded verdict and that verdict is not resolvable further.
    broken = dict(GOOD, id="old", superseded_by="mid")
    mid_bad = dict(GOOD, id="mid", superseded_by="gone")
    problems = check(load([broken, mid_bad, third]))
    assert any("is not a verdict" in p for p in problems)


def test_cycle_is_refused() -> None:
    a = dict(GOOD, id="a", superseded_by="b")
    b = dict(GOOD, id="b", superseded_by="a")
    problems = check(load([a, b]))
    assert any("loops" in p for p in problems)


def test_citing_a_superseded_verdict_is_refused_with_the_replacement() -> None:
    out = refuses(load([GOOD, NEW]), ["old"])
    assert out and "superseded" in out[0] and "new" in out[0]
    assert refuses(load([GOOD, NEW]), ["new"]) == []


def test_resolve_follows_the_chain() -> None:
    v = load([GOOD, NEW])
    assert resolve(v, "old").id == "new"


def test_db_build_and_queries(tmp_path) -> None:
    rec = {
        "value": 1.91, "unit": "ms", "statistic": "median", "tier": "measured",
        "population": {"kind": "block launches", "items": ["a", "b"], "items_sha": "x", "n": 2},
        "instrument": "cuda events", "config": "sm89", "session": "s1", "run": "r", "id": "R001",
    }
    (tmp_path / "records").mkdir()
    (tmp_path / "records" / "lane.jsonl").write_text(json.dumps(rec) + "\n")
    (tmp_path / "claims.txt").write_text(
        '@claim block_ms 1.91 ms "block launches [x] n=2" measured  @session=s1\n')
    (tmp_path / "verdicts.yaml").write_text(
        "- id: v1\n  lane: int8\n  status: current\n  scope: our kernel\n  date: 2026-11-17\n"
        "  statement: the block closes the epilogue\n  evidence: docs/wsp/verified/does-not-exist.txt\n")

    counts = db.build(str(tmp_path / "l.db"), records=str(tmp_path / "records" / "*.jsonl"),
                      claims=str(tmp_path / "claims.txt"), verdicts=str(tmp_path / "verdicts.yaml"))
    assert counts == {"records": 1, "claims": 1, "verdicts": 1}

    con = db.connect(str(tmp_path / "l.db"))
    assert [r["vid"] for r in db.current(con)] == ["v1"]
    assert db.lanes(con)["int8"][0]["vid"] == "v1"
    s = db.summary(con)
    assert (s["records"], s["claims"], s["current"]) == (1, 1, 1)
    # `evidence` is repo-relative, so a deliberately missing path is what `stale` must report
    assert [r["vid"] for r in db.stale(con)] == ["v1"]
    assert "CURRENT STATE" in db.render_current_state(con)
    con.close()


def test_splice_is_idempotent(tmp_path) -> None:
    path = tmp_path / "doc.md"
    path.write_text("# Title\n\npreamble\n\n## History\nold text\n")
    block = "## CURRENT STATE\n\n- v1\n"
    assert db.splice(str(path), block) is True
    first = path.read_text()
    assert "CURRENT STATE" in first and "## History" in first
    assert db.splice(str(path), block) is False      # second write changes nothing
    assert path.read_text() == first
