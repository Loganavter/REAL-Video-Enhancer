"""Attacks on the evidence harness, as negative tests.

Each case is a false claim that the harness used to certify. A red team found 23 of them on
2026-11-16; these are the ones that can be expressed as an assertion, so the bypass cannot come back
silently. The rule for every test: the call must RAISE, not print - a gate that reports and continues
is the failure mode the harness exists to prevent (its own `migrate` once flipped FAIL to PASS by
reading back the records it had just written).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "harness" / "scripts"))

from harness import (  # noqa: E402
    Bar, Claim, Population, Record, RecordError, Store, below, check_claims, parse_ptxas, refuse,
)
from harness.run import claimable  # noqa: E402
from harness.gates import AA  # noqa: E402
from harness.run import run  # noqa: E402


def rec(store: Store, value: float, unit: str = "%", *, items=("1", "2"), stat="worst_of",
        tier="measured", instrument="acceptance_stat", config="cfg", extra=None, **kw) -> Record:
    kw.setdefault("run", "run-1")
    return store.emit(Record(value=value, unit=unit, population=Population.of("frames", items),
                             instrument=instrument, config=config, session=store.session,
                             statistic=stat, tier=tier, extra=extra or {}, **kw))


def test_two_scales_of_one_unit_are_not_a_ratio():
    s = Store()
    slow, fast = rec(s, 2000.0, "ms", stat="single"), rec(s, 2.0, "s", stat="single")
    with pytest.raises(RecordError, match="units differ"):
        s.ratio(slow.id, fast.id)


def test_a_bar_is_not_judged_in_another_scale():
    s = Store()
    two_seconds = rec(s, 2.0, "s", stat="single", extra={"of": "frame_time"})
    with pytest.raises(RecordError, match="cannot be judged"):
        below(two_seconds, Bar(name="500ms", value=500.0, unit="ms", of="frame_time", statistic="single"))


def test_nan_cannot_be_a_record():
    s = Store()
    with pytest.raises(RecordError, match="non-finite"):
        rec(s, float("nan"))


def test_an_unknown_formula_is_not_a_pass():
    s = Store()
    a, b = rec(s, 10.0, "ms", stat="single"), rec(s, 4.0, "ms", stat="single")
    s.derived("invented", 999.0, "x", "a_speedup_model_nobody_wrote", (a.id, b.id))
    problems = s.verify_derived()
    assert problems and "not one this verifier can recompute" in problems[0]


def test_assumed_is_not_evidence_against_a_bar():
    s = Store()
    assumed = rec(s, 0.009, tier="assumed", extra={"of": "share"})
    with pytest.raises(RecordError, match="assumed"):
        below(assumed, Bar(name="bar", value=0.1, unit="%", of="share"))


def test_one_frame_is_not_a_spread_sample():
    s = Store()
    one = rec(s, 0.0, items=("7",), stat="worst_of", extra={"of": "share"})
    with pytest.raises(RecordError, match="n>=2"):
        below(one, Bar(name="bar", value=0.1, unit="%", of="share", statistic="worst_of"))


def test_a_cause_needs_the_same_items_and_one_varying_field():
    s = Store()
    a = rec(s, 1.0, items=("1", "2", "3"))
    b = rec(s, 2.0, items=("1", "2", "4"))
    problems, _ = check_claims(s, [Claim("C1", "cause", (a.id, b.id), mechanism="write-allocate")])
    assert problems and "population" in problems[0]

    c = rec(s, 3.0, items=("1", "2", "3"), config="cfg-no-rfo")
    problems, _ = check_claims(s, [Claim("C2", "cause", (a.id, c.id), mechanism="RFO")])
    assert problems == []


def test_a_forged_aa_cannot_unlock_a_refusal():
    s = Store()
    base = rec(s, 1.0)
    forged = AA(identical=True, max_delta=0.0, n=2, record=base)
    with pytest.raises(RecordError):
        refuse(s, "C", base.id, forged, "no A/A was ever run")


def test_provenance_cannot_be_deleted_after_emit():
    s = Store()
    r = rec(s, 0.01, extra={"of": "share"})
    r.run = ""
    r.artifact = ""
    with pytest.raises(RecordError, match="provenance was deleted"):
        r.validate()


def test_an_artifact_without_a_digest_is_not_stored():
    s = Store()
    with pytest.raises(RecordError, match="digest"):
        s.emit(Record(value=1.0, unit="%", population=Population.of("p", ["1"]), instrument="i",
                      config="c", session=s.session, statistic="single",
                      artifact="/nonexistent/capture-never-taken.txt"))


def test_a_parse_with_every_field_missing_is_not_claimable():
    r = run(["echo", "ptxas info: nothing here"], parse=parse_ptxas)
    assert r.status == "ok" and r.parsed.missing
    assert not claimable(r)


def test_a_verdict_validates_the_records_it_cites():
    s = Store()
    a = rec(s, 1.0, stat="single", artifact="/nonexistent/x", extra={"artifact_sha256": "d"})
    b = rec(s, 2.0, stat="single", artifact="/nonexistent/y", extra={"artifact_sha256": "d"})
    a.extra.pop("artifact_sha256")
    problems, _ = check_claims(s, [Claim("V", "verdict", (a.id, b.id))])
    assert problems and "digest" in problems[0]


def test_a_duplicate_id_or_an_edited_item_hash_cannot_be_loaded(tmp_path):
    s = Store()
    r = rec(s, 1.0)
    line = json.dumps(r.to_dict())
    p = tmp_path / "records.jsonl"
    p.write_text(line + "\n" + line + "\n")
    with pytest.raises(RecordError, match="duplicate record id"):
        Store.load(str(p))

    rows = [json.loads(line)]
    rows[0]["population"]["items_sha"] = "0" * 16
    p.write_text(json.dumps(rows[0]) + "\n")
    with pytest.raises(RecordError, match="edited"):
        Store.load(str(p))


def test_a_delta_across_sessions_is_assumed_not_derived():
    s = Store()
    a = rec(s, 10.0, "%", stat="single")
    b = rec(s, 12.0, "%", stat="single", extra={})
    b.session = "s-2026-11-15T00:00:00Z"
    with pytest.raises(RecordError, match="session differs"):
        s.delta(a.id, b.id)
    assert s.delta(a.id, b.id, cross_session=True).tier == "assumed"
