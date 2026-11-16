"""The harness itself: the tuple, the gates, the three-state probe, the artifact parsers.

No GPU, no torch (the checkpoint parser is torch-free on purpose), no network. Every test here is a
retraction from docs/RESEARCH-LEDGER.md turned into a rule that can no longer be violated quietly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from harness import (  # noqa: E402
    Bar, Claim, Population, Record, RecordError, ResultsTable, Scope, Store, aa_gate, below,
    check_claims, claimable, graph_delta, parse_checkpoint, parse_engine_inspector, parse_ptxas, probe,
    refuse,
    require_config_pair, run, shared_calibration, suspicious_identity,
)
from harness.astcheck import dead_stores, used_before_assignment  # noqa: E402

MODEL_2X = Path("/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2/custom_models/"
                "2x-Futsuu-Anime(1).pth")


def mk(store: Store, value: float, unit: str = "%", *, stat: str = "worst_of",
       items=("0", "1", "2"), kind: str = "rendered frames", instrument: str = "acceptance_stat",
       config: str = "cfg-64-fp16", session: str | None = None, **kw) -> Record:
    kw.setdefault("run", "run-test-1")
    return store.emit(Record(value=value, unit=unit, population=Population.of(kind, items),
                             instrument=instrument, config=config,
                             session=session or store.session, statistic=stat, **kw))


# ---------------------------------------------------------------------------- 1. the tuple
def test_same_label_different_frames_is_not_the_same_population():
    a = Population.of("rendered frames", [1, 2, 3], label="rendered frames")
    b = Population.of("rendered frames", [1, 2, 4], label="rendered frames")
    assert a.label == b.label
    assert a.claims_text() != b.claims_text()
    assert a.items_sha != b.items_sha


def test_ratio_between_different_item_sets_is_refused():
    s = Store()
    a = mk(s, 0.009, items=(1, 2, 3))
    b = mk(s, 0.174, items=(1, 2, 4))
    with pytest.raises(RecordError, match="populations differ"):
        s.ratio(a.id, b.id)


def test_ratio_between_two_instruments_is_refused():
    s = Store()
    a = mk(s, 24.88, unit="ms", stat="single", instrument="gpu_events")
    b = mk(s, 56.28, unit="ms", stat="single", instrument="wall_clock")
    with pytest.raises(RecordError, match="instrument differs"):
        s.ratio(a.id, b.id)


def test_ratio_across_configs_is_refused():
    s = Store()
    a = mk(s, 40.24, unit="ms", stat="single", config="cfg-int8-scalar")
    b = mk(s, 56.28, unit="ms", stat="single", config="cfg-64-fp16")
    with pytest.raises(RecordError, match="config differs"):
        s.ratio(a.id, b.id)


def test_worst_of_over_median_is_refused_unless_declared():
    s = Store()
    a = mk(s, 0.009, stat="worst_of")
    b = mk(s, 0.05, stat="median")
    with pytest.raises(RecordError, match="statistic differs"):
        s.ratio(a.id, b.id)
    rec = s.ratio(a.id, b.id, cross_statistic=True)
    assert rec.tier == "assumed"


def test_cross_session_ratio_downgrades_to_assumed():
    s = Store()
    a = mk(s, 45.7, unit="ms", stat="single")
    b = mk(s, 51.37, unit="ms", stat="single", session="s-2026-11-15T00:00:00Z")
    with pytest.raises(RecordError, match="session differs"):
        s.ratio(a.id, b.id)
    assert s.ratio(a.id, b.id, cross_session=True).tier == "assumed"


def test_measured_record_needs_a_run_or_an_artifact():
    s = Store()
    with pytest.raises(RecordError, match="must name its run"):
        s.emit(Record(value=1.0, unit="%", population=Population.of("x", [1]), instrument="i",
                      config="c", session=s.session))


def test_derived_record_needs_formula_and_inputs():
    s = Store()
    with pytest.raises(RecordError, match="derived record must carry"):
        s.emit(Record(value=1.0, unit="x", population=Population.of("x", [1]), instrument="i",
                      config="c", session=s.session, tier="derived"))


def test_record_roundtrip_keeps_the_item_set(tmp_path):
    s = Store()
    a = mk(s, 0.009)
    out = tmp_path / "records.jsonl"
    s.dump(str(out))
    s2 = Store.load(str(out))
    assert s2.get(a.id).population.items_sha == a.population.items_sha


def test_verify_derived_recomputes_and_catches_tampering():
    s = Store()
    a = mk(s, 10.0, unit="ms", stat="single")
    b = mk(s, 5.0, unit="ms", stat="single")
    good = s.derived("half", 2.0, "x", f"{a.id}/{b.id}", (a.id, b.id))
    assert s.verify_derived() == []
    good.value = 2.5                      # a number edited after the fact
    problems = s.verify_derived()
    assert problems and good.id in problems[0]


# ---------------------------------------------------------------------------- 3/4. lifecycle
def test_probe_exception_is_error_never_fail():
    p = probe("c2 read", lambda: undefined_name == 1, synthetic=True)  # noqa: F821
    assert p.status == "error"
    assert not p.is_result


def test_results_table_refuses_an_error_row():
    t = ResultsTable()
    t.add(probe("ok", lambda: True, synthetic=True))
    with pytest.raises(Exception, match="an error is not a result"):
        t.add(probe("boom", lambda: 1 / 0, synthetic=True))
    assert len(t.rows) == 1 and len(t.skipped) == 1


def test_run_timeout_kills_the_group_and_is_an_error():
    r = run(["sleep", "5"], timeout=0.3)
    assert r.timed_out and r.status == "error" and r.error_kind == "timeout"
    assert r.wall_s < 3


def test_run_parse_failure_is_an_error_not_an_empty_result():
    r = run(["echo", "not json"], parse=json.loads)
    assert r.status == "error" and r.error_kind == "unparsed"


def test_run_ok_and_parsed_is_claimable():
    r = run(["echo", '{"a": 1}'], parse=json.loads)
    assert r.status == "ok" and r.parsed == {"a": 1}


# ---------------------------------------------------------------------------- 3. A/A and refusal
def test_aa_gate_and_refusal_against_a_verified_base():
    s = Store()
    a = mk(s, 0.009)
    a2 = mk(s, 0.009)
    aa = aa_gate(s, a.id, a2.id)
    assert aa.identical
    refuse(s, "C-refuse-fp16", a.id, aa, "the fused kernel measured slower on a verified base")

    bad = mk(s, 0.011)
    aa_bad = aa_gate(s, a.id, bad.id)
    assert not aa_bad.identical
    with pytest.raises(RecordError, match="A/A"):
        refuse(s, "C-false-refusal", a.id, aa_bad, "rejecting a correct variant")


# ---------------------------------------------------------------------------- 7. bars
def test_below_refuses_a_peak_against_a_share_threshold():
    s = Store()
    peak = mk(s, 7.0, unit="levels", stat="max", extra={"of": "peak"})
    share_threshold = Bar(name="share_threshold", value=8, unit="levels", of="pixel_defect",
                          statistic="max")
    with pytest.raises(RecordError, match="is a bar of"):
        below(peak, share_threshold)


def test_below_refuses_a_single_sample_against_a_worst_of_bar():
    s = Store()
    one = mk(s, 0.000, stat="single", items=(7,))
    bar = Bar(name="pixel_share", value=0.1, unit="%", of="share", statistic="worst_of")
    one.extra["of"] = "share"
    with pytest.raises(RecordError, match="statistic"):
        below(one, bar)


def test_below_passes_the_shipping_configuration():
    s = Store()
    share = mk(s, 0.009, extra={"of": "share"})
    below(share, Bar(name="pixel_share", value=0.1, unit="%", of="share", statistic="worst_of"))


# ---------------------------------------------------------------------------- 2. cause
def test_cause_needs_two_records_that_differ_in_one_field():
    s = Store()
    a = mk(s, 0.060, stat="mean", items=(1, 2, 3))
    b = mk(s, 26.1, stat="mean", items=(1, 2, 3))
    problems, scope = check_claims(s, [Claim("C1", "cause", (a.id, b.id), mechanism="write-allocate")])
    assert problems and "never varied" in problems[0]

    c = mk(s, 0.060, stat="mean", items=(1, 2, 3), config="cfg-no-rfo")
    problems, scope = check_claims(s, [Claim("C2", "cause", (a.id, c.id), mechanism="RFO")])
    assert problems == [] and "cause" in scope.checked


# ---------------------------------------------------------------------------- 8. coverage
def test_verdict_mixes_populations():
    s = Store()
    a = mk(s, 1.0, items=(1, 2))
    b = mk(s, 1.0, items=(1, 2, 3))
    problems, _ = check_claims(s, [Claim("V1", "verdict", (a.id, b.id))])
    assert problems and "mixes" in problems[0]


def test_scope_refuses_a_green_over_an_uncovered_kind():
    sc = Scope(expected={"below", "ratio"})
    sc.cover("below")
    with pytest.raises(RecordError, match="uncovered scope"):
        sc.finalize()


def test_scope_reports_what_it_did_not_check():
    sc = Scope(expected={"below"})
    sc.cover("below")
    sc.miss("cause", "no causal claim in this file")
    sc.finalize()                                  # the miss is not an expected kind
    assert "NOT checked" in sc.report() and "cause" in sc.report()


# ---------------------------------------------------------------------------- 5. artifacts
def test_ptxas_static_smem_is_not_smem():
    sch = parse_ptxas("ptxas info: Used 165 registers, 0 bytes smem, 0 bytes spill stores")
    assert sch["registers"] == 165 and sch["static_smem"] == 0
    with pytest.raises(Exception, match="dynamic_smem"):
        sch.require("dynamic_smem")


def test_engine_inspector_counts_partitioning():
    names = ["__myl_CorrMulLtSele", "__myl_CorrMulLtSele", "Reformat", "PWN(/body.1/PRelu)",
             "CaskConvolution", "__myl_TranReshMoveAdd"]
    sch = parse_engine_inspector(json.dumps([{"Name": n} for n in names]))
    assert sch["total"] == 6 and sch["fused_mylin"] == 3
    assert sch["reformat"] == 1 and sch["pointwise"] == 1


@pytest.mark.skipif(not MODEL_2X.exists(), reason="the 2x model is outside the repo")
def test_checkpoint_is_read_not_recalled():
    sch = parse_checkpoint(str(MODEL_2X))
    assert sch["tensors"] == 53 and sch["params"] == 600652
    assert sch["body_layers"] == 18                 # head + 16 body + tail
    assert len(sch["slope_tensors"]) == 17          # 1088 per-channel slopes
    assert sch["shapes"]["body.0.weight"] == (64, 3, 3, 3)


# ---------------------------------------------------------------------------- 6. pairs
def test_config_pair_requires_both_quantities():
    s = Store()
    mk(s, 24.88, unit="ms", stat="single", config="cfg-int8", extra={"quantity": "speed"})
    with pytest.raises(RecordError, match="accuracy"):
        require_config_pair(s, "cfg-int8")
    mk(s, 0.174, config="cfg-int8", extra={"quantity": "accuracy"})
    require_config_pair(s, "cfg-int8")


def test_shared_calibration_rejects_two_sets():
    s = Store()
    a = mk(s, 1.0, extra={"calibration": "cal-8frames"})
    b = mk(s, 1.0, extra={"calibration": "cal-16frames"})
    with pytest.raises(RecordError, match="different calibration"):
        shared_calibration(s, [a.id, b.id])
    b.extra["calibration"] = "cal-8frames"
    assert shared_calibration(s, [a.id, b.id]) == "cal-8frames"


def test_graph_change_needs_layer_counts():
    s = Store()
    base = mk(s, 56.28, unit="ms", stat="single", config="cfg-split",
              extra={"graph": {"total": 25, "reformat": 2, "pointwise": 1, "fused_mylin": 18}})
    new = mk(s, 71.0, unit="ms", stat="single", config="cfg-fitpad")
    with pytest.raises(RecordError, match="layer counts"):
        graph_delta(base, new)
    new.extra["graph"] = {"total": 26, "reformat": 3, "pointwise": 2, "fused_mylin": 17}
    assert graph_delta(base, new)["reformat"] == (2, 3)


def test_two_implementations_agreeing_to_five_digits_is_a_warning():
    s = Store()
    a = mk(s, 2.15278, stat="single", instrument="triton_kernel")
    b = mk(s, 2.15278, stat="single", instrument="trt_engine")
    assert suspicious_identity(a, b)
    c = mk(s, 2.15278, stat="single", instrument="triton_kernel")
    assert suspicious_identity(a, c) == ""


# ---------------------------------------------------------------------------- static checks
def test_dead_store_is_found():
    src = "def f(quantise, other):\n    buf = quantise(1)\n    buf = other(2)\n    return buf\n"
    found = dead_stores(src, "probe.py")
    assert len(found) == 1 and found[0].name == "buf" and found[0].first_line == 2


def test_dead_store_is_not_flagged_when_the_value_is_read():
    src = "def f(a, b):\n    x = a()\n    print(x)\n    x = b()\n    return x\n"
    assert dead_stores(src) == []


def test_use_before_assignment_is_found():
    src = "def f():\n    if cond():\n        pass\n    print(c2)\n    c2 = 1\n"
    found = used_before_assignment(src, "probe.py")
    assert found and found[0].name == "c2" and found[0].first_line == 4


# ---------------------------------------------------------------------------- migration of old findings
def _claims(tmp_path, text: str) -> str:
    p = tmp_path / "claims.txt"
    p.write_text(text)
    return str(p)


def test_migrate_flags_one_name_with_two_values(tmp_path):
    from harness.migrate import conflicts, migrate
    path = _claims(tmp_path, '@claim ratio_idle 1.489 x "one process, 1920x1080"\n'
                             '@claim ratio_idle 1.322 x "one process, 1920x1080"\n')
    found, _, _ = migrate([path])
    assert {f.status for f in found} == {"conflicted"}
    reuse, pop, _ = conflicts(found)
    assert reuse and reuse[0]["name"] == "ratio_idle"


def test_migrate_does_not_call_three_arms_a_contradiction(tmp_path):
    from harness.migrate import migrate
    path = _claims(tmp_path, '@claim fp16_window_ms 51.06 ms "one process, steady state"\n'
                             '@claim int8_window_ms 46.52 ms "one process, steady state"\n'
                             '@claim int8_window_ms2 42.75 ms "one process, steady state"\n')
    found, _, meta = migrate([path])
    assert {f.status for f in found} == {"unverifiable"}
    assert meta["pop_conflicts"] and not meta["name_reuse"]


def test_migrate_excludes_the_withdrawn_corpus(tmp_path):
    from harness.migrate import migrate
    d = tmp_path / "claims-example.txt"
    d.write_text('@claim margin 18.6 pp "Chainsaw, margin"\n')
    found, _, _ = migrate([str(d)])
    assert found[0].status == "withdrawn"
    assert "withdrawn by the source file" in found[0].missing


def test_migrate_gate_fails_while_a_relation_cites_an_unprotected_claim(tmp_path):
    from harness.migrate import migrate, report
    path = _claims(tmp_path, '@claim a 1.0 % "x"\n@claim b 2.0 % "x"\n@below a bar_share\n')
    found, bars, meta = migrate([path])
    text, failed = report(found, bars, meta)
    assert failed and "DECISION-BEARING AND UNPROTECTED" in text


def test_migrate_emits_assumed_records_that_cannot_be_evidence(tmp_path):
    from harness.migrate import emit_records, migrate
    from harness.gates import Claim, check_claims
    path = _claims(tmp_path, '@claim a 1.0 % "x"\n@claim b 2.0 % "x"\n')
    found, _, _ = migrate([path])
    st = emit_records(found, Store())
    assert {r.tier for r in st.records.values()} == {"assumed"}
    problems, _ = check_claims(st, [Claim("V1", "verdict", ("a", "b"))])
    assert problems and "assumed" in problems[0]


def test_migrate_protects_a_claim_when_a_record_matches(tmp_path):
    from harness.migrate import migrate
    path = _claims(tmp_path, '@claim a 1.0 % "x"\n')
    st = Store()
    st.emit(Record(value=1.0, unit="%", population=Population(kind="x", label="x"),
                   instrument="acceptance_stat", config="cfg", session=st.session,
                   statistic="worst_of", run="run-1"))
    found, _, _ = migrate([path], st)
    assert found[0].status == "protected"


def test_migrate_parses_a_signed_value_instead_of_dropping_the_line(tmp_path):
    """`@claim net_24min_now +31.8 s` was invisible to both parsers: the value pattern had no '+',
    and the file still reported PASS. A directive nobody reads is not a zero-finding."""
    from harness.migrate import migrate
    path = _claims(tmp_path, '@claim net_24min_now +31.8 s "a 24-minute episode now" derived\n')
    found, _, meta = migrate([path])
    assert [f.name for f in found] == ["net_24min_now"] and found[0].value == 31.8
    assert meta["unparsed"] == []


def test_migrate_fails_the_gate_on_a_directive_it_cannot_read(tmp_path):
    from harness.migrate import migrate, report
    path = _claims(tmp_path, '@claim broken 31.8 "no unit"\n')
    found, bars, meta = migrate([path])
    assert meta["unparsed"], "a line the parser cannot read must be reported"
    text, failed = report(found, bars, meta)
    assert failed and "NO PARSER READS" in text


def test_no_false_positive_on_a_lambda_parameter_or_a_closure():
    """`lambda _m, _i, out: ...` and a dict filled after the function that reads it are both correct
    Python; reporting them is how a checker teaches its reader to ignore it."""
    lam = "def f(x):\n    h = [lambda _m, _i, out, k=1: out for _ in x]\n    out = 1\n    return h, out\n"
    assert used_before_assignment(lam) == []
    closure = ("def main():\n"
               "    def body():\n"
               "        return cache.get('k')\n"
               "    cache = {}\n"
               "    return body()\n")
    assert used_before_assignment(closure) == []


# ------------------------------------------------ red-team residuals (F11/F17/F22 + noqa)
def test_measured_provenance_cannot_be_deleted_after_emit():
    s = Store()
    r = mk(s, 0.01, items=("1", "2"), extra={"of": "share"})   # n=2 so the sample rule cannot mask this
    r.run = ""
    r.artifact = ""
    with pytest.raises(RecordError, match="provenance was deleted"):
        r.validate()
    with pytest.raises(RecordError, match="provenance was deleted"):
        below(r, Bar(name="b", value=0.1, unit="%", of="share", statistic="worst_of"))


def test_a_parse_that_produced_nothing_is_not_claimable():
    s = run(["echo", "ptxas info: nothing here"], parse=parse_ptxas)
    assert s.status == "ok" and s.parsed.missing
    assert not claimable(s)


def test_verdict_validates_its_records():
    s = Store()
    a = s.emit(Record(value=1.0, unit="%", population=Population.of("p", ["1", "2"]), instrument="i",
                      config="c", session=s.session, statistic="single", artifact="/nonexistent/x",
                      extra={"artifact_sha256": "deadbeef"}))
    b = s.emit(Record(value=2.0, unit="%", population=Population.of("p", ["1", "2"]), instrument="i",
                      config="c", session=s.session, statistic="single", artifact="/nonexistent/y",
                      extra={"artifact_sha256": "deadbeef"}))
    a.extra.pop("artifact_sha256")            # provenance edited after emit: the verdict must notice
    problems, _ = check_claims(s, [Claim("V", "verdict", (a.id, b.id))])
    assert problems and "digest" in problems[0]


def test_a_forged_aa_object_cannot_unlock_a_refusal():
    from harness.gates import AA
    s = Store()
    base = mk(s, 1.0, items=("1", "2"))
    forged = AA(identical=True, max_delta=0.0, n=2, record=base)
    with pytest.raises(RecordError):
        refuse(s, "C", base.id, forged, "no A/A was ever run")


def test_a_bare_noqa_does_not_silence_a_check():
    bare = "def f(a, b):\n    x = a()  # noqa\n    x = b()\n    return x\n"
    specific = "def f(a, b):\n    x = a()  # noqa: dead-store\n    x = b()\n    return x\n"
    wrong = "def f(a, b):\n    x = a()  # noqa: used-before-assignment\n    x = b()\n    return x\n"
    assert len(dead_stores(bare)) == 1
    assert len(dead_stores(specific)) == 0
    assert len(dead_stores(wrong)) == 1


def test_a_record_that_cannot_be_evidence_is_not_stored():
    s = Store()
    with pytest.raises(RecordError, match="digest"):
        s.emit(Record(value=1.0, unit="%", population=Population.of("p", ["1"]), instrument="i",
                      config="c", session=s.session, statistic="single",
                      artifact="/nonexistent/capture-never-taken.txt"))
    assert not s.records
