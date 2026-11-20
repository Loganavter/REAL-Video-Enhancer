"""A render pair must declare its reference, or the number cannot say what it measures.

The failure this pins: on 2026-11-18 two int8 renders were compared against an FP16 ENGINE render and the
result was read as the quantiser's error. It is not - the engine deviates from the model by 0.055-0.087%
and the colour pipe alone moves 0.21% of pixels past the 8-level bar, while the quantiser against the MODEL
is 0.004-0.019%. The tool now refuses to guess and the records carry the declaration.

CPU only: the checker and the emitter's contract need no GPU, which is the point - the guard has to run on a
machine that has nothing loaded.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "harness" / "scripts"))


def test_an_undeclared_pair_is_refused():
    sweep = pytest.importorskip("gate_sweep")
    with pytest.raises(SystemExit) as exc:
        sweep.declare_reference("undeclared", force=False)
    assert "reference" in str(exc.value).lower()


def test_a_forced_undeclared_pair_says_it_cannot_say_anything():
    sweep = pytest.importorskip("gate_sweep")
    note = sweep.declare_reference("undeclared", force=True)
    assert "cannot say" in note.lower()


def test_engine_reference_is_declared_as_the_pair_not_the_quantiser():
    sweep = pytest.importorskip("gate_sweep")
    note = sweep.declare_reference("engine-render")
    assert "PAIR" in note and "0.21" in note


def test_emitted_records_carry_the_reference_kind():
    sweep = pytest.importorskip("gate_sweep")
    rows = [{"frame": 0, "share": 0.0, "block_mean": 0.0, "peak": 0.0, "sum_abs": 0}]
    s = sweep.summary(rows, block=64, bar=0.1)
    args = SimpleNamespace(clip_name="clip.mkv", label="l", session="s", run="r", artifact="",
                          instrument="channel-max |delta| in RGB", config="c", bar=0.1,
                          id_prefix="Z", reference_kind="model", force_undeclared=False)
    recs = sweep.emit_records(rows, s, args, digest="0" * 64)
    assert all(r["extra"]["reference_kind"] == "model" for r in recs)
    assert all("reference: model" in r["instrument"] for r in recs)


def test_the_checker_sees_an_undeclared_record(tmp_path):
    checker = pytest.importorskip("check_reference_declarations")
    good = tmp_path / "good.jsonl"
    bad = tmp_path / "bad.jsonl"
    common = {"instrument": "channel-max |delta| in RGB, 64px blocks", "id": "X001"}
    good.write_text(json.dumps({**common, "extra": {"reference_kind": "model"}}) + "\n")
    bad.write_text(json.dumps({**common, "extra": {}}) + "\n")
    assert checker.undeclared([str(good)]) == []
    assert len(checker.undeclared([str(bad)])) == 1
    assert checker.main(["--records", str(bad), "--quiet"]) == 1
    assert checker.main(["--records", str(good), "--quiet"]) == 0
