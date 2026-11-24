"""The evidence gate: a record's artifact is a file in this tree, and its digest still matches.

Why this exists (owner, 2026-11-24): the records used to cite `~/rve-scratch/...`, a 21 GB directory on one machine -
"что это за реестр доказательств на чьём-то диске". A record that points outside the repository stops being evidence
the moment that directory is cleaned. The gate makes the next outside path, missing file or changed-file-since-the-record
loud, and the baseline is the accepted debt it may only shrink from.

Three halves, on purpose:
1. the tree passes with today's baseline (`48` known entries: 24 files outside the repository, 24 whose content changed
   after the record was written);
2. a synthetic violation FAILS - a gate that cannot fail is a comment;
3. a baseline entry that no longer reproduces is REPORTED, because leaving it there trains everyone to ignore the gate.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "harness" / "scripts"))

import check_artifacts  # noqa: E402


def _records() -> pathlib.Path:
    return ROOT / "harness" / "docs" / "wsp" / "verified" / "records"


def test_the_shipped_tree_passes_the_gate():
    assert check_artifacts.main([]) == 0


def test_a_changed_artifact_fails_the_gate(monkeypatch, tmp_path):
    """A record whose digest no longer matches its file must fail, unless it is in the baseline."""
    verified = tmp_path / "harness" / "docs" / "wsp" / "verified"
    records = verified / "records"
    records.mkdir(parents=True)
    artifact = verified / "artifact.txt"
    artifact.write_text("what was read\n", encoding="utf-8")
    relative = "harness/docs/wsp/verified/artifact.txt"      # a repo-relative path: the only kind the gate accepts
    wrong = "0" * 64
    (records / "lane.jsonl").write_text(
        json.dumps({"id": "X1", "artifact": relative, "extra": {"artifact_sha256": wrong}}) + "\n",
        encoding="utf-8")
    monkeypatch.setattr(check_artifacts, "ROOT", tmp_path)
    monkeypatch.setattr(check_artifacts, "VERIFIED", verified)
    monkeypatch.setattr(check_artifacts, "RECORDS", records)
    monkeypatch.setattr(check_artifacts, "BASELINE", verified / "artifacts-baseline.json")

    found = check_artifacts.findings()
    assert [entry["rule"] for entry in found] == ["artifact-stale"], found
    assert check_artifacts.main([]) == 1, "a new stale artifact must fail the gate"

    check_artifacts.write_baseline(found)
    assert check_artifacts.main([]) == 0, "the baseline accepts today's known entry"
    assert "stale" in (verified / "artifacts-baseline.json").read_text(encoding="utf-8")


def test_an_absolute_path_is_a_violation(monkeypatch, tmp_path):
    """The class this gate exists for: an artifact on someone's disk, not in the tree."""
    verified = tmp_path / "verified"
    records = verified / "records"
    records.mkdir(parents=True)
    (records / "lane.jsonl").write_text(
        json.dumps({"id": "X2", "artifact": "/home/someone/scratch/render.mkv"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(check_artifacts, "ROOT", tmp_path)
    monkeypatch.setattr(check_artifacts, "VERIFIED", verified)
    monkeypatch.setattr(check_artifacts, "RECORDS", records)
    found = check_artifacts.findings()
    assert [entry["rule"] for entry in found] == ["artifact-outside-repo"], found


def test_a_digest_in_the_artifact_field_is_not_a_path(monkeypatch, tmp_path):
    """Nine records carried a bare sha256 in `artifact`: the field is a path, the digest has its own key."""
    verified = tmp_path / "verified"
    records = verified / "records"
    records.mkdir(parents=True)
    (records / "lane.jsonl").write_text(
        json.dumps({"id": "X3", "artifact": "a" * 64}) + "\n", encoding="utf-8")
    monkeypatch.setattr(check_artifacts, "ROOT", tmp_path)
    monkeypatch.setattr(check_artifacts, "VERIFIED", verified)
    monkeypatch.setattr(check_artifacts, "RECORDS", records)
    found = check_artifacts.findings()
    assert [entry["rule"] for entry in found] == ["artifact-not-a-path"], found


def test_an_empty_corpus_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(check_artifacts, "RECORDS", tmp_path / "nowhere")
    assert check_artifacts.main([]) == 2, "'nothing to check' must never print as a pass"


def test_the_baseline_has_no_bare_digests_left():
    """The nine records that named a digest instead of a file were resolved in the 2026-11-24 cleanup."""
    data = json.loads((ROOT / "harness" / "docs" / "wsp" / "verified" / "artifacts-baseline.json")
                      .read_text(encoding="utf-8"))
    for item in data["known"]:
        assert not (len(item["artifact"]) == 64 and "/" not in item["artifact"]), item
