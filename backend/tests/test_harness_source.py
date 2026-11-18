"""The source check: a false population label is a checkable fact, not a judgement call.

The failure this covers: 2026-11-15..18, every content number in the int8/gate lane was measured on
h264 re-encodes at 4.04/3.98 Mbps while the prose called them "the BD remaster" (37.2-37.8 Mbps).
`harness.cli check` was green throughout, because it hashes the population and does not read it.

CPU only, no ffprobe required: the prober and the proof are injected.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from harness import source  # noqa: E402


def facts(mbps: float, packets: int = 480, **fields):
    base = {"pix_fmt": "yuv420p", "color_range": "", "has_b_frames": "4", "codec_name": "h264"}
    base.update(fields)
    return source.Facts(bit_rate_mbps=mbps, packets=packets, fields=base)


REAL = {
    "bd": facts(37.715, 495),
    "reencode": facts(4.038, 480, color_range="tv", has_b_frames="2"),
}


def manifest(**over):
    clip = {"file": "bd.mkv", "label": "BDRemux stream copy", "must_exist": True}
    clip.update(over.pop("clip", {}))
    m = {"min_bitrate_mbps": 35.0, "clips": [clip]}
    m.update(over)
    return m


def test_a_reencode_under_a_bd_label_is_refused(tmp_path):
    (tmp_path / "bd.mkv").write_bytes(b"x")
    problems, rows = source.check(manifest(), str(tmp_path), probe_fn=lambda p: REAL["reencode"])
    assert rows[0]["checked"] is True
    assert any("4.04 Mbps is below the 35.0" in p for p in problems)


def test_the_same_file_is_accepted_at_bd_rate(tmp_path):
    (tmp_path / "bd.mkv").write_bytes(b"x")
    problems, _ = source.check(manifest(), str(tmp_path), probe_fn=lambda p: REAL["bd"])
    assert problems == []


def test_a_contaminated_clip_is_not_refused_for_declaring_its_own_rate(tmp_path):
    (tmp_path / "bd.mkv").write_bytes(b"x")
    m = manifest(clip={"contaminated": True, "label": "4 Mbps re-encode"})
    problems, rows = source.check(m, str(tmp_path), probe_fn=lambda p: REAL["reencode"])
    assert problems == []
    assert rows[0]["contaminated"] is True


def test_a_manifest_that_disagrees_with_the_file_is_refused(tmp_path):
    (tmp_path / "bd.mkv").write_bytes(b"x")
    m = manifest(clip={"facts": {"bit_rate_mbps": 37.715, "packets": 481}})
    problems, _ = source.check(m, str(tmp_path), probe_fn=lambda p: REAL["bd"])
    assert any("manifest says packets=481" in p for p in problems)


def test_an_absent_file_is_not_checked_and_not_a_pass(tmp_path):
    problems, rows = source.check(manifest(), str(tmp_path), probe_fn=lambda p: REAL["bd"])
    assert rows[0]["checked"] is False
    assert "NOT CHECKED" in rows[0]["status"]
    assert any("must_exist" in p for p in problems)


def test_a_stream_copy_claim_needs_a_proof_block_when_not_reproved(tmp_path):
    (tmp_path / "bd.mkv").write_bytes(b"x")
    m = manifest(clip={"cut": {"mode": "stream_copy"}})
    problems, _ = source.check(m, str(tmp_path), probe_fn=lambda p: REAL["bd"])
    assert any("cut block is incomplete" in p for p in problems)


def test_reproved_offset_must_match_the_manifest(tmp_path, monkeypatch):
    (tmp_path / "bd.mkv").write_bytes(b"x")
    src = tmp_path / "src.mkv"
    src.write_bytes(b"x")
    m = manifest(clip={"source": str(src),
                       "cut": {"mode": "stream_copy", "offset": 14376, "compared": 494,
                               "packets": 495}})
    monkeypatch.setattr(source, "packet_sizes", lambda p, ffprobe="ffprobe":
                        [7] + [1, 2, 3] * 20 if "src" in str(p) else [9, 1, 2, 3])
    # the clip's packets never appear in the source at 14376
    problems, rows = source.check(m, str(tmp_path), probe_fn=lambda p: REAL["bd"], prove=True)
    assert any("packet sequence does not appear" in p for p in problems)


def test_packet_subsequence_finds_the_offset():
    clip = [100] + [i % 7 for i in range(60)]
    src = [5, 6, 7, 8] + clip
    calls = {"n": 0}

    def fake_sizes(path, ffprobe="ffprobe"):
        calls["n"] += 1
        return src if calls["n"] == 2 else clip

    import harness.source as mod
    old = mod.packet_sizes
    mod.packet_sizes = fake_sizes
    try:
        proof = mod.stream_copy_proof("clip", "src")
    finally:
        mod.packet_sizes = old
    assert proof["ok"] is True
    assert proof["offset"] == 4      # the clip's first packet is the source's 5th
    assert proof["compared"] == len(clip)


def test_contamination_propagates_by_item_set_and_respects_name_boundaries(tmp_path):
    recs = tmp_path / "r.jsonl"
    def rec(rid, items):
        return json.dumps({"id": rid, "population": {"items": items}}) + "\n"
    recs.write_text(
        rec("R1", ["violet20s.mkv#frame0", "violet20s.mkv#frame1"]) +
        rec("R2", ["bd_violet20s.mkv#frame0"]) +          # the BD cut CONTAINS the old name
        rec("R3", ["/tmp/opencode/violet20s.mkv#frame7"]) +
        rec("R4", ["body-1#0"]))
    m = {"clips": [{"file": "violet20s.mkv", "contaminated": True}]}
    hits = source.contaminated_records(m, str(recs))
    ids = [h[2].split()[0] for h in hits]
    assert ids == ["R1", "R3"]


def test_a_clean_manifest_yields_no_contamination(tmp_path):
    m = {"clips": [{"file": "bd.mkv", "contaminated": False}]}
    assert source.contaminated_records(m, str(tmp_path / "*.jsonl")) == []
