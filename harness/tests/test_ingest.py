"""A run report becomes records, so a render's cost can be compared across commits.

The point of these tests is the shape, not a number: every phase the backend names must arrive as a record carrying
the commit and the card from the report, and a report without a gate must not invent a reuse record.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.ingest import records_from  # noqa: E402

REPORT = {
    "provenance": {"measured_at": "2026-11-20T13:00:00+03:00", "commit": "abc123", "dirty": False,
                   "device": "NVIDIA GeForce RTX 4060 Ti", "capability": "8.9", "driver": "590.1",
                   "torch": "2.9.0", "triton": "3.6.0", "tensorrt": "10.12", "model_sha": "deadbeef"},
    "run": {"input": "clip.mkv", "output": "out.mkv", "backend": "tensorrt", "model": "model.pth",
            "precision": "float16", "frames": 100},
    "settings": {"temporal_reuse": True, "temporal_reuse_correction": True},
    "phases_ms_per_frame": {"gate": 0.4, "up_wall": 31.2, "pack": 3.0, "read": 0.5, "frame": 35.1},
    "gate": {"processed": 100, "reused": 62, "reuse_pct": 62.0, "depth_histogram": {"1": 40, "2": 22},
             "vetoes": 0, "scene_cuts": 1},
    "autotuner": None,
}


def test_every_named_phase_becomes_a_record():
    records = records_from(REPORT, start=900)
    units = {r.unit for r in records}
    phases = [r for r in records if r.unit == "ms"]
    assert len(phases) == 4, [r.instrument for r in phases]
    assert units == {"ms", "%"}


def test_the_provenance_travels_with_every_record():
    for record in records_from(REPORT, start=900):
        assert record.commit == "abc123"
        assert record.device.startswith("NVIDIA")
        assert record.capability == "8.9"
        assert record.measured_at.startswith("2026-11-20")


def test_the_reuse_rate_is_recorded_and_a_gateless_run_does_not_invent_one():
    with_gate = records_from(REPORT, start=900)
    assert any(r.unit == "%" and abs(r.value - 62.0) < 1e-9 for r in with_gate)
    without = json.loads(json.dumps(REPORT))
    without["gate"] = None
    assert not any(r.unit == "%" for r in records_from(without, start=900))


def test_ids_do_not_collide_between_ingests():
    first = records_from(REPORT, start=900)
    second = records_from(REPORT, start=900 + len(first))
    assert not ({r.id for r in first} & {r.id for r in second})
