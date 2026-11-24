"""The seek profile's join: five stages from two processes, and a one-sided generation is never silently dropped.

Why these pins exist (owner, 2026-11-25): "seeking is extremely slow" could not be answered because each log held one
end of the measurement. The pane's `turnaround_ms` is a sum; the backend's `backend_total` starts when the renderer
already acted. Between them sit the wait for the renderer's loop and the wait for the pane's poll, and neither process
can see the other's clock monothonically - so both stamp `time.time()` and the join is arithmetic on the generation.

The numbers below are synthetic on purpose: this pins the JOIN, not a run. A real run is measured by the supervisor on
a leased instance.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "harness" / "scripts"))

import seek_profile  # noqa: E402


PANE = """\
preview: seek decide press=7 route=display target_s=5.972 target_frame=143 produced=0.3192 edge_s=6.590 answerable=1 ahead=0 generation=3 ts=1000.000
preview: seek turnaround_ms=578.2 target_frame=143 press=7 target_s=5.972 landed_seq=122 landed_time=5.972 delta_s=+0.000 target_gen=3 landed_gen=3 ts=1000.5782
preview: seek decide press=8 route=display target_s=10.237 target_frame=245 produced=0.3475 edge_s=7.174 answerable=1 ahead=1 generation=4 ts=1010.000
preview: seek turnaround_ms=800.0 target_frame=245 press=8 target_gen=4 landed_gen=4 ts=1010.800
preview: seek decide press=9 route=tail target_s=1.000 generation=5 ts=1020.000
"""

BACKEND = """\
seek: requested 5.972s (generation 3) from the running render
seek: displayed the frame at 5.972s (generation 3, not written to the export); stages reader=1.0ms decode=120.0ms bytes=2.0ms model=300.0ms publish=4.0ms backend_total=427.0ms pickup_wall=1000.100
seek: displayed the frame at 10.237s (generation 4, not written to the export); stages reader=0.5ms decode=90.0ms bytes=1.5ms model=250.0ms publish=3.0ms backend_total=345.0ms pickup_wall=1010.250
"""


def test_the_five_stages_of_one_seek_are_arithmetic(tmp_path):
    pane = tmp_path / "preview-log.txt"
    backend = tmp_path / "log.txt"
    pane.write_text(PANE, encoding="utf-8")
    backend.write_text(BACKEND, encoding="utf-8")

    assert seek_profile.main(["--pane", str(pane), "--backend", str(backend), "--json"]) == 0
    rows = seek_profile.join(*seek_profile.parse_pane(pane.read_text(encoding="utf-8").splitlines()),
                             seek_profile.parse_backend(backend.read_text(encoding="utf-8").splitlines()))
    by_gen = {row["generation"]: row for row in rows}

    served = by_gen[3]
    assert served["press"] == 7
    assert served["press_to_pickup_ms"] == pytest.approx(100.0), \
        "0.1 s passed between the press and the renderer acting on it"
    assert served["turnaround_ms"] == 578.2
    assert served["publish_to_paint_ms"] == pytest.approx(578.2 - 427.0 - 100.0), \
        "what is left after the queue wait and the backend's own work is the pane's poll"
    assert (served["reader"], served["decode"], served["bytes"], served["model"], served["publish"]) == \
        (1.0, 120.0, 2.0, 300.0, 4.0), "the model is its own stage, not buried in prepare"
    assert served["unmatched"] == []

    # Generation 4 was served and closed too, with its own poll; the last row is the one that has no backend.
    assert by_gen[4]["unmatched"] == []
    assert by_gen[4]["turnaround_ms"] == pytest.approx(800.0)
    assert by_gen[4]["publish_to_paint_ms"] == pytest.approx(800.0 - 345.0 - 250.0)
    # Generation 5 was a tail seek: the display channel never answers it, so no backend row exists - and that is a
    # different fact from "the backend was slow".
    assert by_gen[5]["unmatched"] == ["backend", "close"]


def test_a_backend_log_without_any_seek_is_a_finding_not_an_empty_answer(tmp_path):
    pane = tmp_path / "preview-log.txt"
    backend = tmp_path / "log.txt"
    pane.write_text(PANE, encoding="utf-8")
    backend.write_text("nothing about seeks here\n", encoding="utf-8")

    assert seek_profile.main(["--pane", str(pane), "--backend", str(backend)]) == 2


def test_nothing_to_join_fails_instead_of_printing_an_empty_table(tmp_path):
    pane = tmp_path / "preview-log.txt"
    pane.write_text("preview: stats frames=0 paints=10\n", encoding="utf-8")
    assert seek_profile.main(["--pane", str(pane), "--backend", str(pane)]) == 1
