"""Locator resolution: the evidence corpus, and the plane that no longer lives here.

The failure these exist for: documents were moved out of this repository into the framework's
knowledge plane, and a locator that pointed at the old place silently stopped resolving - a verdict
whose evidence does not exist reads as stale, and a comment that names a missing path sends the next
reader to a 404. ``resolve`` is where both are decided, so it is tested directly.
"""
from __future__ import annotations

from pathlib import Path

from harness import paths


def test_doc_is_anchored_to_the_package() -> None:
    assert paths.doc("wsp/verdicts.yaml") == str(paths.DOCS / "wsp/verdicts.yaml")


def test_absolute_locator_passes_through() -> None:
    assert paths.resolve("/tmp/x.txt") == "/tmp/x.txt"


def test_old_docs_prefix_still_means_the_harness_corpus() -> None:
    assert paths.resolve("docs/wsp/verdicts.yaml") == str(paths.DOCS / "wsp/verdicts.yaml")


def test_unknown_relative_locator_is_left_alone() -> None:
    assert paths.resolve("no/such/path.txt") == "no/such/path.txt"


def test_plane_locator_resolves_into_the_framework_checkout(tmp_path, monkeypatch) -> None:
    plane = tmp_path / "SLI-framework"
    note = plane / "projects/rve/docs/notes/PRIOR-ART-reuse-detail-2026-11-20.md"
    note.parent.mkdir(parents=True)
    note.write_text("moved", encoding="utf-8")
    monkeypatch.setattr(paths, "FRAMEWORK", plane)
    locator = "projects/rve/docs/notes/PRIOR-ART-reuse-detail-2026-11-20.md"
    assert Path(paths.resolve(locator)) == note


def test_plane_locator_stays_as_written_when_the_checkout_is_absent(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(paths, "FRAMEWORK", tmp_path / "missing")
    locator = "projects/rve/docs/notes/PRIOR-ART-reuse-detail-2026-11-20.md"
    assert paths.resolve(locator) == locator
