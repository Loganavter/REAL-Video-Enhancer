"""A pair is counted where the DECISION is made, and the seam has exactly one writer.

Two infrastructure pieces from `projects/rve/docs/notes/blinktable-infrastructure-2026-11-27.md`:

1. **A counter at the commit.** The pane used to count pairs at PAINT time. During the 2026-11-27 defect every result
   was held (no source presented, `held_block=24`), so no paint carried a pair - and the stats looked normal, because
   paints of a stale pair are still paints. `pair_commits` is incremented in `PairWriter._paint`, the one place a pair
   becomes the picture on screen, so "results are advancing and no pair is committed" is one field, not an inference.
2. **Ownership by existence.** `offer_result`/`offer_source` may be called from the pane and nowhere else: a second
   writer is the class the refactor exists to make impossible (CODE_PATTERNS §2, §5).
"""

from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("PySide6")

from src.ui.preview.pair import PairWriter, Half  # noqa: E402
from src.ui.video import telemetry  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]


class Sink:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def set_frame(self, frame) -> None:
        self.calls.append(("result", frame))

    def set_original(self, frame) -> None:
        self.calls.append(("source", frame))


@pytest.fixture(autouse=True)
def fresh():
    telemetry.set_writer(lambda line, which: None)
    telemetry.set_clock(lambda: 0.0)
    telemetry._reset(session=True)
    yield
    telemetry.set_writer(None)
    telemetry.set_clock(None)


def _writer() -> PairWriter:
    writer = PairWriter(Sink(), clock=lambda: 0.0)
    writer.set_view_mode("sbs")
    writer.set_tolerance(0.0)
    return writer


def test_a_committed_pair_increments_the_commit_counter():
    writer = _writer()
    writer.offer_result(Half(frame="r", moment=1.0, generation=0, name=("frame", 1)))
    writer.offer_source(Half(frame="s", moment=1.0, generation=0, name=("frame", 1)))

    assert writer.paint() is True

    assert telemetry.snapshot()["pair_commits"] == 1


def test_a_held_result_is_not_a_commit():
    """The defect's shape: the result is offered every frame, no source ever arrives, nothing is committed."""
    writer = _writer()
    writer.offer_result(Half(frame="r", moment=1.0, generation=0, name=("frame", 1)))

    assert writer.paint() is False

    assert telemetry.snapshot()["pair_commits"] == 0, "a hold must never read as a committed pair"


def test_a_lone_half_is_not_a_commit():
    writer = _writer()
    writer.offer_result(Half(frame="r", moment=1.0, generation=0, name=("frame", 1)))
    # No source and the hold expired: the writer paints the result alone, which is a decision but not a pair.
    writer._hold_started = -10_000.0
    writer.paint()

    assert telemetry.snapshot()["pair_commits"] == 0


def test_the_pair_writer_has_exactly_one_caller():
    """A second caller of `offer_result`/`offer_source` is a second owner of the picture - the class this seam kills."""
    offenders = []
    for path in (ROOT / "src").rglob("*.py"):
        if path.name == "pair.py":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "_pair.offer_result(" in text and path.name != "video_preview.py":
            offenders.append(f"{path.relative_to(ROOT)} calls offer_result")
        if "_pair.offer_source(" in text and path.name != "video_preview.py":
            offenders.append(f"{path.relative_to(ROOT)} calls offer_source")
    assert not offenders, "the pair writer must have one caller (the pane):\n" + "\n".join(offenders)
