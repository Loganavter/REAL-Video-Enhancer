"""The press ticket: one decision and one outcome per press, and silence is not a state it can be in.

Why these pins exist (owner, 2026-11-24): the preview log could name where a seek was asked to go and - on the happy
path - where it landed, but three of his own sessions left NO line about the result at all while the pane did change.
A press could end in silence, so the log could not answer "which frame did we draw". The ticket is the object that
makes that impossible: it is created per press, it states its route and whether that route can answer, and every
terminal path goes through one of `settled`/`refused`/`superseded`/`abandoned`.
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from src.ui.video import telemetry  # noqa: E402
from src.ui.video.press import PressTicket  # noqa: E402


class _Lines:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []
        self.seeks: list[tuple[float, dict]] = []

    def event(self, kind, **fields):
        self.events.append((kind, fields))

    def note_seek(self, turnaround_ms, **fields):
        self.seeks.append((turnaround_ms, fields))


def _capture(monkeypatch) -> _Lines:
    lines = _Lines()
    monkeypatch.setattr("src.ui.video.press.telemetry.event", lines.event)
    monkeypatch.setattr("src.ui.video.press.telemetry.note_seek", lines.note_seek)
    return lines


def test_a_press_states_its_route_and_whether_that_route_can_answer_it(monkeypatch):
    lines = _capture(monkeypatch)
    ticket = PressTicket(target_s=5.0, target_frame=50, produced=0.0, edge_s=0.0)

    ticket.decide(route="display", answerable=False, why="target-past-produced-edge", generation=3)

    kind, fields = lines.events[-1]
    assert kind == "seek decide"
    assert fields["press"] == ticket.id
    assert fields["route"] == "display"
    assert fields["target_s"] == "5.000"
    assert fields["produced"] == "0.0000"
    assert (fields["answerable"], fields["why"], fields["generation"]) == (0, "target-past-produced-edge", 3)
    assert ticket.open, "a decision alone does not end the press"


def test_the_answer_names_the_picture_it_settled_on(monkeypatch):
    lines = _capture(monkeypatch)
    ticket = PressTicket(target_s=1.888, target_frame=45)
    ticket.decide(route="display", answerable=True, generation=7)

    assert ticket.settled(landed=("seq", 99), landed_time=1.888, turnaround_ms=578.2, landed_gen=7) is True

    turnaround, fields = lines.seeks[-1]
    assert turnaround == 578.2
    assert fields["press"] == ticket.id, "the outcome points back at the decision"
    assert (fields["landed_time"], fields["target_gen"], fields["landed_gen"]) == (1.888, 7, 7)
    assert not ticket.open
    assert ticket.settled(landed=("seq", 100), landed_time=1.9, turnaround_ms=1.0, landed_gen=7) is False, \
        "a second picture cannot settle a press twice"


def test_every_terminal_path_is_named_and_only_one_of_them_happens(monkeypatch):
    lines = _capture(monkeypatch)
    refused = PressTicket(target_s=9.0, target_frame=90)
    refused.decide(route="display", answerable=False, why="target-past-produced-edge")
    assert refused.refused("target-past-produced-edge") is True
    assert lines.events[-1][0] == "seek refused" and lines.events[-1][1]["press"] == refused.id
    assert refused.settled(landed=None, landed_time=None, turnaround_ms=1.0, landed_gen=None) is False
    assert refused.abandoned("late") is False, "a press that already ended cannot end again"

    abandoned = PressTicket(target_s=1.0, target_frame=10)
    abandoned.decide(route="tail", answerable=True)
    assert abandoned.abandoned("no-picture-from-the-tail") is True
    assert lines.events[-1][0] == "seek abandoned"

    replaced = PressTicket(target_s=2.0, target_frame=20)
    replaced.decide(route="display", answerable=True, generation=1)
    assert replaced.superseded(99) is True
    assert lines.events[-1][0] == "seek superseded" and lines.events[-1][1]["by"] == 99
    assert replaced.settled(landed=None, landed_time=None, turnaround_ms=1.0, landed_gen=1) is False, \
        "a replaced press must not also close: one outcome, always"


def test_a_streaming_picture_is_named_once_per_press_and_is_not_an_outcome(monkeypatch):
    """The line that was entirely missing: the pane painted the newest upscaled frame while a press was open."""
    lines = _capture(monkeypatch)
    ticket = PressTicket(target_s=1.888, target_frame=45)
    ticket.decide(route="display", answerable=True, generation=7)

    assert ticket.note_stream(seq=98, media_time=4.046, landed_gen=None) is True
    assert ticket.note_stream(seq=99, media_time=4.05, landed_gen=None) is False, "one line per press, not per frame"
    kind, fields = lines.events[-1]
    assert kind == "seek stream"
    assert (fields["seq"], fields["t"], fields["not_the_answer"]) == (98, "4.046", 1)
    assert ticket.open, "a streaming frame does not end the press - only its own answer does"
