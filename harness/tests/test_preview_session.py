"""The preview press session: ONE clamp, one ticket, one epoch - proved with a fake clock, no window.

What plan P1 changed (plan-rve-preview-architecture-2026-11-26 §Phases): a strip press used to be clamped twice - once
in `PreviewController._seek` against `PreviewState.seekable_ceiling()` and again in `ProcessTab._on_preview_seek`
against the page's own notion of the produced band - and the press record that makes a seek legible lived on the page.
`PreviewSession` is now the only place a target is compared with the reach and the only owner of the ticket.

Pinned here, in numbers:

* a seek inside the reach opens EXACTLY ONE ticket and asks the clock ONCE;
* a seek past the reach returns False, opens NOTHING and names the reach (never a silent clamp);
* a negative seek is 0 (the strip's left edge, not a refusal);
* a painted frame within two frames (floor 50 ms) of the target CLOSES the ticket; one outside it does not;
* a JobChanged clears it (P4 adds the deadline that calls the same `abandon()`);
* two seeks in a row: the newest ticket wins, and only the newest target can close it;
* snapshot/dump carry the epoch and the target for the debug layer (P5 serves `dump()`).

No Qt and no window: the module under test is Qt-free by construction, and a source check below keeps it that way.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

import src.ui.preview.session as session_module
from src.ui.preview.session import JobChanged, Painted, PreviewSession
from src.ui.preview.state import PreviewCommand

CLIP_S = 20.0
REACH_S = 12.0
FRAME_S = 1.0 / 24.0           # tolerance = max(0.05, 2 / 24) ~= 0.0833 s


class FakeClock:
    """The pane's clock as the session sees it: `seek_to(seconds) -> bool`, nothing else."""

    def __init__(self, accepts: bool = True) -> None:
        self.calls: list[float] = []
        self.accepts = bool(accepts)

    def seek_to(self, seconds: float) -> bool:
        self.calls.append(float(seconds))
        return self.accepts


def _session(clock: FakeClock | None = None, *, reach: float = REACH_S,
             duration: float = CLIP_S, frame_interval: float = FRAME_S) -> PreviewSession:
    session = PreviewSession()
    session.bind(clock=clock or FakeClock(), duration=duration, reach=reach, frame_interval=frame_interval)
    return session


def _spy_tickets(monkeypatch) -> list:
    """Record every record the session opens (the real `PressTicket`, so `open`/`route` are its own)."""
    made: list = []
    original = session_module.PressTicket

    def factory(**kwargs):
        ticket = original(**kwargs)
        made.append(ticket)
        return ticket

    monkeypatch.setattr(session_module, "PressTicket", factory)
    return made


# --- the clamp is here and only here -----------------------------------------------------------------


def test_a_seek_inside_the_reach_opens_one_ticket_and_asks_the_clock_once(monkeypatch):
    clock = FakeClock()
    session = _session(clock)
    made = _spy_tickets(monkeypatch)

    assert session.seek_to(5.0) is True

    assert clock.calls == [pytest.approx(5.0)], "the clock is asked exactly once, with the target"
    assert len(made) == 1, f"one press, one record, got {len(made)}"
    ticket = made[0]
    assert session.ticket is ticket, "the session owns the record"
    assert ticket.open is True
    assert ticket.target_s == pytest.approx(5.0)
    assert session.epoch == 1


def test_a_seek_past_the_reach_is_refused_opens_nothing_and_names_the_reach(monkeypatch):
    clock = FakeClock()
    session = _session(clock)
    made = _spy_tickets(monkeypatch)

    assert session.seek_to(REACH_S + 5.0) is False

    assert clock.calls == [], "a refused press never reaches the clock"
    assert made == [], "and it opens no ticket, so the pane cannot claim a time nothing can show"
    assert session.ticket is None
    refusal = session.refusal() or ""
    assert f"{REACH_S:.3f}" in refusal, f"the reason names the reach, got {refusal!r}"
    assert session.refused_for_reach() is True
    assert session.dump()["refusal"] == refusal, "P5 serves the same reason from dump()"
    assert session.epoch == 0


def test_a_negative_seek_is_the_left_edge_not_a_refusal(monkeypatch):
    clock = FakeClock()
    session = _session(clock)
    made = _spy_tickets(monkeypatch)

    assert session.seek_to(-3.0) is True

    assert clock.calls == [0.0], "the clamp is to 0.0, and it happens before the clock sees the number"
    assert len(made) == 1 and made[0].target_s == 0.0
    assert session.snapshot()["target"] == 0.0


# --- the ticket's life ---------------------------------------------------------------------------------


def test_a_painted_frame_within_the_tolerance_closes_the_ticket():
    session = _session()
    assert session.seek_to(5.0) is True
    ticket = session.ticket

    session.observe(Painted(media_time=5.0 + 2.0 * FRAME_S / 2.0, index=121))

    assert ticket is not None and ticket.open is False, "the answer closes the press"
    assert session.snapshot()["landed"] == pytest.approx(5.0 + FRAME_S)
    assert session.snapshot()["open"] is False


def test_a_painted_frame_outside_the_tolerance_does_not():
    session = _session()
    assert session.seek_to(5.0) is True
    ticket = session.ticket

    session.observe(Painted(media_time=5.0 + 4.0 * FRAME_S, index=130))   # ~0.167 s off: four frames

    assert ticket is not None and ticket.open is True, "a frame that is not the answer must not close the press"
    assert session.snapshot()["landed"] is None


def test_a_job_change_clears_the_ticket():
    session = _session()
    assert session.seek_to(5.0) is True
    ticket = session.ticket

    session.observe(JobChanged())

    assert ticket is not None and ticket.open is False, "a press cannot outlive its job"
    assert session.snapshot()["open"] is False
    assert session.dump()["refusal"] is None, "an abandonment is not a refusal"


def test_two_seeks_in_a_row_replace_the_ticket_and_only_the_newest_target_closes_it():
    session = _session()
    assert session.seek_to(4.0) is True
    first = session.ticket
    assert session.seek_to(9.0) is True
    second = session.ticket

    assert first is not None and second is not None and second is not first
    assert first.open is False, "the replaced press says it was superseded instead of waiting forever"
    assert second.open is True
    assert session.epoch == 2, "one ticket per accepted press"

    session.observe(Painted(media_time=4.0))
    assert second.open is True, "the old target must not close the new press"
    session.observe(Painted(media_time=9.0))
    assert second.open is False
    assert session.snapshot()["target"] == pytest.approx(9.0)


# --- the readings the debug layer serves ---------------------------------------------------------------


def test_snapshot_and_dump_carry_the_epoch_and_the_target():
    session = _session()
    assert session.seek_to(6.0) is True

    snapshot = session.snapshot()
    dump = session.dump()

    assert snapshot["epoch"] == 1 and dump["epoch"] == 1
    assert snapshot["target"] == pytest.approx(6.0) and dump["target"] == pytest.approx(6.0)
    assert snapshot["open"] is True
    assert dump["refusal"] is None
    assert set(dump) == set(snapshot) | {"refusal"}, "dump() is the snapshot plus the reason, not a second shape"


# --- the command route and the clock's own answer -------------------------------------------------------


def test_dispatch_routes_seek_to_through_the_bound_inputs():
    clock = FakeClock()
    session = _session(clock)

    session.dispatch(PreviewCommand.SEEK_TO, 6.0)

    assert clock.calls == [pytest.approx(6.0)]
    assert session.snapshot()["target"] == pytest.approx(6.0)


def test_a_press_that_goes_through_the_clock_keeps_ONE_record(monkeypatch):
    """The real seam: `seek_to` opens the record, then the page (via the clock adapter) claims THAT one.

    Without the claim the page would open its own record and supersede the fresh one, i.e. one press would leave two
    tickets and a bogus `seek superseded` line - the telemetry the owner reads must stay one line per press.
    """

    class RoutingClock:
        """What `_HostBackend.seek_to` does: hand the press to the page, which claims the session's record."""

        def __init__(self, session: PreviewSession) -> None:
            self.session = session
            self.calls: list[float] = []

        def seek_to(self, seconds: float) -> bool:
            self.calls.append(float(seconds))
            ticket = self.session.press(target_s=float(seconds), target_frame=7,
                                        produced=0.25, edge_s=5.0)
            ticket.decide(route="tail", answerable=True)
            return True

    session = PreviewSession()
    clock = RoutingClock(session)
    session.bind(clock=clock, duration=CLIP_S, reach=REACH_S, frame_interval=FRAME_S)
    made = _spy_tickets(monkeypatch)

    assert session.seek_to(5.0) is True

    assert clock.calls == [pytest.approx(5.0)]
    assert len(made) == 1, f"one press, one record - the page's claim must adopt it, got {len(made)}"
    ticket = made[0]
    assert session.ticket is ticket
    assert ticket.route == "tail"
    assert ticket.target_frame == 7 and ticket.produced == pytest.approx(0.25)
    assert ticket.edge_s == pytest.approx(5.0)
    assert session.epoch == 1


def test_a_clock_that_cannot_show_the_time_closes_its_ticket_and_says_so(monkeypatch):
    clock = FakeClock(accepts=False)
    session = _session(clock)
    made = _spy_tickets(monkeypatch)

    assert session.seek_to(5.0) is False

    assert clock.calls == [pytest.approx(5.0)], "the clock WAS asked - it answered no"
    assert len(made) == 1 and made[0].open is False, "a press the route cannot answer is closed, not left open"
    assert "nothing can show 5.000 s" == session.refusal()
    assert session.refused_for_reach() is False, "the controller keeps its own wording for this one"


def test_the_session_module_carries_no_qt():
    source = pathlib.Path(session_module.__file__).read_text(encoding="utf-8")
    for pattern in ("import PySide6", "from PySide6", "QtCore", "QtWidgets"):
        assert pattern not in source, f"the press session must import on an interpreter without PySide6 ({pattern})"
