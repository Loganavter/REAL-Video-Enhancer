"""The preview's press session: one seek route, one ticket, one epoch - and no Qt.

Why this module exists. A strip press used to be clamped TWICE: `PreviewController._seek` compared the target against
the state's own reach, and `ProcessTab._on_preview_seek` then routed the survivor against its notion of the produced
band. Three commits in three days fixed the same shape (plan-rve-preview-architecture-2026-11-26, audit §3 C3), and
the press record that makes a seek legible lived on the page - where it is reset never, so a ticket could outlive its
job (`PressTicket.abandoned()` had no production caller at all).

This is the ONE place a seek is expressed. `seek_to` clamps once - a negative target is 0, a target past the reach is a
refusal with one reason and no ticket - opens exactly one `PressTicket` carrying the next epoch, and asks the clock.
The ticket closes on the picture the painter reports (`observe(Painted)`) when that picture is the one that was asked
for. The page still decides the ROUTE (queue / on-demand / tail / decoder / export display): it supplies the facts only
it knows and uses the session's ticket; the session owns the record and the epoch.

No Qt here on purpose: the numbers and the FSM are testable with a fake clock and a fake painted event - no window, no
renderer, no PySide6. `PreviewCommand` (Enum) and `PressTicket` (telemetry + plain fields) are Qt-free by construction.

What is NOT here yet: the deadline that closes a press nothing answers. `abandon()` is the method P4 wires (it already
ends a ticket on a job change, which is the one closure this phase needs). The landing timeout stays where it is in the
page (`ProcessTab.SEEK_WATCH_TIMEOUT_S`) until P4 moves it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..video.press import PressTicket
from .state import PreviewCommand

__all__ = ["JobChanged", "Painted", "PreviewSession", "SEEK_SLACK"]

#: A target this far past the reachable ceiling is a refusal, not a round-off (half a millisecond). The controller's
#: phase derivation reads the same tolerance (`state.produced < state.duration - SEEK_SLACK`), so there is one number.
SEEK_SLACK = 5e-4


@dataclass(frozen=True)
class Painted:
    """The picture the painter just put on the pane: its media time and (when it has one) its frame identity."""

    media_time: float
    index: int | None = None


@dataclass(frozen=True)
class JobChanged:
    """The page opened a different job: a press that outlived its job cannot be answered."""


def _clock_seek(clock, seconds: float) -> bool:
    """Ask the clock to move, once.

    `seek_to` is this phase's port (`backends.PlaybackBackend`). The `seek(seconds, exact=True)` fallback is the pre-P1
    spelling a caller outside this phase's write scope still uses (the harness's own backend test double); `exact` said
    "never snap to the edge", which is the only behaviour left, so nothing is lost. Keeping the fallback in ONE function
    means the route still has exactly one call site.
    """
    if clock is None:
        return False
    seek_to = getattr(clock, "seek_to", None)
    if callable(seek_to):
        try:
            return bool(seek_to(float(seconds)))
        except Exception:
            return False
    seek = getattr(clock, "seek", None)
    if callable(seek):
        try:
            try:
                return bool(seek(float(seconds), exact=True))
            except TypeError:               # a clock older than the keyword
                return bool(seek(float(seconds)))
        except Exception:
            return False
    return False


class PreviewSession:
    """The press record and the one seek clamp. One instance per preview pane (the controller owns it)."""

    def __init__(self) -> None:
        self._clock = None
        self._duration = 0.0
        self._reach = 0.0
        self._frame_interval = 1.0 / 24.0
        self._epoch = 0
        self._ticket: PressTicket | None = None
        self._pending = False            # the open ticket was opened by seek_to and the page has not claimed it yet
        self._landed: float | None = None
        self._opened_at: float | None = None
        self._refusal: str | None = None
        self._refusal_was_reach = False

    # -- inputs ------------------------------------------------------------------------------------

    def bind(self, *, clock, duration: float, reach: float, frame_interval: float) -> None:
        """Record the inputs a `dispatch()` uses: the clock, the clip's length, the reach and one frame's interval."""
        self._clock = clock
        self._duration = max(0.0, float(duration or 0.0))
        self._reach = max(0.0, float(reach or 0.0))
        self._frame_interval = max(0.0, float(frame_interval or 0.0))

    # -- readings ----------------------------------------------------------------------------------

    @property
    def epoch(self) -> int:
        """How many presses have opened a ticket (the sweep generation a painted frame must belong to)."""
        return int(self._epoch)

    @property
    def ticket(self) -> PressTicket | None:
        """The current record (open or closed), so the page can route and settle it."""
        return self._ticket

    @property
    def target(self) -> float | None:
        """The time the current ticket asked for, in seconds."""
        ticket = self._ticket
        return None if ticket is None or ticket.target_s is None else float(ticket.target_s)

    def refusal(self) -> str | None:
        """Why the last press was refused, in words that can be printed as-is (the ONE status line's tail)."""
        return self._refusal

    def refused_for_reach(self) -> bool:
        """Whether that refusal was the reach clamp (the controller's `seek ... refused: ...` line) or the clock's."""
        return bool(self._refusal_was_reach)

    def snapshot(self) -> dict:
        """The press state as data: epoch, whether a ticket is open, its target, and the last landed picture."""
        ticket = self._ticket
        return {
            "epoch": int(self._epoch),
            "open": bool(ticket is not None and ticket.open),
            "press": None if ticket is None else int(ticket.id),
            "route": None if ticket is None else ticket.route,
            "target": None if ticket is None or ticket.target_s is None else float(ticket.target_s),
            "landed": None if self._landed is None else float(self._landed),
        }

    def dump(self) -> dict:
        """`snapshot()` plus the last refusal reason - what the debug channel serves (plan P5)."""
        state = self.snapshot()
        state["refusal"] = self._refusal
        return state

    # -- the command --------------------------------------------------------------------------------

    def dispatch(self, command, value: object = None) -> None:
        """Route a `PreviewCommand` to the session's own route, using the inputs `bind()` recorded.

        Only SEEK_TO is the session's: every other verb is the controller's (play/pause through a backend, the view
        modes through the surface) and is deliberately not re-implemented here.
        """
        try:
            command = PreviewCommand(command)
        except ValueError:
            return
        if command is not PreviewCommand.SEEK_TO:
            return
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            self._refuse(f"not a time: {value!r}", past_reach=False)
            return
        self.seek_to(seconds)

    def seek_to(self, seconds: float, *, clock=None, duration: float | None = None,
                reach: float | None = None, frame_interval: float | None = None) -> bool:
        """The one seek: clamp, open one ticket, ask the clock. True when the clock took the target.

        Omitted inputs fall back to the ones `bind()` recorded. A target past the reach is NOT clamped to the reach:
        it is refused with one reason and opens no ticket, so the pane never claims a time nothing can show.
        """
        if clock is not None:
            self._clock = clock
        if duration is not None:
            self._duration = max(0.0, float(duration))
        if reach is not None:
            self._reach = max(0.0, float(reach))
        if frame_interval is not None:
            self._frame_interval = max(0.0, float(frame_interval))

        try:
            target = float(seconds)
        except (TypeError, ValueError):
            self._refuse(f"not a time: {seconds!r}", past_reach=False)
            return False
        if target < 0.0:
            target = 0.0
        if target > self._reach + SEEK_SLACK:
            self._refuse(f"the pane can show up to {self._reach:.3f} s", past_reach=True)
            return False

        ticket = self._begin(target_s=target)
        self._pending = True             # the page's route claims this record when it is called next
        if not _clock_seek(self._clock, target):
            reason = f"nothing can show {target:.3f} s"
            self._pending = False
            ticket.refused(reason)
            self._refuse(reason, past_reach=False)
            return False
        self._refusal = None
        self._refusal_was_reach = False
        self._landed = None
        self._opened_at = time.perf_counter()
        return True

    # -- the page's half ----------------------------------------------------------------------------

    def press(self, *, target_s: float | None, target_frame: int | None = None,
              produced: float | None = None, edge_s: float | None = None) -> PressTicket:
        """The ticket for the press the page is routing RIGHT NOW.

        A press that came through `seek_to` already owns a record: the page fills in the route facts only it knows
        (which frame the share names, what the renderer had produced) and routes it. A press that reached the page
        directly (the pane's own strip signal, a hand-wired tab) opens its record here, superseding an unanswered one.
        """
        ticket = self._ticket
        if self._pending and ticket is not None and ticket.open:
            self._pending = False
            if target_s is not None:
                ticket.target_s = float(target_s)
            ticket.target_frame = target_frame
            ticket.produced = produced
            ticket.edge_s = edge_s
            return ticket
        ticket = self._begin(target_s=target_s, target_frame=target_frame, produced=produced, edge_s=edge_s)
        self._pending = False
        return ticket

    # -- what the picture says ----------------------------------------------------------------------

    def observe(self, event) -> None:
        """Take a fact from the painter or the page: a painted picture, or a new job."""
        if isinstance(event, Painted):
            self._note_painted(event)
        elif isinstance(event, JobChanged):
            self.abandon("job-changed")

    def abandon(self, reason: str) -> bool:
        """End the open ticket without an answer. P4 wires the deadline; the job change already calls it.

        `JobChanged` uses it today (a press cannot outlive its job). P4 adds the timeout that calls it for a press
        nothing answered, which is why the method is here and not in the page.
        """
        ticket = self._ticket
        self._pending = False
        if ticket is None or not ticket.open:
            return False
        return bool(ticket.abandoned(str(reason)))

    # -- internals ----------------------------------------------------------------------------------

    def _begin(self, *, target_s: float | None, target_frame: int | None = None,
               produced: float | None = None, edge_s: float | None = None) -> PressTicket:
        """Open one record with the next epoch; an unanswered predecessor is SUPERSEDED, not dropped."""
        self._epoch += 1
        ticket = PressTicket(target_s=target_s, target_frame=target_frame, produced=produced, edge_s=edge_s)
        previous = self._ticket
        if previous is not None and previous.open:
            previous.superseded(ticket.id)
        self._ticket = ticket
        self._landed = None
        return ticket

    def _note_painted(self, painted: Painted) -> None:
        """Close the open ticket on the picture that ANSWERS it; two frames' worth of slack, at least 50 ms."""
        try:
            media_time = float(painted.media_time)
        except (TypeError, ValueError):
            return
        ticket = self._ticket
        if ticket is None or not ticket.open:
            self._landed = media_time
            return
        if ticket.target_s is None:
            return
        tolerance = max(0.05, 2.0 * self._frame_interval)
        if abs(media_time - float(ticket.target_s)) > tolerance:
            return
        opened = self._opened_at
        turnaround_ms = 0.0 if opened is None else max(0.0, (time.perf_counter() - opened) * 1000.0)
        # `settled()` names the picture the way telemetry does: `("frame", n)` when the producer numbered it, absent
        # when it did not. The session knows the frame's index, never the pane's display sequence.
        landed = None if painted.index is None else ("frame", int(painted.index))
        if ticket.settled(landed=landed, landed_time=media_time,
                          turnaround_ms=turnaround_ms, landed_gen=None):
            self._pending = False
            self._landed = media_time

    def _refuse(self, reason: str, *, past_reach: bool) -> None:
        self._refusal = str(reason)
        self._refusal_was_reach = bool(past_reach)
