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

The lifecycle is closed HERE (plan P4): a ticket ends on the picture that answers it (`observe(Painted)`), on the
deadline a tick reports (`check_deadline`, `SEEK_WATCH_TIMEOUT_S`) or on a job change (`observe(JobChanged)`) - in every
case `abandon()` is the session's own call, never the page's. The page supplies one callback (`set_settle`) and only
formats the line the owner reads; `ProcessTab._seek_watch` no longer decides whether a press was answered.

`dump()` is the debug layer's ONE read of the pane (plan P5): the snapshot (this session's facts plus the state numbers
the page feeds through `set_dump_source`), the ticket, the epoch, the scrubber's last paint and the owner decision. The
control channel serves it as the preview verb and `preview-log.txt` writes the same state numbers once a second; a dump
never writes the session, the ticket or a widget.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from ..video.press import PressTicket
from .state import PreviewCommand

__all__ = ["JobChanged", "Painted", "PreviewSession", "SEEK_SLACK", "SEEK_WATCH_TIMEOUT_S", "Settled"]

#: The state numbers the page feeds into `dump()` (plan P5). Same names as the aggregate log line's keys, so the debug
#: channel and `preview-log.txt` cannot describe the pane in two different vocabularies.
DUMP_STATE_KEYS = ("kind", "duration", "produced", "played", "playhead", "finished")

#: A target this far past the reachable ceiling is a refusal, not a round-off (half a millisecond). The controller's
#: phase derivation reads the same tolerance (`state.produced < state.duration - SEEK_SLACK`), so there is one number.
SEEK_SLACK = 5e-4

#: How long a press may stand unanswered before the session gives up on it. ONE definition (plan P4): the page's
#: `ProcessTab.SEEK_WATCH_TIMEOUT_S` is an alias of this name, and the 100 ms tick reports the elapsed time to
#: `check_deadline`, which closes the ticket. Without it a frame that never arrives parks the marker on the requested
#: time forever (`PressTicket.abandoned()` had no production caller).
SEEK_WATCH_TIMEOUT_S = 1.5


@dataclass(frozen=True)
class Painted:
    """The picture the painter just put on the pane: its media time and (when it has one) its frame identity."""

    media_time: float
    index: int | None = None


@dataclass(frozen=True)
class JobChanged:
    """The page opened a different job: a press that outlived its job cannot be answered."""


@dataclass(frozen=True)
class Settled:
    """What the page formats when a ticket ends on a picture or on its deadline (never a second press record).

    `asked` is the time the ROUTE was given (the tail maps a strip share onto its own timeline, so it can differ from
    the ticket's `target_s`); `landed` is the picture's own media time; `matched` is the session's answer to "is this
    the frame that was asked for"; `gave_up` says the deadline closed it. The page's landing line is a projection of
    these fields and nothing more.
    """

    asked: float | None
    landed: float | None
    matched: bool
    gave_up: bool
    turnaround_ms: float | None


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

    def __init__(self, *, settle: Callable[[Settled], None] | None = None,
                 deadline_s: float | None = None) -> None:
        self._clock = None
        self._duration = 0.0
        self._reach = 0.0
        self._frame_interval = 1.0 / 24.0
        self._epoch = 0
        self._ticket: PressTicket | None = None
        self._pending = False            # the open ticket was opened by seek_to and the page has not claimed it yet
        self._landed: float | None = None
        self._last_painted: float | None = None
        self._watch_s: float | None = None
        self._opened_at: float | None = None
        self._refusal: str | None = None
        self._refusal_was_reach = False
        #: The page's one read of what the session cannot see (P5): the state's numbers, the scrubber's paint, the owner.
        self._dump_source: Callable[[], dict] | None = None
        #: The page's one formatter for the landing line (P4): the session owns the closure, the page owns the words.
        self._settle = settle
        self._deadline_s = SEEK_WATCH_TIMEOUT_S if deadline_s is None else max(0.0, float(deadline_s))

    # -- the page's callback ----------------------------------------------------------------------

    def set_settle(self, settle: Callable[[Settled], None] | None) -> None:
        """Point the landing line at the page's formatter (idempotent; the page sets it on every access)."""
        self._settle = settle

    def set_dump_source(self, source: Callable[[], dict] | None) -> None:
        """Point `dump()` at the page's one read of the state's numbers, the scrubber's paint and the owner.

        Idempotent, and a plain assignment on purpose: the page re-points it on every session access, so a tab wired by
        hand and a tab with a controller both answer with their own page's facts. The source is only CALLED by `dump()`;
        it must read, never write - the dump is the channel's view of the pane, not an action on it.
        """
        self._dump_source = source

    @property
    def deadline_s(self) -> float:
        """Seconds a press may stand unanswered before `check_deadline` abandons it."""
        return float(self._deadline_s)

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

    @property
    def playhead(self) -> float | None:
        """Where the marker may stand: the requested time while a press is open, else the last painted picture.

        This is the session's half of "the marker follows the picture": once the deadline (or a job change) closes the
        ticket, the request stops being a place and the picture's own time takes over - never `target_s` again.
        """
        ticket = self._ticket
        if ticket is not None and ticket.open and ticket.target_s is not None:
            return float(ticket.target_s)
        return self._last_painted

    def watch(self, seconds: float | None) -> None:
        """Record the time the press actually asked the OWNER for.

        The tail maps a strip share onto its own timeline (the flushed edge when the clip length is unknown), so the
        asked time can differ from the ticket's `target_s`. The closure compares the painted picture with THIS number;
        the page passes it in the route that knows it.
        """
        self._watch_s = None if seconds is None else float(seconds)

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
        """The whole preview state as data, in ONE read: `{snapshot, ticket, epoch, paint, owner}` (plan P5).

        * `snapshot` - this session's `snapshot()` (epoch, open, press, route, target, landed, refusal) merged with the
          state numbers only the page can read (kind, duration, produced, played, playhead, finished). One snapshot for
          the whole pane, so the control channel and the aggregate log line carry the same keys;
        * `ticket` - the press record's own detail (id, route, target, generation, outcome), `None` before a press;
        * `epoch` - the sweep generation (`snapshot()["epoch"]`, repeated at the top for the dump's own shape);
        * `paint` - the scrubber's last painted rects (`scrub_report()["paint"]`), `None` without a scrubber;
        * `owner` - the one painter decision: the kind, its three permissions and whether a clock/surface exist.

        A READ: nothing here writes the session, the ticket or a widget. The page's source is only called.
        """
        extra: dict = {}
        source = self._dump_source
        if callable(source):
            try:
                extra = dict(source() or {})
            except Exception:
                # A page that cannot answer must not take the debug channel down: the session's own half still answers.
                extra = {}
        snapshot = self.snapshot()
        snapshot["refusal"] = self._refusal
        for key in DUMP_STATE_KEYS:
            snapshot[key] = extra.get(key)
        return {
            "snapshot": snapshot,
            "ticket": self._ticket_dump(),
            "epoch": int(self._epoch),
            "paint": extra.get("paint"),
            "owner": extra.get("owner"),
        }

    def _ticket_dump(self) -> dict | None:
        """The current record as data (`None` before the first press): what was asked, where it went, how it ended."""
        ticket = self._ticket
        if ticket is None:
            return None
        produced = ticket.produced
        return {
            "id": int(ticket.id),
            "open": bool(ticket.open),
            "route": ticket.route,
            "target_s": None if ticket.target_s is None else float(ticket.target_s),
            "target_frame": ticket.target_frame,
            "produced": None if produced is None else float(produced),
            "edge_s": None if ticket.edge_s is None else float(ticket.edge_s),
            "generation": ticket.generation,
            "reason": ticket.reason,
            "outcome": getattr(ticket, "_outcome", None),
        }

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
        """The one seek: clamp, open one ticket, ask the clock. True when the clock TAKES the target.

        The clock is the page's press route (`backends._HostBackend.seek_to`), and since audit §6.4 its answer is the
        route's own: False for every refusal path, with that path's reason carried into `refusal()` (not the generic
        "nothing can show X s" the session would otherwise print over it). Omitted inputs fall back to the ones
        `bind()` recorded. A target past the reach is NOT clamped to the reach: it is refused with one reason and
        opens no ticket, so the pane never claims a time nothing can show.
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
            # A route that refused already closed the ticket with ITS reason (`ticket.refused(...)` in the use case);
            # `refused()` here then answers False and the record keeps that reason. The session's one status line
            # must carry the reason the reader can act on ("preview-without-a-block"), not the generic fallback -
            # audit §6.4's complaint was that the three layers that say "this press worked" disagreed, and the
            # wording was the only one of them the log never showed.
            if not ticket.refused(reason) and ticket.reason:
                reason = str(ticket.reason)
            self._refuse(reason, past_reach=False)
            return False
        self._refusal = None
        self._refusal_was_reach = False
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
            # The route has not armed a watch yet; the ticket's own target is what it asks for until `watch()` says
            # otherwise (the tail's share -> edge mapping is the one route that differs).
            self._watch_s = ticket.target_s
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
            # A press cannot outlive its job: this is the session's own call, not the page's.
            self.abandon("job-changed")

    def check_deadline(self, now: float | None = None) -> bool:
        """Close the open ticket when the bound deadline elapsed (the page calls this from its 100 ms tick).

        `now` is a `time.perf_counter()` reading; omitted, the current one is used. The ticket is abandoned through the
        same `abandon()` a job change uses - the only difference is that the page is told the picture it stood on, so
        the owner's landing line can say the seek gave up. True when this call closed a ticket.
        """
        ticket = self._ticket
        if ticket is None or not ticket.open:
            return False
        opened = self._opened_at
        if opened is None:
            return False
        moment = time.perf_counter() if now is None else float(now)
        if (moment - opened) < self._deadline_s:
            return False
        asked = self._asked()
        landed = self._last_painted
        if not self.abandon("watch-timeout"):
            return False
        self._notify(asked=asked, landed=landed, matched=False, gave_up=True,
                     turnaround_ms=max(0.0, (moment - opened) * 1000.0))
        return True

    def abandon(self, reason: str) -> bool:
        """End the open ticket without an answer: the deadline and a job change are both the session's call.

        `JobChanged` uses it for a press that outlived its job; `check_deadline` uses it for a press nothing answered.
        An abandonment is not a refusal: it says the answer never came, which is a different fact from "this route
        cannot show the time" (`refused`).
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
        self._watch_s = target_s
        self._opened_at = time.perf_counter()
        return ticket

    def _asked(self) -> float | None:
        """The time the open press asked its owner for: the route's watch, else the ticket's own target."""
        if self._watch_s is not None:
            return float(self._watch_s)
        ticket = self._ticket
        if ticket is None or ticket.target_s is None:
            return None
        return float(ticket.target_s)

    def _note_painted(self, painted: Painted) -> None:
        """Close the open ticket on the picture that ANSWERS it; two frames' worth of slack, at least 50 ms.

        This method IS the closure (plan P4): a painted frame inside the tolerance closes the ticket and reports the
        landing through the page's callback; a frame outside it is not the answer and only records where the picture
        stands. Once the ticket is closed - by this frame, by the deadline, by a job change - a later frame updates the
        picture and nothing else, so a late answer cannot resurrect a press that already gave up.
        """
        try:
            media_time = float(painted.media_time)
        except (TypeError, ValueError):
            return
        self._last_painted = media_time
        ticket = self._ticket
        if ticket is None or not ticket.open:
            self._landed = media_time
            return
        asked = self._asked()
        if asked is None:
            return
        tolerance = max(0.05, 2.0 * self._frame_interval)
        if abs(media_time - asked) > tolerance:
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
            self._notify(asked=asked, landed=media_time, matched=True, gave_up=False,
                         turnaround_ms=turnaround_ms)

    def _notify(self, *, asked: float | None, landed: float | None, matched: bool, gave_up: bool,
                turnaround_ms: float | None) -> None:
        """Tell the page a ticket ended on a picture or on its deadline; the page only formats the line."""
        callback = self._settle
        if callback is None:
            return
        try:
            callback(Settled(asked=asked, landed=landed, matched=bool(matched), gave_up=bool(gave_up),
                             turnaround_ms=turnaround_ms))
        except Exception:
            # The closure must not depend on the formatter: a page without a working log still closes its ticket.
            pass

    def _refuse(self, reason: str, *, past_reach: bool) -> None:
        self._refusal = str(reason)
        self._refusal_was_reach = bool(past_reach)
