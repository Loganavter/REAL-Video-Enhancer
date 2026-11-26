"""One press, one record: what was asked, what was decided, what was drawn.

Why this module exists (owner, 2026-11-24): *"we can say where the seek was, but not which frame we drew"*. Three of
his own sessions pressed the strip four times each and produced no line about the result at all, while the pane did
change; a fourth closed one press out of three. The reason was structural, not a missing field: the seek's state lived
in eight flags on the tab, four different painters decided independently whether to paint and whether to close, and the
picture's identity was read from side state at paint time. The intent and the result never shared an object, so a press
could end in silence and the log could not say which picture it settled on.

A press is a request/answer protocol, so it gets one object with a lifecycle:

* `decide()` - written the moment the press is routed: where it goes, what it asked for, what the renderer had produced
  at that instant, and whether that route can answer it at all (`answerable=0` always carries `why=`). No stopwatch:
  the answer to "can this be answered" is known immediately (a frame past the produced edge does not exist yet);
* `settled(...)` - the picture that answers it, with the frame's own identity (`seq`/`index` + media time + seek
  generation), so the numbers come from the picture rather than from the tab's current state;
* `refused(why)` - the route cannot answer;
* `superseded(by)` - a newer press replaced it before it was answered (the seek slot is newest-wins, so this is
  normal, but it must be visible);
* `abandoned(waited_ms, answered_gen)` - it was open and nothing arrived; named so the reader never has to guess
  between "dropped on purpose" and "the answer never came".

Exactly one `decide` and exactly one outcome per ticket. Silence is not a state this object can be in.
"""

from __future__ import annotations

import itertools

from . import telemetry

#: Press ids are per process and monotonic: a log line's `press=` is what ties a decision to its outcome.
_IDS = itertools.count(1)


def _seconds(value: float | None) -> str | None:
    return None if value is None else f"{float(value):.3f}"


def _share(value: float | None) -> str | None:
    return None if value is None else f"{float(value):.4f}"


class PressTicket:
    """One strip press: the route it took, the answer it wanted, and how it ended."""

    def __init__(self, *, target_s: float | None, target_frame: int | None,
                 produced: float | None = None, edge_s: float | None = None) -> None:
        self.id = next(_IDS)
        self.target_s = target_s
        self.target_frame = target_frame
        self.produced = produced
        self.edge_s = edge_s
        self.route = "undecided"
        self.generation: int | None = None
        self.reason: str | None = None
        self._decided = False
        self._outcome: str | None = None
        self._stream_noted = False

    # -- the lifecycle ------------------------------------------------------------------------------

    @property
    def open(self) -> bool:
        """True while the press has no outcome: the pane may still park on this target."""
        return self._outcome is None

    def decide(self, *, route: str, answerable: bool, why: str | None = None,
               generation: int | None = None, ahead: bool | None = None) -> "PressTicket":
        """Record where the press went and whether that route can answer it. Written once, at routing time.

        `answerable` is about the ROUTE, `ahead` about the FRAME: the display route decodes any timestamp on demand, so
        it is answerable even when the requested second is past what the export has written (`ahead=1`). That split is
        the whole difference between "the renderer has not got there yet" and "nothing can answer this", and conflating
        them is what made a working press look like a refusal (measured 2026-11-24: `answerable=0` on a press the
        renderer then answered in 940 ms).
        """
        if self._decided:
            return self
        self.route = str(route)
        self.generation = None if generation is None else int(generation)
        self.reason = why
        self._decided = True
        telemetry.event("seek decide", ts=telemetry.wall(), press=self.id, route=self.route,
                        target_s=_seconds(self.target_s), target_frame=self.target_frame,
                        produced=_share(self.produced), edge_s=_seconds(self.edge_s),
                        answerable=int(bool(answerable)), ahead=None if ahead is None else int(bool(ahead)),
                        why=why, generation=self.generation)
        return self

    def settled(self, *, landed, landed_time: float | None, turnaround_ms: float,
                landed_gen: int | None) -> bool:
        """The picture that answers this press: its identity, its media time, and the generation it carries."""
        if not self.open:
            return False
        self._outcome = "closed"
        telemetry.note_seek(float(turnaround_ms), target=self.target_frame, landed=landed,
                            target_s=self.target_s, landed_time=landed_time,
                            target_gen=self.generation, landed_gen=landed_gen, press=self.id,
                            ts=telemetry.wall())
        return True

    def refused(self, reason: str) -> bool:
        """This route cannot answer the press; said out loud instead of leaving the request open."""
        return self._finish("refused", reason=reason)

    def abandoned(self, reason: str) -> bool:
        """The press was open and nothing arrived (a backend that died, a decode that failed, a route that went away)."""
        return self._finish("abandoned", reason=reason)

    def superseded(self, by: int) -> bool:
        """A newer press replaced this one before it was answered: the seek slot holds one request, newest wins."""
        return self._finish("superseded", by=int(by))

    def note_stream(self, *, seq: int | None, media_time: float | None, landed_gen: int | None) -> bool:
        """A streaming picture arrived while this press is open, naming itself. Logged once per press.

        This is the line that was missing entirely: the pane painted the newest upscaled frame and moved the marker to
        the produced edge, and nothing said so. Without it a reader sees a press with no outcome and cannot tell a
        picture the press waits for from a picture it must not accept.
        """
        if not self.open or self._stream_noted:
            return False
        self._stream_noted = True
        telemetry.event("seek stream", ts=telemetry.wall(), press=self.id, seq=seq, t=_seconds(media_time),
                        gen=landed_gen, not_the_answer=1)
        return True

    def _finish(self, outcome: str, **fields) -> bool:
        if not self.open:
            return False
        self._outcome = outcome
        # The reason belongs to the TICKET, not only to the log line: a caller (or a test) that asks "why could this
        # press not be served" read `ticket.reason` and got None, because `refused(reason)` passed it straight to
        # telemetry and `_finish` ignored it. Found twice - audit 2026-11-27 and again by the P3 deletion's own test.
        if "reason" in fields:
            self.reason = str(fields["reason"])
        telemetry.event(f"seek {outcome}", ts=telemetry.wall(), press=self.id, route=self.route,
                        target_s=_seconds(self.target_s), **fields)
        return True
