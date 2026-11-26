"""The painted pair: the ONE writer of what the preview pane hands the renderer.

Before this module the pane held two queues (`_result_queue`/`_source_queue`), a valve timer
(`_release_pair_paint`), a blocked-at stamp (`_pair_blocked_at`) and a predicate (`pair_in_sync`) - five data asserting
conclusions about one picture, each able to disagree with the others. The renderer was handed the newest result alone
whenever its source was missing, which is the normal state after every refused source and before every first source
frame, so a compare mode painted "the upscale, then the original" (audit 2026-11-27 §6.1, probe D2/E2). The queues also
survived a clear, so a late source re-keyed itself by its own moment and painted half a picture onto a pane told to be
empty (§6.1b, probe G2).

The policy, in full - this module is the only place that decides it, and the only place in the tree that calls
`renderer.set_frame` / `renderer.set_original`:

* **A compare mode (`sbs`/`split`) paints a PAIR or holds the last one.** While a result's source has not arrived the
  renderer keeps painting the previous matched pair; the pane hands it nothing in between. That hold is what stops a
  strip drag blanking the left half: measured in the owner's log (2026-11-27), the decoder answered one ask late, every
  answer was refused as a near neighbour, and the source half went black for ~1.3 s over 31 draws.
* **One hard limit: `MAX_PAIR_STALE_S`.** A source that never arrives must not freeze the picture for ever, so after
  that long the newest result is painted alone. That is the ONLY path that paints one half of a compare mode without
  its partner: `CommittedPair.lone_half` names which half it was and a `pair stale` telemetry line says why. (The other
  documented half-picture is the opposite case - no result exists at all, so the source is the only picture there is;
  it is painted, and also says so.)
* **A source is not a pair unless it is the same moment and the same generation.** Distance is measured against the
  tolerance the pane passes in (`half a frame` of its own learned cadence, see `VideoPreviewSurface.
  source_half_frame_s`), so a snapshot more than half a frame from the result is refused rather than drawn as a
  neighbour; a picture from an older generation is refused too, and the pair already painted survives both.
* **A clear is a clear.** `clear()` drops the pair, paints an empty pane and refuses a source offered for a moment from
  before the clear (`2026-11-27 f1ed720e`).
* **`single` has one surface.** There is no pair to form, so the writer forwards the result and the LMB flash picture;
  the flash names the moment the host was ASKED for and is deliberately not the result's moment - that is the mode's
  policy, not a pair.

The writer is Qt-free on purpose (the sink is duck-typed and the frame payloads are opaque here): its whole policy is
testable with a recording sink and a fake clock, which is how `harness/tests/test_pair_writer.py` proves it.
"""

from __future__ import annotations

import time

from dataclasses import dataclass
from typing import Any, Callable

from src.ui.video import telemetry

__all__ = ["CommittedPair", "Half", "MAX_PAIR_STALE_S", "PairWriter"]

#: How long the last matched pair may stand in for a missing partner, in seconds.
#:
#: The number comes from the measured defect it answers: a strip drag during a render left the source half blank for
#: ~1.3 s (31 draws in the owner's log, 2026-11-27) because every answer arrived one frame behind the result and was
#: refused as a neighbour. 1.5 s covers that outage with headroom, and is short enough that a decoder which died cannot
#: freeze the picture noticeably. It also subsumes the old seek valve's stand-down (`PAIR_JUMP_WAIT_S = 1.0`): a seek
#: holds the old pair for less than this, then paints the new result alone if its source is still missing.
MAX_PAIR_STALE_S = 1.5


@dataclass(frozen=True)
class Half:
    """One picture as the writer holds it: the renderer's payload plus the identity the logs need.

    `frame` is opaque here (`PreviewFrame` in the pane, anything a test likes): the writer only ever hands it back to
    the sink. `image` is the same picture as a QImage when one exists - the pane reads it for the draw line's
    fingerprint, never for pixels; a YUV (planes) frame has none. `name` is `("frame", n)` or `("seq", n)`, the
    identity the producer gave the picture.
    """

    frame: Any | None
    moment: float | None
    generation: int = 0
    image: Any | None = None
    name: tuple[str, int] | None = None


@dataclass(frozen=True)
class CommittedPair:
    """What the renderer holds: the two halves of one moment, or the one half the policy allows.

    In a compare mode `result` and `source` are both present and both name `moment` - that invariant is the point of
    this module. The two exceptions are explicit, never implicit: `lone_half="result"` is the `MAX_PAIR_STALE_S` valve
    (a source that never came), `lone_half="source"` is a pane with no result at all (the source is the only picture),
    and `single` records the result with `source` being the LMB flash picture.
    """

    result: Half | None
    source: Half | None
    moment: float | None
    generation: int
    lone_half: str | None = None


def _same(left: Half | None, right: Half | None) -> bool:
    """Identity of a payload, not equality of pixels: a QImage compares its bytes, a `PreviewFrame` its planes."""
    if left is None or right is None:
        return left is None and right is None
    return left.frame is right.frame


def _moment(value: float | None) -> str:
    return "-" if value is None else f"{float(value):.3f}"


class PairWriter:
    """Owns the painted pair: the pane offers halves, this object decides what reaches the renderer.

    The pane's flow is: accept a picture through its own guards, `offer_result`/`offer_source`, then ask the pane's one
    render gate (`VideoPreviewSurface._request_frame_paint`) to `paint()`. Offers only record; `paint()` is the single
    decision point, so there is one place where "a pair, a hold or the lone half" is chosen.
    """

    def __init__(self, sink, *, clock: Callable[[], float] | None = None) -> None:
        """`sink` needs `set_frame(frame | None)` and `set_original(frame | None)` (the renderer's own contract)."""
        self._sink = sink
        self._clock = clock or time.monotonic
        self._mode = "single"
        #: How far a source may sit from its result and still be the same picture. The pane sets this from its own
        #: learned cadence (half a frame) before every decision; 0.0 means "exact" and only tests run it that way.
        self._tolerance = 0.0
        self._generation = 0
        self._pending_result: Half | None = None
        self._pending_source: Half | None = None
        #: `single`'s LMB flash: the picture the host decoded for an ask. Not a pair - one surface, one flash.
        self._flash: Half | None = None
        self._committed: CommittedPair | None = None
        #: When the current hold began (monotonic), or None while nothing is waiting.
        self._hold_started: float | None = None
        #: True once `MAX_PAIR_STALE_S` has been paid: the newest result keeps painting alone until a pair completes.
        self._lone_result = False
        self._cleared = False
        #: The moment of the picture the clear dropped, or None when the clear found the pane already empty. It is the
        #: deadline a late source is measured against: a picture of that moment (or older) belongs to the pane that no
        #: longer exists, while a pane that had no moment at all has nothing a source could be "late" for.
        self._cleared_moment: float | None = None

    # -- configuration ---------------------------------------------------

    def set_view_mode(self, mode: str) -> None:
        """`single` (one surface) or a compare mode (`sbs`/`split`, a pair or nothing)."""
        if mode == self._mode:
            return
        self._mode = str(mode)
        self._hold_started = None
        self._lone_result = False
        if mode != "single" and self._committed is not None:
            # `single` is one surface, so what it painted is not a pair: entering a compare mode the writer does not
            # claim it as one (the renderer still shows it until the first pair arrives - the user just switched modes
            # and the source was asked for - but `painted()` stops answering "a pair" about one half).
            pair = self._committed
            if pair.result is None or pair.source is None or not self._pairs(pair.result, pair.source):
                self._committed = None

    def set_tolerance(self, seconds: float) -> None:
        """The largest source distance that is still the same picture, in seconds (the pane's half frame)."""
        self._tolerance = max(0.0, float(seconds))

    @property
    def cleared(self) -> bool:
        """Whether the pane was told to be empty and nothing has been published since."""
        return self._cleared

    # -- offering halves -------------------------------------------------

    def offer_result(self, half: Half) -> None:
        """A result the pane accepted. It waits for a source of its own moment; `paint()` decides when it shows."""
        self._cleared = False
        self._cleared_moment = None
        self._generation = max(self._generation, int(half.generation))
        if (self._pending_result is not None and not _same(self._pending_result, half)
                and telemetry.level() >= 2):
            telemetry.event("pair drop", half="result", moment=_moment(self._pending_result.moment),
                            why="superseded", newest=_moment(half.moment))
        self._pending_result = half

    def offer_source(self, half: Half | None) -> None:
        """A source picture, or `None` for a gap. A gap does NOT take the painted pair off the screen."""
        if half is None:
            self._pending_source = None
            if self._mode == "single":
                self._flash = None
            return
        if self._cleared:
            # A pane whose painted picture had a moment refuses every source until a result is published again: the
            # moment no longer exists, so a source can only be a picture of the past (audit 2026-11-27 §6.1b, f1ed720e).
            # A clear that found the pane already empty has no such deadline - the source is then the ONLY picture there
            # is (a compare mode opened before its first result), which is the one case a lone source is painted.
            if self._cleared_moment is not None:
                telemetry.event("pair refused", half="source", why="cleared-pane",
                                moment=_moment(half.moment), generation=int(half.generation),
                                cleared_moment=_moment(self._cleared_moment))
                return
        if int(half.generation) < self._generation:
            telemetry.event("pair refused", half="source", why="older-generation",
                            moment=_moment(half.moment), generation=int(half.generation),
                            held_generation=int(self._generation))
            return
        if self._mode == "single":
            self._flash = half
            return
        self._pending_source = half

    # -- the one decision -------------------------------------------------

    def paint(self) -> bool:
        """Decide what the renderer holds now and hand over whatever changed. True when something was handed over.

        This is the whole policy in one place (see the module docstring): a pair when both halves of one moment are in
        hand, the last matched pair while one is missing, the lone result only once `MAX_PAIR_STALE_S` has passed.
        """
        if self._mode == "single":
            return self._paint_single()
        pending = self._pending_result
        if pending is not None:
            source, consume = self._partner_for(pending)
            if source is not None:
                if consume:
                    self._pending_source = None
                self._pending_result = None
                self._lone_result = False
                if telemetry.level() >= 2:
                    telemetry.event("pair commit", moment=_moment(pending.moment),
                                    generation=int(pending.generation))
                return self._paint(pending, source, moment=pending.moment,
                                   generation=int(pending.generation))
            if self._lone_result or self._hold_expired():
                self._pending_result = None
                self._lone_result = True
                self._hold_started = None
                # The one documented way a compare mode paints half a picture. It must be visible in the log, which is
                # the difference between "the pane is broken again" and "the decoder gave up".
                telemetry.event("pair stale", half="result", why="no-source-within-hold",
                                waited_s=f"{MAX_PAIR_STALE_S:.2f}", moment=_moment(pending.moment),
                                generation=int(pending.generation))
                return self._paint(pending, None, moment=pending.moment,
                                   generation=int(pending.generation), lone_half="result")
            if self._hold_started is None:
                self._hold_started = self._clock()
                telemetry.event("pair hold", half="result", why="no-source",
                                moment=_moment(pending.moment), generation=int(pending.generation))
            return False
        source = self._pending_source
        if source is not None:
            committed = self._committed
            if committed is not None and committed.result is not None and self._pairs(committed.result, source):
                # The result half has not advanced; this source completes the pair standing behind the hard limit.
                self._pending_source = None
                self._lone_result = False
                return self._paint(committed.result, source, moment=committed.moment,
                                   generation=int(committed.generation))
            if committed is None or committed.result is None:
                # No result exists at all: the source is the one picture there is, and a pane that waits for a result
                # that may never come is blank while a decoded picture sits in hand (owner, 2026-11-26: "очень долго
                # ждем оригинал на side by side").
                self._pending_source = None
                telemetry.event("pair lone", half="source", why="no-result-on-screen",
                                moment=_moment(source.moment), generation=int(source.generation))
                return self._paint(None, source, moment=source.moment,
                                   generation=int(source.generation), lone_half="source")
            # A source one moment ahead of the painted pair with no result to pair with: hold, never paint it alone.
            if telemetry.level() >= 2:
                telemetry.event("pair hold", half="source", why="no-result",
                                moment=_moment(source.moment), painted=_moment(committed.moment))
            return False
        # Nothing new: the renderer keeps what it has. A forced re-hand exists for the paths that lost the textures
        # (`refresh`), not for a gap notification - a gap arrives many times a second during a drag.
        return False

    def refresh(self) -> bool:
        """Re-hand the pair the renderer already holds: a mode change or a re-init needs the frames again."""
        committed = self._committed
        if committed is None:
            return False
        return self._paint(committed.result, committed.source, moment=committed.moment,
                           generation=int(committed.generation), lone_half=committed.lone_half, force=True)

    # -- lifecycle -------------------------------------------------------

    def clear(self) -> None:
        """The pane was told to be empty: drop every half, empty the renderer, refuse a late source.

        The refusal is not a memory (`_cleared_empty` was): the pair is gone, so there is no moment a late source could
        belong to, and `offer_source` refuses it until a new picture is published. What the clear records is the MOMENT
        it dropped - that is the deadline a late source is judged against, and a clear that found the pane empty has
        none, so the source-only state (a compare opened before its first result) still works.
        """
        painted = self._committed
        self._cleared = True
        self._cleared_moment = None if painted is None else painted.moment
        self._pending_result = None
        self._pending_source = None
        self._flash = None
        self._lone_result = False
        self._hold_started = None
        self._paint(None, None, moment=None, generation=int(self._generation), force=True)
        # `painted()` answers "what is on screen"; after a clear that is NOTHING, not a pair whose halves happen to be
        # None. The empty object invited exactly the kind of read this module exists to remove (a caller asking
        # `is not None` and concluding a picture is there), so the committed pair goes away with the picture.
        self._committed = None

    def release(self) -> None:
        """Forget what was only WAITING (a released RHI resource invalidates no decision).

        The committed pair stays: the renderer keeps its ingestion state across a release and re-publishes the standing
        pictures on the next `initialize()`, so dropping them here would blank a pane that is about to be redrawn.
        """
        self._pending_result = None
        self._pending_source = None
        self._flash = None
        self._hold_started = None

    # -- what is on screen ------------------------------------------------

    def painted(self) -> CommittedPair | None:
        """The pair the renderer holds, or None before the first hand-over."""
        return self._committed

    def committed_time(self) -> float | None:
        """The media time of the pair the renderer holds - the picture on screen, not the newest arrival."""
        pair = self._committed
        return None if pair is None else pair.moment

    def oldest_result_time(self) -> float | None:
        """The next picture to pair: the result waiting for its partner, or None when nothing is waiting."""
        pending = self._pending_result
        if pending is None or pending.moment is None:
            return None
        return float(pending.moment)

    # -- internals -------------------------------------------------------

    def _paint_single(self) -> bool:
        """One surface: the result is the picture and the source is the LMB flash beside it."""
        result = self._pending_result
        if result is None and self._committed is not None:
            result = self._committed.result
        self._pending_result = None
        return self._paint(result, self._flash, moment=None if result is None else result.moment,
                           generation=int(self._generation if result is None else result.generation))

    def _partner_for(self, result: Half) -> tuple[Half | None, bool]:
        """This result's other half: the source waiting for it, else the source already painted.

        The second candidate is a real case, not a fallback: a compare mode opened before its first result asks the host
        for the source at the playhead first, so by the time the result lands the picture is already on screen. The flag
        says whether the pending source was the one used (it is then spent).
        """
        pending = self._pending_source
        if pending is not None and self._pairs(result, pending):
            return pending, True
        painted = None if self._committed is None else self._committed.source
        if painted is not None and self._pairs(result, painted):
            return painted, False
        return None, False

    def _pairs(self, result: Half, source: Half) -> bool:
        """Whether `source` is the other half of `result`: same generation, same moment within the tolerance."""
        if result.moment is None or source.moment is None:
            return False
        if int(result.generation) != int(source.generation):
            return False
        return abs(float(source.moment) - float(result.moment)) <= self._tolerance

    def _hold_expired(self) -> bool:
        return self._hold_started is not None and (self._clock() - self._hold_started) >= MAX_PAIR_STALE_S

    def _paint(self, result: Half | None, source: Half | None, *, moment: float | None, generation: int,
               lone_half: str | None = None, force: bool = False) -> bool:
        """Hand the renderer whatever changed, and remember the pair as the one on screen.

        Only the halves whose PAYLOAD changed are handed over: a repaint of the same pair must not mark a 6 MB texture
        dirty (a source gap arrives 10-60 times a second during a drag, and an unconditional hand-over would re-upload
        the whole picture for each one).
        """
        pair = CommittedPair(result=result, source=source, moment=moment, generation=int(generation),
                             lone_half=lone_half)
        previous = self._committed
        handed = False
        if force or previous is None or not _same(previous.result, pair.result):
            self._sink.set_frame(None if result is None else result.frame)
            handed = True
        if force or previous is None or not _same(previous.source, pair.source):
            self._sink.set_original(None if source is None else source.frame)
            handed = True
        self._committed = pair
        # Every call here IS a decision (a pair, a lone half, an empty pane, a re-hand): a hold never reaches `_paint`,
        # so the clock that measured the hold is spent by definition.
        self._hold_started = None
        return handed
