"""The compare's source half: every picture that exists, named by the moment it belongs to.

One register, two producers, no clocks. The renderer's block and the comparison decoder both OFFER pictures here; the
page's one synchroniser asks for the moment the pane is actually showing and hands the pane *that* picture (or nothing).
Nothing in this module decides liveness: a picture either exists for a moment or it does not. The previous seam carried
a one-second "the block is alive" stamp, an "already shown" memo and a "newest decoded" pool, and each of them could
claim a picture the pane did not hold - seven failures in one day, one per flag. A register cannot make that claim: the
only question it answers is "do I hold a picture for this moment, and if so whose is it".

`picture_for` is BOUNDED on purpose. Outside the tolerance it answers ``None`` - never "the nearest" - because a
neighbour's picture is the desync this module exists to remove. `owner_of` is derived from what is actually held (the
millisecond key of the moment), not from a timestamp, so an old offer from a producer that has since gone quiet owns
nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QImage

__all__ = ["ResultPicture", "SourcePicture", "SourceRegister"]


@dataclass(frozen=True)
class SourcePicture:
    """One picture and the moment it belongs to: the identity every consumer must use.

    `generation` names the render/file/seek that produced it. A picture from a previous generation is not an answer for
    the pane's current result, even if its moment happens to be close - the same guard the result half carries.
    """

    moment: float
    picture: QImage
    origin: str
    generation: int = 0


@dataclass(frozen=True)
class ResultPicture:
    """One RESULT picture: its moment, its pixels, and the generation that produced it.

    The result defines the pane's clock, so it cannot be checked against "its own" moment; what can be checked is its
    identity: a generation that is not older than the one already accepted, a moment that does not go backwards within
    that generation, and a name when the compare needs a pair. `moment=None` with pixels is an UNNAMED publication; it
    must not inherit the previous pane moment.
    """

    moment: float | None
    picture: QImage | None
    generation: int


class SourceRegister:
    """A bounded set of `moment -> SourcePicture`, with a bounded (never nearest) lookup.

    The register is deliberately dumb: it has no notion of "live", "current" or "newest enough". A moment is an
    identity, quantised to a millisecond (the same grid the pane's pair queue uses); a picture offered for a moment
    replaces any previous picture for that moment, and the last producer to answer owns it.
    """

    def __init__(self, capacity: int = 32) -> None:
        self._capacity = max(1, int(capacity))
        self._held: dict[int, SourcePicture] = {}
        self.generation = 0
        # The moment somebody last ASKED for (the pane's own, via `picture_for`). Eviction is measured from here, not
        # from "newest": see `offer`.
        self._anchor: float | None = None
        #: Why the last `picture_for` answered nothing (`{"why": "generation"|"tolerance"|"empty", ...}`), or None.
        self.last_refusal: dict | None = None
        #: Level-2 trace of evictions and offers, drained by the owner (`take_trace`) - the register is Qt-free, so it
        #: records strings and the tab logs them.
        self._trace = False
        self._trace_events: list[str] = []

    def trace(self, on: bool = True) -> None:
        self._trace = bool(on)

    def take_trace(self) -> list[str]:
        events, self._trace_events = self._trace_events, []
        return events

    # -- writing ---------------------------------------------------------

    def offer(self, moment: float, picture: QImage, origin: str,
              generation: int = 0) -> SourcePicture | None:
        """Hold `picture` as the picture of `moment`, offered by `origin` ("block" or "decoder").

        A picture without a moment or without pixels cannot be matched to anything, so it is refused rather than
        stored under a lie. Re-offering the same moment replaces the picture (the last answer wins) and bumps the
        generation; once the register is full the moment FARTHEST FROM THE ONE THAT WAS ASKED FOR is dropped.
        `generation` names the render/seek/file that produced the picture, so a later lookup can refuse an answer from
        a previous one.

        Eviction is measured from the last `picture_for` call, not from "newest wins". The old rule was wrong in the
        one case a comparison actually spends its life in: a PAUSED pane asks for one moment over and over while the
        fill keeps a decode lead ahead of it, so every lead frame made the pane's own picture the oldest entry and it
        was evicted by the next walk - the half went blank, then reappeared, then blanked again (owner, 2026-11-27:
        "теперь оригинал умеет пару раз померцать даже без запуска плей"). A cache must never drop the answer it is
        being asked for.
        """
        if moment is None or picture is None:
            return None
        key = self._key(moment)
        if key is None:
            return None
        held = SourcePicture(float(moment), picture, str(origin), int(generation))
        self._held[key] = held
        if self._trace:
            self._trace_events.append(f"offer pts={held.moment:.3f} origin={held.origin} gen={held.generation}")
        self._prune()
        self.generation += 1
        return held

    def _prune(self) -> None:
        """Drop the entries farthest from the anchored moment until the register fits its capacity.

        Without an anchor (nobody has asked yet) this is the previous behaviour, oldest first. With one, the entry
        that answers the anchor is the LAST thing to go: ties go to the older entry, so a walk in one direction
        cannot pin a stale neighbour just because its distance is equal.
        """
        while len(self._held) > self._capacity:
            anchor = self._anchor
            if anchor is None:
                del self._held[min(self._held)]
                continue
            # Protect the picture that ANSWERS the anchor; otherwise drop the OLDEST. "Farthest from the anchor" threw
            # away the LEAD - the frames the pane is about to need - while spent frames behind the playhead stayed, so
            # the pool looked full (`held=24`) and sat one to three frames behind the result for the whole render
            # (measured live 2026-11-27: `compare ask newest=18.101 playhead=18.143`, `pair_hold` on every other tick).
            anchor_key = self._key(anchor)
            candidates = [key for key in self._held if key != anchor_key]
            if not candidates:
                break                              # nothing but the anchored picture: capacity 1 keeps it
            victim = min(candidates)
            if self._trace:
                held = self._held[victim]
                self._trace_events.append(f"evict pts={held.moment:.3f} origin={held.origin} anchor={anchor:.3f}")
            del self._held[victim]


    def clear(self) -> None:
        """Forget every picture (a decoder/file change must not leave the previous file's frames answerable)."""
        if self._held:
            self._held.clear()
            self.generation += 1
        # A new file/generation has its own moments: the previous anchor names a place in a timeline that is gone.
        self._anchor = None

    def counts(self) -> dict[str, int]:
        """How many pictures each producer holds, by origin - so a full `held=` cannot be read as "the source is here"."""
        counts: dict[str, int] = {}
        for held in self._held.values():
            counts[held.origin] = counts.get(held.origin, 0) + 1
        return counts

    def drop_origin(self, origin: str) -> None:
        """Forget the pictures one producer supplied (its block/decoder closed; its frames are not answers any more)."""
        stale = [key for key, held in self._held.items() if held.origin == origin]
        if not stale:
            return
        for key in stale:
            del self._held[key]
        self.generation += 1

    # -- reading ---------------------------------------------------------

    def picture_for(self, moment: float, tolerance_s: float,
                    generation: int | None = None) -> SourcePicture | None:
        """The picture held for `moment`, or `None` when none is within `tolerance_s` of it.

        Bounded by the caller's tolerance and NEVER a fallback to the nearest: a picture more than half a frame away
        is a picture of a different moment, and showing it is the desync. Within the tolerance the closest held moment
        wins (ties go to the later moment); the tolerance is the caller's statement of "these are the same picture".
        When `generation` is given, only pictures from that generation answer.
        """
        if moment is None:
            return None
        try:
            moment = float(moment)
            tolerance = max(0.0, float(tolerance_s))
        except (TypeError, ValueError):
            return None
        # The ask IS the anchor: this is the moment the register must be able to answer next time, so eviction is
        # measured from it (see `offer`).
        self._anchor = moment
        wanted_generation = None if generation is None else int(generation)
        best: SourcePicture | None = None
        best_distance: float | None = None
        #: Why the last lookup answered nothing, for the instrument: a hold that cannot say which filter refused it
        #: leaves the next reader guessing (2026-11-27: 24 pictures held, every result held, and no line said whether
        #: the generation or the tolerance was the gate).
        nearest_any: SourcePicture | None = None
        nearest_any_distance: float | None = None
        nearest_same_generation: SourcePicture | None = None
        nearest_same_generation_distance: float | None = None
        for key in sorted(self._held):
            held = self._held[key]
            distance = abs(float(held.moment) - moment)
            if nearest_any_distance is None or distance < nearest_any_distance:
                nearest_any, nearest_any_distance = held, distance
            if wanted_generation is not None and int(held.generation) != wanted_generation:
                continue
            if nearest_same_generation_distance is None or distance < nearest_same_generation_distance:
                nearest_same_generation, nearest_same_generation_distance = held, distance
            # A tolerance written as `1.0 / fps` and a moment subtracted in floats land a hair over it (2.0 - 1/24 is
            # 0.04166666666666674 from 2.0 against a 0.041666666666666664 bound) - an exact one-frame answer must not
            # be refused by the last bit of a double.
            if distance > tolerance + 1e-9:
                continue
            if best_distance is None or distance < best_distance or (
                    distance == best_distance and held.moment > best.moment):
                best, best_distance = held, distance
        if best is None:
            if nearest_same_generation is None and nearest_any is not None:
                reason = "generation"
            elif nearest_same_generation is not None:
                reason = "tolerance"
            else:
                reason = "empty"
            self.last_refusal = {
                "why": reason,
                "wanted": round(float(moment), 3),
                "tolerance": round(float(tolerance), 4),
                "wanted_generation": wanted_generation,
                "nearest_moment": None if nearest_any is None else round(float(nearest_any.moment), 3),
                "nearest_generation": None if nearest_any is None else int(nearest_any.generation),
                "nearest_distance": None if nearest_any_distance is None else round(nearest_any_distance, 4),
                "held": len(self._held),
            }
        else:
            self.last_refusal = None
        return best

    def owner_of(self, moment: float, generation: int | None = None) -> str | None:
        """Which producer holds a picture FOR THIS MOMENT ("block"/"decoder"), or `None`.

        Identity, not adjacency: the moment is looked up on the register's millisecond grid. A block that last
        published five seconds ago - or one moment ago, at a different moment - owns nothing here. That is the fact
        `_source_fill` needs to decide whether the decoder has anything left to supply. A picture from another
        `generation` owns nothing for this one.
        """
        key = self._key(moment)
        if key is None:
            return None
        held = self._held.get(key)
        if held is None:
            return None
        if generation is not None and int(held.generation) != int(generation):
            return None
        return held.origin

    def newest_moment(self, generation: int | None = None) -> float | None:
        """The latest moment the register holds - the decode lead's own frontier, not a clock."""
        moments = [held.moment for held in self._held.values()
                   if generation is None or int(held.generation) == int(generation)]
        return max(moments) if moments else None

    def __len__(self) -> int:
        return len(self._held)

    def __contains__(self, moment: object) -> bool:
        key = self._key(moment)
        return key is not None and key in self._held

    @staticmethod
    def _key(moment: object) -> int | None:
        try:
            return int(round(float(moment) * 1000.0))
        except (TypeError, ValueError):
            return None
