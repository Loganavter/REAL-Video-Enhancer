"""Presentation: hand the pane the compare's source half for the moment the pane ITSELF shows.

Owns, and is the only path for: taking the pane's own moment (`shown_moment()`), asking the register for the picture of
that moment (a bounded lookup), and handing the answer to the pane (`set_source`, or the legacy `set_original` for a
double that predates the register API). A moment with no picture is handed over as `None` - a gap is the honest answer,
a neighbour's picture is not.

`paint_pair` is the same path under the name the paint routes look up when a result has just been painted, so both
halves change in one repaint.

May not touch: the view mode, the choice of a picture, a queue, a decoder, the register's contents, or any other state
on the tab. There is deliberately no mode/picture/queue branch in here - each of those was a place where a flag could
disagree with what the pane actually held, which is the whole defect class this wave removes. No widget is constructed
and no signal is connected: the tab keeps its wiring and its state.

Reads from the tab: `page_handles.proc.previewLabel`, `_source_register`, `_result_generation`,
`_compare_playhead()`, `_source_tolerance()`, `_preview_playhead` (the fallback when `_compare_playhead` raises).
"""

from __future__ import annotations

__all__ = ["paint_pair", "sync_compare_source"]


def sync_compare_source(tab, *, paint: bool = False) -> None:
    """Hand the pane the register's picture FOR THE MOMENT IT IS SHOWING, or nothing.

    The only presentation path for the compare's source half. `shown_moment()` is the result's own moment (the
    pane's state, not a tab clock), `picture_for` is a BOUNDED match, and `set_source` is the pane's only source API.
    A moment with no picture is handed over as `None`: a gap is the honest answer, a neighbour's picture is not.
    There is deliberately no mode/picture/queue branch here - each of those was a place where a flag could disagree
    with what the pane actually held, which is the whole defect class this wave removes.
    """
    pane = getattr(getattr(getattr(tab, "page_handles", None), "proc", None), "previewLabel", None)
    if pane is None:
        return
    # Tell the pair writer whether the source side is alive, on EVERY sync (a paint and an arrival both come through
    # here): while it is, the hard limit holds rather than painting the result alone - at the start of a render the
    # decoder is still walking to the playhead and the valve used to expire first (owner's log 2026-11-27: three
    # `pair stale ... waited_s=1.50` in the first five seconds of media).
    note_alive = getattr(pane, "note_source_alive", None)
    if callable(note_alive):
        try:
            import time as _time

            now = _time.monotonic()
            arrived = getattr(tab, "_source_last_arrival_mono", None)
            asked = getattr(tab, "_source_last_ask_mono", None)
            fresh_arrival = arrived is not None and (now - float(arrived)) < 2.0
            # An ASK without an arrival is a rewind in progress - the decoder is working, it is simply silent until the
            # walk reaches the target. Only while the register has produced something in the last five seconds, so a
            # decoder that is gone still lets the hard limit fire.
            asking = (asked is not None and (now - float(asked)) < 3.0
                      and arrived is not None and (now - float(arrived)) < 5.0)
            # Neither stamp yet: the switch has just OPENED the decoder (a 1440p open costs a few hundred ms) and not a
            # single ask has gone out. That is not a dead source - it is a source that has not been asked yet - so the
            # hard limit waits for the first ask instead of painting the result alone in the meantime (measured
            # 2026-11-27: `pair stale ... moment=2.000/2.125/3.500 generation=1`, all before the first arrival).
            opening = (arrived is None and asked is None
                       and getattr(tab, "_source_decoder", None) is not None)
            note_alive(fresh_arrival or asking or opening)
        except Exception:
            pass
    moment_of = getattr(pane, "shown_moment", None)
    moment = None
    if callable(moment_of):
        try:
            moment = moment_of()
        except Exception:
            moment = None
    if moment is None:
        # An unnamed result (a queue item, a compare opened before the first frame) still has a playhead; the
        # register can answer for it, so the source half is not forced to wait for a result that does not exist.
        try:
            moment = float(tab._compare_playhead())
        except Exception:
            moment = float(getattr(tab, "_preview_playhead", 0.0) or 0.0)
    register = getattr(tab, "_source_register", None)
    snapshot = None
    if register is not None:
        snapshot = register.picture_for(moment, tab._source_tolerance(),
                                        generation=int(getattr(tab, "_result_generation", 0)))
        if snapshot is None and paint:
            # WHICH filter refused it, at level 1 - and ONLY for a PAINT, once per PAINTED RESULT. This function is also
            # called from the source-feed offer path (`_sync_compare_source`), so logging every miss made the counter
            # measure the fill's questions rather than the picture: one window had 1020 gap lines for 3 genuinely missing
            # painted moments (measured 2026-11-27). `paint=True` comes from `paint_pair`, the one path where a result has
            # just been painted and the pane needs its other half NOW - and that path runs more than once per painted
            # result (the pane's own repaint and the decoder's arrival both reach it), so the moment is deduped: one
            # line per result, not per call. The set lives on the tab and is dropped when the result generation moves,
            # because a new generation is a new picture and its miss is news again.
            why = getattr(register, "last_refusal", None)
            if why and _first_gap_for(tab, why):
                try:
                    from ...video import telemetry

                    if telemetry.level() >= 1:
                        telemetry.event("compare gap", **{k: v for k, v in why.items() if v is not None})
                except Exception:
                    pass
    setter = getattr(pane, "set_source", None)
    if callable(setter):
        setter(snapshot)
        return
    # A pane from before the register API (an old harness double): one compatibility call, through the same moment
    # check the pane's `set_source` applies.
    legacy = getattr(pane, "set_original", None)
    if callable(legacy):
        if snapshot is None:
            legacy(None)
        else:
            legacy(snapshot.picture, media_time=snapshot.moment)


def _first_gap_for(tab, why: dict) -> bool:
    """True the first time THIS painted moment is reported missing; False for its repeats.

    A gap is a property of a picture, not of a call: `paint_pair` runs for the pane's repaint and again when the
    decoder lands, so an undeduped counter reports the same missing moment several times and the number stops meaning
    "how many painted moments had no source". The key is the moment plus the generation it was wanted in; the set is
    capped so a long job cannot grow it without bound.
    """
    try:
        key = (int(why.get("wanted_generation") or 0), float(why.get("wanted")))
    except Exception:
        return True
    seen = getattr(tab, "_compare_gap_seen", None)
    generation = key[0]
    if not isinstance(seen, set) or getattr(tab, "_compare_gap_generation", None) != generation:
        seen = set()
        tab._compare_gap_seen = seen
        tab._compare_gap_generation = generation
    if key in seen:
        return False
    seen.add(key)
    if len(seen) > 8192:
        seen.clear()
        seen.add(key)
    return True


def paint_pair(tab, playhead: float | None = None) -> None:
    """One entry point for "the result has just been painted; give it its other half in the same repaint".

    A minimal host (a harness that builds the tab without `__init__`) reaches the paint paths with nothing to pair,
    which is why the call sites look it up instead of calling it. The presentation rule itself is the pane's: the
    frame it now shows, the register's bounded answer, `set_source`. A caller that wants to force one moment can
    still name it, but the moment is only ever used to READ the register - never to assert a picture exists.
    """
    # The same one path as `sync_compare_source`, without the delegator hop: the tab's `_source_paint_pair` exists to
    # BE this call, so it calls the module directly rather than looping through its own `_sync_compare_source`.
    # `paint=True`: this call happens because a result was just painted, so a miss here is a gap on screen.
    sync_compare_source(tab, paint=True)
