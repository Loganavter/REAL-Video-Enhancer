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
            # WHICH filter refused it, at level 1 - and ONLY for a PAINT. This function is also called from the
            # source-feed offer path (`_sync_compare_source`), so logging every miss made the counter measure the fill's
            # questions rather than the picture: one window had 1020 gap lines for 3 genuinely missing painted moments
            # (measured 2026-11-27). `paint=True` comes from `paint_pair`, the one path where a result has just been
            # painted and the pane needs its other half NOW.
            why = getattr(register, "last_refusal", None)
            if why:
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
