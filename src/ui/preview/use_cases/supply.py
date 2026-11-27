"""Supply: keep the compare's source decoder warm and the register topped up ahead of the playhead.

Owns: the decision to open the compare decoder (warm), the ask for the decode lead (one ask in flight, the arrival
counter's `asks` side), the pause policy (a still playhead is asked for ONE picture at its own moment, never for a lead
nobody will watch), and the tolerance at which two moments are still the same picture.

May not touch: the pane (no `set_source`, no picture is handed over here - presentation is `present.py`), the view mode,
the register's contents (producers `offer`, this module only asks), the result generation, or any state the tab does not
already own. No widget is constructed and no signal is connected: the tab keeps its wiring and its state.

Reads from the tab: `page_handles.proc.previewLabel`, `currentRenderOptions.inputFile`, `isPreview`, `_source_decoder`,
`_source_warm_tried`, `_source_register`, `_source_fill_pending`, `_result_generation`, `_preview_paused`,
`_compare_fill_stats`, `_compare_from_tick`, and the methods `_compare_playhead()`, `_compare_fps()`,
`_compare_lead()`, `_on_original_requested()`, `_open_source_decoder()`. It writes exactly what the tab already wrote:
`_source_fill_pending`, `_source_warm_tried`, `_compare_fill_stats["asks"/"target"/"span"/"newest"]`, `_compare_from_tick`.
"""

from __future__ import annotations

import time

from ..owner import owner_of
from ..state import PreviewBackendKind

__all__ = ["COMPARE_FILL_WINDOW_S", "source_fill", "source_tolerance", "supply_compare_source",
           "warm_source_decoder"]

#: How long one source ask counts as "in flight". A span of `COMPARE_LEAD_FRAMES` costs ~40 ms of decode, so this is a
#: retry deadline, not a throttle: an ask nothing answers is re-sent on the next tick after it, which keeps a paused
#: pane's first frame arriving as quickly as the retry itself. It must stay BELOW the picture interval of a live tail
#: (~60 ms at 16 fps): at 250 ms a pending blocked two to three fills per frame, the pool fell 1-3 frames behind the
#: result and every pair timed out (measured 2026-11-26, split during inference).
#:
#: Moved here with `source_fill` (the only reader). `ProcessTab.COMPARE_FILL_WINDOW_S` is the same number and becomes
#: an alias of this one when the tab's body is deleted - one fact, one owner.
COMPARE_FILL_WINDOW_S = 0.08


def supply_compare_source(tab) -> None:
    """Warm the compare decoder and ask for the lead while the register does not hold what it would ask for.

    `single` keeps the LMB flash (one picture on demand, no standing fill); a queue item IS the source and has no
    second surface. Those are SUPPLY decisions, not presentation ones - `sync_compare_source` above needs none.
    """
    pane = getattr(getattr(getattr(tab, "page_handles", None), "proc", None), "previewLabel", None)
    view_mode = getattr(pane, "view_mode", None)
    if view_mode is None:
        return
    if owner_of(tab).kind is PreviewBackendKind.QUEUE_FILE:
        return
    try:
        mode = view_mode()
    except Exception:
        return
    if mode == "single":
        # Warm first: the point of warming is to be ready when the mode CHANGES, and during a render the pane is
        # `single` - a warm-up below this return never ran, which is how the decoder ended up opening on the switch
        # itself (~0.8 s of the owner's wait).
        warm_source_decoder(tab)
        return
    warm_source_decoder(tab)
    options = getattr(tab, "currentRenderOptions", None)
    if not (getattr(options, "inputFile", "") or ""):
        return
    # The SAME clock presentation uses: the pane's own moment. `_result_playhead` prefers the transport while a tail
    # or a file decoder exists, and after a render hands over the finished file it reads 0.0 while the pane already
    # paints its first frame - so this guard returned before asking, and the source half never arrived (found live
    # 2026-11-27 with the pair gate in place: `compare fill asks=0 newest=0.0 playhead=0.167` for a whole run, the
    # left half blank with the upscale painted beside it).
    playhead = tab._compare_playhead()
    if playhead <= 0.0:
        return
    # The fill asks at the moment PRESENTATION reads - the pane's own. It used to ask at
    # `_compare_supply_playhead()` (the pair queue's oldest result, since deleted with its wrapper), a third clock; the answer was then refused by
    # the pane's moment guard and the half stayed blank while the decoder refilled a moment nobody showed (measured
    # live 2026-11-27: `playhead=0.167 newest=1.709 target=0.0` with the register full of render-head frames, the
    # left half empty beside the upscale). One clock: what the pane shows is what the supply fetches.
    source_fill(tab, playhead)


def source_fill(tab, playhead: float) -> None:
    """Top the register up to `COMPARE_LEAD_FRAMES` ahead of the playhead, one ask per call.

    The next call comes from the frame's ARRIVAL (`_on_original_frame`): the decoder then runs flat out until the
    lead is full and stops, instead of being polled at a timer's rate - a 10 Hz poll cannot feed a 24 fps stream, so
    the lead would drain exactly when it is needed. The register is the pool; there is no separate prune, no "newest
    decoded" flag, and no clock deciding who owns a moment.
    """
    playhead = float(playhead)
    register = getattr(tab, "_source_register", None)
    if register is None:
        return
    interval = 1.0 / tab._compare_fps()
    generation = int(getattr(tab, "_result_generation", 0))
    if register.owner_of(playhead, generation=generation) == "block":
        # The renderer is publishing THIS moment's source: decoding the same file a second time is exactly the cost
        # this block exists to remove. An older block offer owns nothing, so it cannot suppress the decoder.
        return
    newest = register.newest_moment(generation=generation)
    # ONE ask in flight at a time: the arrival of each frame calls this again, and without the guard a burst would
    # start a second span for the same moment (the decoder keeps only the last request, so the work is wasted).
    pending = getattr(tab, "_source_fill_pending", None)
    if pending is not None:
        aged = time.monotonic() - float(pending[1])
        if newest is not None and newest >= float(pending[0]) - 0.5 * interval:
            tab._source_fill_pending = None
        elif aged < COMPARE_FILL_WINDOW_S:
            # ONE ask in flight, and it is the frame on screen: the decoder keeps a single request slot, so a burst of
            # asks collapses to its LAST one and the moment in the middle of the burst is lost (measured 2026-11-27:
            # 285 requests reached the decoder, 28 pump runs executed). Blocking here is what keeps the request that
            # matters: the strict-miss ask is always the playhead, so the pending is the pane's own frame.
            try:
                from ...video import telemetry

                if telemetry.level() >= 2:
                    telemetry.event("compare fill blocked", aged_ms=round(aged * 1000.0, 1),
                                    target=round(float(pending[0]), 3),
                                    newest=None if newest is None else round(float(newest), 3))
            except Exception:
                pass
            return
        else:
            tab._source_fill_pending = None      # a request nothing answered: ask again

    span = True
    # A PAUSED pane asks for one picture at its moment and never for a lead. The lead is a playback cost (the
    # decoder's ~8 frames/s latency is hidden only when the playhead is moving); with a still playhead it decodes
    # frames nobody will see - and, because the oldest entry went first, every lead walk used to evict the very
    # picture the pane was showing (owner, 2026-11-27: the original blinked a couple of times with playback never
    # started). The register's own anchor rule now protects that picture; this one stops the waste at the ask.
    paused = bool(getattr(tab, "_preview_paused", False))
    # IS THE PLAYHEAD MOVING, or did it JUMP? Both are "not paused", and the difference decides what to ask for: a
    # continuous advance (~a frame per tick) is playback and needs the LEAD, or every ask chases a moment that has
    # already moved on - measured live 2026-11-27 in the owner's session: the source lagged 21 frames, 29 results were
    # refused as older than the picture, and the hold expired 41 times, leaving 41 sbs draws without an original.
    # A jump (a scrub, a press) is one moment and gets exactly one frame.
    previous = getattr(tab, "_source_fill_last", None)
    moving = False
    if previous is not None:
        step = playhead - float(previous)
        moving = 0.0 < step <= 4.0 * interval     # a tick or two of slip still counts as playback, not a jump
    tab._source_fill_last = playhead
    # 1.5 frames: the fill's "is this moment worth a decode" is deliberately LOOSER than the pane's half-frame
    # identity bound. Matching the pane's bound exactly was tried and measured WORSE (delivered-at-the-moment 32/91 vs
    # 88/101): it turned every neighbour into a hole, the fill asked at the playhead on every result, every ask became a
    # keyframe rewind (~176 ms), and the decoder spent the window rewinding instead of leading (`decode rewind` 37 in
    # one window against 20). A looser coverage test with a strict presenter is the trade that keeps the decoder
    # leading; the hole to fix is on the PRODUCER side (the source channel publishes on a 2-frame step).
    # TWO facts, not one: `strict` is the PANE's own bound (would its lookup find a picture?), `covered` is the looser
    # 1.5-frame "is the lead worth asking for". The loose one must never suppress the strict one - that conflation sent
    # the ask to the lead edge while the frame on screen was missing, the walk then started AFTER it, and the frame was
    # never delivered (`compare ask target=18.06 playhead=18.018`, no delivery of 18.018, measured 2026-11-27).
    strict = register.picture_for(playhead, tab._source_tolerance(), generation=generation) is not None
    covered = register.picture_for(playhead, 1.5 * interval, generation=generation) is not None
    # A PROBE is not a presentation: the `compare gap` line belongs to the painter (`present.sync_compare_source`), or
    # the counter measures every coverage question the fill asks and cannot answer the acceptance ("zero gaps at the
    # moment the result paints"). Measured 2026-11-27: 1020 gap lines for 3 genuinely missing painted moments.
    register.last_refusal = None
    if not strict and moving and not paused:
        # The pane's frame is missing: ask PAST it by the lead, not AT it.
        #
        # Asking AT the playhead is what the earlier finding demanded (`8ef9493f`: "a register that does not reach the
        # playhead asks for the playhead, not for the lead beyond it") - and that was right for the decoder of the time,
        # which published only the LAST frame of a walk and CONSUMED the rest, so an ask past the playhead skipped it.
        # The walk now publishes EVERY frame it decodes (`358a50bc`, streamed in `192f6388`), so a target past the
        # playhead covers the playhead on the way and leaves the buffer the pane needs at the NEXT paint.
        #
        # Measured 2026-11-27 with the ask AT the playhead: in a window of 300 painted moments the source was already
        # in the register at paint time for ZERO of them - every walk ended exactly where the playhead had been when the
        # ask was made, and the next result painted a moment further on, so the source was structurally one burst
        # behind. The lead is what turns "decode what was just shown" into "decode what is about to be shown".
        target = playhead + tab._compare_lead()
        span = True
    elif not covered and moving and not paused:
        # Playback with nothing held for this moment: ask AHEAD, over a span, so one decode covers the stretch the
        # pane is about to show. A single frame here cannot win a race against the next tick - UNLESS the register does
        # not even REACH the playhead (a fresh decoder after a restart, a register cleared by a mode/owner change):
        # asking ahead then walks away from the one moment the pane is showing, and nothing ever fills it - measured
        # live 2026-11-27: `compare gap why=tolerance wanted=16.600` (frame 398 at 23.976) with 397 and 399 held, while
        # `compare put pts=16.6` proves the frame HAD been offered before the register was emptied. The playhead's own
        # picture comes first; the lead starts on the next arrival.
        reaches_playhead = newest is not None and float(newest) >= playhead - interval
        # ... and the symmetric hole: the pool can be AHEAD of the frame on screen (the decoder walked past it), which
        # is what the counters showed - `nearest_moment` one frame AFTER the wanted one, and that wanted frame never
        # delivered at all. Asking ahead there fetches the future and leaves the hole; ask AT the playhead, which is
        # behind the read position, so the decoder takes its rewind path.
        # ONE frame ahead is already a hole at the moment on screen: that is what the counters showed (`wanted=14.056`,
        # `nearest_moment=14.097` - its neighbour, one frame ahead, and `decode rewind/step target=14.056` never
        # happened). The ask must go to the playhead (a rewind) whenever the pool holds nothing at or BEFORE it.
        skipped_playhead = newest is not None and float(newest) > playhead + 0.5 * interval
        if not reaches_playhead or skipped_playhead:
            target = playhead
            span = False
        else:
            target = playhead + tab._compare_lead()
            span = True
    elif not covered:
        # Nothing in the register answers for THIS playhead (an empty register, a seek, a jump back): ask at it, and
        # take the single frame that answers. A seek's answer is one picture, not a burst from wherever the decoder
        # stands.
        target = playhead
        span = False
    elif paused:
        # The pane holds the moment it is showing and is not moving: there is nothing to ask for.
        return
    elif newest is None:
        target = playhead
        span = False
    elif playhead - newest > 2.0 * interval:
        # The register is BEHIND the playhead - a scrub, or a SPARSE tail (its lateness rule skips pictures, so two
        # consecutive ones can be six frames apart). One frame per call cannot close a gap the playhead reopens on
        # every picture: measured 2026-11-26, `pool=1` for ever and every pair a timeout. Ask for the far edge of the
        # lead and take the whole walk - the burst IS the point. It is decoded FORWARD, not rewound: a rewind's ceil
        # landed a frame off the tail's sparse grid while the forward walk lands on the frame itself.
        target = playhead + tab._compare_lead()
        span = True
    elif newest < playhead + tab._compare_lead():
        # Ask for the WHOLE lead in one call and take every frame of the walk: one round trip per lead, not one per
        # frame (a frame-by-frame ask is latency-bound at ~8 frames/s; the decode itself runs at 80).
        target = playhead + tab._compare_lead()
    else:
        return
    if register.owner_of(target, generation=generation) is not None and \
            register.owner_of(playhead, generation=generation) is not None:
        # Both ends of the ask are already held, so the decode would fetch nothing new. The PLAYHEAD has to be part of
        # this test: the source feed is a lossy preview channel (it published a 2-frame step in the owner's sbs window:
        # 16.558 -> 16.642 while the pane wanted 16.600), so a held LEAD EDGE says nothing about the frame on screen -
        # and with a two-frame step the edge is held almost every tick, which made the fill stand down in silence
        # (`compare gap wanted=16.600`, no offer of 16.600 by any producer, measured 2026-11-27).
        return
    tab._source_fill_pending = (float(target), time.monotonic())
    try:
        from ...video import telemetry

        if telemetry.level() >= 2:
            telemetry.event("compare ask", target=round(float(target), 3), span=int(bool(span)),
                            playhead=round(playhead, 3), newest=None if newest is None else round(float(newest), 3),
                            held=len(register))
    except Exception:
        pass
    stats = getattr(tab, "_compare_fill_stats", None)
    if stats is not None:
        stats["asks"] += 1
        stats["target"] = round(float(target), 3)
        stats["span"] = int(bool(span))
        stats["newest"] = None if newest is None else round(float(newest), 3)
    tab._compare_from_tick = True
    try:
        tab._on_original_requested(max(0.0, target), span=span)
    finally:
        tab._compare_from_tick = False


def warm_source_decoder(tab) -> None:
    """Open the source decoder while a job runs, so the first switch to a compare mode is just a seek.

    Called from the GUI tick (`_tick_second_surface`), synchronously: a `QTimer` scheduled from here never fired in a
    live run (measured 2026-11-26 - a whole render went by with no `source decoder` line and the first switch still
    paid the open), and `renderToPipeThread` is a worker thread where a timer cannot live at all. One open per job,
    on the thread that owns the decoder, guarded so the tick cannot retry a failing file 10 times a second.

    Why it matters: the open plus the first seek is the ~0.8 s the owner measured between the result half appearing
    and the source half ("апскейл сразу справа отобразился, а оригинал - н-ое количество времени ожидания").
    """
    if getattr(tab, "_source_decoder", None) is not None or getattr(tab, "_source_warm_tried", False):
        return
    options = getattr(tab, "currentRenderOptions", None)
    if not (getattr(options, "inputFile", "") or ""):
        return
    tab._source_warm_tried = True
    if getattr(tab, "isPreview", False):
        # The owner's case: a preview's first switch to a compare mode paid the open. A preview is the one flow where
        # the pane is guaranteed to be shown and the switch is a human action a second away - open ONCE, here, on the
        # GUI thread (a `QTimer` scheduled from this tick never fired in a live run: measured 2026-11-26, a whole
        # render went by with no `source decoder` line).
        tab._open_source_decoder()
        return
    from PySide6.QtCore import QTimer

    QTimer.singleShot(700, tab._open_source_decoder)


def source_tolerance(tab) -> float:
    """Half a source frame: the largest distance at which a decoded picture is THE SAME picture.

    Kept at half a frame deliberately - the committed pair's halves must name one moment. The one-frame gap that made
    every source un-matchable during a live render is fixed at the ASK (the half-frame bias in the caller), not by
    widening this bound: measured live 2026-11-27, `compare gap why=tolerance nearest_distance=0.041` (exactly one
    frame) against `tolerance=0.0209`.
    """
    return 0.5 / max(1.0, tab._compare_fps())

