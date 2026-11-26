"""Seek use cases: a press on the strip, and the renderer's one-frame request.

What this module owns:

* routing a press/drag to the object that can answer it (queue item, live tail, finished-file decoder, the renderer's
  on-demand block), and refusing when no route can;
* the result-generation bump: it happens ONLY where a route actually asks (`bump_generation`), never at the top of the
  press. That is the fix for audit §6.3 (commit `b33df4c1`): a refused press used to bump `_result_generation` and clear
  the compare register while nothing on screen moved, so every source offer was refused as the wrong generation until
  the next result - the blank left half after a click, and a register wiped 10-60x/second during a drag;
* the display-seek request (`(seconds, generation + 1)` in the renderer's one slot, newest wins) and its parking.

What it may NOT touch: widgets, signal connections, the pane's picture APIs, or any state of its own. Every helper it
calls is a method with its current name on the tab (the owner), and the tab keeps a delegator with the old name.
"""

from __future__ import annotations

import time

from ...video import telemetry
from ..owner import owner_of
from ..state import PreviewBackendKind

__all__ = ["on_preview_seek", "request_display_seek"]


def _log(message: str) -> None:
    """Write one line through the app's log, imported lazily so this module imports without the app's deps."""
    from ....Util import log

    log(message)


def on_preview_seek(tab, fraction: float) -> None:
    """A press or drag on the strip seeks whoever owns the picture - the ROUTE is all that is decided here.

    Live, that is the tail: it repositions inside what the renderer has already written, and the next poll
    (120 ms) shows the target. A press PAST the produced band is different only while the pane's block is usable:
    the tail has nothing there, but the renderer can render that one frame on demand - so the press becomes a
    newest-wins request (`_request_display_seek`) and the pane holds it until play hands it back. A finished file
    seeks its own decoder. The strip reports a share of the WHOLE timeline, so while the timeline length is
    unknown the produced share stands in for it - which is the same assumption the strip itself makes when it
    fills its white band.

    The REACH is not checked here: `PreviewSession.seek_to` has already refused a target past it before this
    method is called (plan P1 - the clamp lives in one place). The press record is the session's too: this method
    asks for the ticket for the press it is routing and fills in the route facts only the page knows.

    The generation is minted by a route that asks, never by the attempt (audit §6.3, fix `b33df4c1`).
    """
    fraction = float(fraction)
    # A SEEK starts a new result generation - but only a seek that is actually ASKED FOR. The bump used to run here,
    # before any route was chosen, so a press the pane could not serve invalidated the picture identity and cleared
    # the compare register while nothing on screen moved (audit 2026-11-27 §6.3, probe B: a refused press took
    # generation 7 -> 8 and the register 3 -> 0, and every source offer was refused as the wrong generation until
    # the next result arrived - the owner's blank left half after a press). Every route below bumps when it asks.
    bumped = False

    def bump_generation() -> None:
        nonlocal bumped
        if not bumped:
            bumped = True
            tab._next_result_generation()
    tab._seek_pressed_at = time.perf_counter()
    # The press names the FRAME it points at, not only the share: a turnaround in milliseconds says nothing about
    # whether the picture is the one asked for (owner, 2026-11-24: "write where the seek went and what is shown").
    total_frames = float(getattr(tab, "max_value", 0) or 0)
    if total_frames <= 0:
        fps = float(getattr(getattr(tab, "_preview_decoder", None), "fps", 0) or 0)
        duration = float(getattr(tab, "_preview_duration", 0.0) or 0.0)
        total_frames = round(duration * fps)
    if total_frames <= 0:
        # The render publishes its total late; the CLIP's own frame count is known from the probe at load time and
        # is the right denominator for a share -> frame conversion in the meantime.
        total_frames = float(getattr(tab, "videoFrameCount", 0) or 0)
    target_frame = int(round(fraction * total_frames)) if total_frames > 0 else None
    tab._seek_target_frame = target_frame
    # The produced share goes with the press: a press past what the renderer has written is a different event from
    # a press inside it, and only the log can tell them apart afterwards.
    produced = 0.0
    total = float(getattr(tab, "max_value", 0) or 0)
    if total > 0:
        produced = max(0.0, min(1.0, float(getattr(tab, "currentFrame", 0) or 0) / total))
    target_s = None
    owner = owner_of(tab)
    duration_s = float(tab._clip_duration(owner) or 0.0)
    if duration_s > 0.0:
        target_s = fraction * duration_s
    tab._seek_target_s = target_s
    telemetry.event("seek press", fraction=f"{fraction:.4f}", produced=f"{produced:.4f}",
                    target_s=None if target_s is None else f"{target_s:.3f}",
                    target_frame=target_frame)
    # One record per press, created BEFORE a route is chosen: the intent, the route it took, whether that route
    # can answer it, and (from whichever painter answers) the picture. Without this a press could end in silence -
    # the owner's three sessions pressed four times each and left no line about the result at all (2026-11-24).
    # The record lives in the pane's press session now (plan P1): a press that came through the controller already
    # owns one, and `press()` hands that back instead of opening a second; a direct press opens its own.
    edge_s = produced * duration_s if duration_s > 0.0 else None
    ticket = tab._press_session().press(
        target_s=target_s, target_frame=target_frame, produced=produced, edge_s=edge_s)
    tab._press = ticket
    if owner.kind is PreviewBackendKind.QUEUE_FILE:
        decoder = getattr(tab, "_queue_pane_decoder", None)
        duration = float(getattr(decoder, "duration", 0.0) or 0.0)
        if decoder is not None and duration > 0.0:
            ticket.decide(route="queue", answerable=True)
            # The queue item is a finished file: the frame that lands answers the press, so the session watches
            # the same seconds the decoder is asked for (plan P4).
            tab._press_session().watch(fraction * duration)
            try:
                decoder.seek(fraction * duration)
            except Exception as exc:
                ticket.refused(f"queue-decode-failed: {exc}")
                _log(f"queue pane: seek failed ({exc})")
        else:
            ticket.refused("queue-has-no-duration")
        return
    live = getattr(tab, "_live_tail", None)
    produced = tab._live_produced_fraction()
    clip = tab._clip_duration()
    # The instrument: every press says what it meant, in the units the answer will be checked against. One line per
    # press (a drag sends many), which is why the target is what is logged and not the fraction alone.
    _log(f"seek: press {fraction:.1%} of the strip (clip {clip:.3f}s, produced {produced:.1%}, "
        f"tail={'yes' if live is not None else 'no'}, preview={bool(getattr(tab, 'isPreview', False))})")
    # Two edges, not one: the band is the RENDERER's progress (`currentFrame / max_value`) and the tail can only
    # serve what the muxer has FLUSHED - a preview at 60% whose file holds 20% refused every press between them
    # inside the tail ("not in the flushed file"), with the strip advertising the whole 60% as reachable (owner's
    # log, 2026-11-26: press at 8.164 s with produced=0.603, no seek record, the thumb stayed at 4.17 s).
    tail_edge = float(getattr(live, "produced_seconds", 0.0) or 0.0) if live is not None else 0.0
    tail_tolerance = 0.5 / max(1.0, float(getattr(tab, "fps", 0) or 0.0) or 24.0)
    past_tail = (live is not None and target_s is not None and tail_edge > 0.0
                 and float(target_s) > tail_edge + tail_tolerance)
    if (bool(getattr(tab, "isPreview", False))
            and owner.may_seek_any
            and (live is None or fraction > produced + 1e-6 or past_tail)):
        # The renderer can render this frame even though the tail cannot: the same block an export uses. Inside
        # the produced band the tail stays the answer (exact, and it costs the renderer nothing); past it the
        # press is one request and a hold on the picture it yields.
        if tab._request_display_seek(fraction, hold=True):
            bump_generation()
            return
        ticket.refused("no-process-or-unknown-length")
        _log("seek: refused - the running render cannot answer that press (no process, or the clip length is unknown)")
        return
    if live is not None:
        edge = float(getattr(live, "produced_seconds", 0.0) or 0.0)
        if edge <= 0.0:
            ticket.decide(route="tail", answerable=False, why="nothing-written-yet")
            ticket.refused("nothing-written-yet")
            _log("seek: refused - nothing is written yet")
            return                      # nothing is written yet, so there is nothing to seek to
        target = fraction * tab._live_timeline(edge)
        # The tail maps a strip share onto ITS timeline (the clip length, or the flushed edge when it is unknown),
        # so the asked time it must be judged against is this one, not the ticket's whole-clip target.
        bump_generation()
        tab._press_session().watch(target)
        live.seek(target)
        ticket.decide(route="tail", answerable=True)
        _log(f"seek: tail asked for {target:.3f}s (published edge {edge:.3f}s)")
        return
    decoder = getattr(tab, "_preview_decoder", None)
    duration = float(getattr(tab, "_preview_duration", 0.0) or 0.0)
    if decoder is not None and duration > 0:
        ticket.decide(route="decoder", answerable=True)
        # The same landing instrument as the tail's: the session watches the time the decoder was asked for, and
        # the frame it publishes closes the press - not the page's timeout.
        bump_generation()
        tab._press_session().watch(fraction * duration)
        decoder.seek(fraction * duration)
        _log(f"seek: decoder asked for {fraction * duration:.3f}s")
        return
    # EXPORT, third route: no tail and no file decoder, but a renderer is alive and publishing into the pane's
    # block. The strip's share becomes a newest-wins request - `(seconds, generation + 1)` - and the renderer
    # decodes that one frame through the model path and publishes it without writing it to the export. A request
    # only ever goes out while something renders; with no process the block is left alone (nothing would answer).
    # A PREVIEW with no usable block keeps today's refusal: its tail is all there is.
    if bool(getattr(tab, "isPreview", False)):
        ticket.refused("preview-without-a-block")
        return
    # An export press used to pass `hold=False` - "its feed keeps painting the moving band" - which is exactly what
    # the owner saw: the newest upscaled frame painted over his request and the marker dragged to the produced edge,
    # pause notwithstanding (2026-11-24). A press on a PAUSED pane parks on the requested picture; while playing it
    # stays a live follow, and the answer is one publication in the stream.
    if tab._request_display_seek(fraction, hold=bool(getattr(tab, "_preview_paused", False))):
        bump_generation()
    else:
        # Nobody will answer: the render has finished (or never started), and an export has no file decoder behind
        # the pane. Refusing keeps the pane honest - it also clears the pending flag, because a parked pane that
        # waits forever for a picture is the "video became unavailable after the render" the owner reported
        # (2026-11-25). Reported instead of silent: the press was logged as decided and then vanished.
        ticket.refused("no-render-and-no-file-decoder")
        tab._display_seek_pending = False
        tab._update_display_wanted()


def request_display_seek(tab, fraction: float, *, hold: bool) -> bool:
    """Ask the running render for ONE frame at `fraction` of the clip; True when the request went out.

    The block holds one `(seconds, generation + 1)` slot, so a drag replaces its predecessor (newest wins) and the
    renderer decodes the requested timestamp between frames (`RenderVideo._seek_preview`), publishing it without
    writing it to the export. `_on_display_frame` paints it.

    `hold` is a PREVIEW press: its tail cannot pass a frame it has not produced, so the transport parks on the
    requested picture (and play hands the pane back, following the produced edge as before). An export's press
    does not park anything - its feed keeps painting the moving band.

    Returns False without writing anything when no block is attached, no process is alive, or the clip length is
    unknown: nothing would answer, so the seek slot is left alone.
    """
    reader = getattr(tab, "_display", None)
    if reader is None or not bool(getattr(reader, "attached", False)):
        return False
    process = getattr(tab, "renderProcess", None)
    if process is None or process.poll() is not None:
        return False                # nothing would answer a request; the block is left alone
    clip = tab._clip_duration()
    if clip <= 0.0:
        return False                # the timeline length is unknown, so a share cannot become seconds
    target = fraction * clip
    # The seek slot belongs to the reader's block; with a feed attached the write goes through the feed, which
    # is the object that owns "the pane asked for this generation" (newest wins, one slot).
    chain = tab._display_chain()
    if hold and not bool(getattr(tab, "_preview_paused", True)):
        # Park the transport on the frame it asked for. A playing tail would otherwise paint its own (produced)
        # frame over it a moment later; play resumes the follow from the produced edge, as before.
        tab._on_preview_pause_toggled(True)
    unanswered = getattr(tab, "_seek_pressed_at", None) is not None
    previous_generation = getattr(tab, "_seek_generation", None)
    generation = chain.request_seek(target)
    tab._seek_generation = generation or None
    spot = getattr(tab, "_press", None)
    if spot is not None and spot.open:
        # "Can this be answered" is known now, in milliseconds: a produced frame exists or it does not. No stopwatch
        # - `answerable=0` with `why=target-past-produced-edge` is the explanation for a press that used to leave no
        # line at all (owner, 2026-11-24: four presses, produced=0.0000, nothing drawn, nothing logged).
        produced_s = float(tab._live_produced_fraction() or 0.0) * clip
        ahead = target > produced_s + 1e-6
        spot.decide(route="display", answerable=True, ahead=ahead,
                    why="decoded-on-demand" if ahead else None, generation=generation)
    if hold or bool(getattr(tab, "_preview_paused", True)):
        # A held pane has `wanted` 0, so the renderer would stage the requested frame and never publish it. Ask
        # for exactly that one frame: `_on_display_frame` paints it and clears this again.
        tab._display_seek_pending = True
        chain.set_wanted(True)
    if hold:
        # The marker and the compare's source clock stand on the requested frame; the produced band keeps
        # advancing from the renderer's own counters (republished with the frame).
        tab._preview_playhead = float(target)
    _log(f"seek: requested {target:.3f}s (generation {generation}) from the running render")
    return True
