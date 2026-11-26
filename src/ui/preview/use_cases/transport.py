"""Transport use cases: the play/pause state, the decoder's arrived picture, and the job's playback hand-over.

What this module owns:

* the button's painted state (`_sync_preview_transport`), including the cross-thread hand-off: `.setChecked`, the
  tooltip and `setEnabled` are widget writes, so a caller on a worker thread (the render's `Thread` starting a tail)
  must not perform them. The three-argument `QTimer.singleShot(0, button, fn)` is the sanctioned form - the button is
  the CONTEXT object, so the retry lands on the button's own thread;
* the pause handler's effect on the three possible pane owners (live tail, finished-file decoder, queue item);
* one decoder picture's arrival (`on_original_frame`): crop, offer to the compare register, and the `arrivals`
  counter. That counter is the fix for audit §6 / `rules.md` §6: `_compare_fill_note_arrival` had lost its only write
  site in the compare-seam refactor while the fill line kept printing `arrivals=0`, so "never asked" and "asked and
  never answered" read the same in every log.

What it may NOT touch: signal connections, widget construction, the pane's picture APIs (`set_result`/`set_source`
belong to the presenter), or any state of its own. Every helper it calls is a method with its current name on the tab.
"""

from __future__ import annotations

from ...video import telemetry
from ..owner import owner_of
from ..state import PreviewBackendKind

__all__ = ["sync_preview_transport", "on_preview_pause_toggled", "on_original_frame"]


def _log(message: str) -> None:
    """Write one line through the app's log, imported lazily so this module imports without the app's deps."""
    from ....Util import log

    log(message)


def sync_preview_transport(tab) -> None:
    """Paint the toggle's state, its tooltip and whether it can do anything at all (GUI thread only).

    The glyphs are the ACTION a press performs, like a player's own button: playing shows a pause symbol, paused
    shows a play symbol. `enabled` is honest about the idle pane - there is nothing to hold when nothing owns it.
    """
    button = getattr(tab.page_handles.proc, "previewPauseButton", None)
    if button is None:
        tab._push_preview_state()
        return
    from PySide6.QtCore import QThread, QTimer

    if QThread.currentThread() != button.thread():
        # `_stop_live_tail` runs in the render's worker thread, which has no event loop to paint from: hand the
        # widget work to the button's own thread instead of writing to a widget cross-thread.
        #
        # This branch is NOT about the tick: the 100 ms tick already arrives on the GUI thread (its plain-callable
        # connection is queued there - see `startGUIUpdate`, plan P6), so `updateProcessTab` never takes it. It is
        # for the other callers of this method: `renderToPipeThread` starts the job's tail from a plain `Thread`
        # (`_start_live_tail` -> `_stop_live_tail` -> here, ProcessTab.py:781/1432), and a widget write from there
        # is a cross-thread write whatever the tick does. The three-argument `singleShot` is the sanctioned form:
        # `button` is the CONTEXT object, so the retry lands on the button's thread and not on the caller's.
        QTimer.singleShot(0, button, lambda: sync_preview_transport(tab))
        return
    paused = bool(getattr(tab, "_preview_paused", True))
    try:
        # `emit=False`: painting the state must not look like a user toggle (that would call this handler again).
        button.setChecked(paused, emit=False)
    except TypeError:      # a toolkit older than the keyword
        button.blockSignals(True)
        try:
            button.setChecked(paused)
        finally:
            button.blockSignals(False)
    button.setToolTip(
        "Paused: the preview holds this frame. Press to follow the render."
        if paused else "Following the render. Press to hold this frame."
    )
    # A render paints the pane through the thumbnail even before a tail exists, so a running render is an owner
    # too: the transport must be able to hold THAT picture, and must not look pressable when nothing paints.
    process = getattr(tab, "renderProcess", None)
    rendering = process is not None and process.poll() is None
    owns_picture = (getattr(tab, "_live_tail", None) is not None
                    or getattr(tab, "_preview_decoder", None) is not None
                    or getattr(tab, "_queue_pane_decoder", None) is not None
                    or rendering)
    button.setEnabled(bool(owns_picture))
    # The bars read the same truth the legacy button does: one push per change, no second timer.
    tab._push_preview_state()


def on_preview_pause_toggled(tab, paused: bool) -> None:
    telemetry.event("transport", paused=int(bool(paused)))
    """Hold the preview on this frame, or let it follow the render again (the button's `toggled(bool)`).

    One state for both owners of the pane: the live tail simply stops polling (no decode, no I/O, the picture
    stands, and a strip press still repositions it), the finished-file decoder pauses on its frame. The state
    comes IN with the signal, so the handler cannot act on a half-applied toggle.
    """
    tab._preview_paused = bool(paused)
    # A one-shot seek frame belongs to the state it was asked in: leaving or re-entering the hold ends it.
    tab._display_seek_pending = False
    # One line per change, with the owner: the pane can be painted by three different things, and "the button did
    # nothing" needs to be answerable from log.txt alone.
    owner = "tail" if getattr(tab, "_live_tail", None) is not None else (
        "queue" if getattr(tab, "_queue_pane_decoder", None) is not None else (
            "file" if getattr(tab, "_preview_decoder", None) is not None else "thumbnail"))
    _log(f"preview transport: {'paused' if tab._preview_paused else 'playing'} (owner={owner})")
    holding = bool(tab._preview_paused)
    # ONE producer at a time: while the queue tab owns the pane, pressing play starts the QUEUE item, not the
    # render's tail/decoder behind it - a decoder playing under a picture nobody shows is pure cost.
    queue_owns = owner_of(tab).kind is PreviewBackendKind.QUEUE_FILE
    if holding and getattr(tab, "_playback_tail", False):
        # The tail this PLAY created exists only while the transport plays; pausing drops it and the feed's cheap
        # path (the renderer's own composited frame) takes the pane back. A PREVIEW's tail and a tail somebody else
        # owns (the queue tab's hand-back) are NOT ours to stop - they are the picture there.
        tab._stop_live_tail()
    tail = getattr(tab, "_live_tail", None)
    decoder = getattr(tab, "_preview_decoder", None)
    if not holding and decoder is not None and getattr(tab, "_playback_tail", False):
        # A finished file has a real player (media-rate pacing, one frame per tick, box-capped). The playback tail
        # polls the file every 120 ms and hands a backlog burst to the GUI, which paints once per burst - measured
        # 2026-11-25: 18 frames/s delivered, 9 painted, the drawn index stepping by 2. The tail is the player for a
        # file still being WRITTEN; once the job is done its decoder owns the timeline.
        tab._stop_live_tail()
        tail = None
    if (not holding and tail is None and not queue_owns and not bool(getattr(tab, "isPreview", False))
            and decoder is None):
        # PLAY, export, no tail and no finished-file decoder: the pane was following the renderer's newest frames,
        # which is not playback. The output file is the timeline; start the tail here (the GUI tick builds it) at
        # the pane's own position, and the feed stands down while it paints. Pausing drops it again, so the feed's
        # cheap path is what a held pane uses (owner, 2026-11-25: "воспроизведение всё ещё просто переключает нас
        # на слежение за последними апкад кадрами").
        options = getattr(tab, "currentRenderOptions", None)
        path = str(getattr(options, "outputPath", "") or "")
        if path:
            tab._start_live_tail(path)
            tab._ensure_live_tail()
            tab._playback_tail = getattr(tab, "_live_tail", None) is not None
            tail = getattr(tab, "_live_tail", None)
    if tail is not None:
        if not holding and not queue_owns:
            # Its own try: a resume that cannot work out the head must still unpause the tail (the first version
            # put both in one block, and an AttributeError in the head arithmetic skipped `set_paused` entirely).
            try:
                tab._resume_tail_at_the_pane(tail)
            except Exception as exc:
                _log(f"preview transport: resume at the head failed ({exc})")
        try:
            tail.set_paused(holding or queue_owns)
        except Exception as exc:
            _log(f"preview transport: {exc}")
    if decoder is not None:
        try:
            decoder.pause() if (holding or queue_owns) else decoder.play()
        except Exception as exc:
            _log(f"preview transport: {exc}")
    queue_decoder = getattr(tab, "_queue_pane_decoder", None)
    if queue_decoder is not None:
        try:
            queue_decoder.pause() if (holding or not queue_owns) else queue_decoder.play()
        except Exception as exc:
            _log(f"queue pane: {exc}")
    # A held picture clears `wanted`: the renderer must stop staging frames for a pane that is not following
    # (the whole cost of the channel when nobody watches is one byte read per frame).
    tab._update_display_wanted()
    tab._sync_preview_transport()


def on_original_frame(tab, frame) -> None:
    """The source frame from the decoder, cropped to the box THE RENDER used, then handed to the pane.

    The crop is applied here rather than in the decoder because the decoder exposes no crop and the surface
    (`src/ui/widgets/video_preview.py`) is another lane's file: `QImage.copy` of the same rectangle is the same
    pixels, and the surface then scales it to the rendered frame's size - which is what makes the two pictures
    comparable. Without the crop the compare shows the whole source (black bars included) and the eye reads the
    letterbox as a scale difference.

    The arrival counter (`_compare_fill_note_arrival`) is written HERE, at the decoder's arrival, because that is
    the only place a picture actually comes back (audit §6 / `rules.md` §6); a counter written anywhere else makes
    "asked and never answered" indistinguishable from "never asked".
    """
    from PySide6.QtGui import QImage

    # The parser is module-level in `ProcessTab` and a lazy import keeps this module free of a cycle while the
    # page imports its use cases.
    from ...ProcessTab import crop_region

    image = QImage(bytes(frame.buffer), frame.width, frame.height, frame.stride,
                   QImage.Format.Format_RGB888).copy()
    box = getattr(tab, "_source_crop", None)
    if box is not None:
        # The box is written in SOURCE pixels; this decoder hands over a fitted picture, so it is scaled by the same
        # ratio (`stream_size` is the coded size, the one the render's BORDERS line describes).
        source_size = None
        decoder = getattr(tab, "_source_decoder", None)
        size_of = getattr(decoder, "stream_size", None)
        if callable(size_of):
            try:
                source_size = size_of()
            except Exception:
                source_size = None
        if source_size and source_size[0] > 0 and image.width() != source_size[0]:
            scale = image.width() / float(source_size[0])
            box = tuple(int(round(value * scale)) for value in box)
    region = crop_region(box, image.width(), image.height())
    if region is None:
        # Today's picture (the whole source) is the fallback, and the log says once why: a silent
        # uncropped frame is indistinguishable from "crop detection found nothing".
        if box is None:
            tab._log_source_crop_unknown("this render's output carried no readable BORDERS line")
        else:
            tab._log_source_crop_unknown(
                f"the render reported {box} but the source frame is {image.width()}x{image.height()}")
    else:
        image = image.copy(*region)
    if region is None:
        # No crop: the QImage still points into `raw`, which dies with this function - own the pixels.
        image = image.copy()
    # What the decoder actually produced, before the crop and the scale: a compare half that ends up black is
    # either an empty decode or a viewer-side problem, and only this line can tell them apart.
    if telemetry.level() >= 2:
        # Three `pixelColor` reads per source frame exist for one line of evidence; at 24 fps they are 72 a second
        # on the GUI thread for a log nobody is reading. Level 2 is the "one line per frame" mode, so ask there.
        _samples = []
        for _fx, _fy in ((1, 1), (image.width() // 2, image.height() // 2),
                         (max(0, image.width() - 2), max(0, image.height() - 2))):
            _c = image.pixelColor(int(_fx), int(_fy))
            _samples.append(f"{_c.red()},{_c.green()},{_c.blue()}")
        telemetry.event("source frame", w=image.width(), h=image.height(), fmt=image.format().name,
                        px="/".join(_samples))
    else:
        telemetry.event("source frame", w=image.width(), h=image.height(), fmt=image.format().name)
    watched = getattr(tab, "_compare_watch", None)
    tab._compare_watch = None
    pts = float(getattr(frame, "pts", 0.0) or 0.0)
    index = getattr(frame, "index", None)
    pane = tab.page_handles.proc.previewLabel
    # The LAST step's other half: what the pane was showing when the press asked for this source frame. The labels
    # can agree and the pictures still differ (a shifted file timeline names every frame late), so the draw line in
    # the pane's own log carries a content fingerprint next to this one.
    result_time = None
    result_kind = None
    try:
        result_time = pane.shown_time() if hasattr(pane, "shown_time") else None
        result_kind = pane.shown() if hasattr(pane, "shown") else None
    except Exception:
        pass
    if watched is not None:
        fps = 0.0
        try:
            fps = float(tab._source_decoder.fps()) if getattr(tab, "_source_decoder", None) is not None else 0.0
        except Exception:
            fps = 0.0
        fps = fps if fps > 0.0 else 24.0
        gap = "" if result_time is None else (
            f", result {float(result_time):.3f}s {result_kind}"
            f" -> gap {pts - float(result_time):+.3f}s ({(pts - float(result_time)) * fps:+.1f} frames)")
        _log(f"compare: the source frame is {pts:.3f}s idx={index} for an ask of {watched:.3f}s "
            f"(delta {pts - watched:+.3f}s, {abs(pts - watched) * fps:.1f} frames at {fps:.2f} fps{gap})")
    # OFFER, never write the pane: the frame is one candidate for the moment it names, and `_sync_compare_source`
    # decides which candidate the pane's own moment gets. A producer that wrote directly could bypass the pane's
    # half-frame refusal; this one cannot.
    register = getattr(tab, "_source_register", None)
    if register is not None:
        arrived = register.offer(pts, image, "decoder",
                                 generation=int(getattr(tab, "_result_generation", 0)))
        if arrived is not None:
            tab._compare_fill_note_arrival()
    if telemetry.level() >= 2:
        telemetry.event("compare put", pts=round(float(pts), 3),
                        pool=len(register) if register is not None else 0)
    if not callable(getattr(pane, "set_source", None)):
        # A pane from before the register API (an old harness double) still gets the picture here; the real pane
        # only ever receives `set_source` from `_sync_compare_source`.
        legacy = getattr(pane, "set_original", None)
        if callable(legacy):
            pane.set_original(image, media_time=pts, index=index)
        return
    tab._sync_compare_source()
    mode = getattr(pane, "view_mode", None)
    single = True
    if callable(mode):
        try:
            single = mode() == "single"
        except Exception:
            single = True
    if not single:
        # The PANE's clock, like every other supply entry point: an arrival must not re-target the fill at a third
        # clock (the pair queue's oldest result). That was the last path still doing it, so a source answered at the
        # queue's moment while the pane showed another - the mismatch the fill's own target/playhead line reports
        # (audit 2026-11-27 §4, CODE_PATTERNS §4: one clock).
        tab._source_fill(tab._compare_playhead())
