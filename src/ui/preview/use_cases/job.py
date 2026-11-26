"""Job use cases: the one result-generation token, and the render-end hand-over.

What this module owns:

* `next_result_generation` - the ONE token that says "which render/file/seek is current". It is minted only by a
  landing site (a new job, a new file, a seek that was actually asked for), never by an attempt that was refused,
  and it clears the compare register because a stale source is exactly as wrong as a stale result;
* `on_job_finished` - what happens after the job's painters are stopped: a preview starts its finished-file playback;
  a successful export hands its finished file to the pane at the render's own position, or keeps the last painted
  frame when there is no readable file. The thread code (`renderToPipeThread`) and the queue's own state stay in the
  tab; this function only performs the hand-over.

What it may NOT touch: threads (no start/stop), widgets (no signal, no picture write), or any state of its own.
Every helper it calls is a method with its current name on the tab.
"""

from __future__ import annotations


__all__ = ["next_result_generation", "on_job_finished"]


def _log(message: str) -> None:
    """Write one line through the app's log, imported lazily so this module imports without the app's deps."""
    from ....Util import log

    log(message)


def next_result_generation(tab) -> int:
    """Start a new result generation: a render/file/seek changed, so the previous pictures are not answers.

    Returning it lets every producer in this generation stamp its offers with the same number; the pane refuses
    anything older than what it holds, and the register is emptied because a stale source is exactly as wrong as a
    stale result. Call it only where something actually landed - a refused press must be a no-op (audit §6.3).
    """
    tab._result_generation = int(getattr(tab, "_result_generation", 0)) + 1
    register = getattr(tab, "_source_register", None)
    if register is not None:
        register.clear()
    return tab._result_generation


def on_job_finished(tab) -> None:
    """The render-end hand-over: decide what the finished job leaves on the pane.

    A PREVIEW job repaints from its own finished file a moment later, so the queue it set aside is restored and its
    playback starts. An EXPORT whose render succeeded hands the output file to the pane, held at the position the
    render stood on: the strip scrubs it, Play runs it at the media rate, and the picture is the real output rather
    than the last live frame (the "the last seeks do nothing on a fully rendered file" report, owner 2026-11-25).
    An unreadable output keeps the last painted frame.

    The failure flag is recomputed from the tab's own return codes rather than passed in, so the function answers
    "did this job finish successfully" from the state the tab already owns.
    """
    tab._restoreQueueAfterPreview()
    if tab.currentRenderOptions.isPreview:
        tab._start_preview_playback(tab.currentRenderOptions.outputPath)
        return
    finished_ok = all(code == 0 or code == 3221225477 for code in tab.return_codes) or tab.userKilled
    if not finished_ok:
        return
    # The export is done, and its pane had NO source after this point: the display feed is released, the tail
    # was stopped, and `_request_display_seek` refuses a press because no process will answer it. That is why
    # "the last seeks do nothing on a fully rendered file" (owner, 2026-11-25). The finished file becomes the
    # pane's source, HELD at the position the render stood on: the strip scrubs it, Play runs it at the media
    # rate (the decoder paces by PTS), and the picture is the real output rather than the last live frame.
    output = str(getattr(tab.currentRenderOptions, "outputPath", "") or "")
    try:
        import os as _os

        readable = bool(output) and _os.path.exists(output) and _os.path.getsize(output) > 0
    except Exception:
        readable = False
    if readable:
        tab._preview_paused = True
        tab._start_preview_playback(output)
        tab._sync_preview_transport()
        _log(f"preview: the finished file is the pane's source ({output}); the strip scrubs it")
    else:
        _log(f"preview: no finished file to hand the pane ({output!r}); the last painted frame stays")
