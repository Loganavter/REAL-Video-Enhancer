"""One picture, one painter - decided once.

Why this module exists. "Who paints this pane" used to be eight separate predicates on `ProcessTab`
(`_display_paints_pane`, `_display_owns_pane`, `_preview_backend_kind`, `_tail_should_run`, `_display_should_want`,
`_display_can_seek_any_frame` and the two hand-written frame gates). Every one of them asked the same handful of raw
facts - is the queue tab showing its item, is there a live tail, is there a preview decoder, is the display block
attached, is this a preview job, is a render process alive - and answered a different slice of the same question. The
painter races the audit records (`3791f9e8` two painters on one pane, `48455ff9` the thumbnail undoing a seek,
`f17cbfff` the queue hand-back, `b3e4b7b1` the one-frame loan) are each a place where two of those predicates disagreed
for a frame. Nine readers of one fact is nine chances to add a tenth; one function is the guard that cannot drift.

`owner_of(page)` is that function: a pure read of the page's raw inputs, in the order a painter is installed.
`may_paint` answers "who paints the NEXT frame", `may_want` "may the renderer be asked to stage one", `may_seek_any`
"is every timeline frame reachable on demand", `kind` "who answers the transport". What the owner deliberately does NOT
answer is anything that depends on the picture's own identity (the pane's `shown`, a seek generation, a frame index):
those checks stay in the page, at the frame they are about, because they are about THIS picture and not about the pane.

No Qt here: no widget is called and nothing is imported from PySide6, so the decision is testable as a table of page
shapes with no window. `page` is duck-typed - any object with the raw attributes works, which is what lets the page's
harness build one with `object.__new__` and a test build one with `SimpleNamespace`.

The two readings that need a widget stay out on purpose:

* the pane's VISIBILITY (`pane.isVisible()`) is ANDed onto `may_want` by the page: `may_want` is "the transport wants a
  frame", visibility is "somebody is there to see it", and folding a widget call in here would make the decision
  untestable without a window;
* `has_frame()`/`shown()`/`shown_time()` are the pane's own facts, read where the frame is handled.
"""

from __future__ import annotations

from dataclasses import dataclass

from .state import PreviewBackendKind

__all__ = ["Owner", "owner_of"]


@dataclass(frozen=True)
class Owner:
    """Who owns the pane right now: the transport, the painter, the write permission and the reach.

    `clock` is the object that answers this pane's timeline when `kind` names one - the live tail, the finished file's
    decoder, or the feed chain over the display block - and `None` when nobody does. It is the raw object, not the
    transport adapter: the page still wraps it in a `ports.Clock` (`backends.backend_for`) this phase, and P4 binds the
    session's clock from it.

    `surface` is the pane itself when it may paint (`None` otherwise), so a caller that wants to hand the picture
    somewhere has the one object that is allowed to take it.
    """

    kind: PreviewBackendKind
    clock: object | None
    surface: object | None
    may_paint: bool
    may_want: bool
    may_seek_any: bool


def owner_of(page) -> Owner:
    """The one painter decision: read the page's raw inputs, answer who paints, who may ask, how far a press reaches.

    The raw inputs, and nothing else (no widget is called, nothing is written):

    * `_queue_pane_owns` - the queue tab is showing one of its own items on the pane;
    * `_live_tail` - the file-following thread of the current job, if one was started;
    * `_preview_decoder` - the finished preview file's own player;
    * `_display.attached` - the renderer's frame-on-demand block is usable;
    * `isPreview` - this job is a preview, not an export;
    * `renderProcess` - alive (`poll()` is None) while the renderer can answer;
    * `_preview_paused` / `_display_seek_pending` - the transport is holding, and whether it asked for one frame anyway.

    The answers, in the rules the deleted gates each held:

    * **kind** - the queue item is not the preview's clock (`NONE`); a finished file's decoder is; a live tail is; an
      export whose block is usable is the display feed; otherwise nobody.
    * **may_paint** - the display feed paints an export whenever the block is usable, UNLESS a playing tail is the
      timeline (the two-painter race: the feed publishing at the same time as a tail is the 18.9 s ↔ 8.97 s playhead).
      A preview's pane belongs to its tail, except for the ONE frame a strip press asked the block for.
    * **may_want** - paints, the transport is not holding (or it asked for one frame), and the renderer is alive: a
      held picture costs the renderer nothing, which is the whole reason the request byte exists.
    * **may_seek_any** - the block is attached, i.e. the renderer can render ANY timeline frame on demand. This is the
      strip's REACH, not the band: what has been produced is `facts.produced_seconds`, and they are different facts.

    The queue item's own decoder is not an input: `_queue_pane_owns` is its only producer (the decoder is built when the
    tab enters and released when it leaves), so reading it twice would be the second copy of that predicate.
    """
    queue_owns = bool(getattr(page, "_queue_pane_owns", False))
    tail = getattr(page, "_live_tail", None)
    decoder = getattr(page, "_preview_decoder", None)
    reader = getattr(page, "_display", None)
    attached = reader is not None and bool(getattr(reader, "attached", False))
    preview = bool(getattr(page, "isPreview", False))
    paused = bool(getattr(page, "_preview_paused", True))
    pending = bool(getattr(page, "_display_seek_pending", False))
    process = getattr(page, "renderProcess", None)
    rendering = process is not None and process.poll() is None

    if queue_owns:
        # The queue item is the pane's one owner there: the feed stages nothing, the tail's frames are dropped.
        kind = PreviewBackendKind.NONE
    elif decoder is not None:
        kind = PreviewBackendKind.FINISHED_FILE
    elif tail is not None:
        kind = PreviewBackendKind.LIVE_TAIL
    elif attached and not preview:
        kind = PreviewBackendKind.DISPLAY_FEED
    else:
        kind = PreviewBackendKind.NONE

    if not attached or queue_owns:
        # No block, nothing to show through it; a queue item is not the render's picture even when a block exists.
        may_paint = False
    elif not preview:
        # An export: the feed owns the pane unless a PLAYING tail is the timeline. A held export has no tail, so the
        # one-frame loan is untouched.
        may_paint = not (tail is not None and not paused)
    else:
        # A preview: the tail is the painter, except for exactly the frame a strip press asked the block for.
        may_paint = pending

    may_want = bool(may_paint and (not paused or pending) and rendering)
    may_seek_any = bool(attached)

    if kind is PreviewBackendKind.LIVE_TAIL:
        clock = tail
    elif kind is PreviewBackendKind.FINISHED_FILE:
        clock = decoder
    elif kind is PreviewBackendKind.DISPLAY_FEED:
        clock = getattr(page, "_feed", None) or reader
    else:
        clock = None

    surface = None
    if may_paint:
        surface = getattr(getattr(getattr(page, "page_handles", None), "proc", None), "previewLabel", None)

    return Owner(kind=kind, clock=clock, surface=surface, may_paint=bool(may_paint),
                 may_want=may_want, may_seek_any=may_seek_any)
