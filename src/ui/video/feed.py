"""The pane's feed: the newest picture, its identity, its drop count, and the seek request.

The chain used to be spread over three owners: the reader knew the block (`DisplayReader`), the window module
polled it on its 100 ms tick and read the identity off the reader in the middle of a paint, and the pane kept
what it was showing. One object now holds the four facts that only make sense together:

* **one picture** - the newest accepted one, never a queue. The renderer produces faster than the pane polls,
  and a preview that lags is worse than a preview that skips: an accepted picture that nobody has read before
  the next one arrives is replaced, and that replacement is the drop counter;
* **its identity** - the publisher's `seq` and the picture's own media time, read from the reader at the moment
  the picture is accepted. A sequence number proves a NEW picture; the time says WHICH one, and it is the only
  thing that may close a seek (`media_time is not None`) because the renderer keeps publishing unnamed frames
  right after the one a seek asked for;
* **its drop count** - how many accepted pictures were replaced before anyone took them;
* **the seek request** - a newest-wins `(seconds, generation)` write into the block the reader owns.

What this module does *not* own: the block's layout and names (`channel.py`, a contract with the writer), the
"is a picture wanted right now" rules (the window module's transport state), and the paint itself (the pane).
Composing rather than absorbing is deliberate: the layout is the half that can draw a torn picture with no
exception, so it stays where its byte tests are.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QImage

from .channel import DEFAULT_DISPLAY_BOX, DisplayReader


def _is_attached(reader) -> bool:
    """`attached` is a property on the reader and a method on the feed; the block question is the same one.

    A test stub is allowed to answer with either shape, and a stub that answers with neither is detached rather
    than an error: "no block" is the ordinary state before a render, not a failure.
    """
    value = getattr(reader, "attached", False)
    try:
        return bool(value() if callable(value) else value)
    except Exception:
        return False


class PreviewFeed(QObject):
    """The block's chain as one object: one picture in, its identity beside it, a drop count, and the seek slot.

    Typical life (the window module drives all of it): the GUI builds the feed before a render starts, `attach`
    allocates the block, the transport sets `wanted` on its 100 ms tick, `poll` accepts at most one picture per
    tick and emits `picture_changed`, and the pane reads `image`/`shown`/`shown_time` to paint. A strip press
    becomes `request_seek`, whose generation the renderer acts on.
    """

    #: Emitted once per accepted picture - i.e. after `poll` returns True, and only then. The receiver reads the
    #: picture through `image()`; the signal itself carries nothing, because the identity belongs to the feed
    #: (`shown`, `shown_time`) and not to whoever happens to be connected.
    picture_changed = Signal()

    def __init__(self, name: str, *, box: tuple[int, int] | None = None,
                 seek_name: str | None = None, parent: QObject | None = None,
                 reader=None) -> None:
        """Name the block; nothing is allocated until `attach`.

        `reader` is the injection seam: normally omitted, and then `attach` builds the `DisplayReader` the
        channel module defines from `name`/`box`/`seek_name`. Tests hand in a stub so the drop and identity
        rules can be pinned without a render, and the window module hands in the reader it already allocated
        (it holds the block name and the box in its own state, and the pane's reach predicates read that reader).
        """
        super().__init__(parent)
        self.name = str(name)
        self.seek_name = str(seek_name) if seek_name else None
        if box is None:
            self.box = DEFAULT_DISPLAY_BOX
        else:
            self.box = (max(1, int(box[0])), max(1, int(box[1])))
        self._reader = reader
        self._attached = False
        self._wanted = False
        self._image: QImage | None = None
        self._shown: tuple[str, int] | None = None
        self._shown_time: float | None = None
        self._dropped = 0
        #: A picture counts as taken once `image()` was read for it; only an untaken one is a drop when replaced.
        self._taken = True

    # -- the block ----------------------------------------------------------------------------------

    def attach(self) -> bool:
        """Create (or adopt) the pane's block; False when it cannot be used, and then nothing is held.

        A block that cannot be allocated or is not usable is the ordinary "no render, no pane" case: the caller
        logs it and runs without a feed. The name carries the pid, so a block found under it is this pane's own.
        """
        if self._attached:
            return True
        reader = self._reader
        if reader is None:
            try:
                reader = DisplayReader(self.name, box=self.box, seek_name=self.seek_name)
            except Exception:
                self._reader = None
                return False
            self._reader = reader
        if not _is_attached(reader):
            return False
        self._attached = True
        return True

    def detach(self) -> None:
        """Release the block and forget the picture; safe twice and never raises.

        `DisplayReader.release` closes and unlinks only the blocks this reader opened (the names carry the pid),
        so this can never take a live pane's channel away.
        """
        reader, self._reader = self._reader, None
        self._attached = False
        self._wanted = False
        self._image = None
        self._shown = None
        self._shown_time = None
        self._taken = True
        if reader is not None:
            releaser = getattr(reader, "release", None)
            if callable(releaser):
                try:
                    releaser()
                except Exception:
                    pass

    def attached(self) -> bool:
        return bool(self._attached)

    def set_wanted(self, wanted: bool) -> None:
        """One byte to the renderer: is a picture wanted right now (the transport's rules live in the caller)."""
        self._wanted = bool(wanted)
        setter = getattr(self._reader, "set_wanted", None)
        if callable(setter):
            try:
                setter(self._wanted)
            except Exception:
                pass

    def wanted(self) -> bool:
        return bool(self._wanted)

    # -- the tick -----------------------------------------------------------------------------------

    def poll(self) -> bool:
        """Pump the reader once; True only when a NEW picture was accepted.

        The reader already answers "nothing new" with None (an unchanged `seq`, nobody asking, an incomplete
        slot), so this is the one place a picture enters the feed.
        """
        if not self._attached or self._reader is None:
            return False
        poller = getattr(self._reader, "poll", None)
        if not callable(poller):
            return False
        try:
            image = poller()
        except Exception:
            return False
        if image is None:
            return False
        self._accept(image)
        return True

    def _accept(self, image) -> None:
        """Keep `image` as the one picture, with the identity the reader carries for it right now.

        Latest wins: the picture being replaced is a DROP only if nobody read it (see `image`). Nothing is
        queued, so a pane that polls slower than the renderer never replays old frames.
        """
        if self._image is not None and not self._taken:
            self._dropped += 1
        self._image = image
        self._taken = False
        seq = getattr(self._reader, "seq", 0)
        try:
            seq = int(seq or 0)
        except Exception:
            seq = 0
        self._shown = ("seq", seq)
        media_time = getattr(self._reader, "media_time", None)
        try:
            self._shown_time = None if media_time is None else float(media_time)
        except Exception:
            self._shown_time = None
        self.picture_changed.emit()

    def image(self) -> QImage | None:
        """The newest accepted picture, or None before the first one.

        Reading it is taking it: the next accepted picture then replaces a picture somebody already had, which
        is not a drop. The drop counter itself is never cleared here - it is a count of the run, not of the hold.
        """
        self._taken = True
        return self._image

    def shown(self) -> tuple[str, int] | None:
        """`("seq", n)` for the picture `image()` returns, or None before the first one."""
        return self._shown

    def shown_time(self) -> float | None:
        """That picture's media time in seconds, or None when the writer could not name it."""
        return self._shown_time

    def dropped(self) -> int:
        """How many accepted pictures were replaced before anyone took them (latest-wins, no queue)."""
        return self._dropped

    # -- the render status --------------------------------------------------------------------------

    def status(self) -> dict | None:
        """The render's own status record (`DisplayReader.status`), or None when there is none.

        The status is the reader's byte contract, not the feed's picture state: the feed only forwards it so
        the pane has one object to ask. A stub reader without the call, a detached reader, or a torn record
        all answer None - which the pane treats as "nothing new", never as zeroes.
        """
        getter = getattr(self._reader, "status", None)
        if not callable(getter):
            return None
        try:
            return getter()
        except Exception:
            return None

    # -- the newest-wins seek -----------------------------------------------------------------------

    def request_seek(self, seconds: float) -> int:
        """Write `(seconds, generation + 1)` into the reader's seek block; returns the generation written."""
        requester = getattr(self._reader, "request_seek", None)
        if not callable(requester):
            return 0
        try:
            return int(requester(float(seconds)))
        except Exception:
            return 0

    def seek_generation(self) -> int:
        """The generation currently in the block (0 when the feed holds no reader)."""
        getter = getattr(self._reader, "seek_generation", None)
        if not callable(getter):
            return 0
        try:
            return int(getter())
        except Exception:
            return 0
