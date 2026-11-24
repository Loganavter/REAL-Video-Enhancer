"""Following the preview file while the renderer is still writing it.

The plan moved the live picture off the shared-memory thumbnail and onto the preview file itself
("Full resolution on both sides", owner decision 2026-11-23), because the file costs the renderer
nothing: the encoder already writes it. The catch is that we are reading a file that is being
appended to at 4K, and a demuxer is not built for that. The rules this class exists to keep:

* **Never block the writer.** The file is opened read-only, the reader never writes, and it only ever
  asks libavformat for what has been written: `poll` is called by the owner (a timer), and a poll
  that finds no growth does no I/O at all. Nothing here sleeps or waits.
* **EOF is "wait", not an error.** The writer has caught up with the reader; the next growth is the
  next picture.
* **Never read past the last complete fragment.** libavformat will not hand over a frame it could not
  decode, so a truncated tail is simply "nothing this time". A failure at the file's end drops the
  read position and resumes from the last published frame's PTS (backward seek, i.e. always inside
  written data) on the next growth; nothing is skipped without being decoded.
* **No duplicates.** Resuming re-decodes from the previous keyframe; frames at or before the last
  published PTS are skipped, so the surface sees each frame exactly once.
* **A seek repositions inside what is written.** The owner can drag the strip while the render still
  runs. The target becomes the de-duplication floor, the container is dropped, and the next `poll`
  reopens the file at that frame - so the picture follows the strip within one poll instead of at the
  renderer's next fragment. Past the produced edge there is nothing to show, so such a target is
  clamped to the edge, which is exactly follow-live - unless written frames are still buffered behind
  the playhead (a held pane publishes one frame and stops reading), where the target is attempted and
  the edge is only the fallback.
* **A held pane seeks ONE frame.** A press on a paused preview is answered by `poll(1)`: its target,
  with no backlog behind it and no overshoot. A playing pane keeps the normal per-poll rate.

`import av` happens in the default container factory, not at module scope: the surface can import
this module on an interpreter without PyAV and gets an `error` signal only if it actually tails a
file.
"""

from __future__ import annotations

import os

from PySide6.QtCore import QObject, QThread, Signal

from .frames import FORMATS, frame_pts, pack_frame

_MICROS = 1_000_000

#: How many consecutive failed resumes at the file's end are treated as "this file really does not
#: parse" instead of "the next fragment is half written". One truncated fragment is the normal state
#: of a growing file; three in a row, with the file growing between them, is not.
_ERROR_AFTER_FAILURES = 3

NO_PYAV = "PyAV is not installed: the live preview needs `pip install av` to follow the preview file"


def _default_container_factory(path: str):
    """The real demuxer. Kept behind a factory so the growth logic is testable without PyAV."""
    import av

    return av.open(path, mode="r")



class TailReader(QObject):
    """Decode what a growing file has already written; poll for the rest."""

    #: One `FrameRef`, in presentation order.
    frame_ready = Signal(object)
    #: The file's new size in bytes, each time `poll` sees growth.
    grew = Signal(int)
    #: A failure worth reporting; a successful read clears the de-duplication.
    error = Signal(str)

    def __init__(self, parent: QObject | None = None, *, pixel_format: str = "rgb24",
                 max_frames_per_poll: int = 4, codec_threads: int = 2,
                 container_factory=None, open_grace_polls: int = 40) -> None:
        super().__init__(parent)
        if pixel_format not in FORMATS:
            raise ValueError(f"unknown frame format {pixel_format!r}, expected one of {FORMATS}")
        self._pixel_format = pixel_format
        self._max_frames_per_poll = max(1, int(max_frames_per_poll))
        self._codec_threads = max(1, int(codec_threads))
        self._container_factory = container_factory or _default_container_factory
        #: A file whose header is still being written fails to open for a while; that is a wait. The
        #: count turns an indefinite silence into one honest error if the header never appears.
        self._open_grace_polls = max(1, int(open_grace_polls))

        self._path: str | None = None
        self._size = 0
        self._container = None
        self._stream = None
        self._demux = None
        self._frame_iter = None
        self._fps = 0.0
        self._emitted = 0
        self._last_pts: float | None = None
        #: The highest PTS this reader has published - the produced edge as far as the reader knows it.
        #: Unlike `_last_pts` (the de-duplication floor) a seek never lowers it: it is what a seek is
        #: clamped to.
        self._edge_pts: float | None = None
        self._exhausted = True      # the current demux position has nothing left to give
        #: The last poll stopped because its per-poll cap was reached, not because the demuxer ran dry: written
        #: frames are buffered behind the playhead, so the published edge is NOT the end of the file.
        self._backlog = False
        #: The published edge a seek past it should fall back to, when the seek was allowed to try for a target
        #: deeper than anything published yet (see `seek`). `None` = no such seek is outstanding.
        self._seek_fallback: float | None = None
        self._open_failures = 0
        self._decode_failures = 0
        self._signature: tuple | None = None   # de-duplicates a repeated error

    # -- lifecycle ---------------------------------------------------------------------------------

    def open(self, path: str) -> bool:
        """Point the reader at a growing file; the container itself opens lazily on the first bytes.

        A missing file is NOT an error here: the renderer creates its output when the encoder starts, so a reader that
        begins at "the render just launched" legitimately sees nothing for a moment. `False` means "wait and try again",
        and it is deliberately silent - a preview that spams the log before the first byte is worse than one that waits
        (measured against a real preview render: ten "no such file" lines before the first fragment).
        """
        self.close()
        path = str(path)
        if not os.path.isfile(path):
            return False
        self._path = path
        return True

    def close(self) -> None:
        """Release the demuxer; the next `open` starts a new file."""
        self._path = None
        self._size = 0
        self._emitted = 0
        self._last_pts = None
        self._edge_pts = None
        self._exhausted = True
        self._backlog = False
        self._seek_fallback = None
        self._open_failures = 0
        self._decode_failures = 0
        self._signature = None
        self._drop_container()

    # -- the poll ----------------------------------------------------------------------------------

    def poll(self, max_frames: int | None = None) -> int:
        """Decode what the file has grown by; return how many frames were emitted.

        Called from the owner's timer. A poll that finds no growth and an exhausted demuxer does no
        I/O at all, which is the common case while the renderer is busy with a frame; a poll that
        finds buffered frames left over from a capped read continues them without waiting for growth.

        `max_frames` caps what THIS call may emit, for a poll that is not a read of the file but the
        answer to one press: a seek on a held pane asks for a single frame, its target, and must not
        drag the backlog behind it onto the surface (see `seek`). `None` is the reader's own cap.
        """
        if self._path is None:
            return 0
        try:
            size = os.path.getsize(self._path)
        except OSError as exc:
            self._fail(("stat",), f"tail: cannot stat {self._path}: {exc}")
            return 0
        if size < self._size:
            # The writer replaced the file (a new render at the same path): start over rather than
            # reading the old file's tail as if it belonged to the new one.
            self._restart()
        if size <= self._size and self._exhausted:
            return 0
        if size > self._size:
            self._size = size
            self.grew.emit(size)

        if self._container is None:
            if not self._open_container():
                return 0
        elif self._exhausted:
            # The last read ended at the file's end. A fragmented container picks the new bytes up
            # where it stopped, so try that before paying for a reopen.
            self._reset_stream()

        emitted = self._read(max_frames)
        if emitted == 0 and self._exhausted and self._container is not None:
            # libavformat did not notice the growth (its EOF flag is set). Reopening and seeking back
            # to the last published frame is the resume that always works; it re-decodes the current
            # fragment only, and `_decode_available` skips what is already on screen.
            self._drop_container()
            if not self._open_container():
                return 0
            emitted = self._read(max_frames)
        if emitted == 0 and self._seek_fallback is not None:
            # The target is past the last COMPLETE frame the file holds (the renderer is mid-fragment,
            # or the press reached just beyond what it flushed). Standing on the produced edge is the
            # honest answer - the deepest frame that exists - instead of silently ignoring the press.
            edge = self._seek_fallback
            self._seek_fallback = None
            self._last_pts = max(0.0, edge - self._interval() * 0.5) if edge > 0.0 else None
            self._drop_container()
            if self._open_container():
                emitted = self._read(max_frames)
        return emitted

    # -- seeking -----------------------------------------------------------------------------------

    @property
    def produced_seconds(self) -> float:
        """Media time of the last frame this reader published: the produced edge, as far as it knows."""
        return float(self._edge_pts or 0.0)

    @property
    def published_frames(self) -> int:
        """How many frames the surface has seen from this reader (0 until the first one)."""
        return int(self._emitted)

    def seek(self, seconds: float) -> float:
        """Move the playhead inside the already-written part; return where it landed, in seconds.

        Only `poll` reads, so a seek is cheap and a drag coalesces: it moves the de-duplication
        floor, drops the demuxer and marks the reader as having work. The next `poll` reopens the
        file at the target and emits its frames from there, which is what makes the picture follow
        the strip within one poll instead of at the renderer's next produced frame.

        Past what has been PUBLISHED the target is attempted as asked when the reader knows written
        frames are still buffered (`_backlog`), and clamped to the published edge only when it does
        not: a HELD pane stops polling, so its last published frame is not the end of the file -
        clamping to it made every strip press on a held preview land on that one frame, i.e. "any
        attempt to move it puts me at the current end of the upscale" (owner, 2026-11-24). The poll
        itself falls back to the published edge if the file holds no frame at the target (a
        half-written fragment), so a press past the end still lands on the deepest picture that exists.
        """
        if self._path is None:
            return 0.0
        try:
            size = os.path.getsize(self._path)
        except OSError as exc:
            self._fail(("stat",), f"tail: cannot stat {self._path}: {exc}")
            return 0.0
        if size < self._size:
            # A new render at the same path: the old playhead says nothing about this file.
            self._restart()
        target = max(0.0, float(seconds))
        edge = self.produced_seconds
        self._seek_fallback = None
        if target > edge:
            if self._backlog:
                self._seek_fallback = edge
            else:
                target = edge
        # The floor sits just under the target: the frame AT the target is emitted, while the frames
        # of the keyframe re-read behind it stay de-duplicated.
        self._last_pts = target - self._interval() * 0.5 if target > 0.0 else None
        self._drop_container()
        self._exhausted = False      # there is work at the new position even if the file has not grown
        return target

    def _read(self, max_frames: int | None = None) -> int:
        """One decode pass at the current end of the file, with the failure policy in one place."""
        try:
            emitted = self._decode_available(max_frames)
        except Exception as exc:
            # The read position is now inside a fragment libavformat could not finish. Dropping the
            # container is what makes the next growth re-read from the last complete frame instead of
            # continuing from a broken position.
            self._decode_failures += 1
            self._exhausted = True
            self._drop_container()
            if self._decode_failures >= _ERROR_AFTER_FAILURES:
                self._fail(("decode",),
                           f"tail: {self._path}: {self._decode_failures} consecutive failed reads "
                           f"at {self._size} bytes, last: {exc}")
            return 0
        if emitted:
            self._decode_failures = 0
            self._signature = None
        return emitted

    def _decode_available(self, limit: int | None = None) -> int:
        """Pull frames until the demuxer runs dry or this call's cap is reached, skipping what is published.

        `limit` is what THIS call may emit; `None` means the reader's per-poll cap. A `while/else` tells
        the two stop reasons apart: reaching the cap leaves written frames buffered behind the playhead
        (`_backlog`), which is what lets a held pane seek deeper than the last frame it published.
        """
        cap = self._max_frames_per_poll if limit is None else max(1, int(limit))
        emitted = 0
        fps = self._fps
        self._backlog = False
        while emitted < cap:
            try:
                decoded = next(self._frame_iter)
            except StopIteration:
                self._exhausted = True
                break
            pts = frame_pts(decoded)
            if getattr(decoded, "time", None) is None:
                pts = (self._last_pts if self._last_pts is not None else 0.0) + self._interval()
            if self._last_pts is not None and pts <= self._last_pts:
                continue  # a re-read after a resume: this frame is already on screen
            self._emitted += 1
            index = int(round(pts * fps)) if fps > 0 else self._emitted
            self._last_pts = pts
            if self._edge_pts is None or pts > self._edge_pts:
                self._edge_pts = pts
            self.frame_ready.emit(pack_frame(decoded, index, self._pixel_format, pts=pts))
            emitted += 1
        else:
            # The cap stopped the read, not the file: there is more written data behind the playhead.
            self._backlog = True
        return emitted

    def _interval(self) -> float:
        return 1.0 / self._fps if self._fps > 0 else 0.04

    # -- stream handling ---------------------------------------------------------------------------

    def _open_container(self) -> bool:
        try:
            container = self._container_factory(self._path)
        except ImportError:
            self._fail(("pyav",), NO_PYAV)
            return False
        except Exception as exc:
            self._open_failures += 1
            if self._open_failures >= self._open_grace_polls:
                # Still unreadable after the grace window: the header is not coming, say so once.
                self._fail(("open",),
                           f"tail: {self._path} is still not readable after "
                           f"{self._open_failures} polls: {exc}")
            return False

        video = getattr(container.streams, "video", None)
        if not video:
            container.close()
            self._fail(("no-video",), f"tail: {self._path} has no video stream")
            return False

        stream = video[0]
        stream.codec_context.thread_count = self._codec_threads
        self._container = container
        self._stream = stream
        self._fps = float(stream.average_rate) if stream.average_rate else 0.0
        self._open_failures = 0
        self._resume_at_last_frame()
        self._reset_stream()
        return True

    def _resume_at_last_frame(self) -> None:
        """Seek back to the last published PTS so a resume re-decodes only the current fragment.

        Backward seek, so it cannot look past what the writer has written. A container that cannot
        seek yet (a fragmented file with no index for that position) falls back to reading from the
        start; `_decode_available` skips what was already shown, so the picture stays correct.
        """
        if self._last_pts is None:
            return
        try:
            self._container.seek(int(max(0.0, self._last_pts) * _MICROS), backward=True)
        except Exception as exc:
            self._fail(("seek",),
                       f"tail: cannot resume {self._path} at {self._last_pts:.3f} s ({exc}); "
                       f"reading from the start and skipping what was shown")

    def _reset_stream(self) -> None:
        self._stream.codec_context.flush_buffers()
        self._demux = self._container.demux(self._stream)
        self._frame_iter = self._frame_stream()
        self._exhausted = False

    def _frame_stream(self):
        """Decoded frames from the current demux position, one packet at a time.

        One generator across polls: dropping it between polls would throw away the frames of a packet
        that is already in the decoder.
        """
        while True:
            try:
                packet = next(self._demux)
            except StopIteration:
                return
            for frame in packet.decode():
                yield frame

    def _drop_container(self) -> None:
        self._demux = None
        self._frame_iter = None
        self._stream = None
        self._exhausted = True
        self._backlog = False
        container, self._container = self._container, None
        if container is not None:
            try:
                container.close()
            except Exception as exc:  # a container that will not close is a bug, not a silent state
                self.error.emit(f"tail: closing {exc}")

    def _restart(self) -> None:
        self._drop_container()
        self._size = 0
        self._emitted = 0
        self._last_pts = None
        self._edge_pts = None
        self._seek_fallback = None
        self._decode_failures = 0
        self._signature = None

    # -- reporting ---------------------------------------------------------------------------------

    def _fail(self, signature: tuple, message: str) -> None:
        """Emit one error per distinct failure, not one per poll."""
        if signature == self._signature:
            return
        self._signature = signature
        self.error.emit(message)


class LiveTailThread(QThread):
    """Drive a `TailReader` from its own thread.

    The reader needs a clock. The first attempt gave it a `QTimer` built inside the render's worker
    (`threading.Thread`) - a thread with no Qt event loop, so the timer never fired and the pane never
    showed a live frame (measured 2026-11-23: a 12 s preview left the pane on the 640x360 thumbnail; the
    only trace was one swallowed `TypeError` in log.txt). The GUI thread is not an option either: one
    poll decodes up to `max_frames_per_poll` frames, which is exactly the work a 4K preview must not do
    between paints.

    So the loop lives here, and the two things that cross the thread boundary are explicit:

    * `frame_ready` / `error` are signals, delivered to whoever the owner connected (the GUI thread);
    * a seek arrives as a *pending target* - `seek` only stores it, and the loop applies it between
      polls. `TailReader`'s state is then touched by one thread only.
    """

    #: One `FrameRef`, in presentation order (re-emitted from the reader).
    frame_ready = Signal(object)
    #: A failure worth reporting (re-emitted from the reader).
    error = Signal(str)

    def __init__(self, path: str, parent: QObject | None = None, *, interval_ms: int = 120,
                 paused: bool = True, **reader_kwargs) -> None:
        super().__init__(parent)
        self._path = str(path)
        self._interval_ms = max(20, int(interval_ms))
        self._reader_kwargs = reader_kwargs
        self._reader: TailReader | None = None
        self._pending_seek: float | None = None
        self._stop_requested = False
        #: A preview stands still until its transport is pressed (the owner's default), so `paused` starts True: the
        #: loop still reads until the FIRST frame exists (an empty pane would be "paused on nothing"), then holds it.
        self._paused = bool(paused)

    # -- what the owner reads while the loop runs ----------------------------------------------------

    @property
    def path(self) -> str:
        return self._path

    @property
    def published_frames(self) -> int:
        reader = self._reader
        return int(reader.published_frames) if reader is not None else 0

    @property
    def produced_seconds(self) -> float:
        reader = self._reader
        return float(reader.produced_seconds) if reader is not None else 0.0

    def seek(self, seconds: float) -> None:
        """Ask the loop to reposition before its next poll; coalesces while a drag is in flight."""
        self._pending_seek = max(0.0, float(seconds))

    @property
    def paused(self) -> bool:
        return self._paused

    def set_paused(self, paused: bool) -> None:
        """Hold the picture (no polling at all) or let it follow the file again.

        While paused the loop does no I/O: the renderer keeps writing and the reader's position is untouched, so
        resuming reads the backlog and catches up at the per-poll frame rate. A pending seek still applies - a strip
        press while paused must show its target - and that poll is the only read a paused loop performs.
        """
        self._paused = bool(paused)

    def stop_and_wait(self, timeout_ms: int = 3000) -> None:
        self._stop_requested = True
        self.wait(timeout_ms)

    # -- the loop ------------------------------------------------------------------------------------

    def run(self) -> None:
        reader = TailReader(pixel_format="rgb24", **self._reader_kwargs)
        reader.frame_ready.connect(self.frame_ready)
        reader.error.connect(self.error)
        self._reader = reader
        opened = False
        try:
            while not self._stop_requested:
                pending, self._pending_seek = self._pending_seek, None
                #: A seek on a HELD pane publishes its target and nothing else: the pane stands on one frame,
                #: so the poll that applies the press is capped at one frame. Without the cap it read a whole
                #: per-poll burst (default 4) and left the picture and the marker that far past the press.
                held_seek = pending is not None and self._paused
                if pending is not None:
                    reader.seek(pending)
                elif self._paused and reader.published_frames > 0:
                    # Held: no read, no decode, the pane keeps this frame. A seek above skips this branch exactly
                    # once, so its target becomes visible.
                    self.msleep(self._interval_ms)
                    continue
                if not opened:
                    # The container does not exist for the first moment of a render; the reader's own grace
                    # period turns "not yet a container" into a wait.
                    opened = bool(reader.open(self._path))
                elif not self._stop_requested:
                    reader.poll(1 if held_seek else None)
                if not self._stop_requested:
                    self.msleep(self._interval_ms)
        finally:
            try:
                reader.close()
            finally:
                self._reader = None
