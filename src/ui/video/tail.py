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

`import av` happens in the default container factory, not at module scope: the surface can import
this module on an interpreter without PyAV and gets an `error` signal only if it actually tails a
file.
"""

from __future__ import annotations

import os

from PySide6.QtCore import QObject, Signal

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
        self._exhausted = True      # the current demux position has nothing left to give
        self._open_failures = 0
        self._decode_failures = 0
        self._signature: tuple | None = None   # de-duplicates a repeated error

    # -- lifecycle ---------------------------------------------------------------------------------

    def open(self, path: str) -> bool:
        """Point the reader at a growing file; the container itself opens lazily on the first bytes."""
        self.close()
        path = str(path)
        if not os.path.isfile(path):
            self.error.emit(f"tail: no such file: {path}")
            return False
        self._path = path
        return True

    def close(self) -> None:
        """Release the demuxer; the next `open` starts a new file."""
        self._path = None
        self._size = 0
        self._emitted = 0
        self._last_pts = None
        self._exhausted = True
        self._open_failures = 0
        self._decode_failures = 0
        self._signature = None
        self._drop_container()

    # -- the poll ----------------------------------------------------------------------------------

    def poll(self) -> int:
        """Decode what the file has grown by; return how many frames were emitted.

        Called from the owner's timer. A poll that finds no growth and an exhausted demuxer does no
        I/O at all, which is the common case while the renderer is busy with a frame; a poll that
        finds buffered frames left over from a capped read continues them without waiting for growth.
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

        emitted = self._read()
        if emitted == 0 and self._exhausted and self._container is not None:
            # libavformat did not notice the growth (its EOF flag is set). Reopening and seeking back
            # to the last published frame is the resume that always works; it re-decodes the current
            # fragment only, and `_decode_available` skips what is already on screen.
            self._drop_container()
            if not self._open_container():
                return 0
            emitted = self._read()
        return emitted

    def _read(self) -> int:
        """One decode pass at the current end of the file, with the failure policy in one place."""
        try:
            emitted = self._decode_available()
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

    def _decode_available(self) -> int:
        """Pull frames until the demuxer runs dry, skipping anything already published."""
        emitted = 0
        fps = self._fps
        while emitted < self._max_frames_per_poll:
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
            self.frame_ready.emit(pack_frame(decoded, index, self._pixel_format, pts=pts))
            emitted += 1
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
        self._decode_failures = 0
        self._signature = None

    # -- reporting ---------------------------------------------------------------------------------

    def _fail(self, signature: tuple, message: str) -> None:
        """Emit one error per distinct failure, not one per poll."""
        if signature == self._signature:
            return
        self._signature = signature
        self.error.emit(message)
