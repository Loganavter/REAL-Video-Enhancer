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
  renderer's next fragment. Past the produced edge the reader does not assume there is nothing: a HELD
  pane stops polling, so "past the edge" only means "past what this reader last published" while the
  muxer may have flushed since the pause. Such a target is attempted, and the press is judged
  unreachable only by the read that answers it.
* **A seek that is not in the file publishes NOTHING.** When the container holds no frame at the
  target (the renderer is mid-fragment, or the press reached beyond what it flushed) the picture and
  its marker stay where they are and the refusal is reported. It is never answered by the published
  edge: on a held pane that edge is stale, so it lands the picture somewhere the finger did not ask
  for - and once behind the standing frame (measured live 2026-11-24: a strip press at 30 % put the
  playhead on 0.79 s after it had stood at 3.88 s, which is the owner's "any attempt to move it puts
  me at the current end of the upscale").
* **A held pane seeks ONE frame.** A press on a paused preview is answered by `poll(1)`: its target,
  with no backlog behind it and no overshoot. A playing pane keeps the normal per-poll rate.

`import av` happens in the default container factory, not at module scope: the surface can import
this module on an interpreter without PyAV and gets an `error` signal only if it actually tails a
file.
"""

from __future__ import annotations

import threading
import time

import os

from PySide6.QtCore import QObject, QThread, Signal

from .frame_names import read_names
from .frames import FORMATS, frame_pts, pack_frame, stream_color_tags
from . import telemetry

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
    #: A seek the container could not satisfy: its target is past the last complete frame the file holds. The picture
    #: does not move; the message is for the owner's log, because a press that silently does nothing reads as a bug.
    seek_unavailable = Signal(str)

    def __init__(self, parent: QObject | None = None, *, pixel_format: str = "rgb24",
                 max_frames_per_poll: int = 4, codec_threads: int = 2,
                 container_factory=None, open_grace_polls: int = 40,
                 target_size: tuple[int, int] | None = None,
                 color_space: str | None = None, color_range: str | None = None) -> None:
        super().__init__(parent)
        if pixel_format not in FORMATS:
            raise ValueError(f"unknown frame format {pixel_format!r}, expected one of {FORMATS}")
        self._pixel_format = pixel_format
        #: The matrix/range the shader converts the planes with (see `VideoDecoder`): the stream's own
        #: tags when the caller has none, and the renderer's bt709/tv fallback when the file has none.
        self._color_space = color_space
        self._color_range = color_range
        #: True when the CALLER stated them: an auto-read pair must be re-read on the next file.
        self._color_explicit = color_space is not None or color_range is not None
        #: `(w, h)` the decoded picture is converted to; None keeps the file's own size. The pane cannot show more
        #: than its box, and converting a 2880x2160 frame to RGB costs 24.9 MB of traffic per frame.
        self._target_size = None if target_size is None else (max(1, int(target_size[0])), max(1, int(target_size[1])))
        self._max_frames_per_poll = max(1, int(max_frames_per_poll))
        self._codec_threads = max(1, int(codec_threads))
        self._container_factory = container_factory or _default_container_factory
        #: A file whose header is still being written fails to open for a while; that is a wait. The
        #: count turns an indefinite silence into one honest error if the header never appears.
        self._open_grace_polls = max(1, int(open_grace_polls))

        self._path: str | None = None
        #: The renderer's own names for the pictures it wrote (`<output>.srcpts`, one source time per OUTPUT frame).
        #: The export is CFR, so the file's PTS is the muxer's grid, not the source moment a held picture carries; the
        #: compare's source half is named by the source's PTS, and the two must be the same number to pair at all.
        self._names: list[float] | None = None
        self._size = 0
        self._container = None
        self._stream = None
        self._demux = None
        self._frame_iter = None
        self._fps = 0.0
        self._emitted = 0
        #: The container's own first timestamp, seconds. A writer that shifts the whole timeline
        #: (`-avoid_negative_ts make_zero` + an encoder's B-frame delay, measured 2026-11-25: the first frame of a
        #: source at 0.000 s lands at 0.083 s) makes every PTS a little LATER than the frame it names. It is subtracted
        #: from every published name and index, and added back to every container seek, so the pane's clock is the
        #: CONTENT's; see `_origin`.
        self._media_origin: float | None = None
        self._last_pts: float | None = None
        #: The highest PTS this reader has published - the produced edge as far as the reader knows it.
        #: Unlike `_last_pts` (the de-duplication floor) a seek never lowers it: it is what a seek is
        #: clamped to.
        self._edge_pts: float | None = None
        self._exhausted = True      # the current demux position has nothing left to give
        #: The last poll stopped because its per-poll cap was reached, not because the demuxer ran dry: written
        #: frames are buffered behind the playhead, so the published edge is NOT the end of the file.
        self._backlog = False
        #: The seek this reader still owes an answer to: `(target seconds, pts of the frame on screen)`. Only set for a
        #: target past everything published, whose reachability only the answering `poll` can decide. `None` = no such
        #: press outstanding.
        self._seek_attempt: tuple[float, float | None] | None = None
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
        self._names = None
        return True

    def close(self) -> None:
        """Release the demuxer; the next `open` starts a new file."""
        self._path = None
        self._names = None
        self._size = 0
        self._emitted = 0
        self._last_pts = None
        self._edge_pts = None
        self._media_origin = None
        self._exhausted = True
        self._backlog = False
        self._seek_attempt = None
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
        if emitted and self._seek_attempt is not None:
            # The press was answered: the target (or the first frame after it) is in the file.
            self._seek_attempt = None
        if emitted == 0 and self._seek_attempt is not None:
            # The target is past the last COMPLETE frame the container holds: the renderer is mid-fragment, or the press
            # reached beyond what it flushed (a held pane does not poll, so its published edge says nothing about the
            # file). Say so and publish NOTHING - the pane keeps the frame it stands on. Falling back to the published
            # edge here is what moved the picture backwards live (2026-11-24): that edge is stale on a held pane.
            target, standing = self._seek_attempt
            self._seek_attempt = None
            self._last_pts = standing           # the de-duplication floor goes back to the frame on screen (FILE time)
            self._drop_container()
            self._exhausted = False             # the next poll reopens at that frame instead of waiting for growth
            where = (f"{max(0.0, float(standing) - self._origin()):.3f} s"
                     if standing is not None else "no frame published yet")
            self.seek_unavailable.emit(
                f"seek to {target:.3f} s is not in the flushed file ({self._path}); "
                f"the picture stays at {where}")
        return emitted

    # -- seeking -----------------------------------------------------------------------------------

    @property
    def produced_seconds(self) -> float:
        """Media time of the last frame this reader published: the produced edge, as far as it knows."""
        edge = self._edge_pts
        if edge is None:
            return 0.0
        return max(0.0, float(edge) - self._origin())

    def _origin(self) -> float:
        """The file timeline's own start, seconds; 0 until the container says otherwise.

        Media time here is CONTENT time: a file written with `-avoid_negative_ts make_zero` starts at the encoder's
        B-frame delay (measured 2026-11-25: 0.083 s, two frames at 23.976), while its first frame is the source's
        first. Without this subtraction the pane names every picture two frames later than the frame it shows, and
        the LMB compare then decodes the source at a time two frames after the result's frame.
        """
        value = self._media_origin
        return 0.0 if value is None else float(value)

    @property
    def media_origin(self) -> float | None:
        """The file's own first timestamp, seconds; None until the container is open.

        It is SUBTRACTED from every published media time and index (the pane's clock is content time) and added back
        to every container seek; it stays readable for the log line and for tests that pin the shift.
        """
        return self._media_origin

    @property
    def published_frames(self) -> int:
        """How many frames the surface has seen from this reader (0 until the first one)."""
        return int(self._emitted)

    def seek(self, seconds: float, *, attempt_past_edge: bool = False) -> float:
        """Move the playhead inside the already-written part; return where it landed, in seconds.

        `seconds` is CONTENT time (the pane's clock); the container is asked for `seconds + media_origin`, the file's
        own timestamp for that content.

        Only `poll` reads, so a seek is cheap and a drag coalesces: it moves the de-duplication
        floor, drops the demuxer and marks the reader as having work. The next `poll` reopens the
        file at the target and emits its frames from there, which is what makes the picture follow
        the strip within one poll instead of at the renderer's next produced frame.

        A target past what has been PUBLISHED is only clamped to the published edge when nothing says
        the file may hold more (`attempt_past_edge` false and no buffered frames - i.e. a playing pane
        that read to the end of the demuxer, where the edge IS the file). A HELD pane passes
        `attempt_past_edge`: it stopped polling at the pause, so its edge is stale and the muxer may
        have flushed since, and only the answering `poll` can tell. That poll publishes the target, or
        nothing at all when the file has no frame there (see `poll`); it never falls back to the edge,
        which would put the picture behind itself.
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
        target = max(0.0, float(seconds))          # CONTENT time (the pane's clock), not the file's
        origin = self._origin()
        edge = self.produced_seconds
        standing = self._last_pts        # the frame on screen, in FILE time (the de-duplication floor's unit)
        self._seek_attempt = None
        # The floor sits just under the target: the frame AT the target is emitted, while the frames
        # of the keyframe re-read behind it stay de-duplicated. Both are FILE time.
        floor = target - self._interval() * 0.5 + origin if target > 0.0 else None
        if target > edge:
            if attempt_past_edge or self._backlog:
                self._seek_attempt = (target, standing)
                if floor is not None and standing is not None:
                    # An attempt past the edge may never land BEHIND the frame on screen: the floor under a target
                    # that is barely past it would otherwise admit one frame the picture already passed.
                    floor = max(floor, standing)
            else:
                target = edge
                floor = target - self._interval() * 0.5 + origin if target > 0.0 else None
        self._last_pts = floor
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
            # The pane's clock is CONTENT time, the file's PTS is not (see `_origin`): subtract the file's own start
            # from the name AND from the index, so a picture's label, its frame number and the source decoder's answer
            # are the same number.
            origin = self._origin()
            shown_pts = pts - origin
            index = int(round(shown_pts * fps)) if fps > 0 else self._emitted
            named = self._name_for(index)
            if named is not None:
                shown_pts = named
            self._last_pts = pts
            if self._edge_pts is None or pts > self._edge_pts:
                self._edge_pts = pts
            self.frame_ready.emit(pack_frame(decoded, index, self._pixel_format, pts=shown_pts,
                                             size=self._target_size,
                                             color_space=self._color_space,
                                             color_range=self._color_range))
            emitted += 1
        else:
            # The cap stopped the read, not the file: there is more written data behind the playhead.
            self._backlog = True
        return emitted

    def _name_for(self, index: int) -> float | None:
        """The SOURCE time the renderer recorded for output picture `index`, or None to keep the container's PTS.

        Read once per file (the list is written when the render ends, so a live tail finds it as soon as the render
        closes; until then the container's own PTS names the frames, which is what this reader always did).
        """
        if self._path:
            # The renderer flushes one name per QUEUED picture, so the list grows while this reader consumes the file:
            # a name is re-read whenever the list does not reach the frame being painted. A file that is not there yet
            # (or one with no sidecar at all) leaves the container's PTS as the name, which is the old behaviour.
            if self._names is None or index >= len(self._names):
                fresh = read_names(self._path)
                if fresh:
                    self._names = fresh
        names = self._names or []
        if 0 <= index < len(names):
            return names[index]
        return None

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
        if not self._color_explicit:
            # The file's own tags, read once per reader: the writer emits bt709/tv for a 10-bit output,
            # but the SHADER's matrix must come from the file rather than from a constant (an HDR or a
            # bt601 source would otherwise be converted with the wrong coefficients).
            self._color_space, self._color_range = stream_color_tags(stream)
        self._container = container
        self._stream = stream
        self._fps = float(stream.average_rate) if stream.average_rate else 0.0
        try:
            start = stream.start_time
            origin = float(start * stream.time_base) if start is not None else 0.0
        except Exception:
            origin = None
        # Pin it: the file's own start is read once, and a reopen of the same growing file may not move it under
        # pictures already named (the header can be parsed before its first cluster's timestamp is known).
        if origin is not None and (self._media_origin is None or self._emitted == 0):
            self._media_origin = origin
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
        self._media_origin = None
        self._seek_attempt = None
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
    #: A press the file could not satisfy: the picture stays, the message goes to the owner's log.
    seek_unavailable = Signal(str)

    #: The longest a single frame may be held back: a gap in the file (or a seek to a distant point) must not park the
    #: thread on a stale estimate - the picture catches up instead of freezing.
    MAX_PRESENT_WAIT_S = 0.5

    #: How many frames in a row may be dropped for being late before the media clock re-anchors to the newest one. At 24
    #: fps this is ~0.25 s of lag: long enough that a normal catch-up still plays the timeline, short enough that a source
    #: slower than realtime (a render in progress) still puts pictures on screen.
    MAX_LATE_STREAK = 6

    #: How close to the newest written frame counts as "following the render". Within this the picture takes the writer's
    #: pace (no pacing, no drops); beyond it there is a backlog and the media clock owns the timeline.
    LIVE_EDGE_S = 0.25

    def __init__(self, path: str, parent: QObject | None = None, *, interval_ms: int = 33,
                 paused: bool = True, clock=None, sleep=None, target_size=None,
                 pixel_format: str = "rgb24", color_space: str | None = None,
                 color_range: str | None = None, **reader_kwargs) -> None:
        super().__init__(parent)
        #: The picture format this player asks the reader for: `rgb24` (the default, the QImage path)
        #: or one of the planar formats (`yuv420p10le` preferred for a 10-bit file, `yuv420p` as the
        #: 8-bit fallback), which the pane uploads as three plane textures. OPT-IN: the queue pane and
        #: the source decoder stay rgb24, so nothing changes for them.
        if pixel_format not in FORMATS:
            raise ValueError(f"unknown frame format {pixel_format!r}, expected one of {FORMATS}")
        self._pixel_format = pixel_format
        self._color_space = color_space
        self._color_range = color_range
        #: Injected by tests; the thread itself uses the real clocks. Pacing by media time is the whole difference
        #: between "the preview plays at the renderer's production rate" and "at the file's own fps".
        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep
        #: The pane's box: the reader converts each decoded frame to it instead of to the file's own (4K-class) size.
        self._target_size = None if target_size is None else (max(1, int(target_size[0])), max(1, int(target_size[1])))
        #: `(pts, wall)` the media clock is anchored to, or None before the first presented frame / after a seek.
        self._present_anchor: tuple[float, float] | None = None
        #: One source interval, learned from the reader once it opens; 24 fps until then.
        self._frame_interval = 1.0 / 24.0
        #: Frames skipped because they were already behind the media clock (evidence for the log, not a cost).
        self._dropped_late = 0
        #: Consecutive frames dropped by the lateness rule (see `_present`): reset by any emitted frame.
        self._late_streak = 0
        self._path = str(path)
        self._interval_ms = max(20, int(interval_ms))
        self._reader_kwargs = reader_kwargs
        self._reader: TailReader | None = None
        #: (seconds, attempt_past_edge): the flag is True when the ask may be ahead of what this reader
        #: has published (a held press, or a resume that jumps to the renderer's head).
        #: The seek slot is written by the GUI thread (`seek`) and taken by the loop's own thread. The take must be
        #: atomic with the clear: `ask, self._pending_seek = self._pending_seek, None` reads, then stores `None`, and a
        #: `seek()` landing between the two stores was silently lost - the tail kept its old position and the press
        #: timed out (audit 2026-11-27 §6.9). The lock makes "take what is there and leave nothing" one step.
        self._seek_lock = threading.Lock()
        self._pending_seek: tuple[float, bool] | None = None
        self._stop_requested = False
        #: A preview stands still until its transport is pressed (the owner's default), so `paused` starts True: the
        #: loop still reads until the FIRST frame exists (an empty pane would be "paused on nothing"), then holds it.
        self._paused = bool(paused)
        self._present_anchor = None

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

    @property
    def media_origin(self) -> float | None:
        """The file's own first timestamp (see `TailReader.media_origin`)."""
        reader = self._reader
        return reader.media_origin if reader is not None else None

    def seek(self, seconds: float, *, attempt_past_edge: bool = False) -> None:
        """Ask the loop to reposition before its next poll; coalesces while a drag is in flight.

        `attempt_past_edge` says the ask is allowed to be ahead of what this reader published - a held press asks the
        renderer-side machinery for a frame, and resuming asks for the RENDERER's head, which the tail has not read yet.
        Without it the ask is clamped to the published edge, which is right for an implicit follow and wrong for both of
        those (measured live 2026-11-24: a resume aimed at 2.544 s landed on 0.334 s because the edge still said 0.1).
        """
        with self._seek_lock:
            self._pending_seek = (max(0.0, float(seconds)), bool(attempt_past_edge))
        # A new position re-anchors the media clock: the next frame is due NOW, not where the old position was.
        self._present_anchor = None

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
        # A pause ends the run the clock was measuring; the resume starts a fresh one (see `_present`).
        self._present_anchor = None

    def stop_and_wait(self, timeout_ms: int = 3000) -> None:
        self._stop_requested = True
        self.wait(timeout_ms)

    # -- pacing --------------------------------------------------------------------------------------

    def _present(self, frame) -> None:
        """Hand the frame on at its MEDIA time, not as fast as the file grew.

        The reader polls every `interval_ms` and hands over whatever it found; with no pacing the picture advances at the
        renderer's production rate, so a 23.976 fps source plays at whatever fps the upscaler happens to run at (owner,
        2026-11-25: "кнопка play ... не с оригинальным фпс"). Here the media clock is anchored to the first presented
        frame and advances with wall time; a frame is held back until its pts is reached, and a frame that is already
        late is shown at once so the picture catches up instead of falling further behind.

        A bounding wait keeps a gap or a distant seek from parking the thread on a stale estimate.
        """
        if self._paused:
            # A held pane shows the ONE frame it has: no pacing and no dropping there, or a seek's target would be
            # skipped for being "late" against a clock that stopped (the anchor is reset by `set_paused`/`seek`).
            self.frame_ready.emit(frame)
            return
        pts = float(getattr(frame, "pts", 0.0) or 0.0)
        now = float(self._clock())
        edge = getattr(getattr(self, "_reader", None), "_edge_pts", None)
        if edge is not None and (float(edge) - pts) <= self.LIVE_EDGE_S:
            # AT THE LIVE EDGE: the renderer is still writing, so the pace of the picture IS the pace of the file. Pacing
            # against the media clock here makes every frame "late" whenever the render runs slower than realtime (the
            # normal case under inference: 13 fps written against a 24 fps clock), and the lateness rule below then drops
            # all of them - measured 2026-11-26: `frames=12..16 dropped_late=12..15`, not one picture on the pane. A
            # BACKLOG (a resume, a scrub back) is still a backlog: it plays at the media rate below.
            self._present_anchor = (pts, now)
            self._late_streak = 0
            self.frame_ready.emit(frame)
            return
        anchor = self._present_anchor
        if anchor is None or pts < anchor[0]:
            anchor = (pts, now)
            self._present_anchor = anchor
        due = anchor[0] + (now - anchor[1])
        ahead = pts - due
        if ahead > 0.0:
            self._sleep(min(ahead, self.MAX_PRESENT_WAIT_S))
        elif -ahead > 1.5 * self._frame_interval:
            # The decoder fell more than a frame behind the media clock (a 4K-class decode does): play the TIMELINE,
            # not every frame. Emitting this one would make the picture advance at the decode rate - the reported "play
            # runs at some other speed than the file's fps" (owner, 2026-11-25). The next frame is closer to `due`.
            self._dropped_late += 1
            self._late_streak += 1
            if self._late_streak < self.MAX_LATE_STREAK:
                return
            # ...but a source that can NEVER keep up must not empty the pane. A render slower than realtime writes a
            # frame every ~75 ms while the media clock runs at 1x, so EVERY frame is "late" and the rule above drops all
            # of them, for ever (measured 2026-11-26: `frames=12..16 dropped_late=12..15` and not one picture painted
            # during inference). The clock is what is wrong there, not the frames: re-anchor to the newest one and show
            # it, so the preview follows the pace the render actually produces at.
            self._present_anchor = (pts, now)
            self._late_streak = 0
            self.frame_ready.emit(frame)
            return
        self._late_streak = 0
        self.frame_ready.emit(frame)

    # -- the loop ------------------------------------------------------------------------------------

    def run(self) -> None:
        reader = TailReader(pixel_format=self._pixel_format, target_size=self._target_size,
                            color_space=self._color_space, color_range=self._color_range,
                            **self._reader_kwargs)
        interval_of = getattr(reader, "_interval", None)
        reader.frame_ready.connect(self._present)
        reader.error.connect(self.error)
        reader.seek_unavailable.connect(self.seek_unavailable)
        self._reader = reader
        opened = False
        #: One line per second of what the LOOP did - the tail had no number at all, so "play did nothing" could only be
        #: answered by reading the code. `frames` is what the reader emitted in that second, `edge` the newest PTS it has
        #: published, `size` the bytes the writer had grown to.
        produced_since = 0
        late_seen = 0
        reported_at = 0.0
        try:
            while not self._stop_requested:
                with self._seek_lock:
                    ask = self._pending_seek
                    self._pending_seek = None
                pending, ask_past_edge = ask if ask is not None else (None, False)
                #: A seek on a HELD pane publishes its target and nothing else: the pane stands on one frame,
                #: so the poll that applies the press is capped at one frame. Without the cap it read a whole
                #: per-poll burst (default 4) and left the picture and the marker that far past the press.
                held_seek = pending is not None and self._paused
                if pending is not None:
                    # A held pane stopped polling at the pause, so its published edge is stale: the press is
                    # attempted against the file rather than clamped to it (the answering poll publishes the
                    # target, or nothing at all - see `TailReader.seek`). A playing seek keeps the old rule.
                    reader.seek(pending, attempt_past_edge=held_seek or ask_past_edge)
                elif self._paused and reader.published_frames > 0:
                    # Held: no read, no decode, the pane keeps this frame. A seek above skips this branch exactly
                    # once, so its target becomes visible.
                    self.msleep(self._interval_ms)
                    continue
                if not opened:
                    # The container does not exist for the first moment of a render; the reader's own grace
                    # period turns "not yet a container" into a wait.
                    opened = bool(reader.open(self._path))
                    if opened:
                        try:
                            self._frame_interval = float(reader._interval())
                        except Exception:
                            pass
                elif not self._stop_requested:
                    produced = reader.poll(1 if held_seek else None)
                    produced_since += int(produced or 0)
                    if not produced:
                        # Nothing was written for a poll or more: the media clock must not keep running through a stall,
                        # or the next frames arrive "late" and the drop rule would throw them away (measured: a paused
                        # write made a growing file's next two frames 0.9 s behind and the tail skipped both). A stall
                        # re-anchors; a burst of frames keeps the clock, which is what makes the drop rule work.
                        self._present_anchor = None
                now = time.monotonic()
                if now - reported_at >= 1.0:
                    reported_at = now
                    dropped = int(getattr(self, "_dropped_late", 0)) - late_seen
                    late_seen += dropped
                    anchor = getattr(self, "_present_anchor", None)
                    telemetry.event("tail", paused=int(self._paused), opened=int(bool(opened)),
                                    frames=produced_since, dropped_late=dropped,
                                    anchor=round(float(anchor[0]), 3) if anchor else "-",
                                    edge=round(float(getattr(reader, "_edge_pts", 0.0) or 0.0), 3),
                                    size=int(getattr(reader, "_size", 0) or 0))
                    produced_since = 0
                if not self._stop_requested:
                    self.msleep(self._interval_ms)
        finally:
            try:
                reader.close()
            finally:
                self._reader = None
