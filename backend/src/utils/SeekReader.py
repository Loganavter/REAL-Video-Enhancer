"""The resident preview decoder: one ffmpeg for a whole scrub, not one per seek.

`RenderVideo._seek_preview` used to open an `FFmpegRead` at the requested time, read one frame and close
it. Every request therefore paid ffmpeg's start-up, a demux seek and a decoder init, and because the pane's
request block is newest-wins (`utils/DisplayChannel.SeekState`) a scrub that arrives faster than one spawn
turns the preview into a chain of ffmpeg processes.

This class keeps ONE decoder alive for the whole render. A request at or ahead of the current position is
served by reading and discarding the frames in between; a request *behind* it, or further ahead than a walk
is worth, re-opens the source at the target - the one case that still pays a spawn.

Frame exactness is not our arithmetic's job, it is ffmpeg's. `-ss T` in front of `-i` seeks to the keyframe
before T and lets the decoder discard up to the first frame whose presentation time is at or after T
(frame-accurate since ffmpeg 2.1), so the frame a spawn hands back first is ffmpeg's own answer for T - the
same frame the one-shot reader returned (measured on a 10 fps synthetic clip: `-ss 1.05` -> frame 11, `-ss
2.999` -> frame 30, `-ss 5.9` -> frame 59, `-ss 5.95` -> none). The walk only has to decide *how many* frames
lie between the current position and the target, and it does that against a frame grid probed once per
render (`probe_frame_grid`) rather than against the render's own frame rate: that rate is the two-decimal
`fps` field of `ffmpeg -i` ("23.98", not 23.976), and an absolute index built on it drifts by a frame every
few minutes of clip. A probed grid is exact for a constant-rate source; a variable-rate source is detected
and every seek re-opens instead, which is slower but never a wrong frame.

Nothing here raises into the render loop: `frame_at` answers `None` and logs, the same contract as
`FFmpegRead.read_frame` returning `None` at the end of the file.
"""

from __future__ import annotations

import math
import re
import subprocess

from ..FFmpegBuffers import FFmpegRead
from .Util import log, subprocess_popen_without_terminal

#: How many frames the grid probe decodes. Two would give a pitch; thirteen makes the *median* delta the
#: pitch, so one odd first delta (an edit-list gap, a dropped frame at a cut) cannot poison the grid.
_PROBE_FRAMES = 13
#: `showinfo` prints one `pts_time:<seconds>` per frame it is given.
_PTS_TIME = re.compile(r"pts_time:(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
#: A source is treated as constant-rate when every probed frame sits within this share of a pitch of the
#: grid fitted to the first one. Wide enough for muxer rounding (a `90k` timebase prints 0.041667-style
#: values), narrow enough that a variable-rate source fails it.
_GRID_TOLERANCE = 0.25
#: Seconds the probe is allowed to take before it is given up on (a seek hint, a decode of ~13 frames).
_PROBE_TIMEOUT = 30.0


def probe_frame_grid(inputFile, ffmpeg_path: str = "ffmpeg", frames: int = _PROBE_FRAMES,
                     near: float = 0.0):
    """`(first_pts, pitch)` of a constant-rate source, or None when it cannot be trusted.

    `first_pts` is any grid point (index 0 need not be the file's first frame: only the *differences*
    between indices are used), so the probe seeks `near` instead of decoding from the start - one cheap
    read near the requested time rather than minutes of clip on a long source.

    None means "no constant grid here" and is not an error: the caller falls back to one spawn per seek,
    which is the behaviour this module replaced. A variable-rate source, a file with fewer than two frames
    or an ffmpeg that will not talk all land here.
    """
    command = [f"{ffmpeg_path}", "-hide_banner", "-nostdin"]
    if near and near > 0:
        command += ["-ss", f"{max(0.0, float(near)):.6f}"]
    command += [
        "-i", f"{inputFile}",
        "-map", "0:v:0",
        "-vf", "showinfo",
        "-frames:v", str(max(2, int(frames))),
        "-an", "-sn",
        "-f", "null", "-",
    ]
    process = subprocess_popen_without_terminal(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    try:
        _, stderr = process.communicate(timeout=_PROBE_TIMEOUT)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        return None
    pts = [float(value) for value in _PTS_TIME.findall(stderr or "")]
    if len(pts) < 2:
        return None
    deltas = [b - a for a, b in zip(pts, pts[1:]) if b > a]
    if not deltas:
        return None
    deltas.sort()
    pitch = deltas[len(deltas) // 2]
    if not (pitch > 0 and math.isfinite(pitch)):
        return None
    for offset, value in enumerate(pts):
        if abs((value - pts[0]) - offset * pitch) > _GRID_TOLERANCE * pitch:
            return None
    return pts[0], pitch


class SeekReader:
    """One resident `FFmpegRead` that serves every preview seek of a render.

    `width`/`height` are the crop size and `reader_kwargs` are the same `FFmpegRead` options the export's
    reader is built with (pixel format, colour tags, ffmpeg path); `start_time` is where the decoder opens
    when `frame_at` has not moved it yet - the render passes the time of the seek that builds this object,
    so the first request is a spawn on the requested frame and nothing else.

    `forward_limit_seconds` is where a forward move stops being worth a walk: the walk decodes and pipes
    the frames in between, so it costs a frame per skipped frame, while a re-open costs one spawn
    (start-up + demux seek + decoder init, plus the one frame it does send). Measured on this machine by
    `harness/tests/test_seek_reader.py` with one spawn per request against a warm resident walk:

    | read size | spawn | walk per frame | break-even |
    |---|---|---|---|
    | 160x120 rgb24 | 50.9 ms | 0.12 ms | ~443 frames / 44.3 s |
    | 1920x1080 rgb24 | 118.7 ms | 3.29 ms | ~36 frames / 1.50 s |
    | 3840x2160 rgb24 | 341.3 ms | 13.00 ms | ~26 frames / 1.09 s |

    The break-even shrinks as the frame grows because the walk pays the pipe per skipped frame while a
    spawn pays it once, so the default sits at the *worst* measured case (1.0 s) rather than the best: an
    over-long walk is strictly worse than a spawn, an under-long one is merely the cost this module
    replaced. A caller can raise it (a small preview box is cheap to walk), and `0` pins the object to one
    spawn per changed request.
    """

    def __init__(self, inputFile, *, width, height, start_time=0.0,
                 forward_limit_seconds: float = 1.0, **reader_kwargs):
        self.inputFile = inputFile
        self.width = int(width)
        self.height = int(height)
        self.start_time = float(start_time or 0.0)
        self.forward_limit_seconds = float(forward_limit_seconds)
        self._kwargs = dict(reader_kwargs)
        # A seek reader reads the whole source from the requested time; the export's segment bound has
        # nothing to say about it (`_open_reader_at` passed `end_time=None` for the same reason).
        self._kwargs.setdefault("end_time", None)
        self._ffmpeg_path = self._kwargs.get("ffmpeg_path") or "ffmpeg"
        #: `(first_pts, pitch)`, probed once, or None for a source that has no constant grid.
        self._grid = None
        self._grid_probed = False
        self._reader = None
        #: Grid index of the frame the next `read_frame()` returns. Indices are internal: only their
        #: differences are used, so an arbitrary origin is fine.
        self._next_index = None
        #: Index the stream ran out at. The seek source is the original file, not the growing preview,
        #: so a request at or past it can be answered without another spawn.
        self._eof_index = None
        self._closed = False
        # Evidence of what this object actually did; logged at close.
        self.seeks = 0
        self.spawns = 0
        self.frames_read = 0
        self.frames_skipped = 0
        self.failures = 0

    # -- lifecycle -----------------------------------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def close(self) -> None:
        """Release the decoder. Safe to call twice and from a `finally`."""
        self._closed = True
        self._drop_reader()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # -- the one public read -------------------------------------------------------------------------
    def frame_at(self, seconds: float) -> bytes | None:
        """The frame a fresh `-ss seconds` reader would return first, or None (end of file, no frame).

        Never raises: a broken source, a closed object or an unparsable time answers None and says so in
        the log, because this is called from the render loop between two exported frames.
        """
        if self._closed:
            return None
        try:
            target = float(seconds)
        except (TypeError, ValueError):
            log(f"seek: {seconds!r} is not a time")
            return None
        if not math.isfinite(target):
            log(f"seek: {target!r} is not a time a frame can be read at")
            return None
        self.seeks += 1
        try:
            return self._frame_at(target)
        except Exception as exc:   # a preview must never take the export down
            self.failures += 1
            log(f"seek: could not decode a frame at {target:.3f}s ({exc})")
            self._drop_reader()
            return None

    # -- one request ---------------------------------------------------------------------------------
    def _frame_at(self, target: float) -> bytes | None:
        grid = self._ensure_grid()
        index = self._index_at(target)
        pitch = grid[1] if grid else 0.0
        if index is None:
            # No grid to walk: serve the request with one decoder, exactly the old path.
            self._drop_reader()
            if not self._open(target, index=None):
                return None
            return self._read()
        if self._eof_index is not None and index >= self._eof_index:
            return None
        if self._reader is None:
            if not self._open(target, index):
                return None
        else:
            ahead = index - self._next_index
            if ahead < 0 or ahead * pitch > self.forward_limit_seconds:
                # Behind: a decoder cannot read backwards. Too far ahead: the walk would decode more
                # than a spawn costs. Both are answered by one seek at the target.
                self._drop_reader()
                if not self._open(target, index):
                    return None
        while self._next_index < index:
            if not self._skip():
                return None
        return self._read()

    def _read(self) -> bytes | None:
        chunk = self._reader.read_frame()
        if chunk is None:
            self._note_eof()
            return None
        self._next_index += 1
        self.frames_read += 1
        return chunk

    def _skip(self) -> bool:
        chunk = self._reader.read_frame()
        if chunk is None:
            self._note_eof()
            return False
        self._next_index += 1
        self.frames_skipped += 1
        return True

    def _note_eof(self):
        self._eof_index = self._next_index
        self._drop_reader()

    # -- decoder management --------------------------------------------------------------------------
    def _open(self, position: float, index: int | None) -> bool:
        try:
            reader = FFmpegRead(
                inputFile=self.inputFile,
                width=self.width,
                height=self.height,
                start_time=float(position),
                **self._kwargs,
            )
        except Exception as exc:
            log(f"seek: could not open the source at {position:.3f}s ({exc})")
            self._reader = None
            return False
        self._reader = reader
        self._next_index = index
        self.spawns += 1
        return True

    def _drop_reader(self):
        reader, self._reader = self._reader, None
        if reader is None:
            return
        try:
            reader.close()
        except Exception:
            pass

    # -- the frame grid ------------------------------------------------------------------------------
    def _ensure_grid(self):
        if self._grid_probed:
            return self._grid
        self._grid_probed = True
        try:
            self._grid = probe_frame_grid(self.inputFile, self._ffmpeg_path, near=self.start_time)
        except Exception as exc:
            log(f"seek: frame-rate probe failed ({exc}); seeks will re-open the source")
            self._grid = None
        if self._grid is None:
            log("seek: no constant frame grid for this source; every preview seek re-opens it "
                "(one spawn per request, which is what the resident reader was added to avoid)")
        else:
            first, pitch = self._grid
            log(f"seek: resident reader on a {1.0 / pitch:.3f} fps grid "
                f"({pitch * 1000.0:.3f} ms/frame, first probe frame {first:.3f}s)")
        return self._grid

    def _index_at(self, seconds: float) -> int | None:
        """Grid index of the first frame at or after `seconds` - the frame `-ss seconds` returns."""
        if self._grid is None:
            return None
        first, pitch = self._grid
        try:
            return max(0, int(math.ceil((seconds - first) / pitch - 1e-6)))
        except (OverflowError, ValueError):
            return None

    # -- evidence ------------------------------------------------------------------------------------
    def stats(self) -> str:
        """One line about what this reader did, for the render log."""
        line = (f"seek: {self.seeks} preview request(s), {self.frames_read} frame(s) decoded, "
                f"{self.spawns} decoder spawn(s), {self.frames_skipped} frame(s) skipped")
        if self.failures:
            line += f", {self.failures} failure(s)"
        return line
