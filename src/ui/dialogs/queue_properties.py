"""The Render Queue's **Properties…** window: what the selected queue item's file actually is, and what RVE will do
with it.

Owner, 2026-11-24: *"we could steal the properties window from BlinkTable for a function -- properties in the render
queue via the context menu (so you can see the video's properties)."* Then, while the first cell was not reading its
mail: MediaInfo is the preferred probe, the window has **two tabs** (MediaInfo / RVE), and the RVE tab carries the
queue's own record - position, timestamps, models, coefficients and the run report's numbers.

Structure taken from BlinkTable's `ImagePropertiesDialog` (``_reference/`` in that checkout): one painted document per
tab with a heading per group and a real table per group, the file's own name as the title, a **Copy all** action, and -
the rule that matters - **a value the probe could not measure renders as an em dash, never as a plausible default**.
Its plugin machinery is deliberately not ported: RVE has no plugin system and one window does not need one.

What is measured, and by which probe
------------------------------------
Three probes, all read-only, all at open time, cheapest first:

* **MediaInfo** (``mediainfo --Output=JSON <file>``, one call) is the preferred source: it answers named fields -
  container/format, video codec, width/height, ``BitDepth``, ``ChromaSubsampling``, frame rate, ``FrameCount``,
  duration, ``colour_primaries``/``transfer_characteristics``/``matrix_coefficients``/``colour_range`` - plus the audio
  and subtitle tracks. The window's caption names it and its version (``probe: mediainfo 26.10``).
* **one ``ffmpeg -i``** (no decoding) is the fallback for builds without MediaInfo: the container, the codec, the frame
  size, the pixel format, the bit depth the pixel format implies, the frame rate, the duration, whatever colour metadata
  the file declares, and the audio/subtitle stream lines. The caption reads ``probe: ffmpeg``.
* **the app's own backend probe** (``python rve-backend.py --print_video_info``, the exact process `VideoLoader` runs)
  fills what the header cannot say, above all the **frame count** - ffmpeg's input header never carries one. It runs
  only when the bundled interpreter (``python/python/bin/python3``) and the backend script are installed, which they
  are not in a source checkout.

Anything none of them answered is an em dash with the reason printed next to it: "the file does not declare a colour
range" is a measurement, "assume tv" is not. Every probe has a hard timeout, so a dead network input cannot hang the
window forever; there is no background thread because the window is a one-shot read of a local file.

The RVE tab is not probed: it is the queue item's own record (``RenderOptions`` + ``JobState``) and, when one exists
next to the output, the run report (``<output>.report.json``). The report is the only honest source for "what this run
cost" and "which settings it actually used" - the live settings can differ from what a past run used, so without a
report those rows are em dashes, not the live values. The tab's **Result** section names the run's outcome (Finished /
Failed / Rendering / Not yet started) and, for a failure, the reason the queue recorded: a finished item is the one a
user opens this window to inspect, and the timestamps alone never said whether the run succeeded. The Files section
reads the output's size from disk the same way it reads the input's, so a run that has not written its output yet
shows an em dash with a reason rather than ``0 B``.
"""

from __future__ import annotations

import errno
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field

from PySide6.QtWidgets import QVBoxLayout, QWidget

from .toolkit_dialogs import TOOLKIT_AVAILABLE, ToolkitDialog, _action_row, _tk_button

if TOOLKIT_AVAILABLE:  # pragma: no branch - the toolkit is what the window is made of
    from sli_ui_toolkit.ui.widgets.text.engine.markdown import (
        HeadingBlock,
        InlineKind,
        InlineSpan,
        ParagraphBlock,
        TableBlock,
    )
    from sli_ui_toolkit.widgets import OverlayScrollArea, Text, TopTabHost

#: A measurement that did not happen. One character, shared by the model, the document and the tests.
EM_DASH = "—"

#: `mediainfo --Output=JSON` reads the file and prints one JSON document; a URL that does not answer is cut off.
MEDIAINFO_TIMEOUT_S = 20.0
#: `ffmpeg -i` opens the input and prints its header; it never decodes a frame.
FFMPEG_TIMEOUT_S = 20.0
#: The backend probe prints one field per line; on an installed app it measured ~0.16 s per file (`video_scan`).
BACKEND_TIMEOUT_S = 30.0

#: Every field the Video group reports, in display order. A field the probes did not answer is an em dash.
VIDEO_FIELDS = (
    "codec",
    "width",
    "height",
    "pixel_format",
    "chroma_subsampling",
    "bit_depth",
    "fps",
    "frames",
    "duration",
    "color_space",
    "color_primaries",
    "color_transfer",
    "color_range",
)

#: Why a field is empty when the probe itself has nothing more specific to say. These strings are shown to the user
#: next to the em dash, so they state what was looked at, not what might be assumed. Deliberately source-neutral: the
#: caption says which probe answered, so a reason must not claim ffmpeg read a file MediaInfo read.
DEFAULT_REASONS = {
    "container": "the file does not declare a container format",
    "codec": "no probe reported a video codec for this file",
    "resolution": "no probe reported the frame size of this file",
    "pixel_format": "no probe reported a pixel format for this file",
    "chroma_subsampling": "the file does not declare chroma subsampling",
    "bit_depth": "no probe reported a bit depth for this file",
    "fps": "no probe reported a frame rate for this file",
    "frames": "no probe reported a frame count for this file",
    "duration": "no probe reported a duration for this file",
    "color_space": "the file does not declare a colour space",
    "color_primaries": "the file does not declare colour primaries",
    "color_transfer": "the file does not declare a transfer characteristic",
    "color_range": "the file does not declare a colour range",
}

#: What MediaInfo leaves unanswered on purpose: it names the chroma layout and bit depth, never ffmpeg's pixel format.
MEDIAINFO_PIXEL_FORMAT_REASON = "mediainfo names the chroma layout and bit depth, not ffmpeg's pixel-format name"

#: ffmpeg prints these inside the pixel format's parentheses when the stream leaves the field unset. They are not
#: measurements, so they must not become one (`bt2020nc/unknown/unknown` is a colour space with two empty fields).
_UNSET_COLOUR_VALUES = frozenset({"unknown", "unspecified", "reserved"})

#: Pixel formats whose name carries no bit-depth suffix. Anything else that also carries no suffix is reported as an
#: em dash rather than assumed to be 8-bit.
_EIGHT_BIT_PIXEL_FORMATS = frozenset({
    "yuv420p", "yuv422p", "yuv444p", "yuv410p", "yuv411p", "yuvj420p", "yuvj422p", "yuvj444p",
    "yuva420p", "yuva422p", "yuva444p", "nv12", "nv21", "nv16", "nv24", "uyvy422", "yuyv422",
    "rgb24", "bgr24", "rgba", "bgra", "argb", "abgr", "rgb0", "bgr0", "gray", "gbrp", "gbrap",
})

_BIT_DEPTHS = (9, 10, 12, 14, 16)

#: ffmpeg's sample-format token -> the bit depth it states (an audio stream's declared sample format, not a guess).
_AUDIO_SAMPLE_FORMATS = {
    "u8": 8, "u8p": 8, "s16": 16, "s16p": 16, "s32": 32, "s32p": 32,
    "flt": 32, "fltp": 32, "dbl": 64, "dblp": 64,
}


# --------------------------------------------------------------------------------------------------------------------
# The model: tabs of sections of labelled rows, and what a row with no value says instead
# --------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PropertyRow:
    """One labelled line. ``value`` empty means "the probe could not answer", and ``reason`` says why."""

    label: str
    value: str = ""
    reason: str = ""

    @property
    def display_value(self) -> str:
        return self.value or EM_DASH


@dataclass(frozen=True)
class PropertySection:
    title: str
    rows: tuple[PropertyRow, ...]


@dataclass(frozen=True)
class PropertiesTab:
    """One tab's document: a title for the tab strip and its grouped rows."""

    title: str
    sections: tuple[PropertySection, ...]


@dataclass(frozen=True)
class QueueProperties:
    """Everything the window shows: the file's name, which row it describes, which probe answered, and the tabs."""

    name: str
    subtitle: str
    tabs: tuple[PropertiesTab, ...]
    probe_caption: str = ""

    def tab(self, title: str) -> PropertiesTab:
        return next(tab for tab in self.tabs if tab.title == title)

    def as_plain_text(self) -> str:
        """The document as text - the Copy all payload, both tabs."""
        lines = [self.name, "", self.subtitle]
        if self.probe_caption:
            lines.append(self.probe_caption)
        for tab in self.tabs:
            lines.extend(["", f"== {tab.title} =="])
            for section in tab.sections:
                lines.extend(["", section.title])
                for row in section.rows:
                    value = row.display_value
                    if not row.value and row.reason:
                        value = f"{value} ({row.reason})"
                    lines.append(f"{row.label}: {value}")
        return "\n".join(lines)


@dataclass(frozen=True)
class ProbeResult:
    """What a probe measured, and why each field it could not measure is empty.

    ``fields`` holds the measured values by :data:`VIDEO_FIELDS` key; ``reasons`` holds the missing ones' reasons;
    ``audio``/``text`` are one dict per audio/subtitle stream (keyed by :data:`AUDIO_FIELDS`/:data:`TEXT_FIELDS`);
    ``error`` is set when every probe failed; ``source`` names the probe that answered, version included, for the
    caption and for the reader of a report.
    """

    fields: dict = field(default_factory=dict)
    reasons: dict = field(default_factory=dict)
    error: str = ""
    source: str = ""
    audio: tuple = ()
    text: tuple = ()


# --------------------------------------------------------------------------------------------------------------------
# Formatting. One place per value shape, so the window and the Copy all text never disagree.
# --------------------------------------------------------------------------------------------------------------------
def format_bytes(size: int) -> str:
    """``1536`` -> ``1.5 KB``. Binary units, one decimal above a kilobyte."""
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{int(value)} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{int(size)} B"  # pragma: no cover - the loop always returns


def format_duration(seconds: float) -> str:
    """``10.01`` -> ``00:00:10.010``: hours, minutes, seconds and milliseconds, never a rounded minute count."""
    total_ms = int(round(float(seconds) * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def format_fps(fps: float) -> str:
    return f"{float(fps):g} fps"


def format_resolution(width: int, height: int) -> str:
    return f"{int(width)} x {int(height)}"


def format_bit_depth(depth: int) -> str:
    return f"{int(depth)} bit"


def format_hz(rate: int) -> str:
    return f"{int(rate)} Hz"


def format_bit_rate(bits_per_second) -> str:
    """A stream's bit rate in a unit a human reads. MediaInfo reports bits/s, ffmpeg prints kb/s."""
    try:
        rate = float(bits_per_second)
    except (TypeError, ValueError):
        return str(bits_per_second)
    if rate >= 1_000_000:
        return f"{rate / 1_000_000:.1f} Mb/s"
    if rate >= 1000:
        return f"{rate / 1000:.0f} kb/s"
    return f"{rate:.0f} b/s"


def format_ms_per_frame(milliseconds: float) -> str:
    return f"{float(milliseconds):g} ms/frame"


def format_pixels(value) -> str:
    """A tile size, in pixels when it is a number (the entry stores it as text)."""
    text = str(value).strip()
    try:
        int(text)
    except ValueError:
        return text
    return f"{text} px"


def format_scale(value) -> str:
    return f"×{int(value)}"


def format_boolean(value) -> str:
    """Booleans read Yes/No; a value that is not a boolean is not one of them."""
    return "Yes" if bool(value) else "No"


def format_wall_clock(timestamp) -> str:
    """A wall-clock timestamp as a human reads it; empty when there is no timestamp to read."""
    if timestamp is None:
        return ""
    try:
        from datetime import datetime

        return datetime.fromtimestamp(float(timestamp)).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError, OSError):
        return ""


def bit_depth_from_pixel_format(name: str) -> int | None:
    """The bit depth a pixel format name states, or ``None`` when the name does not state one.

    ``yuv420p10le`` -> 10, ``p010le`` -> 10, ``yuv420p`` -> 8, ``rgb48le`` -> ``None`` (48 is three channels, not a
    depth). A name this function does not know is not guessed as 8-bit.
    """
    if not name:
        return None
    bare = re.sub(r"(?:le|be)$", "", str(name))
    match = re.search(r"(\d{2})$", bare)
    if match and int(match.group(1)) in _BIT_DEPTHS:
        return int(match.group(1))
    if bare in _EIGHT_BIT_PIXEL_FORMATS:
        return 8
    return None


# --------------------------------------------------------------------------------------------------------------------
# Probe 1 (preferred): MediaInfo's JSON - named fields and every stream in one call
# --------------------------------------------------------------------------------------------------------------------
def resolve_mediainfo(explicit: str | None = None) -> str | None:
    """The MediaInfo binary to probe with: the caller's, else the one on ``PATH``.

    MediaInfo ships no bundled copy in RVE's release runtime (unlike ffmpeg), so ``PATH`` is the only other place.
    """
    if explicit:
        return explicit if os.path.isfile(explicit) and os.access(explicit, os.X_OK) else None
    return shutil.which("mediainfo")


def mediainfo_version(document, binary: str | None = None, *, timeout: float = 5.0) -> str:
    """The MediaInfoLib version, from the JSON's ``creatingLibrary`` or, failing that, ``mediainfo --Version``."""
    if isinstance(document, dict):
        holders = [document, (document.get("media") or {}) if isinstance(document.get("media"), dict) else {}]
        for holder in holders:
            library = holder.get("creatingLibrary") if isinstance(holder, dict) else None
            if isinstance(library, dict) and library.get("version"):
                return str(library["version"])
    if not binary:
        return ""
    try:
        done = subprocess.run(
            [binary, "--Version"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    match = re.search(r"v?(\d+(?:\.\d+)+)", done.stdout or "")
    return match.group(1) if match else ""


def read_mediainfo(
    path: str, mediainfo_path: str | None = None, *, timeout: float = MEDIAINFO_TIMEOUT_S
) -> tuple:
    """``(document, version, reason)`` from one ``mediainfo --Output=JSON``.

    ``(None, "", why)`` when no binary ran or the file did not answer with a media record; the caller then falls back
    to ffmpeg. JSON, not the default text report, because named fields are what the rows are.
    """
    binary = resolve_mediainfo(mediainfo_path)
    if binary is None:
        asked = mediainfo_path or "mediainfo on PATH"
        return None, "", f"no mediainfo binary to probe with ({asked})"
    try:
        done = subprocess.run(
            [binary, "--Output=JSON", str(path)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None, "", f"mediainfo did not finish reading this file within {timeout:.0f} s"
    except (OSError, subprocess.SubprocessError) as error:
        return None, "", f"mediainfo could not be started: {error}"
    text = done.stdout or ""
    try:
        document = json.loads(text)
    except ValueError:
        detail = next((line.strip() for line in (done.stderr or "").splitlines() if line.strip()), "")
        return None, "", f"mediainfo printed no JSON for this file ({detail or 'no reason printed'})"
    if not isinstance(document, dict) or not document.get("media"):
        return None, "", "mediainfo printed no media record for this file"
    return document, mediainfo_version(document, binary), ""


def _mediainfo_tracks(document) -> list:
    tracks = ((document or {}).get("media") or {}).get("track") or ()
    if isinstance(tracks, dict):  # a one-track file: MediaInfo prints an object, not a list
        tracks = (tracks,)
    return [track for track in tracks if isinstance(track, dict)]


def _number(value) -> float | None:
    """A MediaInfo number, or ``None`` when the field is absent, empty or not a number."""
    if value in (None, ""):
        return None
    try:
        number = float(str(value).strip())
    except ValueError:
        return None
    return number if number > 0 else None


def _whole(value) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def _declared(value) -> str:
    """A declared text field, or ``""`` when the file left it unset (MediaInfo's own "unknown" spellings)."""
    text = str(value or "").strip()
    if not text or text.lower() in _UNSET_COLOUR_VALUES or text.lower() == "none":
        return ""
    return text


def _mediainfo_video_fields(track: dict) -> dict:
    fields: dict = {}
    # MediaInfo names the chroma layout and bit depth, never ffmpeg's pixel-format name: `pixel_format` is filled by
    # the ffmpeg path only, and the window states that reason when MediaInfo answered.
    for key, source in (
        ("codec", "Format"),
        ("chroma_subsampling", "ChromaSubsampling"),
        ("color_space", "matrix_coefficients"),
        ("color_primaries", "colour_primaries"),
        ("color_transfer", "transfer_characteristics"),
        ("color_range", "colour_range"),
    ):
        value = _declared(track.get(source))
        if value:
            fields[key] = value
    width, height = _whole(track.get("Width")), _whole(track.get("Height"))
    if width and height:
        fields["width"], fields["height"] = width, height
    for key, source in (("bit_depth", "BitDepth"), ("frames", "FrameCount")):
        number = _whole(track.get(source))
        if number is not None:
            fields[key] = number
    fps = _number(track.get("FrameRate"))
    if fps is not None:
        fields["fps"] = fps
    duration = _number(track.get("Duration"))
    if duration is not None:
        fields["duration"] = duration
    return fields


def _mediainfo_audio(track: dict) -> dict:
    stream: dict = {}
    codec = _declared(track.get("Format"))
    if codec:
        stream["codec"] = codec
    for key, source in (("channels", "Channels"), ("sample_rate", "SamplingRate"), ("bit_depth", "BitDepth")):
        number = _whole(track.get(source))
        if number is not None:
            stream[key] = number
    for key, source in (("language", "Language"), ("bit_rate", "BitRate")):
        value = _declared(track.get(source))
        if value:
            stream[key] = value
    duration = _number(track.get("Duration"))
    if duration is not None:
        stream["duration"] = duration
    return stream


def _mediainfo_text(track: dict) -> dict:
    stream: dict = {}
    codec = _declared(track.get("Format"))
    if codec:
        stream["codec"] = codec
    language = _declared(track.get("Language"))
    if language:
        stream["language"] = language
    duration = _number(track.get("Duration"))
    if duration is not None:
        stream["duration"] = duration
    bit_rate = _declared(track.get("BitRate"))
    if bit_rate:
        stream["bit_rate"] = bit_rate
    return stream


def parse_mediainfo_json(document) -> tuple:
    """``(video_fields, audio_streams, text_streams)`` from one MediaInfo JSON document.

    Only what the document declares is returned: no key means MediaInfo printed nothing for it, and the window
    renders that as an em dash with a reason. ``ColorSpace`` ("YUV") is the colour MODEL, not the matrix, so the
    colour-space row is filled from ``matrix_coefficients`` only.
    """
    fields: dict = {}
    tracks = _mediainfo_tracks(document)
    if not tracks:
        return fields, (), ()

    general = next((track for track in tracks if track.get("@type") == "General"), None)
    if general:
        container = _declared(general.get("Format"))
        if container:
            fields["container"] = container
        duration = _number(general.get("Duration"))
        if duration is not None:
            fields["duration"] = duration

    video = next(
        (
            track for track in tracks
            if track.get("@type") == "Video" and _whole(track.get("Width")) and _whole(track.get("Height"))
        ),
        None,
    ) or next((track for track in tracks if track.get("@type") == "Video"), None)
    if video:
        video_fields = _mediainfo_video_fields(video)
        # The container's duration is the file's; the video stream's is the fallback when General has none.
        for key, value in video_fields.items():
            fields.setdefault(key, value)
        if "duration" not in fields and "duration" in video_fields:
            fields["duration"] = video_fields["duration"]

    audio = tuple(_mediainfo_audio(track) for track in tracks if track.get("@type") == "Audio")
    text = tuple(_mediainfo_text(track) for track in tracks if track.get("@type") == "Text")
    return fields, audio, text


# --------------------------------------------------------------------------------------------------------------------
# Probe 2 (fallback): ffmpeg's input header (one `-i`, no decode)
# --------------------------------------------------------------------------------------------------------------------
_INPUT_RE = re.compile(r"^Input #0,\s*(.+?),\s*from\s", re.MULTILINE)
_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_VIDEO_STREAM_RE = re.compile(r"Stream #\d+:\d+.*?: Video: (.*)$", re.MULTILINE)
_AUDIO_STREAM_LINE_RE = re.compile(r"(Stream #\d+:\d+[^\n]*?): Audio: (.*)$", re.MULTILINE)
_TEXT_STREAM_LINE_RE = re.compile(r"(Stream #\d+:\d+[^\n]*?): Subtitle: (.*)$", re.MULTILINE)
_CODEC_RE = re.compile(r"^([^\s,(]+)")
_FRAME_SIZE_RE = re.compile(r"(\d{2,5})x(\d{2,5})")
_FPS_RE = re.compile(r"([\d.]+)\s*fps")
_SAMPLE_RATE_RE = re.compile(r"(\d+)\s*Hz")
_BIT_RATE_RE = re.compile(r"(\d+)\s*kb/s")
_CHANNELS_RE = re.compile(r"\b(mono|stereo|quad|5\.1|6\.1|7\.1)\b|\b(\d+)\s*channels?\b")
_SAMPLE_FORMAT_RE = re.compile(r"\b(fltp|flt|dblp|dbl|s16p|s16|s32p|s32|u8p|u8)\b")
_LANGUAGE_RE = re.compile(r"\(([a-z]{2,3})\)")
#: The pixel format is the first `name(...)` field after the codec's own parentheses: `h264 (High) (avc1 / 0x…),
#: yuv420p(tv, bt709, progressive), 320x240 …`.
_PIXEL_FORMAT_RE = re.compile(r",\s*([a-z][a-z0-9_]*)\s*(?:\(([^)]*)\))?\s*,")


def parse_input_header(text: str) -> dict:
    """The fields one ``ffmpeg -i`` header states, keyed by :data:`VIDEO_FIELDS`, or ``{}`` when it states none.

    Only what is printed is returned: no key means ffmpeg printed nothing for it, and the window renders that as an
    em dash with a reason. ``unknown`` inside the colour triple is the same "nothing" (`bt2020nc/unknown/unknown` is
    a colour space with no declared primaries or transfer, not the primaries named "unknown").
    """
    fields: dict = {}
    if not text:
        return fields

    container = _INPUT_RE.search(text)
    if container:
        fields["container"] = container.group(1).strip()

    duration = _DURATION_RE.search(text)
    if duration:
        hours, minutes, seconds = duration.groups()
        seconds_total = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        if seconds_total > 0:
            fields["duration"] = seconds_total

    for stream in _VIDEO_STREAM_RE.findall(text):
        parsed = _parse_video_stream(stream)
        # The first video stream that states a frame size is the picture: cover art and thumbnails come later and are
        # usually the smaller stream, and a stream without a size cannot answer the resolution row anyway.
        if parsed.get("width") and parsed.get("height"):
            fields.update(parsed)
            break
    return fields


def _parse_video_stream(line: str) -> dict:
    fields: dict = {}
    codec = _CODEC_RE.match(line.strip())
    if codec:
        fields["codec"] = codec.group(1)
    size = _FRAME_SIZE_RE.search(line)
    if size:
        fields["width"] = int(size.group(1))
        fields["height"] = int(size.group(2))
    pixel = _PIXEL_FORMAT_RE.search(line)
    if pixel:
        fields["pixel_format"] = pixel.group(1)
        depth = bit_depth_from_pixel_format(pixel.group(1))
        if depth is not None:
            fields["bit_depth"] = depth
        fields.update(_parse_pixel_details(pixel.group(2)))
    fps = _FPS_RE.search(line)
    if fps:
        fields["fps"] = float(fps.group(1))
    return fields


def _parse_pixel_details(details: str | None) -> dict:
    """Range and colour triple from the pixel format's parentheses.

    ffmpeg prints ``(tv, bt709/bt709/bt709, progressive)`` or ``(progressive)``; the first comma-separated part is the
    range when it is ``tv``/``pc``, and the part carrying slashes is space/primaries/transfer. Everything else
    (``progressive``, ``top first``, ``default``) is scan order, which this window does not claim to measure.
    """
    fields: dict = {}
    if not details:
        return fields
    parts = [part.strip() for part in details.split(",")]
    if parts and parts[0] in ("tv", "pc"):
        fields["color_range"] = parts[0]
        parts = parts[1:]
    colours = next((part for part in parts if "/" in part), None)
    if colours:
        space, primaries, transfer = (colours.split("/") + ["", "", ""])[:3]
        for key, value in (
            ("color_space", space),
            ("color_primaries", primaries),
            ("color_transfer", transfer),
        ):
            if value and value.lower() not in _UNSET_COLOUR_VALUES:
                fields[key] = value
    return fields


def parse_input_streams(text: str) -> tuple:
    """``(audio_streams, text_streams)`` from the ``Audio:``/``Subtitle:`` lines of one ffmpeg header."""
    if not text:
        return (), ()
    audio = tuple(_parse_audio_stream(line, body) for line, body in _AUDIO_STREAM_LINE_RE.findall(text))
    text_streams = tuple(_parse_text_stream(line, body) for line, body in _TEXT_STREAM_LINE_RE.findall(text))
    return audio, text_streams


def _parse_audio_stream(line: str, body: str) -> dict:
    stream: dict = {}
    codec = _CODEC_RE.match(body.strip())
    if codec:
        stream["codec"] = codec.group(1)
    channels = _CHANNELS_RE.search(body)
    if channels:
        named = channels.group(1)
        stream["channels"] = named if named else int(channels.group(2))
    rate = _SAMPLE_RATE_RE.search(body)
    if rate:
        stream["sample_rate"] = int(rate.group(1))
    bit_rate = _BIT_RATE_RE.search(body)
    if bit_rate:
        stream["bit_rate"] = f"{int(bit_rate.group(1))} kb/s"
    sample_format = _SAMPLE_FORMAT_RE.search(body)
    if sample_format:
        depth = _AUDIO_SAMPLE_FORMATS.get(sample_format.group(1))
        if depth is not None:
            stream["bit_depth"] = depth
    language = _LANGUAGE_RE.search(line)
    if language:
        stream["language"] = language.group(1)
    return stream


def _parse_text_stream(line: str, body: str) -> dict:
    stream: dict = {}
    codec = _CODEC_RE.match(body.strip())
    if codec:
        stream["codec"] = codec.group(1)
    bit_rate = _BIT_RATE_RE.search(body)
    if bit_rate:
        stream["bit_rate"] = f"{int(bit_rate.group(1))} kb/s"
    language = _LANGUAGE_RE.search(line)
    if language:
        stream["language"] = language.group(1)
    return stream


def resolve_ffmpeg(explicit: str | None = None) -> str | None:
    """The ffmpeg binary to probe with: the caller's, else the app's bundled one, else the one on ``PATH``.

    The last fallback exists because a source checkout has no ``bin/ffmpeg`` (the bundled runtime is a release
    artifact) while the development machine has ffmpeg installed. The window reports the binary it used in the log
    line; a binary that is not executable is the same as no binary.
    """
    if explicit:
        return explicit if os.path.isfile(explicit) and os.access(explicit, os.X_OK) else None
    try:
        from ...constants import FFMPEG_PATH

        if FFMPEG_PATH and os.path.isfile(FFMPEG_PATH) and os.access(FFMPEG_PATH, os.X_OK):
            return FFMPEG_PATH
    except ImportError:
        # No app constants on this path (the toolkit module is importable without the app); PATH decides.
        pass
    return shutil.which("ffmpeg")


def read_input_header(path: str, ffmpeg_path: str | None = None, *, timeout: float = FFMPEG_TIMEOUT_S):
    """``(header_text, reason)`` from one ``ffmpeg -i``. ``(None, why)`` when no binary ran or the file did not open."""
    binary = resolve_ffmpeg(ffmpeg_path)
    if binary is None:
        asked = ffmpeg_path or "the app's bin/ffmpeg or PATH"
        return None, f"no ffmpeg binary to probe with ({asked})"
    try:
        done = subprocess.run(
            [binary, "-hide_banner", "-nostdin", "-i", str(path)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None, f"ffmpeg did not finish reading the header within {timeout:.0f} s"
    except (OSError, subprocess.SubprocessError) as error:
        return None, f"ffmpeg could not be started: {error}"
    text = done.stdout or ""
    # ffmpeg exits 1 for every `-i` read ("at least one output file must be specified"), so the exit code says
    # nothing; a file it could not open is the one case where the header never existed.
    if "Error opening input file" in text:
        detail = next((line.strip() for line in text.splitlines() if "Error opening input" in line), "")
        return None, f"ffmpeg could not open this file ({detail or 'no reason printed'})"
    return text, ""


# --------------------------------------------------------------------------------------------------------------------
# Probe 3: the app's own backend probe (the exact process `VideoLoader` runs), for what the header cannot answer
# --------------------------------------------------------------------------------------------------------------------
_BACKEND_KEYS = {
    "Video Codec": "codec",
    "Total Frames": "frames",
    "FPS": "fps",
    "Pixel Format": "pixel_format",
    "Bit Depth": "bit_depth",
    "Color Space": "color_space",
    "Color Primaries": "color_primaries",
    "Color Transfer": "color_transfer",
}


def parse_backend_info(text: str) -> dict:
    """The fields a ``--print_video_info`` run printed, keyed by :data:`VIDEO_FIELDS`.

    The backend prints one ``Label: value`` line per field and its getters answer ``""``/``0``/``None`` when they could
    not measure; all three read as "not measured" here. ``Resolution`` arrives as ``1920x1080`` and
    ``Duration`` as ``12.5 seconds``.
    """
    fields: dict = {}
    if not text:
        return fields
    for line in text.splitlines():
        label, separator, raw = line.partition(":")
        if not separator:
            continue
        label = label.strip()
        value = raw.strip()
        if label == "Resolution":
            match = re.fullmatch(r"(\d{2,5})\s*x\s*(\d{2,5})", value)
            if match:
                fields["width"] = int(match.group(1))
                fields["height"] = int(match.group(2))
            continue
        if label == "Duration":
            number = re.match(r"[\d.]+", value)
            if number and float(number.group(0)) > 0:
                fields["duration"] = float(number.group(0))
            continue
        key = _BACKEND_KEYS.get(label)
        if key is None:
            continue
        if key in ("fps", "frames", "bit_depth"):
            number = re.match(r"\d+", value)
            if not number or int(number.group(0)) <= 0:
                continue
            fields[key] = float(number.group(0)) if key == "fps" else int(number.group(0))
            continue
        if value and value.lower() not in _UNSET_COLOUR_VALUES and value.lower() != "none":
            fields[key] = value
    return fields


def read_backend_info(path: str, *, timeout: float = BACKEND_TIMEOUT_S) -> dict:
    """The app's own probe for one file, or ``{}`` when its bundled runtime is not installed or the probe failed.

    This is ``VideoLoader.loadVideo``'s process with the same options; it is what knows the frame count. It is started
    only when ``python/python/bin/python3`` and the backend script exist (a release install), so a source checkout pays
    nothing and the window says the frame count is not available instead of copying it from a container header.
    """
    try:
        from ...constants import BACKEND_PATH, FFMPEG_PATH, PYTHON_EXECUTABLE_PATH
    except ImportError:
        # No app constants to resolve the bundled runtime with: the window reports the frame count as unmeasured.
        return {}
    script = os.path.join(BACKEND_PATH, "rve-backend.py")
    if not (os.path.isfile(PYTHON_EXECUTABLE_PATH) and os.path.isfile(script)):
        return {}
    command = [
        PYTHON_EXECUTABLE_PATH,
        "-W",
        "ignore",
        script,
        "--print_video_info",
        str(path),
        "--ffmpeg_path",
        f"{FFMPEG_PATH}",
    ]
    try:
        done = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    return parse_backend_info(done.stdout or done.stderr or "")


# --------------------------------------------------------------------------------------------------------------------
# The probe order: MediaInfo, then ffmpeg + the backend; the caption names the winner
# --------------------------------------------------------------------------------------------------------------------
def _probe_result(fields: dict, reasons: dict, error: str, source: str, audio=(), text=()) -> ProbeResult:
    """Normalize a probe's answer: empty values are not measurements, and every gap carries a reason."""
    fields = dict(fields or {})
    reasons = dict(reasons or {})
    for key in VIDEO_FIELDS + ("container",):
        value = fields.get(key)
        if value in (None, "", 0, 0.0):
            fields.pop(key, None)
            reasons[key] = reasons.get(key) or error or DEFAULT_REASONS.get(key, "no probe reported this field")
    return ProbeResult(
        fields=fields,
        reasons=reasons,
        error=error,
        source=source,
        audio=tuple(stream for stream in (audio or ()) if stream),
        text=tuple(stream for stream in (text or ()) if stream),
    )


def probe_video_file(
    path: str,
    *,
    mediainfo_path: str | None = None,
    ffmpeg_path: str | None = None,
    mediainfo_reader=None,
    header_reader=None,
    backend_probe=None,
) -> ProbeResult:
    """Everything that can be measured about ``path`` right now, plus the reason for every field that cannot.

    MediaInfo is asked first and wins when it answers; ffmpeg's header and the app's backend probe are the fallback
    for builds without it. ``mediainfo_reader``/``header_reader``/``backend_probe`` are the test seams: the window
    calls this function with no arguments and the harness passes fakes, so the order and the composition are pinned
    without a media binary or a backend runtime.
    """
    mi_reader = mediainfo_reader or read_mediainfo
    document, version, _mi_reason = mi_reader(str(path), mediainfo_path)
    if document:
        fields, audio, text = parse_mediainfo_json(document)
        reasons = {}
        if "pixel_format" not in fields:
            reasons["pixel_format"] = MEDIAINFO_PIXEL_FORMAT_REASON
        return _probe_result(fields, reasons, "", f"mediainfo {version}".strip(), audio, text)

    reader = header_reader or read_input_header
    text, header_reason = reader(str(path), ffmpeg_path)
    fields = parse_input_header(text) if text else {}
    audio, text_streams = parse_input_streams(text) if text else ((), ())
    backend = dict((backend_probe or read_backend_info)(str(path)) or {})

    # The header wins where both answered (it read the file's own declaration); the backend fills the gaps, and its
    # frame count is the only answer to the frames row.
    for key, value in backend.items():
        if key in fields or value in (None, "", 0, 0.0):
            continue
        fields[key] = value

    reasons: dict = {}
    video_reason = header_reason
    if text and not fields and _VIDEO_STREAM_RE.search(text) is None:
        video_reason = "ffmpeg printed no video stream for this file"
    for key in VIDEO_FIELDS + ("container",):
        if fields.get(key) in (None, "", 0, 0.0) and video_reason:
            reasons[key] = video_reason
    if text:
        source = "ffmpeg"
    elif backend:
        source = "backend probe"
    else:
        source = "no probe answered"
    return _probe_result(fields, reasons, header_reason if not text and not backend else "", source,
                         audio, text_streams)


# --------------------------------------------------------------------------------------------------------------------
# The run report: what a finished run actually cost, beside the output
# --------------------------------------------------------------------------------------------------------------------
def run_report_path(job) -> str:
    """Where `render_command` writes the report: ``<output without extension>.report.json``."""
    output = str(getattr(job, "outputPath", "") or "")
    return f"{os.path.splitext(output)[0]}.report.json" if output else ""


def read_run_report(job) -> tuple:
    """``(document, path, reason)``: the run report beside the output, or ``(None, path, why)``."""
    path = run_report_path(job)
    if not path:
        return None, "", "the queue item carries no output path, so no run report can exist"
    if not os.path.isfile(path):
        return None, path, f"no run report beside the output ({path} is not there)"
    try:
        with open(path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, ValueError) as error:
        return None, path, f"the run report could not be read ({error})"
    if not isinstance(document, dict):
        return None, path, "the run report is not a JSON object"
    return document, path, ""


def _report_settings(report) -> dict:
    settings = (report or {}).get("settings")
    return settings if isinstance(settings, dict) else {}


def _report_phases(report) -> dict:
    phases = (report or {}).get("phases_ms_per_frame")
    return phases if isinstance(phases, dict) else {}


def _report_frames(report):
    run = (report or {}).get("run")
    if isinstance(run, dict):
        return run.get("frames")
    return None


# --------------------------------------------------------------------------------------------------------------------
# The window's model: the probe's tab and RVE's tab
# --------------------------------------------------------------------------------------------------------------------
def _display_name(path: str) -> str:
    name = os.path.basename(str(path).rstrip("/"))
    return name or str(path)


def _reason(probe: ProbeResult, key: str) -> str:
    """Why ``key`` is empty: the probe's own words, else the window's default for that field.

    A row must never render a bare em dash: "it is empty" is not an answer a user can act on.
    """
    return probe.reasons.get(key) or DEFAULT_REASONS.get(key, "no probe reported this field")


def _missing_reason(probe: ProbeResult, *keys: str) -> str:
    """The first reason any of ``keys`` carries, else the first default: a pair of fields can share one row."""
    for key in keys:
        if probe.reasons.get(key):
            return probe.reasons[key]
    for key in keys:
        if DEFAULT_REASONS.get(key):
            return DEFAULT_REASONS[key]
    return "no probe reported this field"


def _container_section(probe: ProbeResult) -> PropertySection:
    container = probe.fields.get("container", "")
    row = PropertyRow("Container", container, "" if container else _reason(probe, "container"))
    return PropertySection("File", (row,))


def _video_section(probe: ProbeResult) -> PropertySection:
    fields = probe.fields
    rows: list[PropertyRow] = []

    def add(label: str, value: str, *keys: str) -> None:
        rows.append(PropertyRow(label, value, "" if value else _missing_reason(probe, *keys)))

    add("Codec", fields.get("codec", ""), "codec")
    width, height = fields.get("width"), fields.get("height")
    resolution = format_resolution(width, height) if width and height else ""
    add("Resolution", resolution, "resolution", "width", "height")
    add("Pixel format", fields.get("pixel_format", ""), "pixel_format")
    add("Chroma subsampling", fields.get("chroma_subsampling", ""), "chroma_subsampling")
    depth = fields.get("bit_depth")
    add("Bit depth", format_bit_depth(depth) if depth else "", "bit_depth")
    fps = fields.get("fps")
    add("Frame rate", format_fps(fps) if fps else "", "fps")
    frames = fields.get("frames")
    add("Frames", str(int(frames)) if frames else "", "frames")
    duration = fields.get("duration")
    add("Duration", format_duration(duration) if duration else "", "duration")
    add("Colour space", fields.get("color_space", ""), "color_space")
    add("Colour primaries", fields.get("color_primaries", ""), "color_primaries")
    add("Transfer", fields.get("color_transfer", ""), "color_transfer")
    add("Colour range", fields.get("color_range", ""), "color_range")
    return PropertySection("Video", tuple(rows))


#: One label per audio-stream field, in display order.
_AUDIO_LABELS = {
    "codec": "Codec",
    "channels": "Channels",
    "sample_rate": "Sample rate",
    "bit_depth": "Bit depth",
    "language": "Language",
    "duration": "Duration",
    "bit_rate": "Bit rate",
}

#: One label per subtitle-stream field, in display order.
_TEXT_LABELS = {
    "codec": "Codec",
    "language": "Language",
    "duration": "Duration",
    "bit_rate": "Bit rate",
}

#: The stream field contract, in display order: the probe dicts use exactly these keys.
AUDIO_FIELDS = tuple(_AUDIO_LABELS)
TEXT_FIELDS = tuple(_TEXT_LABELS)


def _stream_value(key: str, value) -> str:
    if key == "sample_rate":
        return format_hz(value)
    if key == "bit_depth":
        return format_bit_depth(value)
    if key == "duration":
        return format_duration(value)
    if key == "bit_rate":
        return format_bit_rate(value)
    return str(value)


def _stream_section(title: str, stream: dict, labels: dict) -> PropertySection:
    rows: list[PropertyRow] = []
    for key, label in labels.items():
        value = stream.get(key)
        if value in (None, "", 0, 0.0):
            rows.append(PropertyRow(label, reason=f"this stream does not declare {label.lower()}"))
        else:
            rows.append(PropertyRow(label, _stream_value(key, value)))
    return PropertySection(title, tuple(rows))


def _audio_sections(probe: ProbeResult) -> tuple:
    if not probe.audio:
        row = PropertyRow("Streams", reason="the probe found no audio stream in this file")
        return (PropertySection("Audio", (row,)),)
    return tuple(
        _stream_section(f"Audio {index}", stream, _AUDIO_LABELS)
        for index, stream in enumerate(probe.audio, start=1)
    )


def _text_sections(probe: ProbeResult) -> tuple:
    if not probe.text:
        row = PropertyRow("Streams", reason="the probe found no subtitle stream in this file")
        return (PropertySection("Subtitles", (row,)),)
    return tuple(
        _stream_section(f"Subtitle {index}", stream, _TEXT_LABELS)
        for index, stream in enumerate(probe.text, start=1)
    )


def _media_sections(probe: ProbeResult) -> tuple:
    """The MediaInfo tab: the probe's output, grouped by file, video and every audio/subtitle stream."""
    return (_container_section(probe), _video_section(probe)) + _audio_sections(probe) + _text_sections(probe)


#: The model stages a queue item can carry, in the order the render applies them.
_MODEL_STAGES = (
    ("upscaleModelFile", "Upscale model"),
    ("interpolateModelFile", "Interpolation model"),
    ("deblurModelFile", "Deblur model"),
    ("denoiseModelFile", "Denoise model"),
    ("decompressModelFile", "Decompress model"),
    ("sceneChangeModelFile", "Scene-change model"),
)


def _models_section(job) -> PropertySection:
    """One row per model this job carries, by file NAME; a model it does not carry is omitted."""
    rows: list[PropertyRow] = []
    for attribute, label in _MODEL_STAGES:
        value = getattr(job, attribute, None)
        if value:
            rows.append(PropertyRow(label, os.path.basename(str(value).rstrip("/")) or str(value)))
            if attribute == "upscaleModelFile":
                arch = getattr(job, "upscaleModelArch", None)
                if arch:
                    rows.append(PropertyRow("Upscale model arch", str(arch)))
    if not rows:
        rows.append(PropertyRow("Models", reason="this queue item carries no model file"))
    return PropertySection("Models", tuple(rows))


def _coefficients_section(job) -> PropertySection:
    """The entry's own coefficients. A setting the entry does not carry is omitted, not defaulted."""
    rows: list[PropertyRow] = []

    scale = getattr(job, "modelScale", None)
    if scale:
        rows.append(PropertyRow("Scale", format_scale(scale)))
    override = getattr(job, "overrideUpscaleScale", None)
    if override and (not scale or int(override) != int(scale)):
        rows.append(PropertyRow("Override upscale scale", format_scale(override)))
    times = getattr(job, "interpolateTimes", None)
    if times:
        rows.append(PropertyRow("Interpolate times", format_scale(times)))

    for attribute, label in (
        ("ensemble", "Ensemble"),
        # The source spells it `dyanmicScaleOpticalFlow`; the row spells it for a human.
        ("dyanmicScaleOpticalFlow", "Dynamic scale optical flow"),
        ("sloMoMode", "Slow motion mode"),
        ("benchmarkMode", "Benchmark mode"),
        ("tilingEnabled", "Tiling"),
        ("hdrMode", "HDR mode"),
    ):
        value = getattr(job, attribute, None)
        if value is not None:
            rows.append(PropertyRow(label, format_boolean(value)))

    tilesize = getattr(job, "tilesize", None)
    if tilesize not in (None, ""):
        rows.append(PropertyRow("Tile size", format_pixels(tilesize)))
    backend = getattr(job, "backend", None)
    if backend:
        rows.append(PropertyRow("Backend", str(backend)))

    start = getattr(job, "startTime", None)
    if start is not None:
        rows.append(PropertyRow("Segment start", format_duration(start)))
    end = getattr(job, "endTime", None)
    if end is not None:
        rows.append(PropertyRow("Segment end", format_duration(end)))

    encoder = getattr(job, "encoderCommand", None)
    if encoder:
        rows.append(PropertyRow("Encoder", str(encoder).strip()))
    if not rows:
        rows.append(PropertyRow("Coefficients", reason="this queue item carries no render coefficient"))
    return PropertySection("Coefficients", tuple(rows))


def _queue_section(row: int, row_count: int) -> PropertySection:
    return PropertySection("Queue", (PropertyRow("Position", f"{int(row) + 1} of {int(row_count)}"),))


#: The words the window uses for a job's state. They are the queue's own statuses in a human's words, and the hover
#: line (`src/render_jobs.state_text`) says the same states on the row itself.
_STATE_LABELS = {
    "pending": "Not yet started",
    "rendering": "Rendering",
    "done": "Finished",
    "failed": "Failed",
}


def _result_section(job) -> PropertySection:
    """How this item's run ended: its completion state, and the failure's own reason when there is one.

    A finished item is the one a user opens the window to inspect, so the state is a row of its own - the timestamps
    section says WHEN the run ended, never WHETHER it succeeded. A failed item carries the reason the queue recorded
    (`JobState.error`, the same sentence the row's hover line appends); an item that never ran says so.
    """
    state = getattr(job, "state", None)
    rows: list[PropertyRow] = []
    if state is None:
        rows.append(PropertyRow(
            "State", reason="this queue item carries no job state, so its run has no recorded outcome"
        ))
        return PropertySection("Result", tuple(rows))

    status = str(getattr(state, "status", "") or "")
    label = _STATE_LABELS.get(status)
    if label is None:
        rows.append(PropertyRow(
            "State", reason=f"the queue recorded an unknown state ({status or 'empty'})"
        ))
    else:
        rows.append(PropertyRow("State", label))

    if status == "failed":
        error = str(getattr(state, "error", "") or "").strip()
        rows.append(PropertyRow(
            "Failure", error, "" if error else "the run failed and the queue recorded no reason"
        ))
    return PropertySection("Result", tuple(rows))


def _timestamps_section(job, report) -> PropertySection:
    """When the item entered the queue and what its run took. An item that never ran reads em dashes, not zeros."""
    state = getattr(job, "state", None)
    rows: list[PropertyRow] = []

    def add_wall(label: str, timestamp, reason: str) -> None:
        value = format_wall_clock(timestamp)
        rows.append(PropertyRow(label, value, "" if value else reason))

    add_wall("Added", getattr(state, "added_wall", None), "the queue did not record when this item was added")
    add_wall("Started", getattr(state, "started_wall", None), "this item has not started rendering")
    add_wall("Finished", getattr(state, "finished_wall", None), "this item has not finished rendering")

    started = getattr(state, "started_mono", None)
    finished = getattr(state, "finished_mono", None)
    if started is not None and finished is not None:
        rows.append(PropertyRow("Render duration", format_duration(state.duration())))
    elif started is not None:
        rows.append(PropertyRow("Render duration", f"{format_duration(state.duration())} so far"))
    else:
        rows.append(PropertyRow("Render duration", reason="this item has not started rendering"))

    frames = _report_frames(report)
    if frames in (None, "", 0):
        frame_count = getattr(job, "videoFrameCount", None)
        frames = frame_count if finished is not None and frame_count else None
    if frames in (None, "", 0):
        rows.append(PropertyRow(
            "Frames rendered",
            reason=(
                "this run report records no frame count"
                if report is not None
                else "no run report records the frame count, and this item has not finished rendering"
            ),
        ))
    else:
        rows.append(PropertyRow("Frames rendered", str(int(frames))))

    phases = _report_phases(report)
    cost = phases.get("frame")
    if cost in (None, "", 0):
        rows.append(PropertyRow(
            "Per-frame cost",
            reason=(
                "this run report records no per-frame total"
                if report is not None
                else "no run report beside the output records what a frame cost this run"
            ),
        ))
    else:
        rows.append(PropertyRow("Per-frame cost", format_ms_per_frame(cost)))

    report_path = run_report_path(job)
    if report is not None and report_path:
        rows.append(PropertyRow("Run report", report_path))
    return PropertySection("Timestamps", tuple(rows))


def _files_section(job) -> PropertySection:
    """The input's path and size, and where the render writes. The size is read now, not copied from the job."""
    path = str(getattr(job, "inputFile", "") or "")
    rows: list[PropertyRow] = [PropertyRow("Input path", path, "" if path else "this queue item carries no input file")]
    if not path:
        rows.append(PropertyRow("Input size", reason="there is no input path to read a size from"))
    else:
        try:
            stat = os.stat(path)
        except OSError as error:
            rows.append(PropertyRow("Input size", reason=f"the file could not be read: {path} ({error.strerror or error})"))
        else:
            rows.append(PropertyRow("Input size", format_bytes(stat.st_size)))
    output = getattr(job, "outputPath", None)
    rows.append(PropertyRow("Output path", str(output), "" if output else "this queue item carries no output path"))
    if not output:
        rows.append(PropertyRow("Output size", reason="there is no output path to read a size from"))
    else:
        # Read the size now, like the input's: a finished run's output is on disk, and a run that has not written it is
        # not "0 B" - it is unmeasured, with the reason saying so.
        try:
            stat = os.stat(str(output))
        except OSError as error:
            reason = (
                f"the output file is not there ({output})"
                if error.errno == errno.ENOENT
                else f"the output file could not be read: {output} ({error.strerror or error})"
            )
            rows.append(PropertyRow("Output size", reason=reason))
        else:
            rows.append(PropertyRow("Output size", format_bytes(stat.st_size)))
    return PropertySection("Files", tuple(rows))


#: The run-report rows, in display order: (row label, key under the report's `settings`, how to render it).
_REPORT_ROWS = (
    ("Temporal reuse threshold", "temporal_reuse_threshold_effective", str),
    ("Max chain", "temporal_reuse_max_chain", str),
    ("Statistic", "temporal_reuse_statistic", str),
    ("Quality preset", "quality_preset", str),
    ("int8 fit", "int8_fit", str),
)


def _phases_text(phases: dict) -> str:
    parts = [f"{key} {value:g}" for key, value in phases.items() if isinstance(value, (int, float))]
    return " · ".join(parts)


def _report_section(report, report_path: str, reason: str) -> PropertySection:
    """What the run report says a finished run used and cost - labelled with the report's own file name."""
    title = f"From the run report — {os.path.basename(report_path)}" if report is not None else "From the run report"
    rows: list[PropertyRow] = []
    for label, key, render in _REPORT_ROWS:
        value = (_report_settings(report).get(key) if report is not None else None)
        if value in (None, ""):
            rows.append(PropertyRow(label, reason=reason or f"this run report does not record {label.lower()}"))
        else:
            rows.append(PropertyRow(label, render(value)))
    phases = _report_phases(report) if report is not None else {}
    phases_text = _phases_text(phases)
    rows.append(PropertyRow(
        "Phases (ms/frame)",
        phases_text,
        "" if phases_text else reason or "this run report records no per-phase cost",
    ))
    return PropertySection(title, tuple(rows))


def _rve_sections(job, row: int, row_count: int, report, report_path: str, report_reason: str) -> tuple:
    """The RVE tab: the queue's own record - position, outcome, timestamps, files, models, coefficients, the report."""
    return (
        _queue_section(row, row_count),
        _result_section(job),
        _timestamps_section(job, report),
        _files_section(job),
        _models_section(job),
        _coefficients_section(job),
        _report_section(report, report_path, report_reason),
    )


def build_queue_properties(
    job,
    *,
    row: int = 0,
    row_count: int = 1,
    selected_count: int = 1,
    probe=None,
) -> QueueProperties:
    """The window's content for one queue item: the probe's tab and RVE's own tab.

    ``row``/``row_count`` are the item's place in the queue, ``selected_count`` how many rows were selected when the
    menu was asked: the window always describes the FIRST selected row, so it says which one that is and how many
    others the selection held. The probe and the run report are read once, here, before the window exists.
    """
    path = str(getattr(job, "inputFile", "") or "")
    result = (probe or probe_video_file)(path)
    if not isinstance(result, ProbeResult):  # a probe that answers a plain dict is a probe; do not crash the window
        result = ProbeResult(fields=dict(result or {}))
    report, report_path, report_reason = read_run_report(job)

    subtitle = f"Render Queue row {int(row) + 1} of {int(row_count)}"
    if int(selected_count) > 1:
        subtitle += f" · {int(selected_count)} rows selected — showing row {int(row) + 1}"

    return QueueProperties(
        name=_display_name(path) if path else "(no file)",
        subtitle=subtitle,
        probe_caption=f"probe: {result.source or 'no probe answered'}",
        tabs=(
            PropertiesTab("MediaInfo", _media_sections(result)),
            PropertiesTab("RVE", _rve_sections(job, row, row_count, report, report_path, report_reason)),
        ),
    )


# --------------------------------------------------------------------------------------------------------------------
# Rendering: typed document blocks, the way BlinkTable's properties page renders (`HeadingBlock` + `TableBlock`)
# --------------------------------------------------------------------------------------------------------------------
def build_property_blocks(properties: QueueProperties, tab: PropertiesTab) -> tuple:
    """One tab's document: the file's name, which row this is, which probe answered, then a heading and table per group."""
    if not TOOLKIT_AVAILABLE:  # pragma: no cover - the dialog cannot exist without the toolkit
        raise RuntimeError("the properties document needs sli-ui-toolkit")
    blocks: list = [
        HeadingBlock(level=2, text=properties.name),
        ParagraphBlock(spans=(InlineSpan(InlineKind.ITALIC, properties.subtitle),)),
    ]
    if properties.probe_caption:
        blocks.append(ParagraphBlock(spans=(InlineSpan(InlineKind.ITALIC, properties.probe_caption),)))
    for section in tab.sections:
        blocks.append(HeadingBlock(level=3, text=section.title))
        blocks.append(TableBlock(rows=tuple(_row_cells(row) for row in section.rows)))
    return tuple(blocks)


def _row_cells(row: PropertyRow):
    """``(label, value)`` cells for one row. A missing value carries its reason beside the dash, in italics."""
    value_spans = [InlineSpan(InlineKind.TEXT, row.display_value)]
    if not row.value and row.reason:
        value_spans.append(InlineSpan(InlineKind.ITALIC, f"  {row.reason}"))
    return ((InlineSpan(InlineKind.BOLD, row.label),), tuple(value_spans))


class QueuePropertiesDialog(ToolkitDialog):
    """The Properties window: toolkit dialog chrome, two toolkit tabs, Copy all and Close.

    `blocking` defaults to True (the app's path: the menu row opens the window modally); the harness passes False and
    reads the model and both documents without an event loop.
    """

    def __init__(
        self,
        job,
        *,
        row: int = 0,
        row_count: int = 1,
        selected_count: int = 1,
        parent=None,
        probe=None,
        blocking: bool = True,
    ) -> None:
        # The probe runs before the window exists: a failure to measure is a row with a reason, not a half-built
        # window, and the dialog's own construction must not be able to change what was measured.
        self.properties = build_queue_properties(
            job, row=row, row_count=row_count, selected_count=selected_count, probe=probe
        )
        self.job = job
        super().__init__(parent=parent, title="Properties", min_width=700, min_height=560, modal=True)

        # The toolkit's folder-tab host, never a stock QTabWidget: the app's tabs are painted by the toolkit, and a
        # QTabWidget here would be the one stock strip in a window that is otherwise the toolkit's.
        # The MAIN WINDOW's strip, not the toolkit default (owner, 2026-11-24: "the tabs are better styled like the main
        # window's"): `tab_style()` + `TabButtonFactory` are the one look the process and settings pages already share, and
        # a dialog with the stock folder tabs would be the third strip look in one app - the exact drift
        # `src/ui/widgets/tab_strip.py` exists to prevent.
        from ..pages.proc_page import TabButtonFactory
        from ..widgets.tab_strip import TAB_HEIGHT, tab_style

        self.tabs = TopTabHost(
            self,
            tab_height=TAB_HEIGHT,
            pane_radius=0,
            pane_inset=(0, 0, 0, 0),
            chrome_inset=0,
            pane_fill_token=None,
            tab_factory=TabButtonFactory(),
            tab_style=tab_style(),
        )
        self.documents: dict = {}
        for tab in self.properties.tabs:
            page = QWidget(self.tabs)
            layout = QVBoxLayout(page)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            scroll = OverlayScrollArea(page, reserve_scrollbar_space=False, corner_radius=0)
            # `Text(mode="document")` is the toolkit's painted document: one surface that owns selection across all
            # the tables, which is why the rows are blocks and not one widget per line.
            document = Text(mode="document", parent=scroll)
            document.set_panel_fill(True)
            document.set_blocks(build_property_blocks(self.properties, tab))
            scroll.setWidget(document)
            layout.addWidget(scroll)
            self.tabs.addTab(page, tab.title)
            self.documents[tab.title] = document
        self.document = self.documents[self.properties.tabs[0].title]
        self.root_layout.addWidget(self.tabs, 1)

        actions = _action_row()
        self.copy_button = _tk_button("Copy all", self)
        self.copy_button.clicked.connect(self.copy_all)
        self.close_button = _tk_button("Close", self)
        self.close_button.clicked.connect(self.accept)
        actions.addWidget(self.copy_button)
        actions.addWidget(self.close_button)
        self.root_layout.addLayout(actions)

        if blocking:
            self.exec()

    def plain_text(self) -> str:
        """The document as text, both tabs (Copy all's payload, and what a test can read without a screen)."""
        return self.properties.as_plain_text()

    def copy_all(self) -> None:
        from PySide6.QtGui import QGuiApplication

        QGuiApplication.clipboard().setText(self.plain_text())
