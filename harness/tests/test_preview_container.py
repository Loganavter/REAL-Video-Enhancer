"""A preview render must be readable while it is still being written; an export must not change.

The preview surface plays the growing preview file (plan-rve-video-preview-widget-2026-11-23, "Full resolution
on both sides"). That only works if the container is demuxable before the writer closes it: mp4/mov/m4v with
`-movflags +faststart` keeps its index (`moov`) off the file until ffmpeg relocates it **at close**, so until
the render ends a reader sees no stream at all. A preview render therefore writes fragmented mp4
(`+frag_keyframe+empty_moov`: an empty index up front, one fragment per keyframe); a real export keeps exactly
the command line it has always had.

Pure argument-list tests: no ffmpeg, no GPU, no torch/cv2. `object.__new__` skips `FFmpegWrite.__init__` - the
constructor opens the ffmpeg log and spawns the encoder process - and every attribute `command()` reads is set
to an ordinary export's values instead.
"""
from __future__ import annotations

import io
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _path in (_ROOT, os.path.join(_ROOT, "backend")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.FFmpegBuffers import FFmpegWrite, resolve_movflags  # noqa: E402
from src.utils.Encoders import EncoderSettings  # noqa: E402

FRAGMENTED = "+frag_keyframe+empty_moov"
FASTSTART = "+faststart"


def _writer(output: str, preview: bool) -> FFmpegWrite:
    """The real argument builder, with an export's attributes and no process attached."""
    writer = object.__new__(FFmpegWrite)
    writer.ffmpeg_path = "ffmpeg"
    writer.outputFile = output
    writer.outputFileExtension = os.path.splitext(output)[1].lstrip(".")
    writer.inputFile = "input.mkv"
    writer.width = writer.outputWidth = 1920
    writer.height = writer.outputHeight = 1080
    writer.start_time = 0.0
    writer.end_time = 2.0
    writer.fps = 30.0
    writer.outputFPS = 30.0
    writer.interpolateFactor = 1
    writer.upscaleTimes = 1
    writer.crf = "18"
    writer.audio_bitrate = "192k"
    writer.pixelFormat = "yuv420p"
    writer.overwrite = True
    writer.custom_encoder = None
    writer.benchmark = False
    writer.slowmo_mode = False
    writer.mpv_output = False
    writer.hdr_mode = False
    writer.high_bitdepth = False
    writer.pipe_input_pixel_format = None
    writer.color_space = None
    writer.color_primaries = None
    writer.color_transfer = None
    writer.color_range = None
    writer.merge_subtitles = True
    writer.video_encoder = EncoderSettings("libx264")
    writer.audio_encoder = EncoderSettings("aac", "audio")
    writer.subtitle_encoder = EncoderSettings("srt", "subtitle")
    writer.preview = preview
    # `__init__` opens the ffmpeg log; `__del__` closes it. Nothing else in `command()` touches it.
    writer.ffmpeg_log = io.StringIO()
    return writer


def _movflags(command: list[str]) -> str | None:
    """The `-movflags` value, or None when the command carries no index flag at all."""
    if "-movflags" not in command:
        return None
    index = command.index("-movflags")
    assert index + 1 < len(command), "-movflags with no value"
    return command[index + 1]


@pytest.mark.parametrize("extension", ["mp4", "mov", "m4v", "MP4"])
def test_a_preview_mp4_is_fragmented_and_not_faststart(extension, tmp_path):
    command = _writer(str(tmp_path / f"preview.{extension}"), preview=True).command()

    flags = _movflags(command)
    assert flags is not None, "a preview mp4-family render must carry -movflags"
    assert "+frag_keyframe" in flags, flags
    assert "+empty_moov" in flags, flags
    assert "+faststart" not in flags, "faststart would move the index to the end of the file at close"


@pytest.mark.parametrize("extension", ["mp4", "mov", "m4v"])
def test_an_export_mp4_still_gets_faststart_and_no_fragmentation(extension, tmp_path):
    """The regression this change is not allowed to cause: an export's command line is untouched."""
    command = _writer(str(tmp_path / f"export.{extension}"), preview=False).command()

    assert _movflags(command) == FASTSTART


def test_a_preview_render_in_matroska_is_left_alone(tmp_path):
    """mkv (and ts) are readable as they grow; no index flag is added or removed there."""
    command = _writer(str(tmp_path / "preview.mkv"), preview=True).command()

    assert _movflags(command) is None
    assert FRAGMENTED not in command
    assert FASTSTART not in command


def test_the_preview_and_the_export_differ_only_in_the_streamability_flags(tmp_path):
    """The switch is explicit: same parameters, two entries apart - the movflags value and the preview's short GOP.

    The GOP is not cosmetic (measure-video-preview-2026-11-23): `+frag_keyframe` writes one fragment per keyframe, and a
    default 250-frame GOP would hand the tail reader new data once per ~8 s; it also makes a seek cost a GOP of decode
    (480 ms mean / 932 ms p95 measured on a 4K clip with the default).
    """
    preview = _writer(str(tmp_path / "render.mp4"), preview=True).command()
    export = _writer(str(tmp_path / "render.mp4"), preview=False).command()

    preview_at, export_at = preview.index("-movflags"), export.index("-movflags")
    assert preview_at == export_at
    assert preview[preview_at + 1] == FRAGMENTED
    assert export[export_at + 1] == FASTSTART

    # The preview carries `-g <one second of output frames>`; an export carries no GOP at all.
    gop_at = preview.index("-g")
    assert int(preview[gop_at + 1]) >= 1
    assert gop_at > preview_at, "the GOP statement follows the movflags statement"
    assert "-g" not in export

    # Remove both statements from the preview and it must equal the export exactly.
    preview_rest = preview[:gop_at] + preview[gop_at + 2:]
    preview_at = preview_rest.index("-movflags")
    preview_rest = preview_rest[:preview_at] + preview_rest[preview_at + 2:]
    export_rest = export[:export_at] + export[export_at + 2:]
    assert preview_rest == export_rest


def test_resolve_movflags_edge_cases():
    """No output file (benchmark/pipe-shaped jobs) and unknown extensions get no flag at all."""
    assert resolve_movflags(None, None, preview=True) == []
    assert resolve_movflags("null", "null", preview=False) == []
    assert resolve_movflags("out.ts", "ts", preview=True) == []
    assert resolve_movflags("out.mp4", "mp4", preview=False) == [FASTSTART]
    assert resolve_movflags("out.mp4", "mp4", preview=True) == [FRAGMENTED]
