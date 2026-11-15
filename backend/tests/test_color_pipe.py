"""Colour-pipe contracts: the pipe format's tags and the output depth.

Both cases come from measurements recorded in docs/RESEARCH-LEDGER.md C3/C4: a comparison that
crossed an unstated colour conversion cost 5 dB (C3), and the P010 pipe lost 31 dB because it
fell through to the RGB branch and got a full-range reinterpretation (C4). The first test pins
that branch, the second pins the depth the encoder is fed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import FFmpegBuffers as FB  # noqa: E402


class _Encoder:
    def __init__(self, tag: str):
        self._tag = tag

    def getPresetTag(self) -> str:
        return self._tag


def _writer(pipe: str = "p010le", high_bitdepth: bool = True, space: str = "bt709",
            rng: str = "tv"):
    """An FFmpegWrite instance without its subprocess, for the pure argument decisions."""
    writer = FB.FFmpegWrite.__new__(FB.FFmpegWrite)
    writer.high_bitdepth = high_bitdepth
    writer.hdr_mode = False
    writer.custom_encoder = None
    writer.color_space = space
    writer.color_range = rng
    writer.color_primaries = None
    writer.color_transfer = None
    writer.video_encoder = _Encoder("x265_nvenc" if pipe == "p010le" else "hevc_nvenc")
    writer.pipe_input_pixel_format = pipe
    # `__del__` closes this log; the shim never opened one
    import io
    writer.ffmpeg_log = io.StringIO()
    return writer


@pytest.mark.parametrize("pipe", ["yuv420p10le", "p010le"])
def test_the_command_format_is_the_one_that_was_packed(pipe):
    """The `-pix_fmt` on the pipe must be the packer's format, never the encoder's preference.

    Deriving it from the encoder tag meant planar samples were labelled P010 for every
    10-bit source rendered with NVENC: 12.68 dB against the reference, and nothing to see in
    the log because both are 16-bit words at 3 bytes/pixel.
    """
    writer = _writer(pipe=pipe)
    writer.pipe_input_pixel_format = pipe
    writer.video_encoder = _Encoder("x265_nvenc")     # the encoder that "wants" P010
    assert writer._pipe_input_pixel_format() == pipe


def test_a_pack_without_a_declared_format_falls_back_to_planar():
    writer = _writer(pipe="yuv420p10le")
    writer.pipe_input_pixel_format = None
    assert writer._pipe_input_pixel_format() == "yuv420p10le"


@pytest.mark.parametrize("pipe", ["yuv420p10le", "p010le"])
def test_an_already_yuv_pipe_gets_tags_on_the_input_side_and_no_filter(pipe):
    """Neither 10-bit YUV pipe may be converted, and its tags belong to the input.

    Two separate regressions live here. `-vf scale=in_range=pc:...` on a limited-range P010
    stream reinterprets the levels and applies a second matrix (ledger C4, 31 dB). And the tags
    themselves, placed after the input, make ffmpeg convert every frame to the declared space:
    measured 79.0 ms/frame against 16.6 ms/frame at 2880x2160 10-bit - a 4.8x tax for metadata.
    """
    writer = _writer(pipe=pipe)

    assert writer._color_convert_args() == [], "no swscale filter for a YUV pipe"
    tags = writer._color_tags()
    assert tags == ["-colorspace", "bt709", "-color_range", "tv"]


def test_an_rgb_pipe_still_gets_the_matrix_filter():
    args = _writer(pipe="rgb24", high_bitdepth=False)._color_convert_args()
    assert args[0] == "-vf" and "out_color_matrix=bt709" in args[1]


def test_the_tags_exist_only_when_they_were_given():
    writer = _writer(pipe="yuv420p10le")
    writer.color_space = None
    writer.color_range = None
    assert writer._color_tags() == []


def test_the_pipe_format_follows_the_packed_format_and_the_depth():
    assert _writer(pipe="p010le")._pipe_input_pixel_format() == "p010le"
    assert _writer(pipe="yuv420p10le")._pipe_input_pixel_format() == "yuv420p10le"
    # an 8-bit source ignores the 10-bit declaration entirely
    assert _writer(pipe="yuv420p10le", high_bitdepth=False)._pipe_input_pixel_format() == "rgb24"


@pytest.mark.parametrize("requested,high_bitdepth,interpolating,expected", [
    (None, True, False, "yuv420p10le"),      # a 10-bit source keeps its depth (it is free now)
    (None, False, False, "yuv420p"),
    ("yuv420p10le", True, False, "yuv420p10le"),   # opt-in, and worth ~6 dB
    ("yuv420p", True, False, "yuv420p"),     # an explicit request always wins
    ("yuv444p10le", False, False, "yuv444p10le"),
])
def test_output_depth_resolution(requested, high_bitdepth, interpolating, expected):
    assert FB.resolve_output_pixel_format(requested, high_bitdepth, interpolating) == expected
