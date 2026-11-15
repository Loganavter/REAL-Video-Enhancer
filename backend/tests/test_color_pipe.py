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
def test_an_already_yuv_pipe_gets_tags_and_no_scale_filter(pipe):
    """Neither 10-bit YUV pipe may be converted as if it were RGB.

    `-vf scale=in_range=pc:out_color_matrix=...` on a limited-range P010 stream both
    reinterprets the levels and applies a second matrix; that is the C4 regression.
    """
    args = _writer(pipe=pipe)._color_convert_args()

    assert "-vf" not in args
    assert args[0] == "-colorspace" and args[1] == "bt709"
    assert args[2] == "-color_range" and args[3] == "tv"


def test_an_rgb_pipe_does_get_the_matrix_conversion():
    """The RGB pipe must still convert, or bt601 defaults eat 6 chroma levels (C1)."""
    args = _writer(pipe="rgb24", high_bitdepth=False)._color_convert_args()

    assert "-vf" in args
    assert any("out_color_matrix=bt709" in argument for argument in args)


def test_the_pipe_format_follows_the_packed_format_and_the_depth():
    assert _writer(pipe="p010le")._pipe_input_pixel_format() == "p010le"
    assert _writer(pipe="yuv420p10le")._pipe_input_pixel_format() == "yuv420p10le"
    # an 8-bit source ignores the 10-bit declaration entirely
    assert _writer(pipe="yuv420p10le", high_bitdepth=False)._pipe_input_pixel_format() == "rgb24"


@pytest.mark.parametrize("requested,high_bitdepth,interpolating,expected", [
    (None, True, False, "yuv420p"),          # 10 bits at the encoder costs ~2x the NVENC time
    (None, False, False, "yuv420p"),
    ("yuv420p10le", True, False, "yuv420p10le"),   # opt-in, and worth ~6 dB
    ("yuv420p", True, False, "yuv420p"),     # an explicit request always wins
    ("yuv444p10le", False, False, "yuv444p10le"),
])
def test_output_depth_resolution(requested, high_bitdepth, interpolating, expected):
    assert FB.resolve_output_pixel_format(requested, high_bitdepth, interpolating) == expected
