"""The 8-bit planar pipe: what the packer emits, and the arguments it implies.

The 8-bit pipe is opt-in (`RVE_YUV8=1`) because it replaces two conversions in the app with one on
the GPU: rgb24 on the wire (3 bytes/px, 24.9 MB per 4K frame) plus ffmpeg's `scale=` filter, against
planar yuv420p (1.5 bytes/px) with no filter. Measured on 249 frames of Chainsaw Man at 4K: mean
|delta| 0.165 / 0.138 / 0.129 levels on Y/U/V, max 4, and **no pixel** more than 8 levels off - the
acceptance statistic's own bar. These tests pin the parts that would silently break that: the
planar layout, the limited-range mapping, and the ffmpeg arguments for the YUV branch.

CPU only; skipped when torch is not importable, since the packer is a torch op.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from src import FFmpegBuffers as FB  # noqa: E402
from src.pytorch.TorchUtils import TorchUtils  # noqa: E402


class _Encoder:
    def __init__(self, tag: str):
        self._tag = tag

    def getPresetTag(self) -> str:
        return self._tag


def _writer(pipe: str, high_bitdepth: bool = False) -> FB.FFmpegWrite:
    writer = FB.FFmpegWrite.__new__(FB.FFmpegWrite)
    writer.high_bitdepth = high_bitdepth
    writer.hdr_mode = False
    writer.custom_encoder = None
    writer.color_space = "bt709"
    writer.color_range = "tv"
    writer.color_primaries = None
    writer.color_transfer = None
    writer.video_encoder = _Encoder("x265_nvenc")
    writer.pipe_input_pixel_format = pipe
    import io
    writer.ffmpeg_log = io.StringIO()
    return writer


def _solid(rgb, h: int = 4, w: int = 4):
    t = torch.zeros(1, 3, h, w, dtype=torch.float32)
    for c, v in enumerate(rgb):
        t[0, c] = v
    return t


def _unpack(frame: torch.Tensor, h: int = 4, w: int = 4):
    # the constructor builds CUDA streams; the packer only needs the method and a sync hook
    utils = TorchUtils.__new__(TorchUtils)
    utils.sync_all_streams = lambda: None
    packed = utils.tensor_to_yuv420p(frame)
    arr = torch.as_tensor(packed)
    n_y = h * w
    n_c = (h // 2) * (w // 2)
    y = arr[:n_y]
    u = arr[n_y:n_y + n_c]
    v = arr[n_y + n_c:]
    return y, u, v


def test_white_maps_to_limited_range_235_and_neutral_chroma():
    y, u, v = _unpack(_solid((1.0, 1.0, 1.0)))
    assert int(y.max()) == 235
    assert int(u.min()) == int(u.max()) == 128
    assert int(v.min()) == int(v.max()) == 128


def test_black_maps_to_16():
    y, u, v = _unpack(_solid((0.0, 0.0, 0.0)))
    assert int(y.min()) == int(y.max()) == 16
    assert int(u.min()) == int(u.max()) == 128


def test_planar_layout_is_y_then_u_then_v_at_1_5_bytes_per_pixel():
    y, u, v = _unpack(_solid((1.0, 1.0, 1.0)))
    assert y.numel() == 4 * 4
    assert u.numel() == v.numel() == 2 * 2
    assert y.numel() + u.numel() + v.numel() == 4 * 4 * 3 // 2


def test_bt709_red_lands_where_the_matrix_says():
    """BT.709 limited range: Y = 16+219*0.2126 = 63, U = 128-25.7 = 102, V = 240."""
    y, u, v = _unpack(_solid((1.0, 0.0, 0.0)))
    assert int(y.median()) == 63
    assert int(u.median()) == 102
    assert int(v.median()) == 240


def test_the_yuv8_pipe_is_named_what_the_packer_emitted():
    assert _writer("yuv420p")._pipe_input_pixel_format() == "yuv420p"


def test_the_yuv8_pipe_gets_no_rgb_to_yuv_filter():
    """An rgb24 pipe needs `scale=in_range=pc:...`; a yuv420p pipe must not be touched again -
    reinterpreting a limited-range stream cost the P010 pipe 31 dB (ledger C4)."""
    assert _writer("yuv420p")._color_convert_args() == []
    assert _writer("rgb24")._color_convert_args() != []


def test_the_default_8bit_pipe_is_still_rgb24():
    assert _writer("rgb24")._pipe_input_pixel_format() == "rgb24"
