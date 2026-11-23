"""The `--override_upscale_scale` resize: source geometry, the named refusal, and the thread guard.

The defect this file pins (reproduced live 2026-11-22): a 160x120 clip, model
`4xNomos8k_span_otf_weak_no_update_params.pth`, `--override_upscale_scale 2` - the upscaler returned
a 320x240 frame while the caller asserted `width * modelScale` = 640x480, so `resize_image_bytes`
died in `numpy.reshape` with "cannot reshape array of size 230400 into shape (480,640,3)". The
message named neither the frame's own geometry nor the path that produced it, and the exception
lived in the render thread: the process stayed alive at 0 FPS with no error on screen.

Three rules, one test group each:

* the source geometry is the frame's own (its tensor first - the header can lag it), never
  `width * modelScale`; a per-image decision (`payload_geometry`) is cv2-free, so it runs here;
* a payload that cannot be an image of the declared size raises `ValueError` naming both sizes and
  the caller path - never a reshape error;
* `guard_render_thread` logs exactly one `RENDER FAILED: <type>: <msg>` line plus the traceback and
  leaves the process non-zero, which is what ends the job in the GUI.

Group (e) covers the packed planar YUV pipes (`yuv8_output`, `yuv10_output`): the resize must be
plane-wise, land Y and each chroma plane on the override geometry, keep the payload's plane order
and packing (`cat(Y, U, V)` or P010's interleaved `cat(Y, UV)`), and still refuse every layout it
cannot prove. The layout pinned here is the packers' own (`backend/src/pytorch/TorchUtils.py`):
Y `h x w`, chroma `ceil(h/2) x ceil(w/2)`, P010 with the 10-bit sample in the MSBs.

This interpreter has no cv2 (the backend's venv has it). The resize plumbing is therefore exercised
against a numpy stand-in for `cv2.resize`: these tests assert the GEOMETRY decisions - which size is
read from the payload and which array reaches cv2 - not cv2's resampling quality.
"""

import os
import sys
from types import SimpleNamespace

import pytest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

np = pytest.importorskip("numpy")


@pytest.fixture()
def util():
    return pytest.importorskip("src.utils.Util")


class _FakeCv2:
    """`cv2.resize` stand-in: records the request and returns an image of the asked geometry."""

    INTER_AREA = 0
    INTER_LANCZOS4 = 1

    def __init__(self):
        self.calls = []

    def resize(self, array, size, interpolation=None):
        width, height = size
        self.calls.append((tuple(array.shape), (width, height), interpolation))
        rows = np.arange(height) * array.shape[0] // max(1, height)
        columns = np.arange(width) * array.shape[1] // max(1, width)
        return np.ascontiguousarray(array[rows][:, columns])


@pytest.fixture()
def fake_cv2(util, monkeypatch):
    stub = _FakeCv2()
    monkeypatch.setattr(util, "cv2", stub, raising=False)
    return stub


class _FakeFrame:
    """A Frame as far as this path is concerned: a payload, a header and (optionally) a tensor."""

    def __init__(self, payload, width, height, *, tensor_shape=None, **flags):
        self.payload = payload
        self.width = width
        self.height = height
        self._tensor = SimpleNamespace(shape=tensor_shape) if tensor_shape else None
        self._np = None
        for name, value in flags.items():
            setattr(self, name, value)

    def get_frame_bytes(self):
        return self.payload


def _rgb(width, height):
    return np.zeros((height, width, 3), dtype=np.uint8).tobytes()


def _yuv420(width, height, y_value=100, u_value=50, v_value=200, dtype=np.uint8):
    """Packed planar 4:2:0 the way `TorchUtils.tensor_to_yuv420p[10]` packs it.

    `cat(Y, U, V)`, Y `h x w`, each chroma plane `ceil(h/2) x ceil(w/2)` (the packers replicate-pad
    an odd edge before the 2x2 average). A distinct constant per plane is what makes a plane shuffle
    visible: a reordered payload paints Y with chroma.
    """
    chroma_width, chroma_height = (width + 1) // 2, (height + 1) // 2
    y = np.full((height, width), y_value, dtype=dtype)
    u = np.full((chroma_height, chroma_width), u_value, dtype=dtype)
    v = np.full((chroma_height, chroma_width), v_value, dtype=dtype)
    return np.concatenate((y.reshape(-1), u.reshape(-1), v.reshape(-1))).tobytes()


def _p010(width, height, y_value=100, u_value=50, v_value=200):
    """`p010le` as `TorchUtils.tensor_to_yuv420p10` packs it: 10-bit sample in the MSBs, UV interleaved."""
    chroma_width, chroma_height = (width + 1) // 2, (height + 1) // 2
    y = np.full((height, width), y_value << 6, dtype=np.uint16)
    uv = np.empty((chroma_height, chroma_width, 2), dtype=np.uint16)
    uv[:, :, 0] = u_value << 6
    uv[:, :, 1] = v_value << 6
    return np.concatenate((y.reshape(-1), uv.reshape(-1))).tobytes()


# --- (a) a payload already at the declared size is handed back unchanged ---------------------


def test_payload_at_the_declared_size_is_returned_unchanged(util, fake_cv2):
    payload = _rgb(320, 240)

    out = util.resize_image_bytes(payload, 320, 240, 320, 240, path="test")

    assert out is payload
    assert fake_cv2.calls == [], "a no-op resize must not pay a cv2 call"


# --- (b) the resize uses the payload's ACTUAL geometry ---------------------------------------


def test_payload_is_resized_from_its_actual_size_to_the_target(util, fake_cv2):
    payload = _rgb(160, 120)

    out = util.resize_image_bytes(payload, 160, 120, 320, 240, path="test")

    assert len(out) == 320 * 240 * 3
    (source_shape, target, interpolation), = fake_cv2.calls
    assert source_shape == (120, 160, 3), "cv2 must be handed the payload's real geometry"
    assert target == (320, 240)
    assert interpolation == fake_cv2.INTER_LANCZOS4  # upscaling


def test_downscaling_picks_area_interpolation(util, fake_cv2):
    out = util.resize_image_bytes(_rgb(640, 480), 640, 480, 320, 240, path="test")

    assert len(out) == 320 * 240 * 3
    assert fake_cv2.calls[0][2] == fake_cv2.INTER_AREA


def test_a_16_bit_payload_keeps_its_bit_depth(util, fake_cv2):
    payload = np.zeros((120, 160, 3), dtype=np.uint16).tobytes()

    out = util.resize_image_bytes(payload, 160, 120, 320, 240, path="test")

    assert fake_cv2.calls[0][0] == (120, 160, 3)
    assert len(out) == 320 * 240 * 3 * 2


# --- the frame's own geometry wins over the descriptor's scale -------------------------------


def test_the_repro_frame_geometry_wins_over_the_descriptor_scale(util, fake_cv2):
    """160x120 in, model descriptor says 4x, the returned frame is 320x240 (the live repro)."""
    frame = _FakeFrame(_rgb(320, 240), 640, 480, tensor_shape=(1, 3, 240, 320))

    out = util.resize_frame_bytes(frame, 320, 240, path="test")

    assert out is frame.payload, "320x240 is already the override target: nothing to resample"
    assert fake_cv2.calls == []


def test_the_frame_is_rescaled_from_its_tensor_geometry(util, fake_cv2):
    frame = _FakeFrame(_rgb(320, 240), 640, 480, tensor_shape=(1, 3, 240, 320))

    out = util.resize_frame_bytes(frame, 640, 480, path="test")

    assert len(out) == 640 * 480 * 3
    assert fake_cv2.calls[0][0] == (240, 320, 3)
    assert fake_cv2.calls[0][1] == (640, 480)


def test_a_header_only_frame_uses_its_header(util, fake_cv2):
    frame = _FakeFrame(_rgb(160, 120), 160, 120)

    out = util.resize_frame_bytes(frame, 320, 240, path="test")

    assert len(out) == 320 * 240 * 3


# --- (c) a wrong declared size is a named ValueError, not a reshape error ---------------------


def test_a_wrong_declared_size_names_both_geometries_and_the_path(util):
    payload = _rgb(320, 240)  # 230400 bytes

    with pytest.raises(ValueError) as excinfo:
        util.resize_image_bytes(payload, 640, 480, 320, 240,
                                path="RenderVideo.render:override_upscale_scale")

    message = str(excinfo.value)
    assert "640x480" in message, "the declared geometry must be named"
    assert "320x240" in message, "the payload's real geometry must be named"
    assert "230400" in message, "the payload size must be named"
    assert "RenderVideo.render:override_upscale_scale" in message, "the producing path must be named"
    assert "cannot reshape" not in message


def test_a_frame_whose_payload_disagrees_with_its_own_geometry_is_refused(util):
    frame = _FakeFrame(_rgb(320, 240), 640, 480)  # no tensor: the header is all there is

    with pytest.raises(ValueError) as excinfo:
        util.resize_frame_bytes(frame, 320, 240, path="RenderVideo.render:override_upscale_scale")

    message = str(excinfo.value)
    assert "640x480" in message and "320x240" in message


def test_an_empty_declared_geometry_is_refused(util):
    with pytest.raises(ValueError, match="empty"):
        util.payload_geometry(230400, 0, 0, path="test")


def test_a_packed_yuv_frame_is_refused_when_the_ab_switch_is_off(util, monkeypatch):
    """`RVE_OVERRIDE_YUV=0` restores the pre-fix refusal, so the merge decision has a control arm."""
    monkeypatch.setenv("RVE_OVERRIDE_YUV", "0")
    frame = _FakeFrame(b"\x00" * (160 * 120 * 3 // 2), 160, 120, yuv8_output=True)

    with pytest.raises(ValueError, match="packed planar YUV"):
        util.resize_frame_bytes(frame, 320, 240, path="test")


def test_raw_bytes_have_no_geometry_to_read(util):
    with pytest.raises(TypeError, match="no geometry"):
        util.resize_frame_bytes(_rgb(160, 120), 320, 240, path="test")


# --- (d) the thread guard ends the process instead of hanging it -----------------------------


def test_the_thread_guard_logs_one_failure_line_and_returns_non_zero(util, capsys):
    exits = []

    def failing_render():
        raise ValueError("cannot reshape array of size 230400 into shape (480,640,3)")

    code = util.guard_render_thread(failing_render, path="RenderVideo.render",
                                    exit_process=exits.append)

    captured = capsys.readouterr()
    assert code != 0
    assert exits == [1], "the process must be told to leave non-zero"
    failure_lines = [line for line in captured.err.splitlines() if "RENDER FAILED" in line]
    assert len(failure_lines) == 1, captured.err
    assert failure_lines[0].endswith(
        "RENDER FAILED: ValueError: cannot reshape array of size 230400 into shape (480,640,3)")
    assert "Traceback (most recent call last)" in captured.err


def test_the_thread_guard_lets_a_healthy_render_return(util, capsys):
    exits = []

    code = util.guard_render_thread(lambda: None, path="RenderVideo.render",
                                    exit_process=exits.append)

    assert code == 0
    assert exits == []
    assert "RENDER FAILED" not in capsys.readouterr().err


# --- (e) the packed planar YUV pipes: plane-wise resize, layout preserved ---------------------


def test_yuv8_planes_land_on_their_target_sizes(util, fake_cv2):
    """Y -> target size, each chroma plane -> its half, and the payload is exactly one frame."""
    payload = _yuv420(160, 120)

    out = util.resize_yuv420_bytes(payload, 160, 120, 320, 240, pixel_format="yuv420p",
                                   path="test")

    assert len(out) == 320 * 240 + 2 * 160 * 120 + 0, "an 8-bit planar frame is 1.5 bytes/pixel"
    assert len(out) == 115200
    assert [call[0] for call in fake_cv2.calls] == [(120, 160), (60, 80), (60, 80)]
    assert [call[1] for call in fake_cv2.calls] == [(320, 240), (160, 120), (160, 120)]
    assert {call[2] for call in fake_cv2.calls} == {fake_cv2.INTER_LANCZOS4}, "upscaling"


def test_yuv8_constant_colours_stay_on_their_own_plane(util, fake_cv2):
    """A plane shuffle is a wrong picture, not an exception: pin the order with distinct constants."""
    out = util.resize_yuv420_bytes(_yuv420(64, 48), 64, 48, 128, 96, pixel_format="yuv420p",
                                   path="test")

    flat = np.frombuffer(out, dtype=np.uint8)
    y = flat[:128 * 96]
    u = flat[128 * 96:128 * 96 + 64 * 48]
    v = flat[128 * 96 + 64 * 48:]
    assert int(y.min()) == int(y.max()) == 100, "luma stays luma"
    assert int(u.min()) == int(u.max()) == 50, "U stays U"
    assert int(v.min()) == int(v.max()) == 200, "V stays V"


def test_yuv_downscaling_picks_area_interpolation(util, fake_cv2):
    out = util.resize_yuv420_bytes(_yuv420(320, 240), 320, 240, 160, 120, pixel_format="yuv420p",
                                   path="test")

    assert len(out) == 160 * 120 + 2 * 80 * 60
    assert {call[2] for call in fake_cv2.calls} == {fake_cv2.INTER_AREA}, "downscaling"


def test_an_odd_source_geometry_uses_the_ceiling_chroma_size(util, fake_cv2):
    """159x119 -> chroma 80x60 (ceil), the size the packers and ffmpeg both plan for."""
    payload = _yuv420(159, 119)

    out = util.resize_yuv420_bytes(payload, 159, 119, 318, 238, pixel_format="yuv420p", path="test")

    assert [call[0] for call in fake_cv2.calls] == [(119, 159), (60, 80), (60, 80)]
    assert [call[1] for call in fake_cv2.calls] == [(318, 238), (159, 119), (159, 119)]
    assert len(out) == 318 * 238 + 2 * 159 * 119


def test_yuv10_planar_keeps_its_uint16_samples_and_sizes(util, fake_cv2):
    payload = _yuv420(160, 120, y_value=400, u_value=300, v_value=500, dtype=np.uint16)

    out = util.resize_yuv420_bytes(payload, 160, 120, 320, 240, pixel_format="yuv420p10le",
                                   path="test")

    assert len(out) == 2 * (320 * 240 + 2 * 160 * 120), "10-bit planar is 3 bytes/pixel"
    assert [call[0] for call in fake_cv2.calls] == [(120, 160), (60, 80), (60, 80)]
    flat = np.frombuffer(out, dtype=np.uint16)
    y_samples, chroma_samples = 320 * 240, 160 * 120
    assert [int(flat[0]), int(flat[y_samples]), int(flat[y_samples + chroma_samples])] == [400, 300, 500], \
        "planar 10-bit keeps its own sample values (no MSB shift) and its plane order"


def test_p010_keeps_the_interleaved_uv_layout_and_the_msb_shift(util, fake_cv2):
    payload = _p010(64, 48)

    out = util.resize_yuv420_bytes(payload, 64, 48, 128, 96, pixel_format="p010le", path="test")

    assert len(out) == 2 * (128 * 96 + 2 * 64 * 48)
    # U and V are resampled as separate planes, not as one (ch, cw*2) image that smears them
    assert [call[0] for call in fake_cv2.calls] == [(48, 64), (24, 32), (24, 32)]
    assert [call[1] for call in fake_cv2.calls] == [(128, 96), (64, 48), (64, 48)]
    flat = np.frombuffer(out, dtype=np.uint16)
    assert int(flat[0]) == 100 << 6, "Y keeps the 10-bit sample in the MSBs"
    uv = flat[128 * 96:].reshape(64, 48, 2)
    assert (int(uv[0, 0, 0]), int(uv[0, 0, 1])) == (50 << 6, 200 << 6), "U,V still interleaved in order"


def test_a_yuv_payload_that_is_not_a_whole_frame_is_refused_by_name(util):
    with pytest.raises(ValueError) as excinfo:
        util.resize_yuv420_bytes(b"\x00" * 1000, 160, 120, 320, 240, pixel_format="yuv420p",
                                 path="test")

    message = str(excinfo.value)
    assert "160x120" in message, "the declared geometry must be named"
    assert "28800" in message, "the sample count the layout expects must be named"
    assert "1000" in message, "the payload size must be named"
    assert "test" in message, "the producing path must be named"


def test_a_10_bit_target_mismatch_against_an_8_bit_payload_is_refused(util):
    """The same payload cannot be yuv420p10le: the sample count disagrees by exactly 2x."""
    with pytest.raises(ValueError, match="yuv420p10le"):
        util.resize_yuv420_bytes(_yuv420(160, 120), 160, 120, 320, 240,
                                 pixel_format="yuv420p10le", path="test")


def test_an_unknown_yuv_layout_is_refused_by_name(util):
    frame = _FakeFrame(b"\x00" * (160 * 120 * 3), 160, 120, yuv10_output=True,
                       yuv10_format="yuv444p10le")

    with pytest.raises(ValueError, match="not a packed planar 4:2:0"):
        util.resize_frame_bytes(frame, 320, 240, path="test")


def test_a_yuv8_frame_is_resized_from_its_tensor_geometry(util, fake_cv2):
    """The live repro's frame: header says 640x480, the tensor (and payload) are 320x240."""
    frame = _FakeFrame(_yuv420(320, 240), 640, 480, tensor_shape=(1, 3, 240, 320), yuv8_output=True)

    out = util.resize_frame_bytes(frame, 640, 480, path="test")

    assert len(out) == 640 * 480 + 2 * 320 * 240
    assert fake_cv2.calls[0][0] == (240, 320), "the frame's own geometry, not width*modelScale"
    assert fake_cv2.calls[0][1] == (640, 480)


def test_a_yuv10_frame_is_resized_in_its_own_packing(util, fake_cv2):
    frame = _FakeFrame(_p010(160, 120), 160, 120, tensor_shape=(1, 3, 120, 160),
                       yuv10_output=True, yuv10_format="p010le")

    out = util.resize_frame_bytes(frame, 320, 240, path="test")

    assert len(out) == 2 * (320 * 240 + 2 * 160 * 120)
    assert [call[0] for call in fake_cv2.calls] == [(120, 160), (60, 80), (60, 80)]


def test_a_yuv_frame_already_at_the_target_is_returned_untouched(util, fake_cv2):
    """The live repro's no-op: a 320x240 frame with `--override_upscale_scale 2` on a 160x120 clip.

    The payload comes back as `bytes` even though the packer handed a numpy array: the render loop
    branches on that type and would otherwise call `get_frame_bytes()` on an array.
    """
    payload = np.frombuffer(_yuv420(320, 240), dtype=np.uint8)
    frame = _FakeFrame(payload, 320, 240, yuv8_output=True)

    out = util.resize_frame_bytes(frame, 320, 240, path="test")

    assert isinstance(out, bytes), "the render loop's `type(frame) is bytes` branch needs bytes"
    assert out == payload.tobytes()
    assert fake_cv2.calls == [], "a no-op resize must not pay a cv2 call"
