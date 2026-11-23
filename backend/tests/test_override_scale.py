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


def test_a_packed_yuv_frame_is_refused_by_name(util):
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
