"""The file decoder's two halves that can be checked without PyAV: it imports, and it fails loudly.

The real decode loop (open/play/seek/request_frame) cannot run on this repository's interpreter -
PyAV is not installed and the plan ships it with the player, not with the test toolchain - so those
tests are `importorskip`ed rather than faked into a green. What is *not* optional is that the module
stays importable without PyAV (the surface must be able to show the live stream on a machine whose
decoder is missing) and that asking it for a file hands back an `error` signal instead of raising
into the GUI thread.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from src.ui.video.decode import VideoDecoder


def _pyav_installed() -> bool:
    try:
        import av  # noqa: F401
    except ImportError:
        return False
    return True


def test_the_module_imports_without_pyav():
    # The import at the top of this file is the assertion: `decode.py` must not import `av` at module
    # scope, or every interpreter without the wheel would fail to import the preview package.
    assert VideoDecoder.frame_ready is not None


def test_open_without_pyav_reports_an_error_and_returns_false(qapp):
    if _pyav_installed():
        pytest.skip("PyAV is installed here; the no-decoder path is covered where it is missing")

    decoder = VideoDecoder()
    errors: list[str] = []
    decoder.error.connect(errors.append)

    assert decoder.open("/definitely/not/a/video.mp4") is False
    assert len(errors) == 1 and "PyAV is not installed" in errors[0]
    assert decoder.is_playing() is False


def test_transport_calls_before_open_are_reported_not_raised(qapp):
    decoder = VideoDecoder()
    errors: list[str] = []
    decoder.error.connect(errors.append)

    decoder.play()
    decoder.seek(1.0)
    decoder.request_frame(1.0)
    decoder.pause()
    decoder.close()

    assert len(errors) == 3, "play/seek/request_frame each say what is wrong"
    assert all("before open()" in message for message in errors)


def test_an_unknown_pixel_format_is_refused_at_construction(qapp):
    with pytest.raises(ValueError):
        VideoDecoder(pixel_format="nv12")
