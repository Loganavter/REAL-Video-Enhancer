"""Starting a render refuses an input the backend cannot read (the crash that followed an empty path).

Measured 2026-11-27: a job with an EMPTY input path reached the backend, which printed `No video stream found in the
input file` and died with `ZeroDivisionError: float division by zero` inside `VideoInfo.get_duration_seconds`; the pane
then reported `Rendering failed: ... Traceback (most recent call last):`. Two rows here: the probe answers 0.0 without an
fps (backend unit test), and a start refuses a job whose input is empty or missing before anything is spawned.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.ui.render_jobs import missing_input


def test_a_job_with_no_input_is_refused_before_the_backend_is_spawned(tmp_path):
    assert missing_input([SimpleNamespace(inputFile="")]) == "a job has no input file"
    assert missing_input([SimpleNamespace(inputFile="   ")]) == "a job has no input file"
    assert missing_input([SimpleNamespace()]) == "a job has no input file"


def test_a_missing_or_unreadable_path_is_refused_and_named(tmp_path):
    gone = str(tmp_path / "gone.mkv")
    assert missing_input([SimpleNamespace(inputFile=gone)]) == "the input file is missing: gone.mkv"
    folder = tmp_path / "folder.mkv"
    folder.mkdir()
    assert missing_input([SimpleNamespace(inputFile=str(folder))]) == "the input file is missing: folder.mkv"


def test_a_readable_input_passes(tmp_path):
    clip = tmp_path / "clip.mkv"
    clip.write_bytes(b"\x00")
    assert missing_input([SimpleNamespace(inputFile=str(clip))]) == ""
    assert missing_input([SimpleNamespace(inputFile=str(clip)), SimpleNamespace(inputFile=str(clip))]) == ""
    assert missing_input([]) == "", "an empty queue is the other refusal's business"


def test_the_probe_answers_zero_instead_of_dividing_by_a_zero_fps():
    """The backend's duration probe on an unreadable file: 0.0, never a ZeroDivisionError."""
    pytest.importorskip("cv2", reason="VideoInfo imports OpenCV, which ships with the backend, not the app interpreter")
    import importlib.util
    import pathlib
    import sys

    root = pathlib.Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("rve_videoinfo_probe", root / "backend/src/utils/VideoInfo.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["rve_videoinfo_probe"] = module
    spec.loader.exec_module(module)

    class Probe:
        start_time = None
        end_time = None

        class cap:                                    # noqa: N801 - the shape the real class reads
            @staticmethod
            def get(_prop):
                return 0.0

        get_fps = lambda self: 0.0                    # noqa: E731 - the unreadable-file answer

    probe = Probe()
    duration = module.VideoInfo.get_duration_seconds(probe)
    assert duration == pytest.approx(0.0), "an unreadable file has no duration, and the probe must not raise"
