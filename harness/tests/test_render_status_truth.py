"""The status slot's last word is the truth: Finalising while ffmpeg closes, then Done or Failed.

Why this is pinned: the render used to write `Done` (which the pane shows as Idle) *before* the sentinel
that lets the writer close the pipe and wait for the encoder, so the pane could say Idle while the file
was still being finalised. `PHASE_FAILED` existed on both sides of the slot and was never written, so a
failed render left no status at all - the pane stayed on `Rendering` until the process died.

These tests drive the real `Render` methods with a recording status channel and a fake writer. They need
no GPU, no ffmpeg, no torch: the code under test only moves phase bytes, a sentinel and a join.
"""
from __future__ import annotations

import sys
import types

import pytest

# `src.RenderVideo` imports torch at module scope and `utils.VideoInfo` imports cv2; the harness
# interpreter carries neither. The code under test never touches either library, so a bare module stub
# is enough to import the real class. The stubs are removed again below so a later test that genuinely
# needs them still skips instead of importing a fake.
_stubbed = []
for _name in ("torch", "cv2"):
    try:
        __import__(_name)
    except ImportError:
        sys.modules[_name] = types.ModuleType(_name)
        _stubbed.append(_name)

RV = pytest.importorskip("src.RenderVideo")
from src.utils import DisplayChannel as DC  # noqa: E402

for _name in _stubbed:
    sys.modules.pop(_name, None)
# A later `pytest.importorskip("src.RenderVideo")` must re-check the real dependency, not inherit ours.
sys.modules.pop("src.RenderVideo", None)


class _Recorder:
    """One list of events: a status phase, the sentinel, or the writer join, in the order they happen."""

    def __init__(self) -> None:
        self.events: list[tuple] = []
        self.calls: list[dict] = []

    # status channel
    def set(self, **changes) -> int:
        self.calls.append(dict(changes))
        self.events.append(("phase", changes["phase"]) if "phase" in changes else ("status",))
        return len(self.calls)

    # write queue
    def put(self, item) -> None:
        self.events.append(("sentinel", item))

    # write thread
    def join(self) -> None:
        self.events.append(("join",))

    def phases(self) -> list[int]:
        return [event[1] for event in self.events if event[0] == "phase"]


class _Writer:
    """The writer's buffer: the queue the render closes, and the process the writer reaped."""

    def __init__(self, recorder: _Recorder, returncode: int = 0, reason: str | None = None) -> None:
        self.writeQueue = recorder
        self.writeProcess = types.SimpleNamespace(returncode=returncode)
        self.errorReason = reason


def _renderer(recorder: _Recorder, returncode: int = 0, reason: str | None = None) -> RV.Render:
    video = object.__new__(RV.Render)
    video.statusChannel = recorder
    video.writeBuffer = _Writer(recorder, returncode, reason)
    video.ffmpegWriteThread = recorder  # `join` records the wait in the same sequence
    return video


def test_done_is_written_after_the_encoder_exits_not_before_the_sentinel():
    recorder = _Recorder()
    video = _renderer(recorder)

    assert video._close_encoder_and_settle(12) is None

    assert recorder.phases() == [DC.PHASE_FINALISING, DC.PHASE_DONE]
    kinds = [kind for kind, *_ in recorder.events]
    assert kinds == ["phase", "sentinel", "join", "phase"], (
        "Finalising, then the sentinel, then the writer's exit, then Done - in that order")
    frames = [call.get("frames_rendered") for call in recorder.calls]
    assert frames == [12, 12]


def test_a_failed_encoder_writes_failed_with_its_reason():
    recorder = _Recorder()
    video = _renderer(recorder, returncode=1)

    reason = video._close_encoder_and_settle(7)

    assert reason == "ffmpeg exited with code 1"
    assert recorder.phases() == [DC.PHASE_FINALISING, DC.PHASE_FAILED]


def test_a_reason_recorded_by_the_writer_wins_over_the_exit_code():
    recorder = _Recorder()
    video = _renderer(recorder, returncode=1, reason="output path is not writable")

    reason = video._close_encoder_and_settle(3)

    assert reason == "output path is not writable"
    assert recorder.phases()[-1] == DC.PHASE_FAILED


def test_a_raising_writer_marks_the_slot_failed_before_the_guard_ends_the_process():
    recorder = _Recorder()
    video = object.__new__(RV.Render)
    video.statusChannel = recorder
    video._frames_rendered = 5

    class _Boom:
        def write_out_frames(self) -> None:
            raise RuntimeError("pipe closed")

    video.writeBuffer = _Boom()

    with pytest.raises(RuntimeError, match="pipe closed"):
        video._run_writer()

    assert recorder.phases() == [DC.PHASE_FAILED]
    assert recorder.calls[0]["frames_rendered"] == 5, "a failure must not reset the frame count to 0"


def test_an_encoder_error_writes_failed_before_the_original_handler_exits_the_process():
    """`FFmpegWrite.onErroredExit` ends the process; the slot must hear about the failure first."""
    recorder = _Recorder()
    video = object.__new__(RV.Render)
    video.statusChannel = recorder
    video._frames_rendered = 9
    called: list[str] = []

    class _Buffer:
        def __init__(self) -> None:
            self.writeProcess = types.SimpleNamespace(returncode=1)

        def onErroredExit(self) -> None:  # noqa: N802 - the writer's own name
            called.append("original")

        def write_out_frames(self) -> None:
            self.onErroredExit()  # what FFmpegWrite does before `os._exit(1)`

    video.writeBuffer = _Buffer()

    video._run_writer()

    assert called == ["original"], "the writer's own ffmpeg-log path still runs"
    assert recorder.phases() == [DC.PHASE_FAILED]
    assert recorder.calls[0]["frames_rendered"] == 9


def test_a_failing_loop_writes_failed_with_the_error_before_it_is_reraised():
    recorder = _Recorder()
    video = object.__new__(RV.Render)
    video.statusChannel = recorder
    video._frames_rendered = 4

    def _boom() -> int:
        raise RuntimeError("engine died")

    video._render_loop = _boom

    with pytest.raises(RuntimeError, match="engine died"):
        video.render()

    assert recorder.phases() == [DC.PHASE_FAILED]
    assert recorder.calls[0]["frames_rendered"] == 4


def test_the_run_report_records_the_quality_preset():
    """The Properties window reads `settings.quality_preset`; without it the row can only say 'not recorded'."""
    video = object.__new__(RV.Render)
    video.upscaleModel = "4x-UltraSharp.pth"
    video.qualityPreset = "fast"
    video.int8FitPreset = "int8"
    video.temporalReuse = None
    video.temporalReuseCorrection = False
    video.autoTuner = None

    report = video._run_report()

    assert report["settings"]["quality_preset"] == "fast"
    assert report["settings"]["int8_fit"] == "int8", "the report's other settings are untouched"
