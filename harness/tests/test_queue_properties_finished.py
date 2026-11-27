"""The Properties window for a queue item that has FINISHED or FAILED - the state a user is most likely to inspect.

The window's own record was pinned for a pending entry and for a synthetic "ran" state
(`test_queue_properties.py`), but nothing ever named the run's OUTCOME: the RVE tab carried the timestamps of a
finished job and no row said "Finished", and a failed job's reason (`JobState.error`, the line the row's hover line
appends) never reached the window at all. Measured here on the model and on the real toolkit dialog, offscreen:

(a) a finished item's window says ``State: Finished``, and the facts of that run are visible beside it - the render
    duration, the output path AND its size on disk, and the run report's settings row;
(b) a failed item's window says ``State: Failed`` and carries the failure reason the queue recorded;
(c) an item that never ran says ``Not yet started`` - the state row follows the job, it is not a constant;
(d) an output file that does not exist yet is an em dash with a reason, never ``0 B``.

No media, no ffmpeg, no backend and no render: the probe is a fake and the item's state is a real ``JobState``.
"""

from __future__ import annotations

import json
import sys
import types
from datetime import datetime
from pathlib import Path

import pytest

_ROOT = str(Path(__file__).resolve().parents[2])
# The app and the backend both live in a `src` namespace package; the suite's conftest puts the backend first. The
# dialog imports the app's `src.constants`, so the app root goes first for this module (same pattern as
# `test_queue_properties.py`).
_ORIGINAL_SYS_PATH = list(sys.path)
if _ROOT in sys.path:
    sys.path.remove(_ROOT)
sys.path.insert(0, _ROOT)
sys.modules.pop("src.constants", None)

# The app's optional runtime imports: stub them ONLY when the interpreter does not have them.
try:  # pragma: no cover - the branch taken depends on the interpreter
    import cpuinfo  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    _cpuinfo = types.ModuleType("cpuinfo")
    _cpuinfo.get_cpu_info = lambda *args, **kwargs: {}
    sys.modules["cpuinfo"] = _cpuinfo
try:  # pragma: no cover
    import pypresence  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    _presence = types.ModuleType("pypresence")
    _presence.Presence = object
    sys.modules["pypresence"] = _presence
if "resources_rc" not in sys.modules:  # a test tree has no compiled resource blob
    sys.modules["resources_rc"] = types.ModuleType("resources_rc")

pytest.importorskip("PySide6")

from src.ui.RenderQueue import JobState  # noqa: E402
from src.ui.dialogs import queue_properties as properties_module  # noqa: E402

# Leave the suite's path order as it was (the app modules are imported and bound now).
sys.path[:] = _ORIGINAL_SYS_PATH


@pytest.fixture(autouse=True)
def _application(qapp):
    """The suite's one ``QApplication``: a ``QDialog`` cannot be constructed without it (Qt aborts, it does not raise)."""
    return qapp


#: The wall-clock stamps the fake run used; the visibility assertions compare against the rendered form.
ADDED_WALL = 1_766_000_000.0
STARTED_WALL = 1_766_000_010.0
FINISHED_WALL = 1_766_000_022.5


def _wall(timestamp: float) -> str:
    return datetime.fromtimestamp(float(timestamp)).strftime("%Y-%m-%d %H:%M:%S")


def _job(path: Path, *, output: Path, state, video_frame_count: int = 240):
    """A queued item carrying only the attributes the Properties window reads."""
    return types.SimpleNamespace(
        inputFile=str(path),
        outputPath=str(output),
        modelScale=4,
        interpolateTimes=2,
        encoderCommand="-c:v libx264 -crf 18",
        upscaleModelFile="/models/realesrgan-x4plus.pth",
        upscaleModelArch="realesrgan",
        interpolateModelFile="/models/rife-4.25.pth",
        deblurModelFile=None,
        denoiseModelFile=None,
        decompressModelFile=None,
        sceneChangeModelFile=None,
        overrideUpscaleScale=None,
        ensemble=False,
        dyanmicScaleOpticalFlow=True,
        sloMoMode=False,
        benchmarkMode=False,
        tilingEnabled=True,
        tilesize="256",
        backend="tensorrt",
        hdrMode=False,
        startTime=None,
        endTime=None,
        videoFrameCount=video_frame_count,
        state=state,
    )


def _finished_state() -> JobState:
    return JobState(
        status="done",
        progress=1.0,
        added_mono=100.0,
        added_wall=ADDED_WALL,
        started_mono=110.0,
        started_wall=STARTED_WALL,
        finished_mono=122.5,
        finished_wall=FINISHED_WALL,
        source_seconds=10.01,
    )


def _probe(_path: str):
    """A probe that measured nothing: this file is about the queue's own record, not the media."""
    return properties_module.ProbeResult(fields={}, reasons={}, source="fake probe")


def _open_dialog(job):
    return properties_module.QueuePropertiesDialog(job, probe=_probe, blocking=False)


def _write_report(output: Path, **settings) -> Path:
    report = output.with_name(f"{output.stem}.report.json")
    report.write_text(json.dumps({
        "run": {"output": str(output), "frames": 240},
        "settings": settings,
        "phases_ms_per_frame": {"frame": 35.1},
    }), encoding="utf-8")
    return report


# --- (a) a FINISHED item ------------------------------------------------------------------------------


def test_a_finished_items_window_names_the_state_and_the_facts_of_its_run(tmp_path):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"x" * 10)
    output = tmp_path / "clip_2x.mp4"
    output.write_bytes(b"y" * 1536)
    _write_report(output, quality_preset="best")

    dialog = _open_dialog(_job(source, output=output, state=_finished_state()))

    rve_text = dialog.documents["RVE"].text()
    assert "State" in rve_text and "Finished" in rve_text, (
        "the window for a finished item must name the completion state, not only its timestamps"
    )

    text = dialog.plain_text()
    assert "State: Finished" in text
    assert f"Started: {_wall(STARTED_WALL)}" in text
    assert f"Finished: {_wall(FINISHED_WALL)}" in text
    assert "Render duration: 00:00:12.500" in text
    assert f"Output path: {output}" in text
    assert "Output size: 1.5 KB" in text, "a finished run's output exists: the window reads its real size"
    assert "Quality preset: best" in text, "the report half that already landed is visible on the finished item too"


# --- (b) a FAILED item --------------------------------------------------------------------------------


def test_a_failed_items_window_names_the_state_and_the_recorded_reason(tmp_path):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"x" * 10)
    output = tmp_path / "clip_2x.mp4"           # never written: the run died before the encode finished
    reason = "killed by SIGTERM (the system's memory killer stopped the render)"

    state = _finished_state()
    state.status = "failed"
    state.progress = 0.42
    state.error = reason
    dialog = _open_dialog(_job(source, output=output, state=state))

    text = dialog.plain_text()
    assert "State: Failed" in text
    assert f"Failure: {reason}" in text, "the queue's own failure reason must reach the window"
    assert "Per-frame cost: —" in text or "Per-frame cost: 35.1 ms/frame" in text

    rve_text = dialog.documents["RVE"].text()
    assert "Failed" in rve_text and "SIGTERM" in rve_text


# --- (c) the state row follows the job -----------------------------------------------------------------


def test_an_item_that_never_ran_says_so_instead_of_finished(tmp_path):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"x" * 10)
    job = _job(
        source,
        output=tmp_path / "clip_2x.mp4",
        state=JobState(status="pending", added_mono=100.0, added_wall=ADDED_WALL),
    )
    text = _open_dialog(job).plain_text()
    assert "State: Not yet started" in text
    assert "State: Finished" not in text


# --- (d) an output that is not there yet is unmeasured, not 0 B --------------------------------------


def test_an_output_file_that_is_not_there_yet_reads_an_em_dash_with_its_reason(tmp_path):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"x" * 10)
    output = tmp_path / "clip_2x.mp4"
    job = _job(source, output=output, state=JobState(status="rendering", progress=0.5,
                                                     started_mono=110.0, started_wall=STARTED_WALL))
    properties = properties_module.build_queue_properties(job, probe=_probe)
    rows = {
        row.label: row
        for section in properties.tab("RVE").sections
        for row in section.rows
    }
    assert rows["Output size"].value == ""
    assert rows["Output size"].display_value == properties_module.EM_DASH
    assert "not there" in rows["Output size"].reason
    assert "Rendering" in properties.as_plain_text()
