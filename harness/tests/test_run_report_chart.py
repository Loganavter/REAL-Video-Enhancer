"""The RVE tab's processing curve: the per-phase averages become a painted chart (owner, 2026-11-28).

The owner, looking at the Properties window: *"а в чем великий смысл тут у колонки Phases? Показали бы уж лучше
какой-нибудь график обработки"* - and sent an fps-over-time curve. The old row concatenated per-phase AVERAGES
(``phases_ms_per_frame``), which cannot show a stall, a ramp or a section where the pipeline fell apart: it averages
the shape away. The tab now paints ``report["fps_series"]`` (the backend's sparse samples -> window rates) and moves
the averages to that chart's tooltip.

What this file proves on the REAL ``QueuePropertiesDialog``, offscreen, for a finished queue item with a synthetic
report:

(a) the ``Phases (ms/frame)`` row is GONE from the document, while the report's other rows stay;
(b) the chart exists on the RVE tab, is visible, and its size is the design box resolved through ``scaled_px``;
(c) painting it to a QImage draws the theme's ACCENT - measured by sampling pixels, not by "it did not raise";
(d) a report without the series (every report written before this change) prints one reason row and builds no chart;
(e) the tooltip carries the per-phase averages the row used to show.

Why each of these fails on the OLD code (checked by reading the pre-change file, not by running it): the old
``_report_section`` always appended a ``Phases (ms/frame)`` row, so (a) fails; there was no chart widget and no
``dialog.chart``, so (b), (c) fail on the missing object; (d) fails because the old row had no "Processing speed"
alternative and no chart attribute existed to assert absence on; (e) fails because no widget carried the averages.
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
# `test_queue_properties_finished.py`).
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
pytest.importorskip("sli_ui_toolkit")

from PySide6.QtGui import QColor  # noqa: E402

from sli_ui_toolkit.ui.managers.theme_manager import ThemeManager  # noqa: E402
from sli_ui_toolkit.ui.managers.ui_scale import scaled_px  # noqa: E402

from src.ui.RenderQueue import JobState  # noqa: E402
from src.ui.dialogs import queue_properties as properties_module  # noqa: E402

# Leave the suite's path order as it was (the app modules are imported and bound now).
sys.path[:] = _ORIGINAL_SYS_PATH


@pytest.fixture(autouse=True)
def _application(qapp):
    """The suite's one ``QApplication`` plus the app's own palette.

    The chart paints the theme's ``accent`` TOKEN, so the token has to resolve to the colour the app ships (a
    QDialog cannot be constructed without an application either - Qt aborts, it does not raise).
    """
    from src.ui import toolkit_theme

    toolkit_theme.install(qapp)
    return qapp


ADDED_WALL = 1_766_000_000.0
STARTED_WALL = 1_766_000_010.0
FINISHED_WALL = 1_766_000_022.5

#: The synthetic curve: 240 output frames in 8 s. The window rates are `delta_frames / delta_seconds`, so the run
#: starts at 30 fps, drops to 60 for one second, then falls to 20 - the slowdown the averages used to hide.
SERIES = {
    "frames": [0, 60, 120, 180, 240],
    "seconds": [0.0, 2.0, 4.0, 5.0, 8.0],
    "fps": [0.0, 30.0, 30.0, 60.0, 20.0],
}

PHASES = {"gate": 8.915, "infer": 18.019, "up_wall": 18.404, "unaccounted": 15.022}


def _job(path: Path, *, output: Path):
    """A finished queue item carrying only the attributes the Properties window reads."""
    return types.SimpleNamespace(
        inputFile=str(path),
        outputPath=str(output),
        modelScale=4,
        interpolateTimes=2,
        encoderCommand="-c:v libx264 -crf 18",
        upscaleModelFile="/models/realesrgan-x4plus.pth",
        upscaleModelArch="realesrgan",
        interpolateModelFile=None,
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
        videoFrameCount=240,
        state=JobState(
            status="done",
            progress=1.0,
            added_mono=100.0,
            added_wall=ADDED_WALL,
            started_mono=110.0,
            started_wall=STARTED_WALL,
            finished_mono=122.5,
            finished_wall=FINISHED_WALL,
            source_seconds=8.0,
        ),
    )


def _probe(_path: str):
    """A probe that measured nothing: this file is about the queue's own record and its report, not the media."""
    return properties_module.ProbeResult(fields={}, reasons={}, source="fake probe")


def _write_report(output: Path, *, series=None, **settings) -> Path:
    report = output.with_name(f"{output.stem}.report.json")
    document = {
        "run": {"output": str(output), "frames": 240},
        "settings": {"quality_preset": "best", "temporal_reuse_statistic": "median", **settings},
        "phases_ms_per_frame": dict(PHASES),
    }
    if series is not None:
        document["fps_series"] = series
    report.write_text(json.dumps(document), encoding="utf-8")
    return report


def _item(tmp_path: Path, *, series=SERIES):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"x" * 10)
    output = tmp_path / "clip_2x.mp4"
    output.write_bytes(b"y" * 1536)
    _write_report(output, series=series)
    return _job(source, output=output)


def _open(job):
    return properties_module.QueuePropertiesDialog(job, probe=_probe, blocking=False)


def _near_accent(pixel: QColor, accent: QColor, tolerance: int = 40) -> bool:
    return (
        abs(pixel.red() - accent.red()) <= tolerance
        and abs(pixel.green() - accent.green()) <= tolerance
        and abs(pixel.blue() - accent.blue()) <= tolerance
    )


# --- (a) the average row is gone, the rest of the report stays ----------------------------------------


def test_the_phase_average_row_is_replaced_by_the_curve(tmp_path):
    dialog = _open(_item(tmp_path))
    try:
        rve_text = dialog.documents["RVE"].text()
        assert "Phases (ms/frame)" not in rve_text, (
            "the row of per-phase AVERAGES must be gone: the curve shows the run's shape, the row averaged it away"
        )
        assert "Phases (ms/frame)" not in dialog.plain_text()

        # The report's other rows are untouched by the replacement.
        assert "Quality preset: best" in dialog.plain_text()
        assert "Statistic: median" in dialog.plain_text()
        assert "State: Finished" in dialog.plain_text()
    finally:
        dialog.close()


# --- (b) the chart is a real, visible, scaled widget --------------------------------------------------


def test_the_chart_exists_is_visible_and_follows_scaled_px(tmp_path, qapp):
    dialog = _open(_item(tmp_path))
    try:
        chart = dialog.chart
        assert chart is not None, "a report WITH a series must build the chart"
        assert isinstance(chart, properties_module.FpsChart)
        assert dialog.charts.get("RVE") is chart, "the chart belongs to the RVE tab, where the report lives"

        dialog.show()
        # The chart lives on the RVE tab, which is not the tab the window opens on: show it, then assert.
        dialog.tabs.setCurrentIndex(dialog.tabs.indexOf(dialog.charts["RVE"].parentWidget()))
        qapp.processEvents()
        assert chart.isVisible(), "the chart must be on screen when the RVE tab is showing, not a hidden attribute"
        assert chart.width() > 0 and chart.height() > 0

        assert chart.height() == scaled_px(properties_module.FPS_CHART_HEIGHT_PX), (
            "the chart's height is the design box resolved through the toolkit's scale, not a raw pixel literal"
        )
        assert chart.minimumWidth() == scaled_px(properties_module.FPS_CHART_MIN_WIDTH_PX)
        assert chart.width() >= chart.minimumWidth()
    finally:
        dialog.close()


# --- (c) the paint actually draws the accent ----------------------------------------------------------


def test_painting_the_chart_draws_the_accent_line(tmp_path, qapp):
    dialog = _open(_item(tmp_path))
    try:
        dialog.show()
        qapp.processEvents()
        image = dialog.chart.grab().toImage()
        assert not image.isNull()
        accent = QColor(ThemeManager.get_instance().get_color("accent"))

        drawn = 0
        for y in range(image.height()):
            for x in range(image.width()):
                if _near_accent(image.pixelColor(x, y), accent):
                    drawn += 1
        assert drawn >= 50, (
            f"the accent line must be painted ({drawn} accent pixels found); 'it did not raise' is not a pixel proof"
        )
    finally:
        dialog.close()


# --- (d) an old report says why there is no curve -----------------------------------------------------


def test_a_report_without_a_series_prints_the_reason_and_builds_no_chart(tmp_path):
    dialog = _open(_item(tmp_path, series=None))
    try:
        text = dialog.plain_text()
        assert "Phases (ms/frame)" not in text
        assert "Processing speed: —" in text, "an old report must say the curve is missing, not draw an empty one"
        assert "records no per-frame speed series" in text
        assert dialog.chart is None
        assert "RVE" not in dialog.charts
        # The rest of the report is still read from that old report.
        assert "Quality preset: best" in text
    finally:
        dialog.close()


# --- (e) the averages live in the tooltip -------------------------------------------------------------


def test_the_tooltip_carries_the_phase_averages(tmp_path):
    dialog = _open(_item(tmp_path))
    try:
        tooltip = dialog.chart.toolTip()
        assert "Per-phase cost (ms/frame)" in tooltip
        for key, value in PHASES.items():
            assert f"{key} {value:g}" in tooltip, f"the tooltip must carry {key}={value}, the number the row used to show"
    finally:
        dialog.close()
