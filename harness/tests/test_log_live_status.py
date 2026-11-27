"""The Logs tab's human-facing extras: the once-a-second status row, and the engine-build estimate.

Two rules are pinned here, both pure Python except where the page's own tick is exercised through `object.__new__`
(the house style for tab-level tests):

* :class:`LiveStatus` writes at most one line per second, from the numbers the tick already carries, with the clock
  passed in - so the cadence is asserted without sleeping. The page replaces its previous status row instead of
  appending a new one: a 30-minute render leaves ONE row, not 1800.
* :func:`annotate_build_line` names the median of the builds this machine actually timed (GPU + precision + model)
  and says "first build on this GPU" when there is none. The backend reports its own duration, the injected clock is
  the fallback, and the history is a small json beside `settings.txt`.
"""

from __future__ import annotations

import json
import sys
import time
import types
from types import SimpleNamespace

import pytest

from src.ui.render_log import (
    LEVEL_STATUS,
    BUILD_HISTORY_CAP,
    BuildHistory,
    LiveStatus,
    annotate_build_line,
    build_hint,
    build_slot,
    engine_build_history_path,
    format_duration,
    estimate_suffix,
    monitor_available,
    progress_eta,
    status_text,
)

#: A real engine name as the backend prints it (`UpscaleTorch` composes model + precision + device + versions).
ENGINE = ("dncnn_color_blind.pth_fp16_NVIDIA GeForce RTX 4060 Ti_trt-10.8.0.43_torch_tensorrt-2.9.0"
          "_opt-5.engine")
START_LINE = f"BACKEND: Building TensorRT engine {ENGINE}. This may take a while..."
DONE_LINE = "BACKEND: TensorRT engine built in 250.0 seconds."
GPU = "NVIDIA GeForce RTX 4060 Ti"


# --- the curated status line --------------------------------------------------------------------------------------


def test_the_status_line_is_the_owners_shape_from_the_ticks_numbers():
    assert status_text(fps=0.53, eta="0:12:10", frame=120, total=493) == (
        "1.9 s/frame · 0.53 fps · ETA 0:12:10 · frame 120/493")


def test_a_fast_render_states_fps_and_not_its_noise_inverse():
    """29 fps is `29.3 fps`; `0.03 s/frame` beside it would be a number a person has to divide out."""
    assert status_text(fps=29.34, eta="0:01:00", frame=120, total=493) == "29.3 fps · ETA 0:01:00 · frame 120/493"


def test_nothing_has_rendered_yet_means_no_status_line():
    assert status_text(fps=0.0, eta="", frame=0, total=493) == ""


def test_the_cadence_is_at_most_one_line_per_second_with_the_clock_passed_in():
    live = LiveStatus(interval=1.0)
    first = live.maybe_line(0.0, fps=0.53, eta="0:12:10", frame=120, total=493)
    assert first is not None and first.level == LEVEL_STATUS
    assert first.text == "1.9 s/frame · 0.53 fps · ETA 0:12:10 · frame 120/493"
    assert live.maybe_line(0.5, fps=0.53, eta="0:12:09", frame=121, total=493) is None, "two per second"
    second = live.maybe_line(1.0, fps=0.53, eta="0:12:08", frame=122, total=493)
    assert second is not None and "frame 122/493" in second.text and "ETA 0:12:08" in second.text


def test_a_new_job_may_write_its_first_status_at_once():
    live = LiveStatus(interval=1.0)
    assert live.maybe_line(100.0, fps=1.0, eta="0:00:10", frame=1, total=10) is not None
    assert live.maybe_line(100.2, fps=1.0, eta="0:00:10", frame=1, total=10) is None
    live.reset()
    assert live.maybe_line(100.2, fps=1.0, eta="0:00:10", frame=1, total=10) is not None


# --- the page keeps exactly ONE status row ---------------------------------------------------------------------


@pytest.fixture()
def stub_tab():
    """`ProcessTab._append_status_line` over a tab built with `object.__new__`: the shipped method, no window."""
    for name, attrs in (("cpuinfo", {"get_cpu_info": lambda *a, **k: {}}),
                        ("pypresence", {"Presence": object})):
        try:  # pragma: no cover - the branch depends on the interpreter
            __import__(name)
        except ModuleNotFoundError:  # pragma: no cover
            module = types.ModuleType(name)
            for attr, value in attrs.items():
                setattr(module, attr, value)
            sys.modules[name] = module
    if "resources_rc" not in sys.modules:
        sys.modules["resources_rc"] = types.ModuleType("resources_rc")

    from src.ui.ProcessTab import ProcessTab

    tab = object.__new__(ProcessTab)
    tab.renderLogLines = []
    tab._render_started_mono = time.monotonic()
    tab.fps = 0
    tab.fps_value = 0.53
    tab.eta = "0:12:10"
    tab.currentFrame = 120
    tab.max_value = 493
    return tab


def test_the_tick_replaces_the_previous_status_row_instead_of_stacking(monkeypatch, stub_tab):
    from src.ui import ProcessTab as process_tab_module

    now = {"t": 0.0}
    monkeypatch.setattr(process_tab_module.time, "monotonic", lambda: now["t"])
    from src.ui.render_log import LogLine

    stub_tab.renderLogLines.append(LogLine("BACKEND: Rendering"))
    stub_tab._append_status_line()
    now["t"] = 0.4
    stub_tab._append_status_line()
    assert sum(entry.level == LEVEL_STATUS for entry in stub_tab.renderLogLines) == 1, "two rows in one second"

    now["t"] = 1.0
    stub_tab.currentFrame = 121
    stub_tab._append_status_line()
    status_rows = [entry for entry in stub_tab.renderLogLines if entry.level == LEVEL_STATUS]
    assert len(status_rows) == 1, "the previous second's row was not replaced"
    assert "frame 121/493" in status_rows[0].text
    assert stub_tab.renderLogLines[0].text == "BACKEND: Rendering", "the event above the status row survived"


def test_the_console_shows_one_status_row_above_the_events(qapp, monkeypatch, stub_tab):
    """The real toolkit console at the end of the chain: one status row, the events above it untouched.

    `syncRenderConsole` appends what is new and rebuilds when the model was rewritten; replacing the trailing status
    row is a rewrite of one row, so the console must end up with exactly one status line under the events.
    """
    from sli_ui_toolkit.ui.widgets.composite.console_view import ConsoleView

    from src.ui import ProcessTab as process_tab_module
    from src.ui.render_log import LogLine

    console = ConsoleView()
    console.set_max_lines(4000)
    stub_tab.page_handles = SimpleNamespace(logs=SimpleNamespace(renderOutput=console))
    stub_tab.renderLogLines.append(LogLine("BACKEND: Rendering"))
    now = {"t": 0.0}
    monkeypatch.setattr(process_tab_module.time, "monotonic", lambda: now["t"])

    stub_tab._append_status_line()
    stub_tab.syncRenderConsole()
    now["t"] = 1.0
    stub_tab.currentFrame = 121
    stub_tab._append_status_line()
    stub_tab.syncRenderConsole()

    texts = [line.text for line in console.lines()]
    assert texts[0] == "BACKEND: Rendering"
    status = [text for text in texts if "fps" in text]
    assert len(status) == 1, texts
    assert "frame 121/493" in status[0], status


def test_the_status_row_stops_with_the_job(monkeypatch, stub_tab):
    from src.ui import ProcessTab as process_tab_module

    now = {"t": 0.0}
    monkeypatch.setattr(process_tab_module.time, "monotonic", lambda: now["t"])
    stub_tab._append_status_line()
    assert len(stub_tab.renderLogLines) == 1
    stub_tab._render_started_mono = None
    now["t"] = 5.0
    stub_tab.currentFrame = 400
    stub_tab._append_status_line()
    assert len(stub_tab.renderLogLines) == 1, "a finished job kept writing status rows"


# --- the engine-build estimate ------------------------------------------------------------------------------------


def test_the_engine_name_names_the_slot_and_the_human_label():
    slot = build_slot(ENGINE)
    assert (slot.gpu, slot.precision, slot.model) == (GPU, "fp16", "dncnn_color_blind.pth")
    assert slot.label == "dncnn_color_blind.pth fp16"
    # The device line wins when the page has it: the file name may be sanitised by a file system.
    assert build_slot(ENGINE, gpu="NVIDIA GeForce RTX 5090").gpu == "NVIDIA GeForce RTX 5090"


def test_no_history_says_so_plainly():
    assert estimate_suffix([]) == " (first build on this GPU - no estimate yet)"
    assert format_duration(12) == "12 s" and format_duration(250) == "4 min 10 s"


def test_the_page_annotates_the_build_lines_and_records_the_finish(tmp_path, stub_tab):
    """The reader-thread entry point: only build/load lines are touched, and the finish lands in the slot the start
    opened (the finished line itself names no engine)."""
    stub_tab._build_history = BuildHistory.load(tmp_path / "engine-build-times.json")
    stub_tab._gpu_name = GPU
    assert stub_tab._annotate_build_line(START_LINE).endswith("(first build on this GPU - no estimate yet)")
    assert stub_tab._annotate_build_line(DONE_LINE) == DONE_LINE
    assert stub_tab._build_history.durations(build_slot(ENGINE, gpu=GPU).key) == [250.0]
    assert stub_tab._annotate_build_line("BACKEND: Rendering") == "BACKEND: Rendering"


# --- the three tiers: cache hit / builder progress / recorded history -------------------------------------------


def test_a_cache_hit_says_loading_cached_engine_instead_of_an_eta(tmp_path):
    """The engine file exists -> the run loads it. The backend's own line already tells which case this is."""
    history = BuildHistory.load(tmp_path / "engine-build-times.json")
    line = "BACKEND: Loading TensorRT engine from /models/dncnn/dncnn_color_blind.pth_fp16.engine."
    annotated = annotate_build_line(line, history, now=0.0)
    assert annotated == line + " (loading cached engine - no build to wait for)"
    # A cached load must not open a slot: the later `built in` line of some other build cannot close it.
    annotate_build_line(DONE_LINE, history, now=10.0)
    assert history.history == {}
    plan = "BACKEND: Loading raw TensorRT plan from /models/x.plan."
    assert annotate_build_line(plan, history, now=10.0).endswith("(loading cached engine - no build to wait for)")


def test_the_builder_progress_answers_with_its_own_fraction():
    """TensorRT's monitor reports 0..1; elapsed/p*(1-p) is the remaining time from the same clock the builder uses."""
    assert progress_eta(100.0, 0.25) == pytest.approx(300.0)
    assert progress_eta(100.0, 1.0) == pytest.approx(0.0)
    assert progress_eta(100.0, 0.0) is None, "a build that has not started has no rate to extrapolate"
    assert progress_eta(100.0, None) is None
    assert build_hint(fraction=0.42, elapsed=100.0) == " (42% built, ~2 min 18 s left)"


def test_the_fraction_wins_over_the_history_and_the_cache_wins_over_both(tmp_path):
    history = BuildHistory.load(tmp_path / "engine-build-times.json")
    key = build_slot(ENGINE, gpu=GPU).key
    history.note_start(key, 0.0)
    history.note_finished(0.0, seconds=250.0)

    annotated = annotate_build_line(START_LINE, history, now=1000.0, gpu=GPU, fraction=0.5, elapsed=60.0)
    assert annotated.endswith("(50% built, ~1 min 00 s left)"), annotated
    assert build_hint(cached=True, fraction=0.5, elapsed=60.0, durations=[250.0]) == (
        " (loading cached engine - no build to wait for)")


def test_the_monitor_probe_is_a_feature_detection(monkeypatch):
    """`hasattr(trt, "IProgressMonitor")`, asked of the module the CALLER holds - this process imports no TensorRT."""
    fake = SimpleNamespace(IProgressMonitor=object, ProgressMonitorPhase=object)
    assert monitor_available(fake) is True
    assert monitor_available(SimpleNamespace()) is False
    assert monitor_available() is False, "the app process has no TensorRT and must not pretend to have one"


def test_the_finished_build_is_recorded_and_the_median_is_what_the_start_line_names(tmp_path):
    path = tmp_path / "engine-build-times.json"
    history = BuildHistory.load(path)

    first = annotate_build_line(START_LINE, history, now=0.0, gpu=GPU)
    assert first == f"BACKEND: Building TensorRT engine dncnn_color_blind.pth fp16 (first build on this GPU - no estimate yet)"
    annotate_build_line(DONE_LINE, history, now=250.0)
    assert history.durations(build_slot(ENGINE, gpu=GPU).key) == [250.0]

    # A second build reads the FIRST one's duration (the backend's own number is preferred over the clock).
    second = annotate_build_line(START_LINE, history, now=1000.0, gpu=GPU)
    assert second.endswith("(last 1 build: ~4 min 10 s)"), second
    annotate_build_line("BACKEND: TensorRT engine built in 240.0 seconds.", history, now=1240.0)
    third = annotate_build_line(START_LINE, history, now=2000.0, gpu=GPU)
    assert third.endswith("(last 2 builds: ~4 min 05 s)"), third


def test_the_history_is_a_json_beside_the_settings_and_survives_a_reload(tmp_path):
    settings_file = tmp_path / "settings.txt"
    settings_file.write_text("", encoding="utf-8")
    path = engine_build_history_path(str(settings_file))
    assert path.name == "engine-build-times.json" and path.parent == tmp_path

    history = BuildHistory.load(path)
    annotate_build_line(START_LINE, history, now=0.0, gpu=GPU)
    annotate_build_line(DONE_LINE, history, now=250.0)
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["version"] == 1
    assert document["builds"][build_slot(ENGINE, gpu=GPU).key] == [250.0]

    reloaded = BuildHistory.load(path)
    assert reloaded.median(build_slot(ENGINE, gpu=GPU).key) == pytest.approx(250.0)


def test_a_different_gpu_or_precision_is_a_different_slot(tmp_path):
    history = BuildHistory.load(tmp_path / "engine-build-times.json")
    annotate_build_line(START_LINE, history, now=0.0, gpu=GPU)
    annotate_build_line(DONE_LINE, history, now=250.0)
    other = annotate_build_line(START_LINE, history, now=500.0, gpu="NVIDIA GeForce RTX 5090")
    assert other.endswith("(first build on this GPU - no estimate yet)"), other


def test_the_history_is_capped_per_slot(tmp_path):
    history = BuildHistory.load(tmp_path / "engine-build-times.json")
    key = build_slot(ENGINE, gpu=GPU).key
    for index in range(BUILD_HISTORY_CAP + 3):
        history.note_start(key, 0.0)
        history.note_finished(index, seconds=10.0 + index)
    assert len(history.durations(key)) == BUILD_HISTORY_CAP
    assert history.durations(key)[-1] == 10.0 + BUILD_HISTORY_CAP + 2


def test_a_broken_history_file_is_an_empty_history_not_a_crash(tmp_path):
    path = tmp_path / "engine-build-times.json"
    path.write_text("{not json", encoding="utf-8")
    history = BuildHistory.load(path)
    assert history.durations("anything") == []
    assert annotate_build_line(START_LINE, history, now=0.0).endswith("(first build on this GPU - no estimate yet)")


def test_a_finished_line_without_a_start_is_not_attributed_to_a_slot(tmp_path):
    """The finished line names no engine; without a recorded start there is nothing to time, and nothing is invented."""
    history = BuildHistory.load(tmp_path / "engine-build-times.json")
    assert annotate_build_line(DONE_LINE, history, now=10.0) == DONE_LINE
    assert history.history == {}


def test_a_line_that_is_not_a_build_line_is_untouched(tmp_path):
    """Only the two build shapes are touched; every other line goes to the filter byte-identical."""
    history = BuildHistory.load(tmp_path / "engine-build-times.json")
    line = "BACKEND: Rendering"
    assert annotate_build_line(line, history, now=0.0) == line
