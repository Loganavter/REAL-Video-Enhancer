"""The Logs tab filter: what a person reads, and what stays out of the pane.

Every literal in `SAMPLE` was taken from a real backend run (6 s clip, SPAN 4x, tensorrt, the owner's settings) - the
pane showed 106 lines for it, of which the events were a minority. The filter's job is stated as behaviour here: the
per-frame telemetry and the live progress line never reach the console, the events do, warnings/errors keep their level,
and nothing disappears without a counted closing line.
"""

from __future__ import annotations

from src.ui.render_log import (
    LEVEL_ERROR,
    LEVEL_INFO,
    LEVEL_PHASE,
    LEVEL_WARN,
    PHASE_HEADER_WIDTH,
    RenderLogFilter,
    phase_header,
)

#: Lines captured from the real run, in order, with \\r progress chunks already split the way `readline` delivers them.
SAMPLE = [
    "BACKEND: Getting Input Video Properties",
    "BACKEND: Working Directory: /home/jorj/Загрузки/projects/REAL-Video-Enhancer",
    "BACKEND: Handling device: auto, GPU ID: 0",
    "Using Device: NVIDIA GeForce RTX 4060 Ti",
    "BACKEND: Initializing stream for device cuda (GPU ID: 0)",
    "BACKEND: Handling device: cuda, GPU ID: 0",
    "Using Device: NVIDIA GeForce RTX 4060 Ti",
    "BACKEND: Initializing stream for device cuda (GPU ID: 0)",
    "BACKEND: DEBUG after assign: option=139891411680960 int8Fit=int8",
    "BACKEND: CLI Arguments:",
    "['backend/rve-backend.py', '-i', '/tmp/clip6.mp4', '-o', '/tmp/out.mp4', '-b', 'tensorrt']",
    "BACKEND: FFMPEG READ COMMAND: ['/home/jorj/bin/ffmpeg', '-t', '6.000000', '-i', '/tmp/clip6.mp4']",
    "BACKEND: Reference-frame reuse enabled (threshold 5.0/255, statistic coarse, max chain 4, pixel veto 0.0)",
    "FPS: 1119 Current Frame: 1 ETA: 0:00:00",
    "BACKEND: Rendering",
    "host timeline: upload 0.04  upload_sync 0.00  model-call 0.21  frame-ctor 0.00  final-sync 0.49 | total 0.75 ms",
    "host timeline: upload 0.04  upload_sync 0.00  model-call 0.22  frame-ctor 0.00  final-sync 0.49 | total 0.76 ms",
    "fp16 engine GPU: 0.68 ms/frame (last 8); alloc_retries 0",
    "int8 host: 16 launches cost 0.73 ms of HOST (0.046 ms each); whole hop loop 0.92 ms",
    "BACKEND: display: drew source frame 391 named 13.055s (pts 13.055s)",
    "BACKEND: reuse: 0.0% of 250 frames (0 reused), depth up to 0, 0 scene cuts, threshold 5/255",
    "BACKEND: startup: first frame enqueued 2.96 s after process start (interpreter, engine, ffmpeg/NVENC init)",
    "BACKEND: render loop: 0.30 s for 60 frames (4.94 ms/frame), total 3.26 s since process start",
]


def _kept(lines):
    log_filter = RenderLogFilter()
    return [entry.text for entry in log_filter.feed_all(lines)]


def test_per_frame_telemetry_never_reaches_the_pane():
    kept = _kept(SAMPLE)
    assert not any(text.startswith("host timeline:") for text in kept)
    assert not any(text.startswith("fp16 engine GPU:") for text in kept)
    assert not any("BACKEND: reuse: 0.0% of 250 frames" in text for text in kept)


def test_the_shapes_the_backend_prints_today_are_dropped_and_the_event_between_them_survives():
    """The two shapes measured on 2026-11-28: ~9 rows per event, which is why the pane reads as a profiler dump.

    `int8 host:` repeats several times per frame on the int8 path; `display: drew source frame` prints once per
    displayed frame. Both are measurements the run report keeps - the pane keeps the event printed between them.
    """
    log_filter = RenderLogFilter()
    kept = log_filter.feed_all([
        "int8 host: 16 launches cost 0.73 ms of HOST (0.046 ms each); whole hop loop 0.92 ms",
        "BACKEND: engine built in 12.34 seconds.",
        "BACKEND: display: drew source frame 391 named 13.055s (pts 13.055s)",
        "int8 body host: 16 launches cost 0.71 ms of HOST (0.044 ms each); whole hop loop 0.90 ms",
    ])
    assert [entry.text for entry in kept] == ["── BACKEND: engine built in 12.34 seconds. ─────────────────────"]
    assert log_filter.dropped == {"per-frame": 3}, "the counters stay per reason, three per-frame rows"


def test_a_telemetry_line_that_names_a_cause_is_kept():
    """The filter hides numbers, never reasons: a profiler row that says WARN/failed/falling back is a line to read."""
    kept = _kept([
        "int8 host: 16 launches cost 0.73 ms of HOST (0.046 ms each); whole hop loop 0.92 ms "
        "(falling back to the fp16 body)",
    ])
    assert len(kept) == 1 and "falling back to the fp16 body" in kept[0]


def test_the_live_progress_line_is_not_duplicated_into_the_log():
    # The process page already shows FPS, ETA and the bar from the same line.
    assert not any("Current Frame:" in text for text in _kept(SAMPLE))


def test_setup_chatter_is_shown_once_and_debug_leftovers_not_at_all():
    kept = _kept(SAMPLE)
    assert kept.count("Using Device: NVIDIA GeForce RTX 4060 Ti") == 1
    assert sum(text.startswith("BACKEND: Initializing stream") for text in kept) == 1
    assert not any("DEBUG" in text for text in kept)


def test_the_cli_and_ffmpeg_dumps_are_dropped_with_their_continuation():
    kept = _kept(SAMPLE)
    assert not any("CLI Arguments" in text for text in kept)
    assert not any(text.startswith("['backend/rve-backend.py'") for text in kept)
    assert not any("FFMPEG READ COMMAND" in text for text in kept)


def test_the_events_survive_in_order():
    kept = _kept(SAMPLE)
    assert kept[0] == "BACKEND: Getting Input Video Properties"
    # A phase is a header row with the phase in it (the milestone rule), so the phase TEXT is found inside the rule.
    assert any(text == phase_header("BACKEND: Rendering") for text in kept)
    assert phase_header(
        "BACKEND: startup: first frame enqueued 2.96 s after process start (interpreter, engine, ffmpeg/NVENC init)"
    ) in kept
    assert kept[-1].startswith("── BACKEND: render loop:")


def test_levels_drive_the_colour_the_host_paints():
    log_filter = RenderLogFilter()
    error = log_filter.feed("Traceback (most recent call last):")[0]
    warn = log_filter.feed("BACKEND: int8 probe unavailable (invalid literal for int()): int8 stays off")[0]
    phase = log_filter.feed("BACKEND: Rendering")[0]
    info = log_filter.feed("BACKEND: Total Output Frames: 60")[0]
    assert error.level == LEVEL_ERROR
    assert warn.level == LEVEL_WARN
    assert phase.level == LEVEL_PHASE
    assert info.level == LEVEL_INFO


def test_nothing_is_dropped_without_a_counted_closing_line():
    log_filter = RenderLogFilter()
    log_filter.feed_all(SAMPLE)
    summary = log_filter.hidden_summary()
    assert summary is not None
    assert summary.level == LEVEL_INFO
    assert "telemetry lines hidden" in summary.text
    assert f"{log_filter.hidden_total()}" in summary.text

    clean = RenderLogFilter()
    clean.feed("BACKEND: Rendering")
    assert clean.hidden_summary() is None, "a quiet render gets no summary line"


def test_a_blank_line_and_the_bare_backend_prefix_do_not_leave_empty_rows():
    kept = _kept(["", "   ", "BACKEND:", "BACKEND: Rendering"])
    assert kept == [phase_header("BACKEND: Rendering")]


def test_a_phase_renders_as_one_rule_row_with_the_phase_in_it():
    """The milestone treatment: box-drawing rule + the phase on ONE row, painted bold accent by LEVEL_PHASE.

    A separate rule line above the phase was the owner's first sketch ("====== ... хотя наверное лучше было бы
    по-другому придумать"); one row means a cluster of phases costs no rows that say nothing.
    """
    header = phase_header("BACKEND: Rendering")
    assert header.startswith("── BACKEND: Rendering ─") and header.endswith("──")
    assert len(header) == PHASE_HEADER_WIDTH, "a short phase keeps one fixed width, so the pane reads as a block"
    long_text = "BACKEND: Building TensorRT engine dncnn_color_blind.pth fp16 (last 3 builds: ~4 min 10 s)"
    long_header = phase_header(long_text)
    assert long_header.startswith("── ") and long_header.endswith("──")
    assert long_text in long_header, "a long phase is not truncated"
    assert long_header.count("─") == 4, "no filler rule when the text already fills the row"
