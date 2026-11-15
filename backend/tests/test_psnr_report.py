"""Tests for ``backend/tests/tools/psnr_report.py``.

The tool exists to stop two silent measurement bugs (a comparison that crossed an unstated
colour conversion, and an average over files with different frame counts), so the tests are
written around those two: the parser and the statistics are plain functions over text, and
the comparison path takes an injected frame counter and an injected runner, which means
every check below - including the mismatch refusal - runs with no ffmpeg, no torch and no
display. One end-to-end case drives ffmpeg when it is installed: 2 frames of 64x64, a CPU
encoder, skipped rather than failed when either is missing.
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parent / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import psnr_report as PR  # noqa: E402

# The exact shape ffmpeg writes into ``stats_file``, one line per paired frame.
STATS_THREE = (
    "n:1 mse_avg:0.11 psnr_y:51.70 psnr_u:55.80 psnr_v:56.37 psnr_avg:52.73\n"
    "n:2 mse_avg:1.30 psnr_y:46.90 psnr_u:52.10 psnr_v:53.40 psnr_avg:47.12\n"
    "n:3 mse_avg:0.60 psnr_y:49.20 psnr_u:54.00 psnr_v:54.80 psnr_avg:49.90\n"
)


def records(*values: float) -> list[PR.FrameStats]:
    """Frame records with consecutive indices, as ffmpeg emits them."""
    return [PR.FrameStats(index=i, psnr_avg=value)
            for i, value in enumerate(values, start=1)]


def fake_runner(text: str):
    """A runner that writes ``text`` where the command says the stats file belongs.

    Reading the path back out of the filtergraph is deliberate: it also proves
    ``build_command`` puts the stats file where ffmpeg will look for it.
    """

    def run(command):
        graph = command[command.index("-lavfi") + 1]
        match = re.search(r"stats_file='(.*?)'", graph)
        assert match, f"no stats_file in the filtergraph: {graph}"
        Path(match.group(1)).write_text(text, encoding="utf-8")

    return run


def fake_counter(counts: dict[str, int]):
    """A frame counter over a mapping, so no ffmpeg is needed to test the guard."""
    return lambda path: counts[path]


# ------------------------------------------------------------------------------------
# stats-file parsing
# ------------------------------------------------------------------------------------
def test_parse_stats_line_reads_every_field():
    line = "n:1 mse_avg:0.11 psnr_y:51.70 psnr_u:55.80 psnr_v:56.37 psnr_avg:52.73"
    frame = PR.parse_stats_line(line)

    assert frame.index == 1
    assert frame.psnr_avg == pytest.approx(52.73)
    assert frame.psnr_y == pytest.approx(51.70)
    assert frame.psnr_u == pytest.approx(55.80)
    assert frame.psnr_v == pytest.approx(56.37)
    assert frame.mse_avg == pytest.approx(0.11)


def test_parse_stats_line_field_order_does_not_matter():
    reordered = "psnr_avg:52.73 psnr_y:51.70 n:7 mse_avg:0.11"
    frame = PR.parse_stats_line(reordered)

    assert frame.index == 7
    assert frame.psnr_avg == pytest.approx(52.73)
    assert frame.psnr_u is None          # absent rather than guessed


def test_parse_stats_line_accepts_an_infinite_average():
    """A bit-identical frame has mse 0, which ffmpeg prints as ``psnr_avg:inf``."""
    frame = PR.parse_stats_line("n:4 mse_avg:0.00 psnr_avg:inf")

    assert frame.psnr_avg == float("inf")


@pytest.mark.parametrize("line", [
    "garbage",                                     # no fields at all
    "n:1 psnr_y:51.70",                            # no psnr_avg
    "psnr_avg:52.73",                              # no frame index
    "n:two psnr_avg:52.73",                        # index is not an integer
    "n:1 psnr_avg:high",                           # average is not a number
    "n:1 psnr_avg:nan",                            # a NaN would pass every threshold check
])
def test_parse_stats_line_rejects_malformed_input(line):
    with pytest.raises(PR.StatsParseError):
        PR.parse_stats_line(line)


def test_parse_stats_file_names_the_offending_line():
    text = "n:1 psnr_avg:52.73\nnot a record\nn:3 psnr_avg:50.00\n"
    with pytest.raises(PR.StatsParseError) as excinfo:
        PR.parse_stats_file(text)

    assert "line 2" in str(excinfo.value)


def test_parse_stats_file_handles_an_empty_file():
    assert PR.parse_stats_file("") == []
    assert PR.parse_stats_file("\n   \n") == []


def test_parse_stats_file_skips_blank_lines_and_reads_the_rest():
    frames = PR.parse_stats_file(STATS_THREE + "\n")

    assert [frame.index for frame in frames] == [1, 2, 3]
    assert [frame.psnr_avg for frame in frames] == pytest.approx([52.73, 47.12, 49.90])


# ------------------------------------------------------------------------------------
# summary statistics
# ------------------------------------------------------------------------------------
def test_summarize_known_list():
    summary = PR.summarize(records(10.0, 20.0, 30.0, 40.0, 50.0), worst_n=3)

    assert summary.frames == 5
    assert summary.mean == pytest.approx(30.0)
    assert summary.median == pytest.approx(30.0)
    assert summary.worst == pytest.approx(10.0)
    assert [(f.index, f.psnr_avg) for f in summary.worst_frames] == [
        (1, 10.0), (2, 20.0), (3, 30.0)]


def test_summarize_reports_the_indices_of_the_worst_frames():
    # worst 3 of a shuffled list: the values matter, the write order must not
    summary = PR.summarize(records(55.0, 44.0, 61.0, 48.5, 59.0), worst_n=3)

    assert summary.worst == pytest.approx(44.0)
    assert [f.index for f in summary.worst_frames] == [2, 4, 1]
    assert summary.median == pytest.approx(55.0)


def test_identical_frames_are_counted_not_averaged():
    """A gated render re-emits frames byte for byte; their per-frame PSNR is infinite.

    Averaging them in makes the mean infinite and the tool useless on exactly the
    configurations this repo ships (measured on 502 anime frames: 18 identical, naive mean
    `inf`, per-frame mean 52.81 dB, ffmpeg's own "average" 52.73 dB).
    """
    peak_squared = 1023.0 ** 2
    frames = [
        PR.FrameStats(index=1, psnr_avg=50.0, mse_avg=peak_squared / 10.0 ** 5),
        PR.FrameStats(index=2, psnr_avg=float("inf"), mse_avg=0.0),
        PR.FrameStats(index=3, psnr_avg=40.0, mse_avg=peak_squared / 10.0 ** 4),
    ]
    summary = PR.summarize(frames, worst_n=3)

    assert summary.identical == 1
    assert summary.worst == pytest.approx(40.0)      # the identical frame is not the worst
    assert summary.median == pytest.approx(45.0)
    assert summary.mean == pytest.approx(45.0)       # (50 + 40) / 2, not inf
    # `average` is the PSNR of the mean MSE, which is a different (and finite) aggregate
    mean_mse = (peak_squared / 10.0 ** 5 + 0.0 + peak_squared / 10.0 ** 4) / 3
    assert summary.mean_over_mse == pytest.approx(10.0 * math.log10(peak_squared / mean_mse))


def test_every_frame_identical_is_reported_as_infinite():
    frames = [PR.FrameStats(index=i, psnr_avg=float("inf"), mse_avg=0.0) for i in (1, 2, 3)]
    summary = PR.summarize(frames)
    assert summary.identical == 3
    assert math.isinf(summary.mean)
    assert math.isinf(summary.mean_over_mse)
    assert summary.worst_frames == ()


def test_summarize_single_frame():
    summary = PR.summarize(records(51.5))

    assert summary.frames == 1
    assert summary.mean == summary.median == summary.worst == pytest.approx(51.5)
    assert len(summary.worst_frames) == 1
    assert summary.worst_frames[0].index == 1


def test_summarize_all_equal_picks_the_lowest_indices():
    summary = PR.summarize(records(40.0, 40.0, 40.0, 40.0, 40.0), worst_n=3)

    assert summary.mean == summary.median == summary.worst == pytest.approx(40.0)
    assert [f.index for f in summary.worst_frames] == [1, 2, 3]


def test_summarize_worst_n_is_clamped_and_zero_means_no_list():
    five = records(1.0, 2.0, 3.0, 4.0, 5.0)

    assert len(PR.summarize(five, worst_n=10).worst_frames) == 5
    assert PR.summarize(five, worst_n=0).worst_frames == ()


def test_summarize_refuses_an_empty_list():
    with pytest.raises(PR.PsnrError):
        PR.summarize([])


# ------------------------------------------------------------------------------------
# frame counts
# ------------------------------------------------------------------------------------
def test_frame_counts_agree_returns_the_count():
    assert PR.check_frame_counts(94, 94, "ref.mkv", "cand.mkv") == 94


def test_frame_count_mismatch_is_an_error_naming_both_counts():
    with pytest.raises(PR.FrameCountMismatch) as excinfo:
        PR.check_frame_counts(94, 90, "ref.mkv", "cand.mkv")

    message = str(excinfo.value)
    assert "94" in message and "90" in message
    assert "ref.mkv" in message and "cand.mkv" in message


def test_frame_count_of_zero_is_an_error():
    with pytest.raises(PR.PsnrError):
        PR.check_frame_counts(0, 0)


# ------------------------------------------------------------------------------------
# the ffmpeg command
# ------------------------------------------------------------------------------------
def test_build_command_converts_both_inputs_to_the_same_format():
    command = PR.build_command("ffmpeg", "ref.mkv", "cand.mkv", "yuv420p10le", "/x/s.log")

    assert command[0] == "ffmpeg"
    graph = command[command.index("-lavfi") + 1]
    assert graph.count("format=yuv420p10le") == 2
    assert "[0:v]format=yuv420p10le[ref]" in graph
    assert "[1:v]format=yuv420p10le[dist]" in graph
    assert "psnr=stats_file='/x/s.log'" in graph
    assert command[-1] == "-" and command[-3:] == ["-f", "null", "-"]


def test_build_command_quotes_a_stats_path_with_spaces():
    command = PR.build_command("ffmpeg", "a", "b", "yuv420p", "/tmp/my stats/s.log")
    graph = command[command.index("-lavfi") + 1]

    assert "stats_file='/tmp/my stats/s.log'" in graph


def test_build_command_rejects_an_unsupported_format():
    with pytest.raises(PR.PsnrError):
        PR.build_command("ffmpeg", "a", "b", "yuv444p", "/x/s.log")


# ------------------------------------------------------------------------------------
# the comparison, with the two side effects injected
# ------------------------------------------------------------------------------------
def test_compare_summarises_a_complete_run(tmp_path):
    stats = tmp_path / "stats.log"
    report = PR.compare("ref.mkv", "cand.mkv", ffmpeg="ffmpeg",
                        counter=fake_counter({"ref.mkv": 3, "cand.mkv": 3}),
                        keep_stats=str(stats), runner=fake_runner(STATS_THREE))

    assert report.pixel_format == PR.DEFAULT_FORMAT == "yuv420p10le"
    assert report.summary.frames == 3
    assert report.summary.worst == pytest.approx(47.12)
    assert report.summary.worst_frames[0].index == 2
    assert stats.is_file() and report.keep_stats is True
    assert report.stats_path == str(stats)


def test_compare_refuses_a_frame_count_mismatch(tmp_path):
    called = []
    report_args = dict(ffmpeg="ffmpeg", keep_stats=str(tmp_path / "s.log"))

    with pytest.raises(PR.FrameCountMismatch):
        PR.compare("ref.mkv", "cand.mkv",
                   counter=fake_counter({"ref.mkv": 94, "cand.mkv": 90}),
                   runner=lambda command: called.append(command), **report_args)

    assert called == [], "ffmpeg must not run on a mismatched pair"


def test_compare_refuses_a_truncated_stats_file(tmp_path):
    with pytest.raises(PR.PsnrError) as excinfo:
        PR.compare("ref.mkv", "cand.mkv", ffmpeg="ffmpeg",
                   counter=fake_counter({"ref.mkv": 3, "cand.mkv": 3}),
                   keep_stats=str(tmp_path / "s.log"),
                   runner=fake_runner("n:1 psnr_avg:52.73\n"))

    assert "1 frame" in str(excinfo.value)


def test_compare_refuses_gaps_in_the_frame_indices(tmp_path):
    text = "n:1 psnr_avg:52.73\nn:3 psnr_avg:50.00\n"
    with pytest.raises(PR.PsnrError) as excinfo:
        PR.compare("ref.mkv", "cand.mkv", ffmpeg="ffmpeg",
                   counter=fake_counter({"ref.mkv": 2, "cand.mkv": 2}),
                   keep_stats=str(tmp_path / "s.log"), runner=fake_runner(text))

    assert "not 1..2" in str(excinfo.value)


def test_compare_leaves_no_stats_file_behind_by_default():
    report = PR.compare("ref.mkv", "cand.mkv", ffmpeg="ffmpeg",
                        counter=fake_counter({"ref.mkv": 3, "cand.mkv": 3}),
                        runner=fake_runner(STATS_THREE))

    assert report.stats_path is None
    assert report.keep_stats is False


# ------------------------------------------------------------------------------------
# output
# ------------------------------------------------------------------------------------
def _report(tmp_path) -> PR.Report:
    """A report assembled from a fixture, with no ffmpeg and no injection points."""
    stats = tmp_path / "s.log"
    stats.write_text(STATS_THREE, encoding="utf-8")
    frames = PR.parse_stats_file(stats.read_text(encoding="utf-8"))
    return PR.Report(
        reference="ref.mkv",
        candidate="cand.mkv",
        pixel_format=PR.DEFAULT_FORMAT,
        summary=PR.summarize(frames, PR.DEFAULT_WORST),
        command=tuple(PR.build_command("ffmpeg", "ref.mkv", "cand.mkv",
                                       PR.DEFAULT_FORMAT, str(stats))),
        stats_path=str(stats),
        keep_stats=True,
    )


def test_format_report_states_the_format_and_the_command(tmp_path):
    text = PR.format_report(_report(tmp_path))

    assert "format    : yuv420p10le" in text
    assert "frames    : 3 (both inputs)" in text
    assert "frame 2 " in text                     # the worst frame, with its index
    assert "mean      : 49.92 dB" in text
    assert "psnr=stats_file=" in text             # the command: the number is reproducible
    assert "verdict" not in text                  # no --fail-under, no verdict


def test_format_report_gives_a_verdict(tmp_path):
    report = _report(tmp_path)

    passed = PR.format_report(report, fail_under=40.0)
    failed = PR.format_report(report, fail_under=60.0)

    assert "verdict   : PASS (mean 49.92 dB >= 40 dB)" in passed
    assert "verdict   : FAIL (mean 49.92 dB < 60 dB)" in failed


def test_json_payload_carries_the_same_facts(tmp_path):
    payload = PR.build_payload(_report(tmp_path), fail_under=60.0)

    assert payload["format"] == "yuv420p10le"
    assert payload["frames"] == 3
    assert payload["mean"] == pytest.approx(49.9166666, rel=1e-6)
    assert payload["worst_frames"][0] == {"index": 2, "psnr_avg": 47.12}
    assert payload["passed"] is False
    assert payload["command"][0] == "ffmpeg"


# ------------------------------------------------------------------------------------
# argument parsing
# ------------------------------------------------------------------------------------
def test_parse_args_defaults():
    args = PR.parse_args(["ref.mkv", "cand.mkv"])

    assert args.reference == "ref.mkv" and args.candidate == "cand.mkv"
    assert args.format == "yuv420p10le"
    assert args.worst == 3
    assert args.fail_under is None
    assert args.json is False and args.keep_stats is None


def test_parse_args_accepts_the_documented_flags():
    args = PR.parse_args(["ref.mkv", "cand.mkv", "--ffmpeg", "/opt/ffmpeg",
                          "--ffprobe", "/opt/ffprobe", "--format", "rgb24",
                          "--fail-under", "48.2", "--worst", "5", "--json",
                          "--keep-stats", "/tmp/stats.log"])

    assert args.ffmpeg == "/opt/ffmpeg" and args.ffprobe == "/opt/ffprobe"
    assert args.format == "rgb24"
    assert args.fail_under == pytest.approx(48.2)
    assert args.worst == 5
    assert args.json is True
    assert args.keep_stats == "/tmp/stats.log"


def test_parse_args_rejects_an_unknown_format():
    with pytest.raises(SystemExit) as excinfo:
        PR.parse_args(["ref.mkv", "cand.mkv", "--format", "yuv444p"])

    assert excinfo.value.code == 2


def test_parse_args_rejects_a_negative_worst_count():
    with pytest.raises(SystemExit) as excinfo:
        PR.parse_args(["ref.mkv", "cand.mkv", "--worst", "-1"])

    assert excinfo.value.code == 2


# ------------------------------------------------------------------------------------
# exit codes (the CLI wiring, with the comparison replaced)
# ------------------------------------------------------------------------------------
def test_main_returns_zero_and_prints_the_report(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(PR, "default_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(PR, "default_ffprobe", lambda ffmpeg: "ffprobe")
    monkeypatch.setattr(PR, "ffprobe_frame_count", lambda ffprobe, path: 3)
    monkeypatch.setattr(PR, "compare",
                        lambda *a, **kw: _report(tmp_path))

    code = PR.main(["ref.mkv", "cand.mkv"])

    assert code == 0
    assert "format    : yuv420p10le" in capsys.readouterr().out


def test_main_exits_one_below_the_threshold(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(PR, "default_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(PR, "default_ffprobe", lambda ffmpeg: "ffprobe")
    monkeypatch.setattr(PR, "compare", lambda *a, **kw: _report(tmp_path))

    assert PR.main(["ref.mkv", "cand.mkv", "--fail-under", "40"]) == 0
    assert "verdict   : PASS" in capsys.readouterr().out
    assert PR.main(["ref.mkv", "cand.mkv", "--fail-under", "60"]) == 1
    assert "verdict   : FAIL" in capsys.readouterr().out


def test_main_exits_two_on_a_refused_comparison(monkeypatch, capsys):
    monkeypatch.setattr(PR, "default_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(PR, "default_ffprobe", lambda ffmpeg: "ffprobe")

    def refuse(*args, **kwargs):
        raise PR.FrameCountMismatch("frame count mismatch: 94 vs 90")

    monkeypatch.setattr(PR, "compare", refuse)

    code = PR.main(["ref.mkv", "cand.mkv"])

    assert code == 2
    stderr = capsys.readouterr().err
    assert "error:" in stderr and "94 vs 90" in stderr


def test_main_json_mode_prints_a_parseable_payload(tmp_path, monkeypatch, capsys):
    import json

    monkeypatch.setattr(PR, "default_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(PR, "default_ffprobe", lambda ffmpeg: "ffprobe")
    monkeypatch.setattr(PR, "compare", lambda *a, **kw: _report(tmp_path))

    assert PR.main(["ref.mkv", "cand.mkv", "--json", "--fail-under", "60"]) == 1
    payload = json.loads(capsys.readouterr().out)

    assert payload["frames"] == 3 and payload["passed"] is False


# ------------------------------------------------------------------------------------
# end to end: a 2-frame 64x64 clip, CPU encoder only
# ------------------------------------------------------------------------------------
def _ffmpeg_tools():
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip("ffmpeg/ffprobe not installed")
    return ffmpeg, ffprobe


def _make_clip(ffmpeg: str, path: Path, source: str) -> None:
    """Two frames of 64x64 from a lavfi source, encoded with a CPU codec (no GPU here)."""
    command = [ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-y",
               "-f", "lavfi", "-i", f"{source}=size=64x64:rate=10:duration=0.2",
               "-c:v", "ffv1", "-pix_fmt", "yuv420p10le", str(path)]
    proc = subprocess.run(command, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip(f"cannot encode with the local ffmpeg: {proc.stderr.strip()[-200:]}")


def test_compare_end_to_end_on_a_tiny_clip(tmp_path):
    ffmpeg, ffprobe = _ffmpeg_tools()
    reference, candidate = tmp_path / "ref.mkv", tmp_path / "cand.mkv"
    _make_clip(ffmpeg, reference, "testsrc")
    _make_clip(ffmpeg, candidate, "testsrc2")

    report = PR.compare(str(reference), str(candidate), ffmpeg=ffmpeg,
                        counter=lambda path: PR.ffprobe_frame_count(ffprobe, path),
                        keep_stats=str(tmp_path / "stats.log"))

    assert report.summary.frames == 2
    assert 0.0 < report.summary.worst <= report.summary.mean <= report.summary.median + 60.0
    assert (tmp_path / "stats.log").read_text().count("\n") == 2
    print(f"\n[e2e] testsrc vs testsrc2, 2 frames at 64x64, yuv420p10le: "
          f"mean {report.summary.mean:.2f} dB, worst {report.summary.worst:.2f} dB")


def test_end_to_end_refuses_mismatched_clips(tmp_path):
    """The guard, through the real exit code: 2 frames against 3 must not be averaged."""
    ffmpeg, ffprobe = _ffmpeg_tools()
    reference, candidate = tmp_path / "ref.mkv", tmp_path / "cand.mkv"
    _make_clip(ffmpeg, reference, "testsrc")
    command = [ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-y",
               "-f", "lavfi", "-i", "testsrc=size=64x64:rate=10:duration=0.3",
               "-c:v", "ffv1", "-pix_fmt", "yuv420p10le", str(candidate)]
    if subprocess.run(command, capture_output=True).returncode != 0:
        pytest.skip("cannot encode with the local ffmpeg")

    code = PR.main([str(reference), str(candidate), "--ffmpeg", ffmpeg,
                    "--ffprobe", ffprobe])

    assert code == 2
