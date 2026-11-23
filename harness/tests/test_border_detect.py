"""The border decision, pinned on synthetic sample tables - no media, no ffmpeg, no GPU.

`BorderDetect` used to decode the whole file through `cropdetect` and keep the crop with the largest area, so one
bright title card or one noisy frame decided the render. It now samples nine previews and takes the per-side
median (HandBrake's `libhb/scan.c` model). Everything asserted here is the pure half of that change: the side
math, the representative-sample rule, the median/loose/even decision and the probe parsing. The subprocess half
is exercised by the corpus check the supervisor runs.
"""

from __future__ import annotations

import pytest

from src.utils.BorderDetect import (
    BorderDetect,
    decide,
    decide_probes,
    format_decision,
    parse_crop_line,
    parse_header,
    probe_positions,
    sample_valid,
    sampling_enabled,
    sides_from_crop,
)

#: A cropdetect line exactly as ffmpeg n9.1 printed it for a one-frame probe (`skip=0`) of a letterboxed clip.
REAL_CROP_LINE = (
    "[Parsed_cropdetect_0 @ 0x7f5ce0003440] x1:0 x2:319 y1:30 y2:209 w:320 h:176 x:0 y:32 "
    "pts:0 t:0.000000 limit:0.094118 crop=320:176:0:32"
)

#: An input header as ffmpeg prints it with `-hide_banner`.
REAL_HEADER = """Input #0, matroska,webm, from '.scratch/letterbox.mkv':
  Metadata:
    ENCODER         : Lavf63.16.100
  Duration: 00:00:06.00, start: 0.000000, bitrate: 657 kb/s
  Stream #0:0: Video: h264 (Constrained Baseline), yuv420p(tv, progressive), 320x240 [SAR 1:1 DAR 4:3], 25 fps
"""


def _letterboxed(top=120, bottom=120, left=0, right=0):
    return (top, bottom, left, right)


# -- geometry ---------------------------------------------------------------


def test_sides_come_from_the_crop_rectangle():
    # 1440x1080 frame with crop 1432x1072+4+4: four pixels off every side.
    assert sides_from_crop((1432, 1072, 4, 4), 1440, 1080) == (4, 4, 4, 4)


def test_a_full_frame_crop_has_no_borders():
    assert sides_from_crop((1920, 1080, 0, 0), 1920, 1080) == (0, 0, 0, 0)


def test_a_crop_larger_than_the_frame_does_not_produce_negative_sides():
    assert sides_from_crop((1920, 1080, 0, 0), 1280, 720) == (0, 0, 0, 0)


# -- representative samples -------------------------------------------------


def test_a_normal_letterboxed_preview_is_valid():
    assert sample_valid(_letterboxed(), 1920, 1080) is True


def test_a_preview_that_crops_a_quarter_of_the_frame_is_rejected_on_every_side():
    # scan.c:1165-1172 - titles, credits and a fade through black crop far into the picture.
    rejected = [
        (270, 0, 0, 0),  # 1080 / 4 top
        (0, 270, 0, 0),  # 1080 / 4 bottom
        (0, 0, 480, 0),  # 1920 / 4 left
        (0, 0, 0, 480),  # 1920 / 4 right
    ]
    for sides in rejected:
        assert sample_valid(sides, 1920, 1080) is False, sides
    # A pixel less than a quarter is still a representative preview.
    assert sample_valid((269, 269, 479, 479), 1920, 1080) is True


def test_a_preview_that_leaves_no_picture_is_rejected():
    assert sample_valid((540, 540, 0, 0), 1920, 1080) is False


# -- the decision -----------------------------------------------------------


def test_the_median_is_taken_per_side():
    samples = [(4, 4, 4, 4)] * 7 + [(20, 20, 20, 20)] * 2
    decision = decide(samples, 1920, 1080)
    assert decision["median"] == (4, 4, 4, 4)
    assert decision["loose"] is False
    assert decision["crop"] == (1912, 1072, 4, 4)


def test_four_previews_more_than_nine_pixels_smaller_switch_to_per_side_minimum():
    # 5 previews say 12, 4 say 0: the median is 12, and 4 disagreements at 12 px > 9 px take the minimum.
    samples = [(12, 0, 0, 0)] * 5 + [(0, 0, 0, 0)] * 4
    decision = decide(samples, 1920, 1080)
    assert decision["median"] == (12, 0, 0, 0)
    assert decision["loose"] is True
    assert decision["crop"] == (1920, 1080, 0, 0)


def test_three_disagreements_do_not_switch_to_loose():
    samples = [(12, 0, 0, 0)] * 6 + [(0, 0, 0, 0)] * 3
    decision = decide(samples, 1920, 1080)
    assert decision["loose"] is False
    assert decision["crop"] == (1920, 1068, 0, 12)


def test_the_thresholds_are_parameters_not_magic_numbers():
    samples = [(12, 0, 0, 0)] * 6 + [(0, 0, 0, 0)] * 3
    assert decide(samples, 1920, 1080, crop_threshold_frames=3)["loose"] is True
    samples = [(10, 0, 0, 0)] * 6 + [(0, 0, 0, 0)] * 4
    assert decide(samples, 1920, 1080, crop_threshold_pixels=10)["loose"] is False


def test_odd_sides_are_clamped_to_even():
    samples = [(5, 5, 5, 5)] * 5
    decision = decide(samples, 1920, 1080)
    assert decision["median"] == (5, 5, 5, 5)
    assert decision["crop"] == (1912, 1072, 4, 4)


def test_fewer_than_three_valid_samples_means_no_crop():
    decision = decide([(0, 0, 0, 0)] * 2, 1920, 1080)
    assert decision["crop"] is None
    assert "fewer than 3" in decision["reason"]
    assert decide([], 1920, 1080)["crop"] is None


def test_a_crop_that_leaves_less_than_half_the_frame_is_refused():
    decision = decide([(400, 400, 0, 0)] * 5, 1920, 1080)
    assert decision["crop"] is None
    assert "half the frame" in decision["reason"]


def test_a_file_without_borders_returns_the_full_frame():
    decision = decide([(0, 0, 0, 0)] * 5, 1920, 1080)
    assert decision["crop"] == (1920, 1080, 0, 0)
    assert format_decision({**decision, "probes": 9}) == (
        "detecting borders: 9 probes, 5 valid, median 0/0/0/0, loose=false, crop 1920x1080+0+0"
    )


def test_one_full_frame_title_card_loses_to_the_median():
    """The red-before: under "largest area" the title card won, and it is the wrong answer.

    Eight letterboxed previews (120 px top and bottom) and one title card that fills the frame. The old rule
    kept the full frame - the biggest crop - which leaves the black bars in the render.
    """
    letterbox = (120, 120, 0, 0)
    title = (0, 0, 0, 0)
    crops = [
        (1920, 1080 - sum(letterbox[:2]), 0, letterbox[0]),
    ] * 8 + [(1920, 1080, 0, 0)]

    old_choice = max(crops, key=lambda crop: crop[0] * crop[1])
    decision = decide_probes(crops, 1920, 1080)

    assert old_choice == (1920, 1080, 0, 0), "the old max-by-area rule picked the title card"
    assert decision["crop"] == (1920, 840, 0, 120)
    assert decision["crop"] != old_choice


# -- probe aggregation ------------------------------------------------------


def test_a_container_whose_previews_return_nothing_sets_the_fallback_flag():
    decision = decide_probes([None] * 9, 1920, 1080)
    assert decision["fallback"] is True
    assert decision["crop"] is None
    assert decision["probes"] == 9
    assert decision["samples"] == 0


def test_some_failed_previews_do_not_set_the_fallback_flag():
    crops = [None] * 2 + [(1912, 1080, 4, 0)] * 7
    decision = decide_probes(crops, 1920, 1080)
    assert decision["fallback"] is False
    assert decision["samples"] == 7
    assert decision["crop"] == (1912, 1080, 4, 0)


def test_a_quarter_frame_preview_is_dropped_before_the_vote():
    crops = [(1920, 1080, 0, 0)] * 8 + [(1920, 540, 0, 270)]
    decision = decide_probes(crops, 1920, 1080)
    assert decision["probes"] == 9
    assert decision["samples"] == 8
    assert decision["crop"] == (1920, 1080, 0, 0)


def test_the_log_line_names_every_stage():
    decision = {
        "probes": 9,
        "samples": 8,
        "median": (4, 4, 0, 0),
        "loose": False,
        "crop": (1432, 1072, 4, 4),
        "reason": "",
    }
    assert format_decision(decision) == (
        "detecting borders: 9 probes, 8 valid, median 4/4/0/0, loose=false, crop 1432x1072+4+4"
    )


def test_the_loose_log_line_says_so():
    crops = [(1920, 1068, 0, 12)] * 5 + [(1920, 1080, 0, 0)] * 4
    decision = decide_probes(crops, 1920, 1080)
    assert decision["loose"] is True
    assert "loose=true" in decision["log"]


# -- parsing ----------------------------------------------------------------


def test_a_real_cropdetect_line_is_parsed():
    assert parse_crop_line(REAL_CROP_LINE) == (320, 176, 0, 32)


def test_the_last_crop_line_wins():
    text = "crop=1920:1080:0:0\ncrop=1432:1072:4:4\n"
    assert parse_crop_line(text) == (1432, 1072, 4, 4)


def test_a_probe_without_a_crop_line_parses_to_nothing():
    assert parse_crop_line("") is None
    assert parse_crop_line("frame=    1 fps=0.0 q=-0.0 Lsize=N/A time=00:00:00.04") is None
    assert parse_crop_line("crop=0:0:0:0") is None


def test_the_input_header_gives_duration_and_frame_size():
    assert parse_header(REAL_HEADER) == (6.0, 320, 240)


def test_a_header_without_duration_or_stream_is_not_parsed():
    assert parse_header("") is None
    assert parse_header("Input #0, matroska\n  Stream #0:0: Video: h264, 320x240") is None


def test_preview_positions_spread_over_the_file():
    positions = probe_positions(90)
    assert len(positions) == 9
    assert positions[0] == pytest.approx(9.0)
    assert positions[-1] == pytest.approx(81.0)
    assert positions == sorted(positions)
    assert probe_positions(0) == []


# -- the environment gate ---------------------------------------------------


@pytest.mark.parametrize("value, expected", [
    (None, True),
    ("1", True),
    ("true", True),
    ("0", False),
    ("false", False),
    ("FALSE", False),
    ("off", False),
])
def test_the_sampling_gate_defaults_on(value, expected):
    environ = {} if value is None else {"RVE_BORDER_SAMPLE": value}
    assert sampling_enabled(environ) is expected


def test_the_gate_off_takes_the_whole_file_pass(monkeypatch):
    calls = []
    monkeypatch.setenv("RVE_BORDER_SAMPLE", "0")
    monkeypatch.setattr(BorderDetect, "_getBordersSampled", lambda self: calls.append("sampled"))
    monkeypatch.setattr(BorderDetect, "_getBordersWholeFile", lambda self: ("whole",))
    assert BorderDetect("unused.mkv").getBorders() == ("whole",)
    assert calls == []


def test_the_gate_on_takes_the_sampled_pass(monkeypatch):
    calls = []
    monkeypatch.delenv("RVE_BORDER_SAMPLE", raising=False)
    monkeypatch.setattr(BorderDetect, "_getBordersSampled", lambda self: calls.append("sampled") or ("sampled",))
    monkeypatch.setattr(BorderDetect, "_getBordersWholeFile", lambda self: ("whole",))
    assert BorderDetect("unused.mkv").getBorders() == ("sampled",)
    assert calls == ["sampled"]


def test_a_preview_that_never_decodes_falls_back_and_says_so(monkeypatch):
    # A container the probes cannot read: no header at all, so the whole-file pass is used and flagged.
    detect = BorderDetect("not-a-container", ffmpeg_path="definitely-not-ffmpeg")
    monkeypatch.setattr(detect, "_probeHeader", lambda: None)
    assert detect._getBordersSampled() is None
    assert detect.used_fallback is True


def test_the_call_shape_is_unchanged_without_a_binary():
    # The reader expects four values even when nothing can be probed; the old pass answered (None, None, None, None).
    detect = BorderDetect("not-a-container.mkv", ffmpeg_path="definitely-not-ffmpeg")
    assert detect.getBorders() == (None, None, None, None)
