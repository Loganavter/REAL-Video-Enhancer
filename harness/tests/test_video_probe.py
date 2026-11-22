"""The two rules around reading a video's metadata, pinned where a traceback used to be.

Owner, 2026-11-22, on every encoder-settings change while a folder was loaded:

    SettingsTab.updateFFMpegCommand -> VideoLoader.getData -> total_frames / fps -> ZeroDivisionError

The field held the batch sentinel, the backend printed no FPS line, `get_fps()` answered `0.0`, and `getData()` divided
by it. Both halves are asserted here: what counts as probeable input, and that a missing frame rate is a zero duration
rather than an exception.
"""

from __future__ import annotations

import os

import pytest

from src.video_probe import duration_seconds, is_probeable_input


def test_a_real_file_is_probeable(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")
    assert is_probeable_input(str(video)) is True


def test_the_batch_sentinel_is_not_probeable():
    # This is the value the input field carries while a folder is loaded, and the one that crashed the settings page.
    assert is_probeable_input(" { MULTIPLE_FILES } ") is False
    assert is_probeable_input("{MULTIPLE_FILES}") is False


def test_empty_text_and_folders_are_not_probeable(tmp_path):
    assert is_probeable_input("") is False
    assert is_probeable_input("   ") is False
    assert is_probeable_input(str(tmp_path)) is False
    assert is_probeable_input("/definitely/not/here.mp4") is False


def test_a_url_is_probeable():
    # `loadVideo` accepts URLs and the backend fetches them; a filesystem check must not rule them out.
    assert is_probeable_input("https://example.com/clip.mp4") is True
    assert is_probeable_input("http://example.com/clip.mp4") is True


def test_a_missing_frame_rate_reads_as_zero_duration():
    assert duration_seconds(240, 0) == 0.0
    assert duration_seconds(240, 0.0) == 0.0
    assert duration_seconds(0, 0) == 0.0


def test_the_normal_case_still_computes_a_duration():
    assert duration_seconds(240, 24) == 10.0
    assert duration_seconds(1_000, 25.0) == 40.0


def test_nonsense_values_do_not_raise():
    assert duration_seconds(None, None) == 0.0
    assert duration_seconds("not a number", 24) == 0.0
