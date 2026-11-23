"""`settings.txt`: one writer, pinned to the bytes the pre-move class produced.

P3 of the seam plan moved the file's owner out of `SettingsTab` into `src/ui/settings_store.py`. The tab must keep
writing the same file, or every existing `settings.txt` starts being rewritten with a different layout (and a
hand-edited `encoder_command` is the owner's most-reported casualty). The golden below is a capture of the OLD class's
file, taken before the first edit by executing the source at commit eb966ae1
(`git show HEAD:src/ui/SettingsTab.py`, the `Settings` class) with `currentDirectory()` and the default output folder
fixed. It was not produced by the new module.

Both parameters of the new constructor exist for this file: the app calls `Settings()` and the store resolves
`currentDirectory()/settings.txt` and the platform's Videos folder exactly as before, but a test states the path and
the default instead of inheriting the machine's.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ui.settings_store import Settings


#: The file the pre-move class wrote for `default_output_folder="/fixed/Videos"` and nothing else.
SETTINGS_BYTES = ('precision,auto\n'
 'tensorrt_optimization_level,5\n'
 'dynamic_tensorrt_engine,False\n'
 'encoder,libx264\n'
 'video_encoder_speed,medium\n'
 'audio_encoder,copy_audio\n'
 'subtitle_encoder,copy_subtitle\n'
 'audio_bitrate,192k\n'
 'preview_enabled,True\n'
 'scene_change_detection_method,sudo_scene_detect\n'
 'scene_change_detection_enabled,True\n'
 'scene_change_detection_threshold,3.5\n'
 'discord_rich_presence,False\n'
 'video_quality,High\n'
 'output_folder_location,/fixed/Videos\n'
 'last_input_folder_location,/fixed/Videos\n'
 'uhd_mode,True\n'
 'ncnn_gpu_id,0\n'
 'pytorch_gpu_id,0\n'
 'auto_border_cropping,False\n'
 'video_container,mkv\n'
 'video_pixel_format,yuv420p\n'
 'pytorch_version,2.9.0\n'
 'pytorch_backend,CUDA\n'
 'auto_hdr_mode,True\n'
 'quality_preset,best\n'
 'encoder_command,\n'
 'temporal_reuse,True\n'
 'temporal_reuse_statistic,mean_abs\n')


def make(tmp_path):
    return Settings(
        settings_file=str(tmp_path / "settings.txt"),
        default_output_folder="/fixed/Videos",
    )


def read(tmp_path):
    return (tmp_path / "settings.txt").read_text(encoding="utf-8")


def test_the_default_file_is_byte_identical(tmp_path):
    """First launch: no file, so the defaults are written - in this exact order, with these exact separators."""
    make(tmp_path)
    assert read(tmp_path) == SETTINGS_BYTES


def test_a_missing_file_is_created_once(tmp_path):
    """The second construction reads the file it found instead of rewriting it."""
    make(tmp_path)
    make(tmp_path)
    assert read(tmp_path) == SETTINGS_BYTES


def test_a_comma_in_a_value_is_not_a_separator(tmp_path):
    """`encoder_command` is an ffmpeg line with commas (`-vf "scale=1,2"`); the file must round-trip it whole."""
    settings = make(tmp_path)
    command = 'ffmpeg -i in.mkv -vf "scale=1280,720" out.mkv'
    settings.writeSetting("encoder_command", command)
    assert f"encoder_command,{command}\n" in read(tmp_path)

    reread = make(tmp_path)
    assert reread.settings["encoder_command"] == command


def test_an_unknown_key_in_the_file_is_dropped_and_the_file_rewritten(tmp_path):
    """An older/newer build's extra line must not survive a read: the length check is the migration."""
    make(tmp_path)
    path = tmp_path / "settings.txt"
    path.write_text(read(tmp_path) + "from_the_future,1\n", encoding="utf-8")

    reread = make(tmp_path)
    assert "from_the_future" not in reread.settings
    assert read(tmp_path) == SETTINGS_BYTES


def test_a_value_outside_the_allowed_set_is_kept_in_memory_but_never_written(tmp_path):
    """The write filter is the file's: an out-of-range value cannot reach the next launch.

    The line is skipped, not written as-is - so the next read falls back to the default for that key (it is absent
    from the file). That is what the pre-move class did, quirk included.
    """
    settings = make(tmp_path)
    settings.writeSetting("precision", "int4")
    assert settings.settings["precision"] == "int4"
    assert "int4" not in read(tmp_path)
    assert "precision," not in read(tmp_path)
    assert make(tmp_path).settings["precision"] == "auto"


def test_the_defaults_are_what_the_settings_page_expects(tmp_path):
    """The keys the window reads by name; a rename here is a `KeyError` at launch, so it fails in the suite."""
    settings = make(tmp_path)
    for key in (
        "precision", "tensorrt_optimization_level", "encoder_command", "quality_preset",
        "temporal_reuse", "temporal_reuse_statistic", "output_folder_location",
        "last_input_folder_location", "auto_border_cropping", "uhd_mode",
        "preview_enabled", "scene_change_detection_method", "ncnn_gpu_id", "pytorch_gpu_id",
    ):
        assert key in settings.settings
        assert key in settings.allowedSettings
