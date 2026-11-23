"""The settings file's owner: `settings.txt`, its defaults, its format and its migration.

Moved out of `src/ui/SettingsTab.py` (P3 of the seam plan) so the tab keeps the widgets and the signals and stops
knowing the file: the tab asks this object for a value and writes one back, and every byte of `settings.txt` is
produced here. The class name, the attribute names (`settingsFile`, `defaultSettings`, `allowedSettings`, `settings`)
and the four methods are unchanged - `SettingsTab`, `ProcessTab` and the window's entry point build and drive it
exactly as before, and `src.ui.SettingsTab.Settings` re-exports it for the callers that import it there.

The file format is what it was: one `key,value` line per setting, in `defaultSettings` order, written only for a key
the defaults know and a value `allowedSettings` accepts (`"ANY"` accepts everything). A value may contain commas
(`encoder_command` is an ffmpeg command line), so reading partitions on the FIRST comma and never splits.

Python-only on purpose: this module imports the standard library at module scope. The two app values it needs
(`currentDirectory()` and the default output folder) are resolved inside `__init__` - and can be passed in instead -
so the store can be exercised without PySide6, `src.Util` or a window. `harness/tests/test_settings_store.py` pins
the written bytes against a capture of the pre-move class.
"""

from __future__ import annotations

import os


class Settings:
    def __init__(self, settings_file: str | None = None, default_output_folder: str | None = None):
        """Read `settings.txt`, writing the defaults first when there is no file (or when it is incomplete).

        `settings_file` and `default_output_folder` are for tests: the app calls `Settings()` with neither, and both
        then come from the same places as before (`currentDirectory()/settings.txt` and the platform's Videos folder).
        """
        if settings_file is None:
            from ..Util import currentDirectory

            settings_file = os.path.join(currentDirectory(), "settings.txt")
        self.settingsFile = settings_file

        """
        The default settings are set here, and are overwritten by the settings in the settings file if it exists and the legnth of the settings is the same as the default settings.
        The key is equal to the name of the widget of the setting in the settings tab.
        """
        if default_output_folder is None:
            from ..Util import FileHandler

            default_output_folder = FileHandler.getDefaultOutputFolder()
        output_folder_default = default_output_folder
        self.defaultSettings = {
            "precision": "auto",
            "tensorrt_optimization_level": "5",
            "dynamic_tensorrt_engine": "False",
            "encoder": "libx264",
            "video_encoder_speed": "medium",
            "audio_encoder": "copy_audio",
            "subtitle_encoder": "copy_subtitle",
            "audio_bitrate": "192k",
            "preview_enabled": "True",
            # The upscale model the last job ran with: the list's first entry is an accident of the file listing (a 4x
            # model on top, which ran a 4x forward pass for a 2x request until 2026-11-23), and the choice should stick.
            "last_upscale_model": "",
            "scene_change_detection_method": "sudo_scene_detect",
            "scene_change_detection_enabled": "True",
            "scene_change_detection_threshold": "3.5",
            "discord_rich_presence": "False",
            "video_quality": "High",
            "output_folder_location": output_folder_default,
            "last_input_folder_location": output_folder_default,
            "uhd_mode": "True",
            "ncnn_gpu_id": "0",
            "pytorch_gpu_id": "0",
            "auto_border_cropping": "False",
            "video_container": "mkv",
            "video_pixel_format": "yuv420p",
            "pytorch_version": "2.9.0",
            "pytorch_backend": "CUDA",
            "auto_hdr_mode": "True",
            "quality_preset": "best",
            # The generated ffmpeg command line, persisted so a hand-edited one survives a restart. Empty = "generate
            # from the settings above", which is the state the app starts in.
            "encoder_command": "",
            # reuse of unchanged frames: on by default since 2026-11-20 (the gate refuses material it
            # cannot certify, so the default is safe; this switch is for people who want it off entirely)
            "temporal_reuse": "True",
            # what the gate compares. "mean_abs" is the shipped statistic; "coarse" decides on a 3 px low-pass
            # with the translation compensated, i.e. on the drawing rather than on the film's grain, dust and
            # scan wobble (harness/docs/wsp/verified/reuse-structural-gate.txt). Not the default: it certifies
            # its own acceptance at that same coarse scale, which is a picture judgement the owner makes.
            "temporal_reuse_statistic": "mean_abs",
        }
        self.allowedSettings = {
            "encoder_command": "ANY",
            "precision": ("auto", "float32", "float16"),
            "tensorrt_optimization_level": ("0", "1", "2", "3", "4", "5"),
            "dynamic_tensorrt_engine": ("True", "False"),
            "encoder": (
                "libx264",
                "libx265",
                "vp9",
                "av1",
                "prores",
                "ffv1",
                "utvideo",
                "x264_nvenc",
                "x265_nvenc",
                "av1_nvenc (40 series and up)",
            ),
            "video_encoder_speed": ("placebo","slow", "medium", "fast", "fastest"),
            "audio_encoder": ("aac", "libmp3lame", "opus", "copy_audio"),
            "audio_bitrate": "ANY",
            "subtitle_encoder": ("copy_subtitle","srt","ass","webvtt"),
            "preview_enabled": ("True", "False"),
            "last_upscale_model": "ANY",
            "scene_change_detection_method": (
                "mean",
                "mean_segmented",
                "pyscenedetect",
                "sudo_scene_detect",
            ),
            "scene_change_detection_enabled": ("True", "False"),
            "scene_change_detection_threshold": [
                str(num / 10) for num in range(1, 100)
            ],
            "discord_rich_presence": ("True", "False"),
            "video_quality": ("Low", "Medium", "High", "Very_High", "Ultra", "Lossless"),
            "output_folder_location": "ANY",
            "last_input_folder_location": "ANY",
            "uhd_mode": ("True", "False"),
            "ncnn_gpu_id": "ANY",
            "pytorch_gpu_id": "ANY",
            "auto_border_cropping": ("True", "False"),
            "video_container": ("mkv", "mp4", "mov", "webm", "avi"),
            "video_pixel_format": "ANY",
            "pytorch_version": ("2.9.0", "2.8.0", "2.6.0"),
            "pytorch_backend": "ANY",
            "auto_hdr_mode": ("True", "False"),
            "quality_preset": ("best", "fast"),
            "temporal_reuse": ("True", "False"),
            "temporal_reuse_statistic": ("mean_abs", "coarse"),
        }
        self.settings = self.defaultSettings.copy()
        if not os.path.isfile(self.settingsFile):
            self.writeDefaultSettings()
        self.readSettings()
        # check if the settings file is corrupted
        if len(self.defaultSettings) != len(self.settings):
            self.writeDefaultSettings()

    def readSettings(self):
        """
        Reads the settings from the 'settings.txt' file and stores them in the 'settings' dictionary.

        Returns:
            None
        """
        with open(self.settingsFile, "r") as file:
            try:
                for line in file:
                    # `partition`, not `split(",")`: the encoder command is free text and legitimately contains commas
                    # (`-vf "scale=…,…"`), and a bare split raised ValueError and rewrote ALL defaults (owner, 2026-11-22:
                    # "the encoder line resets after any little thing").
                    key, _, value = line.strip().partition(",")
                    if not key:
                        continue
                    self.settings[key] = value
            except (
                ValueError
            ):  # writes and reads again if the settings file is corrupted
                self.writeDefaultSettings()
                self.readSettings()

    def writeSetting(self, setting: str, value: str):
        """
        Writes the specified setting with the given value to the settings dictionary.

        Parameters:
        - setting (str): The name of the setting to be written, this will be equal to the widget name in the settings tab if set correctly.
        - value (str): The value to be assigned to the setting.

        Returns:
            None
        """
        self.settings[setting] = value
        self.writeOutCurrentSettings()

    def writeDefaultSettings(self):
        """
        Writes the default settings to the settings file if it doesn't exist.

        Parameters:
            None

        Returns:
            None
        """
        self.settings = self.defaultSettings.copy()
        self.writeOutCurrentSettings()

    def writeOutCurrentSettings(self):
        """
        Writes the current settings to a file.

        Parameters:
            self (Settings): The instance of the Settings class.

        Returns:
            None
        """
        with open(self.settingsFile, "w") as file:
            for key, value in self.settings.items():
                if key in self.defaultSettings:  # check if the key is valid
                    if (
                        value in self.allowedSettings[key]
                        or self.allowedSettings[key] == "ANY"
                    ):  # check if it is in the allowed settings dict
                        file.write(f"{key},{value}\n")
                else:
                    self.writeDefaultSettings()
