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
import sys

#: The interface languages the app ships: English and ONE translated language. The tuple lives here because this
#: module is the settings owner and must stay importable without PySide6; `src.ui.i18n` re-exports it as its language
#: list, and the validator below is the single answer to "is this a language we can render".
SUPPORTED_LANGUAGES: tuple[str, ...] = ("en", "ru")

#: What an absent, empty or unknown value means. English is the source language of every string in the tree, so it is
#: also the fallback that costs nothing.
DEFAULT_LANGUAGE: str = "en"


def validate_language(value: str | None) -> str:
    """`value` when it is a supported language, `DEFAULT_LANGUAGE` otherwise.

    Unknown values are not an error to raise on: a `settings.txt` written by another build (or hand-edited) must open
    in a language the app can render, not fail to start.
    """
    language = (value or "").strip()
    return language if language in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE


def log(message: str) -> None:
    """Write one diagnostic line through the app's logger, resolved lazily.

    This module is the store and is exercised without a window, so it imports the standard library only. The app's
    logger (`src.Util.log`) opens and ROTATES `log.txt` at import time, so importing it here would make a test that
    constructs a `Settings` truncate the developer's log; by the time a failed read happens in the app the module
    is already loaded (`src.ui.SettingsTab` imports it first), and the lookup below finds the real writer. Without
    it the line goes to stderr - visible either way, which is the point of a failure report.
    """
    util = sys.modules.get("src.Util") or sys.modules.get("Util")
    writer = getattr(util, "log", None)
    if writer is None:
        print(message, file=sys.stderr)
        return
    writer(message)


class Settings:
    def __init__(self, settings_file: str | None = None, default_output_folder: str | None = None):
        """Read `settings.txt`, writing the defaults first when there is no file.

        A file that exists but cannot be read is not touched: the defaults stay in memory and `readError` names the
        exception class (settings audit 2026-11-27, F9). `settings_file` and `default_output_folder` are for tests: the
        app calls `Settings()` with neither, and both then come from the same places as before
        (`currentDirectory()/settings.txt` and the platform's Videos folder).
        """
        if settings_file is None:
            from ..Util import currentDirectory

            settings_file = os.path.join(currentDirectory(), "settings.txt")
        self.settingsFile = settings_file

        # The defaults are stated here and every key the file carries overwrites its own entry in `readSettings`; a
        # key the file does NOT carry keeps the default. An unknown key is ignored rather than treated as damage.
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
            # The interface language (`src.ui.i18n`). English is the source language of every UI string; `ru` is the
            # one translated language the app ships. Appended, not prepended: an existing `settings.txt` keeps its
            # first 30 lines and gains this one at the end on the next write, so nothing a user hand-edited moves.
            "language": DEFAULT_LANGUAGE,
            # The output path the USER chose (`src/ui/output_path.py` owns reading and writing it). Empty means
            # "nothing chosen": the General page then shows a path derived from the input file and the options, and a
            # derived path is never written here. `output_folder_location` above stays as the DIRECTORY fallback an
            # older `settings.txt` already carries, so an existing installation keeps its folder.
            "output_file_path": "",
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
            # The control's own range, 1.0-10.0, and nothing else: the domain used to be 0.1-9.9, so the spinbox
            # offered 10.0 (silently dropped by the writer) and the file could hold 0.5 (unreachable on screen).
            "scene_change_detection_threshold": [
                str(num / 10) for num in range(10, 101)
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
            "language": SUPPORTED_LANGUAGES,
            "output_file_path": "ANY",
        }
        self.settings = self.defaultSettings.copy()
        #: Keys the file carries that this build does not know (a newer build's line, a hand edit). A diagnostic, never
        #: merged into `settings`: the old length check treated one of them as corruption and replaced every value.
        self.ignoredKeys: set[str] = set()
        #: The exception class name of the last read that could not produce values (a file that is not UTF-8, a
        #: directory, a permission failure), or None when the last read succeeded. The app keeps its in-memory values
        #: and does not rewrite the file - the same convention as `ignoredKeys`, a fact the caller can inspect instead
        #: of an exception at launch (settings audit 2026-11-27, F9: the recovery handler caught `ValueError` around
        #: `partition`, which cannot raise, while the failures that do happen took `Settings.__init__` down).
        self.readError: str | None = None
        if not os.path.exists(self.settingsFile):
            self.writeDefaultSettings()
        else:
            self.readSettings()
        # The file is ours to extend, not to police. A key this build does not know is ignored and the known ones keep
        # what the user chose; only a key MISSING from the file falls back to its default (settings audit 2026-11-27:
        # `len(defaults) != len(settings)` wiped five custom values because of one foreign line).
        for key in self.defaultSettings:
            self.settings.setdefault(key, self.defaultSettings[key])

    def readSettings(self):
        """
        Reads the settings from the 'settings.txt' file and stores them in the 'settings' dictionary.

        A file that cannot be READ keeps the values already in memory and is left exactly as it is. Two failures are
        real, and the old handler caught neither: `open()` raises `OSError` (the path is a directory, or the process
        may not read it) and iteration raises `UnicodeDecodeError` (the bytes are not UTF-8). The `except ValueError`
        it had could only ever catch the decode error - and it answered that by calling `writeDefaultSettings()`,
        i.e. by replacing the user's only copy of the file. Nothing is written on this path now; the path and the
        exception class are logged and `readError` records the class for a caller that wants to ask.

        Returns:
            None
        """
        before = dict(self.settings)
        try:
            with open(self.settingsFile, "r") as file:
                for line in file:
                    # `partition`, not `split(",")`: the encoder command is free text and legitimately contains commas
                    # (`-vf "scale=…,…"`), and a bare split raised ValueError and rewrote ALL defaults (owner, 2026-11-22:
                    # "the encoder line resets after any little thing").
                    key, _, value = line.strip().partition(",")
                    if not key:
                        continue
                    # An unknown language is not a value the UI can render; it reads as English instead of being kept
                    # (the writer drops values outside `allowedSettings`, so a kept `de` would vanish on the next save
                    # anyway - this way the current session agrees with the next one).
                    if key == "language":
                        value = validate_language(value)
                    if key not in self.defaultSettings:
                        # Ignored, not fatal - and NOT written back: dropping the line would also drop whatever the
                        # build that wrote it needs. The writer decides the file's shape and never writes an unknown key.
                        self.ignoredKeys.add(key)
                        continue
                    self.settings[key] = value
        except (OSError, UnicodeDecodeError) as error:
            # A partially read file would leave a half-restored configuration; the values from before the read are
            # the honest state (the defaults at construction, the session's dict mid-run).
            self.settings = before
            self.readError = type(error).__name__
            log(f"settings: cannot read {self.settingsFile} ({type(error).__name__}: {error}); "
                f"keeping the values already in memory and leaving the file untouched")
            return
        self.readError = None

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
        """Write the current settings, atomically, and only for keys the defaults know.

        Two guards, both learned the hard way (settings audit, 2026-11-27): the file is written through a temporary and
        replaced, so no reader sees half a file; and an unknown key is SKIPPED instead of calling
        `writeDefaultSettings()` from inside the write loop, which used to leave memory at the defaults and the file at
        the user's values.
        """
        lines = []
        for key, value in self.settings.items():
            if key not in self.defaultSettings:
                continue
            allowed = self.allowedSettings[key]
            if allowed == "ANY" or value in allowed:
                lines.append(f"{key},{value}\n")
        temporary = f"{self.settingsFile}.tmp"
        with open(temporary, "w") as file:
            file.writelines(lines)
        os.replace(temporary, self.settingsFile)
