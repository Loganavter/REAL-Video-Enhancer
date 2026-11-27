"""One setting, three answers: the file, the dict a render obeys, and the widget a user sees.

The settings audit (2026-11-27) names one fault line under four findings, and this module pins the fixes for the
three that are about a value existing in more than one place:

* **F4 - a restore that cannot reach a row.** `connectSettingText` pushed the stored value into every control; a
  combo without that item, a spin box with a lower maximum or a check box whose state has no counterpart kept its own
  value and nothing said so. `SettingsTab.setting_mirror_mismatches()` is the comparator that names each row, and
  `report_setting_mirror()` logs one line per disagreement. Neither the file nor the control is repaired - both
  silent repairs are how the disagreement got here.
* **F5 - the info pane reported the Encoder row while the command line runs.** With a user-owned command the pane's
  Encoder cell says that, shows the command, and names `encoder_command` as the setting it read.
* **F6 - `Reset Settings` was a ten-pass loop.** It is one ordered restore (file -> dict -> controls), and after it
  the three places are asserted equal with the same comparator.
* **F9 - the corruption handler could not fire.** A file that cannot be READ (invalid UTF-8, a directory, a
  permission failure) keeps the defaults in memory, is left byte-for-byte alone, and is logged with its path and the
  exception class.

Everything here is offscreen. What needs a window - the pane's label in a live layout, the reset on a real click -
is `LIVE-PROOF WANTED` in `DONE.md`, never claimed as tested.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("sli_ui_toolkit")

if "resources_rc" not in sys.modules:
    sys.modules["resources_rc"] = types.ModuleType("resources_rc")

from src.ui import settings_store  # noqa: E402
from src.ui.settings_store import Settings  # noqa: E402

#: A loaded clip in the pane's own shape (`test_general_settings_info.VALUES`); the Encoder row's output is what the
#: pane is supposed to report until the user owns the command line.
VALUES: dict[str, tuple[str, str]] = {
    "FPS:": ("24.0", "48.0"),
    "Resolution:": ("1440x1080", "2880x2160"),
    "Frame Count:": ("493", "986"),
    "Encoder:": ("hevc", "libx264"),
    "Container:": ("mkv", "mkv"),
    "Color Space:": ("bt709", "bt709"),
    "Pixel Format:": ("yuv420p10le", "yuv420p10le"),
    "Is HDR:": ("False", "False"),
    "Bit Depth:": ("10 bit", "10 bit"),
}

#: A command line only the user could have typed: it names an encoder the row does not.
USER_COMMAND = "ffmpeg -i in.mkv -c:v libx265 -crf 20 out.mkv"

#: Page roots are windowless here; keeping them alive for the process keeps Qt's C++ side from outliving its Python
#: owner (the same reason `test_general_settings_info.py` parks its root on the page).
_ROOTS: list[object] = []


# --- F9: an unreadable file keeps the defaults, the file untouched, and says so -----------------------------------

def test_a_file_that_is_not_utf8_keeps_the_defaults_and_is_not_rewritten(tmp_path, monkeypatch):
    """The old handler caught `ValueError` and answered it with `writeDefaultSettings()` - replacing the user's only
    copy. The decode error IS a `ValueError`, so that path did fire: it just destroyed the file it claimed to repair.
    """
    path = tmp_path / "settings.txt"
    payload = b"precision,float32\n\xff\xfe not utf-8\n"
    path.write_bytes(payload)
    logged: list[str] = []
    monkeypatch.setattr(settings_store, "log", logged.append, raising=False)

    settings = Settings(settings_file=str(path), default_output_folder="/fixed/Videos")

    assert path.read_bytes() == payload, "the unreadable file is the user's only copy - it must not be rewritten"
    assert settings.settings == settings.defaultSettings
    assert settings.readError == "UnicodeDecodeError"
    assert any("UnicodeDecodeError" in line and str(path) in line for line in logged), logged


def test_a_path_that_is_a_directory_is_readable_as_a_failure_not_a_crash(tmp_path, monkeypatch):
    """`open()` on a directory raises `IsADirectoryError`, which is not the class the old handler caught: the launch
    died in `Settings.__init__`. It is a read failure like any other now - and nothing is written over the directory.
    """
    target = tmp_path / "settings.txt"
    target.mkdir()
    kept = target / "keep.txt"
    kept.write_text("user data", encoding="utf-8")
    logged: list[str] = []
    monkeypatch.setattr(settings_store, "log", logged.append, raising=False)

    settings = Settings(settings_file=str(target), default_output_folder="/fixed/Videos")

    assert target.is_dir() and kept.read_text(encoding="utf-8") == "user data"
    assert not (tmp_path / "settings.txt.tmp").exists(), "the defaults were written next to an unreadable target"
    assert settings.settings == settings.defaultSettings
    assert settings.readError == "IsADirectoryError"
    assert any("IsADirectoryError" in line and str(target) in line for line in logged), logged


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a mode-0 file; the failure cannot be staged")
def test_a_file_the_process_cannot_read_keeps_the_defaults(tmp_path, monkeypatch):
    path = tmp_path / "settings.txt"
    path.write_text("precision,float32\n", encoding="utf-8")
    path.chmod(0o000)
    logged: list[str] = []
    monkeypatch.setattr(settings_store, "log", logged.append, raising=False)
    try:
        settings = Settings(settings_file=str(path), default_output_folder="/fixed/Videos")
        assert settings.settings == settings.defaultSettings
        assert settings.readError == "PermissionError"
        assert any("PermissionError" in line for line in logged), logged
    finally:
        path.chmod(0o600)


# --- the glue the next three findings live in ---------------------------------------------------------------------

def _glue(qapp, tmp_path, monkeypatch, *, half_precision=True, stored_values=None):
    """The settings glue over the real pages, offscreen, with a windowless owner.

    The pages are the shipped ones (`SettingsPagePage` builds the Output/Render/RVE tabs, `Tab6Page` the encoder row,
    `DownloadPagePage` the two combos); the fake window supplies exactly what `SettingsTab` reads from its parent -
    the handle surface, the one settings object, the pane refresh and the page switch. `updateVideoGUIText` mirrors
    the window's real method: it re-reads the file and paints the pane, in that order.
    """
    from src.ui import toolkit_boot, toolkit_theme
    from sli_ui_toolkit.ui.managers.ui_scale import UiScale
    from src.ui.pages.download_page import DownloadPagePage
    from src.ui.pages.general_settings import GeneralSettingsPage
    from src.ui.pages.settings_page import SettingsPagePage
    from src.ui.pages.tab_6 import Tab6Page
    import src.ui.GPUDetect as GPUDetect

    # The card's int8 policy is not this file's subject; the real probe shells out to nvidia-smi / torch.
    monkeypatch.setattr(GPUDetect, "int8_body_available", lambda: (True, "the test card runs int8"))

    UiScale.get_instance().set_factor(1.0)
    toolkit_boot.ensure()
    toolkit_theme.install(qapp)

    general = GeneralSettingsPage()
    _ROOTS.append(general.build())
    settings_page = SettingsPagePage()
    _ROOTS.append(settings_page.build())
    tab6 = Tab6Page()
    _ROOTS.append(tab6.build())
    download = DownloadPagePage()
    _ROOTS.append(download.build())

    handles = types.SimpleNamespace(
        general=general._handles,
        settings=settings_page._handles,
        output=settings_page.tab_2_page._handles,
        render_settings=settings_page.tab_3_page._handles,
        app_settings=settings_page.tab_4_page._handles,
        encoder=tab6._handles,
        download=download._handles,
    )
    # `DownloadTab` (outside this cell) fills these two from the build list; the page builds them empty.
    handles.download.pytorch_version.addItems(["2.9.0", "2.8.0", "2.6.0"])
    handles.download.pytorch_backend.addItems(["CUDA", "ROCm", "xpu"])

    store = Settings(settings_file=str(tmp_path / "settings.txt"), default_output_folder="/fixed/Videos")
    for key, value in (stored_values or {}).items():
        store.writeSetting(key, value)

    class _FakeWindow:
        def __init__(self):
            self.page_handles = handles
            self.settings = store
            self.shell = types.SimpleNamespace(
                settingsPage_page=settings_page,
                procPage_page=types.SimpleNamespace(generalSettings_page=general),
            )
            self.switches = 0

        def updateVideoGUIText(self):
            self.settings.readSettings()
            general.set_video_info(VALUES)

        def switchToSettingsPage(self):
            self.switches += 1

    from src.ui.SettingsTab import SettingsTab

    window = _FakeWindow()
    tab = SettingsTab(
        window,
        halfPrecisionSupport=half_precision,
        pytorch_gpu_count=1,
        ncnn_gpu_count=1,
    )
    return tab, store, general


def _file_values(path: Path) -> dict[str, str]:
    """The `key,value` pairs the file holds (first comma partitions, as the store reads them)."""
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition(",")
        if key:
            values[key] = value
    return values


# --- F4: a restore that cannot reach a row is reported, not papered over ------------------------------------------

def test_a_restore_that_cannot_reach_a_row_is_named_and_logged(qapp, tmp_path, monkeypatch):
    """`precision=float16` on a card without half support and `tensorrt_optimization_level=0` (the store allows 0,
    the combo starts at 1) are the audit's two live examples of a control that cannot show its stored value."""
    from src.ui import SettingsTab as settings_tab_module

    logged: list[str] = []
    monkeypatch.setattr(settings_tab_module, "log", logged.append)
    tab, store, _general = _glue(
        qapp, tmp_path, monkeypatch,
        half_precision=False,
        stored_values={"precision": "float16", "tensorrt_optimization_level": "0"},
    )

    mismatches = tab.setting_mirror_mismatches()
    assert ("precision", "float16", "float32") in mismatches, mismatches
    assert ("tensorrt_optimization_level", "0", "3") in mismatches, mismatches

    # The comparator reports; it does not "fix" either side. The two silent repairs are what the audit blames.
    assert store.settings["precision"] == "float16", "the store was rewritten to match a widget"
    assert store.settings["tensorrt_optimization_level"] == "0"
    assert _file_values(tmp_path / "settings.txt")["precision"] == "float16"

    # One named warning per disagreement, and the same list the test just read.
    warnings = [line for line in logged if "settings mirror mismatch" in line]
    assert any("precision" in line for line in warnings), logged
    assert any("tensorrt_optimization_level" in line for line in warnings), logged


def test_a_restore_that_lands_leaves_no_disagreement(qapp, tmp_path, monkeypatch):
    """The comparator's other half: a healthy restore is silent, so a warning means a real mismatch."""
    tab, _store, _general = _glue(qapp, tmp_path, monkeypatch)
    assert tab.setting_mirror_mismatches() == []


# --- F5: the info pane says which encoder fact it reports ---------------------------------------------------------

def test_a_user_command_is_what_the_pane_reports_and_it_names_the_setting(qapp, tmp_path, monkeypatch):
    """The pane painted the Encoder ROW (`libx264`) while the field's `libx265` command is what executes."""
    tab, _store, general = _glue(qapp, tmp_path, monkeypatch)
    assert general.infoFactValues["Encoder:"].text() == "libx264", "the app-generated row is the fact before a user edit"

    tab.page_handles.encoder.EncoderCommand.setText(USER_COMMAND)
    tab.onEncoderCommandEdited()  # the `textEdited` signal, fired by hand: `setText` does not emit it

    cell = general.infoFactValues["Encoder:"].text()
    assert USER_COMMAND in cell, f"the pane still reports the row: {cell!r}"
    assert "libx264" not in cell, f"the pane shows the encoder the run will not use: {cell!r}"
    assert "encoder_command" in general.infoFactLabels["Encoder:"].text()
    assert general.infoFactValues["Encoder:"].toolTip().startswith("encoder_command:")
    assert USER_COMMAND in general.video_info_text(), "the probe's text disagrees with the pane"
    assert general.encoder_source() == ("encoder_command", USER_COMMAND)


def test_clearing_the_command_hands_the_pane_back_to_the_row(qapp, tmp_path, monkeypatch):
    tab, _store, general = _glue(qapp, tmp_path, monkeypatch)
    tab.page_handles.encoder.EncoderCommand.setText(USER_COMMAND)
    tab.onEncoderCommandEdited()
    assert general.encoder_source()[0] == "encoder_command"

    tab.page_handles.encoder.EncoderCommand.setText("")
    tab.onEncoderCommandEdited()

    assert general.encoder_source() == ("encoder", "libx264")
    assert general.infoFactLabels["Encoder:"].text() == "Encoder"


# --- F6: Reset is one ordered restore and the three places agree afterwards ---------------------------------------

def test_reset_is_one_restore_and_the_file_dict_and_controls_agree(qapp, tmp_path, monkeypatch):
    """Red on the base tree twice over: `writeDefaultSettings`/`connectSettingText` ran TEN times, and a user-owned
    command line survived in the field while the file held the default empty string."""
    tab, store, general = _glue(qapp, tmp_path, monkeypatch)
    handles = tab.page_handles

    # A session that changed several places, including the three that are not simple combos.
    handles.output.scene_change_detection_threshold.setValue(8.0)
    handles.general.frameReuse.setChecked(False)
    handles.general.celReuse.setChecked(True)
    handles.render_settings.precision.setCurrentText("float32")
    handles.encoder.EncoderCommand.setText(USER_COMMAND)
    tab.onEncoderCommandEdited()
    assert tab.encoder_command_user is True

    writes: list[int] = []
    restores: list[int] = []
    original_write = store.writeDefaultSettings
    original_restore = tab.connectSettingText
    monkeypatch.setattr(store, "writeDefaultSettings", lambda: (writes.append(1), original_write())[1])
    monkeypatch.setattr(tab, "connectSettingText", lambda: (restores.append(1), original_restore())[1])

    tab.resetSettings()

    assert len(writes) == 1, f"Reset rewrote the defaults {len(writes)} times"
    assert len(restores) == 1, f"Reset restored the controls {len(restores)} times"
    assert tab.parent.switches == 1

    # 1. the file holds the defaults ...
    file_values = _file_values(tmp_path / "settings.txt")
    for key, value in store.defaultSettings.items():
        assert file_values.get(key) == value, f"{key}: file says {file_values.get(key)!r}, default is {value!r}"
    # 2. the dict equals the file ...
    assert {key: store.settings[key] for key in store.defaultSettings} == file_values
    # 3. ... and every control the restore touches reports its stored value.
    assert tab.setting_mirror_mismatches() == []

    # The command line is a stored setting, not only a field.
    assert tab.encoder_command_user is False
    assert USER_COMMAND not in handles.encoder.EncoderCommand.text()
    assert general.encoder_source()[0] == "encoder"
