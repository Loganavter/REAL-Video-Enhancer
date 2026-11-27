"""One owner for the output path: the reported bug, the rule that fixes it, and the five behaviours around it.

Owner report (2026-11-26): *there are two places to set the output folder, one of them hidden in the settings, and the
hidden one always wipes the choice made in the previous session.* Measured on the tree before the change:

* the General page's `outputFileText` held a full PATH and nothing persisted it;
* the Encoder page's `output_folder_location` held a DIRECTORY, persisted, and it was the value every option change
  regenerated the path from;
* `MainWindow.setDefaultOutputFile` ended in `outputFileText.setText(...)`, so any option checkbox replaced whatever
  the user had chosen, and a hand-typed path was gone after a restart because it had never been saved.

What this file pins, in the order the acceptance names them:

1. a non-empty user path survives an option change (the reported bug);
2. an empty field is filled from the directory default - and only an empty one;
3. the path persists and restores across a settings round-trip, while an OLD `settings.txt` (no path key) still feeds
   the directory fallback;
4. the Encoder row is a READ-ONLY mirror (asserted exactly: the field cannot be edited, its picker button is gone, and
   the mirror follows the path's directory);
5. the path's directory is what the open-folder button opens, not the legacy key's.

Red-before (`harness/docs/CODE_PATTERNS.md` §9 - a test that cannot go red pins nothing): removing the `stored_path`
branch of `output_path.apply_default` fails 1 and 3; removing read-only from the Encoder row fails 4; pointing the
open-folder handler back at the settings key fails 5. The runs are recorded in `DONE.md`.

The window methods are exercised as the shipped source, not a re-implementation: `MainWindow`'s body is extracted from
`REAL-Video-Enhancer.py` by AST and bound to a stand-in window (the pattern `test_render_queue_add_during_run.py`
uses), because the entry module cannot be imported without booting the app. The widgets are the smallest fakes that
have the surface the methods use - the real ones are covered by `test_settings_scale.py` / `test_general_tab_fit.py`.

Runs offscreen only.
"""

from __future__ import annotations

import ast
import dataclasses
import os
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ui import output_path  # noqa: E402
from src.ui.settings_store import Settings  # noqa: E402


# --- the smallest widgets the window methods touch -------------------------------------------------------------


class _Field:
    """A line edit: `text()` / `setText()`, which is the whole surface the path owner uses."""

    def __init__(self, text: str = "") -> None:
        self._text = text

    def text(self) -> str:
        return self._text

    def setText(self, text: str) -> None:  # noqa: N802 - Qt spelling
        self._text = str(text)


class _Combo:
    def __init__(self, text: str = "") -> None:
        self._text = text

    def currentText(self) -> str:  # noqa: N802
        return self._text


class _Check:
    def __init__(self, checked: bool = False) -> None:
        self._checked = checked

    def isChecked(self) -> bool:  # noqa: N802
        return self._checked


class _Spin:
    def __init__(self, value: float) -> None:
        self._value = value

    def value(self) -> float:
        return self._value


def _display_name(model: str) -> str:
    """`src/ModelHandler.getModelDisplayName`, without importing the model catalogue (it needs a backend probe)."""
    cut = model.index(" (") if " (" in model else len(model)
    return model[:cut].lower().replace(" ", "-")


def _page(field: str = "", *, upscale: bool = True) -> SimpleNamespace:
    """The General page's handle group, as far as the path derivation reads it."""
    return SimpleNamespace(
        inputFileText=_Field("/videos/clip.mkv"),
        outputFileText=_Field(field),
        sloMoModeCheckBox=_Check(False),
        interpolateCheckBox=_Check(True),
        decompressCheckBox=_Check(False),
        denoiseCheckBox=_Check(False),
        upscaleCheckBox=_Check(upscale),
        interpolateModelComboBox=_Combo("RIFE 4.25"),
        decompressModelComboBox=_Combo("None"),
        denoiseModelComboBox=_Combo("None"),
        upscaleModelComboBox=_Combo("realesrgan x4"),
        interpolationMultiplierSpinBox=_Spin(2),
        upscaleScaleSpinBox=_Spin(2),
    )


def _settings(tmp_path, **values) -> Settings:
    settings = Settings(
        settings_file=str(tmp_path / "settings.txt"),
        default_output_folder="/fixed/Videos",
    )
    for key, value in values.items():
        settings.writeSetting(key, value)
    return settings


def _window_methods(*names: str, chosen: str | None = None) -> dict:
    """Compile `MainWindow`'s named methods on their own, so what runs is the shipped source.

    The entry module cannot be imported here (it boots the application, probes a backend and pulls `cpuinfo`), so its
    method bodies are extracted by AST and bound to a stand-in window - the same device
    `test_render_queue_add_during_run.py` uses. The module-level names the bodies reference are supplied below: the
    owner module itself, `os`, the display-name formatter, and (when a directory is given) a `QFileDialog` that answers
    it instead of opening a window.
    """
    source = (ROOT / "REAL-Video-Enhancer.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    window = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MainWindow")
    found = [node for node in window.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in found} == set(names), f"missing methods: {set(names) - {n.name for n in found}}"
    namespace: dict = {
        "os": os,
        "output_path_io": output_path,
        "getModelDisplayName": _display_name,
        "QFileDialog": SimpleNamespace(getExistingDirectory=lambda *args, **kwargs: chosen),
    }
    exec(compile(ast.Module(body=found, type_ignores=[]), "REAL-Video-Enhancer.py", "exec"), namespace)
    return {name: namespace[name] for name in names}


_METHODS = (
    "setDefaultOutputFile",
    "derivedOutputFile",
    "_outputDerivation",
    "_output_path_tags",
    "getInterpolationMultiplier",
    "getUpscaleModelScale",
    "openOutputFolder",
)


def _window(tmp_path, *, field: str = "", settings: Settings | None = None, loaded: bool = True) -> SimpleNamespace:
    """A window with the path methods bound; everything else it reads is the fake page above."""
    settings = _settings(tmp_path) if settings is None else settings
    window = SimpleNamespace(
        page_handles=SimpleNamespace(general=_page(field)),
        settings=settings,
        isVideoLoaded=loaded,
        videoFps=24.0,
        videoWidth=1440,
        videoHeight=1080,
        homeDir="/home/user",
    )
    for name, function in _window_methods(*_METHODS).items():
        setattr(window, name, types.MethodType(function, window))
    return window


def _field_of(window) -> _Field:
    return window.page_handles.general.outputFileText


# --- 1. the reported bug: an option change must not overwrite a user's path -----------------------------------


def test_a_user_path_survives_an_option_change(tmp_path):
    """The owner's report: changing an option replaced the path chosen last session.

    The path is non-empty and stored, so `setDefaultOutputFile` - the handler every option checkbox lands on - must
    return it and leave the field alone, even though the options now derive a different name.
    """
    settings = _settings(tmp_path)
    output_path.store_user_path(settings, "/mine/custom.mkv")
    window = _window(tmp_path, field="/mine/custom.mkv", settings=settings)

    # The option change: upscale on, so a derived name would carry the model and the doubled size.
    composed = window._outputDerivation("/fixed/Videos")
    assert output_path.derive(composed) != "/mine/custom.mkv", "the fixture is wrong: the derivation agrees by accident"

    shown = window.setDefaultOutputFile("/videos/clip.mkv", "/fixed/Videos")

    assert shown == "/mine/custom.mkv", "the option change replaced the path the user chose"
    assert _field_of(window).text() == "/mine/custom.mkv"
    assert output_path.stored_path(settings) == "/mine/custom.mkv", "the stored choice was rewritten by a derived default"


def test_a_derived_path_is_refilled_when_the_options_change(tmp_path):
    """The other half of the rule: nothing was chosen, so the name still follows the options (it always has)."""
    settings = _settings(tmp_path)
    window = _window(tmp_path, field="", settings=settings)

    first = window.setDefaultOutputFile("/videos/clip.mkv", "/fixed/Videos")
    assert first == "/fixed/Videos/clip_rife-4.25_realesrgan-x4_48.0fps_2880x2160.mkv"
    assert _field_of(window).text() == first
    assert output_path.stored_path(settings) == "", "a derived default must never be stored as the user's choice"

    # The user turns upscale off: the derived name loses the model tag and the size halves. No path was ever chosen,
    # so the field follows - this is the behaviour that made the name encode the job.
    window.page_handles.general.upscaleCheckBox = _Check(False)
    second = window.setDefaultOutputFile("/videos/clip.mkv", "/fixed/Videos")
    assert second == "/fixed/Videos/clip_rife-4.25_48.0fps_1440x1080.mkv"
    assert _field_of(window).text() == second


# --- 2. an empty field is filled from the directory default ---------------------------------------------------


def test_an_empty_field_is_filled_from_the_directory_default(tmp_path):
    """First use: nothing stored, nothing in the field - the default appears, in the directory it was handed."""
    settings = _settings(tmp_path)
    window = _window(tmp_path, field="", settings=settings)

    shown = window.setDefaultOutputFile("/videos/clip.mkv", "/fixed/Videos")

    assert shown.startswith("/fixed/Videos/")
    assert _field_of(window).text() == shown
    assert output_path.stored_path(settings) == ""


def test_no_video_means_no_path_at_all(tmp_path):
    """The old method answered `None` here, and the Select Output Folder handler then joined it (`TypeError`)."""
    window = _window(tmp_path, field="", loaded=False)
    assert window.setDefaultOutputFile("", "/fixed/Videos") is None
    assert _field_of(window).text() == ""


def test_a_collision_gets_a_numbered_name(tmp_path):
    """The unique-name loop moved with the arithmetic: an existing file must not be silently overwritten."""
    settings = _settings(tmp_path)
    window = _window(tmp_path, field="", settings=settings)
    directory = tmp_path / "out"
    directory.mkdir()
    composed = window._outputDerivation(str(directory))
    taken = output_path.derive(composed)
    Path(taken).write_text("already here", encoding="utf-8")

    shown = window.setDefaultOutputFile("/videos/clip.mkv", str(directory))

    assert shown != taken
    assert shown == f"{taken[: -len('.mkv')]}_(0).mkv", shown


def test_the_batch_sentinel_keeps_its_placeholder_name(tmp_path):
    """A folder load carries `{MULTIPLE_FILES}`; the derived name has always been the `{FILE_NAME}` placeholder."""
    settings = _settings(tmp_path)
    window = _window(tmp_path, field="", settings=settings)
    window.page_handles.general.inputFileText = _Field(" { MULTIPLE_FILES } ")

    shown = window.setDefaultOutputFile(" { MULTIPLE_FILES } ", "/fixed/Videos")

    assert os.path.basename(shown).startswith(" { FILE_NAME } _rife-4.25")


def test_a_batch_add_derives_per_file_instead_of_reusing_the_field(tmp_path):
    """`addToRenderQueue` walks a folder: every file needs its own name even when the field carries a chosen path."""
    settings = _settings(tmp_path)
    output_path.store_user_path(settings, "/mine/custom.mkv")
    window = _window(tmp_path, field="/mine/custom.mkv", settings=settings)

    first = window.derivedOutputFile("/videos/a.mkv", "/fixed/Videos")
    second = window.derivedOutputFile("/videos/b.mkv", "/fixed/Videos")

    assert first != second
    assert os.path.basename(first).startswith("a_rife")
    assert os.path.basename(second).startswith("b_rife")
    assert _field_of(window).text() == "/mine/custom.mkv", "the batch add wrote over the user's path"


# --- 3. persistence, the round-trip, and the old key's migration ----------------------------------------------


def test_the_path_persists_and_restores_across_a_settings_round_trip(tmp_path):
    """A hand-typed path survives a restart - the half of the report about the field coming back wrong."""
    first = _settings(tmp_path)
    output_path.store_user_path(first, "/mine/movies/clip_upscaled.mkv")

    restarted = _settings(tmp_path)  # a new launch reads the same file
    assert output_path.stored_path(restarted) == "/mine/movies/clip_upscaled.mkv"
    assert output_path.user_set(restarted) is True
    assert output_path.stored_directory(restarted) == "/mine/movies"

    window = _window(tmp_path, field="", settings=restarted)
    shown = window.setDefaultOutputFile("/videos/clip.mkv", "/fixed/Videos")

    assert shown == "/mine/movies/clip_upscaled.mkv", "the saved path did not come back into the field"
    assert _field_of(window).text() == "/mine/movies/clip_upscaled.mkv"


def test_an_old_settings_file_with_only_the_directory_key_still_feeds_the_fallback(tmp_path):
    """The migration: a `settings.txt` from before the path key keeps its folder, as the directory fallback."""
    settings = _settings(tmp_path, output_folder_location="/old/place/Videos")
    file = tmp_path / "settings.txt"
    lines = [line for line in file.read_text(encoding="utf-8").splitlines()
             if not line.startswith(f"{output_path.KEY_PATH},")]
    file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    old = _settings(tmp_path)
    assert output_path.stored_path(old) == ""
    assert output_path.user_set(old) is False, "a file with no path key must not read as a chosen path"
    assert output_path.stored_directory(old) == "/old/place/Videos"

    window = _window(tmp_path, field="", settings=old)
    shown = window.setDefaultOutputFile("/videos/clip.mkv", output_path.stored_directory(old))

    assert shown.startswith("/old/place/Videos/"), f"the existing user's folder was lost: {shown}"


def test_clearing_the_field_hands_the_name_back_to_the_derivation(tmp_path):
    """An empty field is the "no choice" state, so the next option change fills it again."""
    settings = _settings(tmp_path)
    output_path.store_user_path(settings, "/mine/custom.mkv")
    assert output_path.user_set(settings) is True

    output_path.store_user_path(settings, "")

    assert output_path.user_set(settings) is False
    window = _window(tmp_path, field="", settings=settings)
    assert window.setDefaultOutputFile("/videos/clip.mkv", "/fixed/Videos").startswith("/fixed/Videos/")


# --- 4. the Encoder row: a read-only mirror, asserted exactly --------------------------------------------------


def test_the_encoder_row_cannot_be_edited(qapp):
    """Asserted exactly: the field is read-only, its picker is gone, and no handle describes a second control."""
    pytest.importorskip("PySide6")
    pytest.importorskip("sli_ui_toolkit")
    if "resources_rc" not in sys.modules:
        sys.modules["resources_rc"] = types.ModuleType("resources_rc")

    from src.ui import toolkit_boot

    toolkit_boot.ensure()
    from src.ui.pages.handles import OutputSettingsHandles
    from src.ui.pages.tab_2 import Tab2Page

    page = Tab2Page()
    page.build()

    assert page.output_folder_location.isReadOnly(), (
        "the Encoder page can still set the output folder - that is the second control the owner reported")
    assert not hasattr(page, "select_output_folder_location_btn"), (
        "the Encoder page's own folder picker survived the merge: it is a second way to the same fact")
    names = {field.name for field in dataclasses.fields(OutputSettingsHandles)}
    assert "output_folder_location" in names
    assert "select_output_folder_location_btn" not in names


def test_the_encoder_row_follows_the_path_s_directory(qapp, tmp_path):
    """The mirror is the path's directory - and the legacy key only while the field names nothing."""
    pytest.importorskip("PySide6")
    pytest.importorskip("sli_ui_toolkit")
    if "resources_rc" not in sys.modules:
        sys.modules["resources_rc"] = types.ModuleType("resources_rc")

    from PySide6.QtWidgets import QLineEdit

    settings = _settings(tmp_path, output_folder_location="/old/place/Videos")
    general = _page("/mine/movies/clip_upscaled.mkv")
    mirror = QLineEdit()

    folder = output_path.mirror_directory(general, mirror, settings)
    assert folder == "/mine/movies"
    assert mirror.text() == "/mine/movies"
    assert mirror.isReadOnly() is False, "the mirror widget itself is generic; the PAGE makes it read-only"

    general.outputFileText = _Field("")
    assert output_path.mirror_directory(general, mirror, settings) == "/old/place/Videos"
    assert mirror.text() == "/old/place/Videos"


# --- 5. the open-folder buttons go to the path's directory -----------------------------------------------------


def _processtab_method(*name: str):
    """`src/ui/ProcessTab.py`'s method, compiled on its own (the module's imports need the whole Qt stack)."""
    source = (ROOT / "src" / "ui" / "ProcessTab.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    tab = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ProcessTab")
    found = [node for node in tab.body if isinstance(node, ast.FunctionDef) and node.name in name]
    assert {node.name for node in found} == set(name)
    opened: list = []
    namespace = {
        "os": os,
        "subprocess": SimpleNamespace(Popen=lambda args, **kwargs: opened.append(args)),
        "PLATFORM": "linux",
    }
    exec(compile(ast.Module(body=found, type_ignores=[]), "ProcessTab.py", "exec"), namespace)
    return namespace[name[0]], opened


def test_the_open_folder_button_opens_the_path_s_directory(tmp_path):
    """The explorer button opens the directory the PATH names - not the legacy settings key beside it."""
    path_dir = tmp_path / "chosen"
    legacy_dir = tmp_path / "legacy"
    path_dir.mkdir()
    legacy_dir.mkdir()

    handler, opened = _processtab_method("openOutputFolderInExplorer")
    tab = SimpleNamespace(page_handles=SimpleNamespace(
        general=SimpleNamespace(outputFileText=_Field(str(path_dir / "clip_upscaled.mkv")))))

    handler(tab)

    assert opened == [["xdg-open", str(path_dir)]], (
        f"the open-folder button went to {opened} instead of the path's own directory {path_dir}")


def test_the_select_folder_action_replaces_the_path_and_persists_the_directory(tmp_path):
    """The one explicit action allowed to win: the user names a DIRECTORY, the app derives the file name in it."""
    settings = _settings(tmp_path)
    output_path.store_user_path(settings, "/mine/custom.mkv")
    chosen = tmp_path / "new-folder"
    chosen.mkdir()
    window = _window(tmp_path, field="/mine/custom.mkv", settings=settings)
    for name, function in _window_methods(*_METHODS, chosen=str(chosen)).items():
        setattr(window, name, types.MethodType(function, window))

    window.openOutputFolder()

    assert _field_of(window).text().startswith(str(chosen) + os.sep), (
        f"the explicit folder choice did not replace the path: {_field_of(window).text()}")
    assert output_path.stored_path(settings) == "", "the derived name was stored as the user's choice"
    assert settings.settings[output_path.KEY_DIRECTORY] == str(chosen)
    assert os.path.dirname(_field_of(window).text()) == str(chosen)


def test_a_cancelled_folder_dialog_changes_nothing(tmp_path):
    """`getExistingDirectory` answers `""` on cancel: no directory write, no path change."""
    settings = _settings(tmp_path)
    output_path.store_user_path(settings, "/mine/custom.mkv")
    window = _window(tmp_path, field="/mine/custom.mkv", settings=settings)
    for name, function in _window_methods(*_METHODS, chosen="").items():
        setattr(window, name, types.MethodType(function, window))

    window.openOutputFolder()

    assert _field_of(window).text() == "/mine/custom.mkv"
    assert output_path.stored_path(settings) == "/mine/custom.mkv"


# --- the invariant behind all five: one writer of the field ----------------------------------------------------


def test_the_app_entry_point_never_writes_the_path_field_itself():
    """Every write of the field goes through `output_path.show_path`, so the rule cannot be bypassed by a caller."""
    source = (ROOT / "REAL-Video-Enhancer.py").read_text(encoding="utf-8")
    assert "outputFileText.setText(" not in source, (
        "the entry point writes the output path directly - route it through src/ui/output_path.py")


def test_the_option_change_handler_lands_on_the_guarded_owner():
    """`updateVideoGUIDetails` (every option checkbox) -> `setDefaultOutputFile` -> `apply_default`."""
    source = (ROOT / "REAL-Video-Enhancer.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    window = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MainWindow")
    bodies = {node.name: ast.get_source_segment(source, node) or ""
              for node in window.body if isinstance(node, ast.FunctionDef)}
    assert "self.setDefaultOutputFile(" in bodies["updateVideoGUIDetails"]
    assert "output_path_io.apply_default(" in bodies["setDefaultOutputFile"]
    assert "setText(" not in bodies["setDefaultOutputFile"]
    assert "output_path_io.derive(" in bodies["derivedOutputFile"]


def test_only_the_owner_module_writes_the_path_field():
    """The grep form of the invariant (the repo's "a second writer is the defect" test)."""
    writers = sorted(
        str(path.relative_to(ROOT))
        for path in (ROOT / "src").rglob("*.py")
        if "outputFileText.setText(" in path.read_text(encoding="utf-8")
    )
    assert writers == ["src/ui/output_path.py"], f"second writer(s) of the output path field: {writers}"
