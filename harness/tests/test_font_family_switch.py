"""The interface face is a setting: one key, one selector, one apply path through the toolkit's `UiFont`.

`harness/tests/test_window_typography.py` already proves the mechanism - `UiFont.set_family` reaches an already-built
`Text` and `Button`. This file pins the feature built on it: RVE Settings offers the families this machine can render
(the bundled face first) above the app's own default, the choice is persisted as `font_family`, a restart opens in it,
and a live switch repaints the built tree without rebuilding it.

The default is the face the app has always opened in: an EMPTY `font_family` means "no `UiFont` override", so the
application face `toolkit_theme.install` states stays in force - the face every layout pin in `harness/tests` was
measured against. The bundled Fira Mono is always listed and reachable, but choosing it is a user act, not the
first launch.

Red before the change: `grep -n "font" src/ui/*.py` found no selector, no persisted key and no apply, so on the
pre-change tree `DEFAULT_FONT_FAMILY` does not import, `Tab4Page` has no `font_family` control, and `UiFont.family()`
never changes when the combo does. Every test below fails there for that reason (measured on the pre-change tree with
the same file: 10 failed; after: 10 passed).

All of it is offscreen and reads the resolved `QFontInfo` family - the requested string would have been green even
while Qt painted a fallback face (the trap `test_bundled_fonts.py` documents). Pixels are a leased-display proof.
"""

from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("sli_ui_toolkit")

from PySide6.QtGui import QFont, QFontDatabase, QFontInfo  # noqa: E402
from PySide6.QtWidgets import QApplication, QMainWindow  # noqa: E402

#: The body label the live switch is read from - a toolkit `Text` the shell builds on the General tab.
BODY_LABEL = "encoderLabel_4"

#: A shot at a face every desktop build carries; only used to pick a *resolvable* second family, never asserted.
FAMILY_PREFERENCES = ("DejaVu Sans", "Liberation Sans", "Noto Sans", "FreeSans")


def _settings(tmp_path):
    """The app's settings owner over a temp file, exactly as the window's owner constructs it."""
    from src.ui.settings_store import Settings

    return Settings(
        settings_file=str(tmp_path / "settings.txt"),
        default_output_folder=str(tmp_path / "Videos"),
    )


def _build(qapp, settings=None):
    """The real shell in a bare `QMainWindow`, offscreen, resources stubbed (the `test_window_geometry_persist` shape)."""
    if "resources_rc" not in sys.modules:
        sys.modules["resources_rc"] = types.ModuleType("resources_rc")
    from src.ui import toolkit_boot, toolkit_theme
    from src.ui.shell.window import MainWindowShell

    toolkit_boot.ensure()
    toolkit_theme.install(qapp)
    window = QMainWindow()
    if settings is not None:
        # The app sets this before `MainWindowShell.build`; the shell reads the face from the window's owner.
        window.settings = settings
    builder = MainWindowShell()
    builder.build(window)
    window.show()
    _settle(qapp)
    return builder, window


def _settle(qapp, turns: int = 8) -> None:
    for _ in range(turns):
        qapp.processEvents()


@pytest.fixture()
def font_state():
    """Leave the process-wide face where it was: the `UiFont` singleton outlives every test's window."""
    from sli_ui_toolkit.ui.managers import UiFont

    from src.ui import i18n

    ui_font = UiFont.get_instance()
    language_before = i18n.current_language()
    yield ui_font
    ui_font.set_family(None)
    i18n.configure_i18n(language_before)


def _other_family(qapp, default: str) -> str:
    """An installed family Qt actually resolves, other than `default` - the live switch's target."""
    available = set(QFontDatabase.families())
    candidates = [name for name in FAMILY_PREFERENCES if name in available and name != default]
    candidates += [name for name in sorted(available) if name and name != default and name not in candidates]
    for family in candidates:
        if QFontInfo(QFont(family)).family() == family:
            return family
    pytest.skip("this interpreter registers only one resolvable font family")


def _combo(builder):
    return builder.settingsPage_page.tab_4_page.font_family


def _body_label(window):
    from sli_ui_toolkit.widgets import Text

    label = window.findChild(Text, BODY_LABEL)
    assert label is not None, f"the shell did not build the {BODY_LABEL!r} label"
    return label


def _resolved_family(widget) -> str:
    """The family Qt actually resolves for the widget's font - what the user sees."""
    return QFontInfo(widget.font()).family()


def _application_family() -> str:
    """The face name the application font carries - what "no family chosen" means to `UiFont`."""
    return QApplication.instance().font().family()


def _resolved_application_family() -> str:
    """The family Qt actually resolves for the application font ("Sans Serif" is an alias, not a face name)."""
    return QFontInfo(QApplication.instance().font()).family()


# --- the validator is the store's, and it refuses instead of crashing ---------------------------------------------


def test_the_default_is_the_application_face_not_a_pinned_bundle():
    """An absent value must not pin a face: it means "no override", which is the look every layout pin measured."""
    from src.ui.settings_store import DEFAULT_FONT_FAMILY, validate_font_family

    assert DEFAULT_FONT_FAMILY == ""
    for absent in (None, "", "   "):
        assert validate_font_family(absent) == DEFAULT_FONT_FAMILY


def test_an_installed_family_is_kept():
    from src.ui.settings_store import validate_font_family

    assert validate_font_family("DejaVu Sans", ("DejaVu Sans", "Fira Mono")) == "DejaVu Sans"


def test_an_unknown_family_is_refused_and_named(monkeypatch):
    """The refusal is a log line naming the value: a settings file from another machine must not silently choose."""
    from src.ui import settings_store

    messages: list[str] = []
    monkeypatch.setattr(settings_store, "log", messages.append)

    got = settings_store.validate_font_family("No Such Face 12345", ("Fira Mono",))
    assert got == settings_store.DEFAULT_FONT_FAMILY
    assert any("No Such Face 12345" in message for message in messages), messages


def test_the_key_round_trips_through_the_settings_file(tmp_path):
    from src.ui.settings_store import DEFAULT_FONT_FAMILY, Settings

    settings = _settings(tmp_path)
    assert settings.settings["font_family"] == DEFAULT_FONT_FAMILY
    assert "font_family,\n" in (tmp_path / "settings.txt").read_text(encoding="utf-8")

    reread = Settings(
        settings_file=str(tmp_path / "settings.txt"),
        default_output_folder=str(tmp_path / "Videos"),
    )
    assert reread.settings["font_family"] == DEFAULT_FONT_FAMILY


# --- the restart path: a saved value is applied at build -----------------------------------------------------------


def test_the_window_opens_in_the_saved_family(qapp, tmp_path, font_state):
    """The row is restored from `settings.txt` and the built labels are painted in it."""
    family = _other_family(qapp, _application_family())
    settings = _settings(tmp_path)
    settings.writeSetting("font_family", family)

    builder, window = _build(qapp, settings)
    try:
        assert font_state.family() == family, "the builder did not apply the saved family through UiFont"
        assert _combo(builder).currentData() == family, "the selector does not show the family in force"
        assert _resolved_family(_body_label(window)) == family, "a built label is not painted in the saved face"
    finally:
        window.close()


def test_a_window_without_settings_opens_in_the_default_face(qapp, font_state):
    """A bare `QMainWindow` (tests, live harness) has no store; that is the first launch, not a missing face."""
    builder, window = _build(qapp, None)
    try:
        assert font_state.family() == _application_family()
        assert _combo(builder).currentData() == ""
        assert _resolved_family(_body_label(window)) == _resolved_application_family()
    finally:
        window.close()


def test_a_garbled_stored_family_falls_back_to_the_default(qapp, tmp_path, font_state, monkeypatch):
    """A settings file from another machine cannot ask Qt for a face nobody has - and the log names the refusal."""
    from src.ui import settings_store

    messages: list[str] = []
    monkeypatch.setattr(settings_store, "log", messages.append)

    settings = _settings(tmp_path)
    settings.writeSetting("font_family", "No Such Face 12345")

    builder, window = _build(qapp, settings)
    try:
        assert font_state.family() == _application_family()
        assert _combo(builder).currentData() == ""
        assert any("No Such Face 12345" in message for message in messages), messages
    finally:
        window.close()


# --- the live path: the selector repaints the built tree, no rebuild ----------------------------------------------


def test_the_selector_lists_the_machines_families_under_the_default(qapp, tmp_path, font_state):
    """The row offers the app's default and then exactly what this machine reports, the bundled face first."""
    from src.ui.typography import available_families

    builder, window = _build(qapp, _settings(tmp_path))
    try:
        combo = _combo(builder)
        listed = [combo.itemData(index) for index in range(combo.count())]
        assert listed[0] == "", "the first entry must mean 'the application's own face'"
        assert listed[1:] == list(available_families()), (
            "the row does not offer exactly this machine's families")
        assert combo.currentData() == ""
        assert "Fira Mono" in listed[1:], "the bundled face is not offered"
        assert len(listed) > 2, "the machine reports no second family, so the switch has nothing to switch to"
    finally:
        window.close()


def test_a_live_choice_repaints_the_built_tree_without_rebuilding_it(qapp, tmp_path, font_state):
    """Set the combo on (rather than calling UiFont) and read the change off an already-built label.

    The page objects and the label must be the SAME objects afterwards: this is `UiFont.set_family` repainting the
    tree, not a rebuild. The choice is persisted for the next launch.
    """
    from sli_ui_toolkit.widgets import Text
    from src.ui.settings_store import DEFAULT_FONT_FAMILY

    settings = _settings(tmp_path)
    builder, window = _build(qapp, settings)
    try:
        combo = _combo(builder)
        page_root = builder.settingsPage
        page = builder.settingsPage_page
        tab = builder.settingsPage_page.tab_4_page
        label = window.findChild(Text, BODY_LABEL)
        assert label is not None, f"the shell did not build the {BODY_LABEL!r} label"
        assert _resolved_family(label) == _resolved_application_family()

        family = _other_family(qapp, _application_family())
        combo.setCurrentIndex(combo.findData(family))
        _settle(qapp)

        assert font_state.family() == family, "the selector did not reach UiFont"
        assert _resolved_family(label) == family, "an already-built label kept the old face"
        assert window.findChild(Text, BODY_LABEL) is label, "the label was rebuilt instead of repainted"
        assert builder.settingsPage is page_root
        assert builder.settingsPage_page is page
        assert builder.settingsPage_page.tab_4_page is tab
        assert _combo(builder) is combo

        assert settings.settings["font_family"] == family, "the live choice was not persisted"
        assert settings.settings["font_family"] != DEFAULT_FONT_FAMILY
        assert f"font_family,{family}\n" in (tmp_path / "settings.txt").read_text(encoding="utf-8")
    finally:
        window.close()


def test_choosing_the_default_again_returns_to_the_application_face(qapp, tmp_path, font_state):
    """The first entry is a real choice too: it must clear the override and be saved, not be skipped as "empty"."""
    settings = _settings(tmp_path)
    family = _other_family(qapp, _application_family())
    settings.writeSetting("font_family", family)

    builder, window = _build(qapp, settings)
    try:
        combo = _combo(builder)
        label = _body_label(window)
        assert font_state.family() == family
        assert _resolved_family(label) == family

        combo.setCurrentIndex(combo.findData(""))
        _settle(qapp)

        assert font_state.family() == _application_family(), "the default entry did not drop the override"
        assert _resolved_family(label) == _resolved_application_family(), "the built label kept the old face"
        assert settings.settings["font_family"] == "", "returning to the default was not persisted"
        assert "font_family,\n" in (tmp_path / "settings.txt").read_text(encoding="utf-8")
    finally:
        window.close()
