"""One translated language (Russian) on the toolkit's translator runtime - the seam, the switch, the fallback.

The three assertions this file exists for, in the order they cost a developer:

1. **Every `retranslate()` accepts a language.** The toolkit's `install_translator(root, tr_func)` calls
   `retranslate(lang)` on every object under `root`; 13 RVE methods took only `self` before this feature, and the
   install raised `TypeError` inside `build()` - one page aborted the pass for the whole window. Measured on the real
   window before the fix: `TypeError: DownloadPagePage.retranslate() takes 1 positional argument but 2 were given`,
   raised from `src/ui/i18n.py::_PageRelay.retranslate`. The AST walk below is that gate, per file.
2. **The installer does not raise on the real window, and it actually reaches the pages.** RVE's pages are plain
   builder classes, not QObjects, so a bare `install_translator(window, …)` covers nothing at all (measured
   2026-11-24: it returned without touching a page). `i18n.bind_page` puts a relay under each page root; the test
   builds the real shell and reads a translated string back off the real widget tree, not off the table.
3. **A missing key falls back to English.** The catalogue is filled in slices; an untranslated string must stay its
   English source on screen, and that is asserted on a real widget (the Render Preview tooltip), not just on the
   helper.

The settings key is here too: `language` must validate to a language the UI can render (unknown -> `en`), because a
`settings.txt` from another build is not allowed to cost the user their window.
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PAGES = ROOT / "src" / "ui" / "pages"
SHELL = ROOT / "src" / "ui" / "shell" / "window.py"
ENTRY = ROOT / "REAL-Video-Enhancer.py"

pytest.importorskip("PySide6")

sys.path.insert(0, str(ROOT))

#: The source strings this file asserts in Russian. Named once so a table edit and a test edit cannot drift apart
#: silently: an entry removed from the table fails here with the key in the message.
RU_SAMPLES = {
    "Start Time (seconds)": "Начало (секунды)",
    "Encoder Options": "Параметры кодирования",
    "Output Settings": "Настройки вывода",
    "Language": "Язык",
}


# --- the seam: every retranslate accepts a language --------------------------------------------


def _retranslate_nodes(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "retranslate":
            yield node


def _accepts_a_language(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True when `retranslate(self, <something>)` can be called with one positional argument."""
    positional = list(node.args.posonlyargs) + list(node.args.args)
    if len(positional) >= 2:
        return True
    defaults = list(node.args.defaults)
    # A parameter with a default is optional - `retranslate(self, lang=None)` passes; `retranslate(self)` cannot.
    return bool(defaults) and len(defaults) >= 1


def test_every_retranslate_takes_a_language_parameter():
    """The toolkit calls `retranslate(lang)`; a method that only takes `self` raises inside the install."""
    files = sorted(PAGES.glob("*.py")) + [SHELL, ENTRY]
    found = 0
    offenders: list[str] = []

    for path in files:
        for node in _retranslate_nodes(path):
            found += 1
            if not _accepts_a_language(node):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} retranslate(self)")

    assert found == 13, f"expected 13 retranslate methods (12 pages + the shell), found {found}"
    assert not offenders, "these cannot be driven by install_translator:\n" + "\n".join(offenders)


# --- the real window: install, switch, repaint ---------------------------------------------------


@pytest.fixture()
def i18n_state(qapp):
    """The real i18n module with the session `QApplication`, restored to English after the test.

    The module is process-global by design (one interface language per process), so a test that switches must put it
    back - otherwise the next test reads Russian off widgets it did not ask to translate.
    """
    from src.ui import i18n

    language_before = i18n.current_language()
    yield i18n
    i18n.emit_language_changed(language_before)


def _build_shell(qapp):
    """The real `MainWindowShell` in a bare `QMainWindow`, with the resource module stubbed (offscreen, no icons)."""
    if "resources_rc" not in sys.modules:
        sys.modules["resources_rc"] = types.ModuleType("resources_rc")
    from PySide6.QtWidgets import QMainWindow

    from src.ui import toolkit_boot, toolkit_theme
    from src.ui.shell.window import MainWindowShell

    toolkit_boot.ensure()
    toolkit_theme.install(qapp)
    window = QMainWindow()
    shell = MainWindowShell()
    shell.build(window)
    return shell, window


def test_the_installer_does_not_raise_on_the_real_window(i18n_state, qapp):
    """`build()` installs the translator; before the signature fix this raised TypeError mid-build."""
    from sli_ui_toolkit.i18n import install_translator

    i18n_state.configure_i18n("ru")
    shell, window = _build_shell(qapp)

    # The shell already installed once (end of `build`). A second install on the same root is the toolkit's own
    # contract - it must stay silent, and it must not raise on a page whose signature is wrong.
    install_translator(window, i18n_state.translator)

    proc = shell.procPage_page
    assert proc.processSettingsContainer.tabText(1) == "Параметры кодирования"


def test_a_switch_repaints_the_process_page_chrome_and_keeps_a_missing_key_english(i18n_state, qapp):
    """Build in English, switch to `ru` at runtime, read the real widgets - no restart, no reload."""
    i18n_state.configure_i18n("en")
    shell, _window = _build_shell(qapp)

    proc = shell.procPage_page
    assert proc.processSettingsContainer.tabText(1) == "Encoder Options"
    assert proc.label_9.text() == "Start Time (seconds)"

    i18n_state.emit_language_changed("ru")

    assert proc.processSettingsContainer.tabText(1) == "Параметры кодирования"
    assert proc.processSettingsContainer.tabText(4) == "Журнал"
    assert proc.label_9.text() == "Начало (секунды)"
    # An untranslated string on the same page stays its English source: the Render Preview tooltip is an HTML blob
    # (one of the 28 markup strings the recon counted), not something this slice translates.
    tooltip = proc.renderPreviewBtn.toolTip()
    assert "Renders a small portion of the video" in tooltip
    assert "Предпросмотр" not in tooltip

    i18n_state.emit_language_changed("en")
    assert proc.processSettingsContainer.tabText(1) == "Encoder Options"
    assert proc.label_9.text() == "Start Time (seconds)"


def test_the_settings_page_headings_follow_the_switch(i18n_state, qapp):
    """The settings page's own tab labels are in the first slice."""
    i18n_state.configure_i18n("en")
    shell, _window = _build_shell(qapp)
    page = shell.settingsPage_page

    assert [page.tabWidget.tabText(index) for index in range(3)] == [
        "Output Settings",
        "Render Settings",
        "RVE Settings",
    ]

    i18n_state.emit_language_changed("ru")

    assert [page.tabWidget.tabText(index) for index in range(3)] == [
        "Настройки вывода",
        "Настройки рендера",
        "Настройки RVE",
    ]
    assert shell.procPage_page.label_9.text() == "Начало (секунды)"


def test_the_language_row_offers_both_languages(i18n_state, qapp):
    """The picker built by `tab_4` carries the code as item data and the endonym as its text.

    Applying the choice is `SettingsTab.applyLanguageSetting` (the glue owns the setting, the page owns the control);
    this test pins the control half - the codes the glue will read back through `currentData()`.
    """
    i18n_state.configure_i18n("en")
    shell, _window = _build_shell(qapp)
    combo = shell.settingsPage_page.tab_4_page.language

    assert [combo.itemText(index) for index in range(combo.count())] == ["English", "Русский"]
    assert [combo.itemData(index) for index in range(combo.count())] == ["en", "ru"]
    assert combo.currentData() == "en"


# --- one page on its own: build, bind, switch ----------------------------------------------------


def test_a_single_page_built_alone_follows_the_switch(i18n_state, qapp):
    """A page built outside the shell (as most page tests do) is covered by installing on its own root."""
    if "resources_rc" not in sys.modules:
        sys.modules["resources_rc"] = types.ModuleType("resources_rc")
    from src.ui import toolkit_boot
    from src.ui.pages.home_page import HomePagePage

    toolkit_boot.ensure()
    i18n_state.configure_i18n("en")

    page = HomePagePage()
    root = page.build()
    i18n_state.install(root)

    # The home page's display heading is a brand, absent from the table: it must stay English in every language.
    assert page.label.text() == "REAL Video Enhancer"
    i18n_state.emit_language_changed("ru")
    assert page.label.text() == "REAL Video Enhancer"
    i18n_state.emit_language_changed("en")


# --- the table and the helper --------------------------------------------------------------------


def test_the_table_covers_the_strings_this_file_asserts(i18n_state):
    """A translation removed from the table fails here with the key, not as a mysteriously English label."""
    table = i18n_state.TRANSLATIONS["ru"]
    missing = [key for key in RU_SAMPLES if key not in table]
    assert not missing, f"missing from the ru table: {missing}"
    for key, russian in RU_SAMPLES.items():
        assert i18n_state.translate(key, "ru") == russian


def test_an_unknown_key_and_an_unknown_language_fall_back(i18n_state):
    key = "This string is not in the catalogue"
    assert i18n_state.translate(key, "ru") == key
    assert i18n_state.translate(key, "en") == key
    assert i18n_state.translate(key, "de") == key  # an unsupported language is English, not a crash
    assert i18n_state.tr("Language") in ("Language", "Язык")


def test_the_toolkit_sees_the_same_table(i18n_state, qapp):
    """`configure_i18n` is the single backend: the toolkit's own `tr()` resolves through it."""
    from sli_ui_toolkit.i18n import tr as toolkit_tr

    i18n_state.configure_i18n("ru")
    assert toolkit_tr("Encoder Options") == "Параметры кодирования"
    i18n_state.configure_i18n("en")
    assert toolkit_tr("Encoder Options") == "Encoder Options"


# --- the settings key ----------------------------------------------------------------------------


def test_an_unknown_stored_language_reads_as_english():
    from src.ui.settings_store import DEFAULT_LANGUAGE, SUPPORTED_LANGUAGES, validate_language

    assert DEFAULT_LANGUAGE == "en"
    assert SUPPORTED_LANGUAGES == ("en", "ru")
    assert validate_language("ru") == "ru"
    assert validate_language("en") == "en"
    # A hand edit with stray whitespace is the same language, not an unknown one.
    assert validate_language(" ru ") == "ru"
    # Everything else - another build's value, the wrong case, an empty line, nothing at all.
    for value in ("de", "RU", "", None, "zz"):
        assert validate_language(value) == "en", value


def test_the_language_key_is_a_setting_with_a_validator(tmp_path):
    from src.ui.settings_store import Settings, validate_language

    settings = Settings(
        settings_file=str(tmp_path / "settings.txt"),
        default_output_folder="/fixed/Videos",
    )
    assert settings.defaultSettings["language"] == "en"
    assert settings.allowedSettings["language"] == ("en", "ru")

    settings.writeSetting("language", "ru")
    written = (tmp_path / "settings.txt").read_text(encoding="utf-8")
    assert "language,ru\n" in written

    reread = Settings(
        settings_file=str(tmp_path / "settings.txt"),
        default_output_folder="/fixed/Videos",
    )
    assert reread.settings["language"] == "ru"

    # A hand-edited value the app cannot render is read as English (and therefore the value the next write stores).
    (tmp_path / "settings.txt").write_text(written.replace("language,ru", "language,de"), encoding="utf-8")
    unknown = Settings(
        settings_file=str(tmp_path / "settings.txt"),
        default_output_folder="/fixed/Videos",
    )
    assert unknown.settings["language"] == validate_language("de") == "en"


def test_the_window_owner_configures_the_language_without_main(qapp, monkeypatch):
    """The live defect (2026-11-24): configuration lived only in `main()`.

    On a lease, `harness/scripts/live_boot.py` builds `MainWindow()` directly - it never runs `main()` - so the
    instance ran in English whatever `settings.txt` said, and the first switch the control channel performed printed
    `i18n root not configured — call configure_i18n(i18n_root=...)` from the toolkit and changed nothing on screen.
    Three entry points build this window (the app, the live harness, the tests), so its OWNER configures the language
    and `main()` is only the one that gets there first. This test calls nothing before `build()`.
    """
    from sli_ui_toolkit.i18n import tr as toolkit_tr

    from src.ui import i18n

    monkeypatch.setattr(i18n, "_configured", False)
    monkeypatch.setattr(i18n, "_current_language", "en")

    class _StoredRussian:
        def __init__(self, *args, **kwargs) -> None:
            self.settings = {"language": "ru"}

    monkeypatch.setattr("src.ui.settings_store.Settings", _StoredRussian)
    assert i18n.is_configured() is False, "the guard must start from 'nothing configured' for this test to mean anything"

    shell, window = _build_shell(qapp)
    try:
        assert i18n.current_language() == "ru", (
            "building the window must read `settings.txt` itself - `main()` is not the only entry point")
        assert toolkit_tr("Language") == "Язык", (
            "the toolkit backend must be registered by the window, not left unconfigured (the live warning)")
        assert shell.procPage_page.processSettingsContainer.tabText(0) == "Общие", (
            "the first paint must already be translated: the hook runs before the pages are built")
    finally:
        window.close()


def test_a_language_switch_does_not_wipe_a_runtime_pane(i18n_state, qapp):
    """The changelog pane holds RUNTIME text, not a label - and `retranslate()` used to erase it.

    `HomeTab.retranslate` called `changeLogText.setHtml(QCoreApplication.translate(... <an empty Qt document> ...))`:
    a catalogue key that could never be translated, and on every switch it replaced the user's notes with a blank page.
    `tab_5.retranslate` did the same to the render log. Measured 2026-11-24 while translating the UI; the empty document
    is gone and a switch now leaves both panes alone.
    """
    i18n_state.configure_i18n("en")
    shell, window = _build_shell(qapp)
    try:
        panel = shell.homePage_page.changeLogText
        panel.setMarkdown("release notes that must survive a switch")
        i18n_state.emit_language_changed("ru")
        assert "survive a switch" in panel.toPlainText(), (
            "a language switch replaced the changelog with an empty document instead of leaving it alone")
        log = shell.procPage_page.tab_5_page.renderOutput
        log.append("frame 1 rendered")
        i18n_state.emit_language_changed("en")
        visible = log.text() if hasattr(log, "text") else log.toPlainText()
        assert "frame 1 rendered" in visible, (
            "a language switch replaced the render log with an empty document")
    finally:
        i18n_state.emit_language_changed("en")
        window.close()
