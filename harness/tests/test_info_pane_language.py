"""The General tab's video-info pane in Russian: nine row names that follow the interface language.

The pane is one `RichTextView` document (`src/ui/pages/general_settings.py::_info_html`). Its nine row names were
built once, in English, at construction: every label around them turned on a language switch and these nine stayed
English (measured live in `ru`, 2026-11-24). Two writers own the pane, and a switch has to answer for both:

* **this page** paints the nine-row placeholder (the names alone, at rest);
* **`MainWindow.updateVideoGUIText`** replaces it with the loaded video's values - the same nine names, each followed
  by its values.

So the assertions below are: the names are Russian after a switch, the values of a loaded pane are byte-identical
across it, and a switch back to English restores the English names (the rewrite has to recognise the spelling it
painted a moment ago, or a second switch would do nothing).
"""

from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("sli_ui_toolkit")

if "resources_rc" not in sys.modules:
    sys.modules["resources_rc"] = types.ModuleType("resources_rc")

from src.ui import i18n  # noqa: E402
from src.ui.pages.general_settings import INFO_LABELS, GeneralSettingsPage  # noqa: E402

#: The nine names in Russian, in the pane's own order. The English keys are the table's keys, byte for byte.
RU_NAMES = {
    "FPS:": "К/с:",
    "Resolution:": "Разрешение:",
    "Frame Count:": "Количество кадров:",
    "Encoder:": "Кодировщик:",
    "Container:": "Контейнер:",
    "Color Space:": "Цветовое пространство:",
    "Pixel Format:": "Формат пикселей:",
    "Is HDR:": "Поддержка HDR:",
    "Bit Depth:": "Глубина цвета:",
}

#: The pane's English rows, as the page paints them at rest.
EN_NAMES = list(INFO_LABELS)

#: The values body `updateVideoGUIText` writes over the placeholder (`setText`, one line per row, `->` for
#: "before -> after"). Copied in shape from `REAL-Video-Enhancer.py:437-448` so a switch is proven against the
#: document a loaded video really produces.
LOADED_VALUES = (
    "FPS: 30.0 -> 60.0\n"
    "Resolution: 1920x1080 -> 3840x2160\n"
    "Frame Count: 900 -> 1800\n"
    "Encoder: h264 -> libx264\n"
    "Container: mp4 -> mkv\n"
    "Color Space: bt709 -> bt709\n"
    "Pixel Format: yuv420p -> yuv420p\n"
    "HDR: False -> False\n"
    "Bit Depth: 8 bit -> 8 bit\n"
)


@pytest.fixture()
def language(qapp):
    """The real i18n module, configured for English and restored to the previous language afterwards."""
    before = i18n.current_language()
    i18n.configure_i18n("en")
    yield i18n
    i18n.emit_language_changed(before)
    i18n.configure_i18n(before)


def _page(qapp) -> GeneralSettingsPage:
    """A real General Settings page, shown, with the translator installed on its own root."""
    from src.ui import toolkit_boot, toolkit_theme
    from sli_ui_toolkit.ui.managers.ui_scale import UiScale

    UiScale.get_instance().set_factor(1.0)
    toolkit_boot.ensure()
    toolkit_theme.install(qapp)
    page = GeneralSettingsPage()
    root = page.build()
    i18n.install(root)
    root.resize(900, 900)
    root.show()
    for _ in range(6):
        qapp.processEvents()
    return page


def _rows(page) -> list[str]:
    return page.inputVideoInfoTextEdit.text().split("\n")


# --- the table -----------------------------------------------------------------------------------


def test_the_table_carries_all_nine_names(language):
    """A name missing from the catalogue would fall back to English and look like a half-done page."""
    table = language.TRANSLATIONS["ru"]
    missing = [name for name in EN_NAMES if name not in table]
    assert not missing, f"missing from the ru table: {missing}"
    for name, russian in RU_NAMES.items():
        assert language.translate(name, "ru") == russian
    # The keys are the English source strings, byte for byte - a trailing space would be a different key.
    assert all(language.translate(name, "en") == name for name in EN_NAMES)


# --- the placeholder: build, switch, switch back -------------------------------------------------


def test_english_first_paint_is_the_nine_english_names(language, qapp):
    page = _page(qapp)
    assert _rows(page) == EN_NAMES


def test_a_switch_repaints_the_nine_names_in_place(language, qapp):
    """No restart: the built page's document is rewritten when the language changes, and back again."""
    page = _page(qapp)
    pane = page.inputVideoInfoTextEdit
    assert _rows(page) == EN_NAMES

    language.emit_language_changed("ru")

    assert _rows(page) == [RU_NAMES[name] for name in EN_NAMES], (
        f"the pane still reads {pane.text()!r} after a switch to ru")
    # The document is still the same one-document rich body: the scaled inline size survives the rewrite.
    assert "font-size:10pt" in pane._markup
    assert "font-size:11pt" in pane._markup

    language.emit_language_changed("en")
    assert _rows(page) == EN_NAMES


# --- a loaded video: the names turn, the values do not -------------------------------------------


def test_a_loaded_panes_value_columns_are_byte_identical_after_the_switch(language, qapp):
    """The strict form: strip the leading name from every row and compare the value columns."""
    page = _page(qapp)
    pane = page.inputVideoInfoTextEdit
    pane.setText(LOADED_VALUES)

    def values(text: str) -> list[str]:
        rows = []
        for line in text.split("\n"):
            prefix = next((spelling for key in EN_NAMES for spelling in (key, RU_NAMES[key])
                           if line.startswith(spelling)), None)
            rows.append(line[len(prefix):] if prefix is not None else line)
        return rows

    before = values(pane.text())
    language.emit_language_changed("ru")
    after = pane.text()
    assert values(after) == before, f"the switch changed a value column:\n{after!r}"
    assert after.startswith("К/с: 30.0 -> 60.0"), after
    assert "Разрешение: 1920x1080 -> 3840x2160" in after
    assert "Глубина цвета: 8 bit -> 8 bit" in after

    language.emit_language_changed("en")
    assert pane.text() == LOADED_VALUES
