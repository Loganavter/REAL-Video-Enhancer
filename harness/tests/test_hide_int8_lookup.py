"""The int8 preset is removed by KEY, not by its translated label.

Settings audit (2026-11-27, F1): `hide_int8_preset_if_unsupported` looked the option up with `findText("Fast")`, while
`general_settings.py:1149` fills that item with `QCoreApplication.translate("MainWindow", u"Fast")`. Under `ru` the item
reads "Быстрое", the lookup missed, the int8 preset stayed offered on a card that cannot run it, the log still said
"Fast (int8) hidden", and `quality_preset` was rewritten to `best` while the combo kept showing Fast - the stored value
and the visible control disagreeing about which body the render runs.

The combo already carries the value as item data (`setItemData(0, "best")`, `setItemData(1, "fast")`), which is what the
writer reads (`currentData()`). This file pins the two halves: the item goes away under any language, and the log stops
claiming a removal that did not happen.
"""

from __future__ import annotations

import sys
import types

import pytest
from PySide6.QtWidgets import QComboBox

if "resources_rc" not in sys.modules:
    sys.modules["resources_rc"] = types.ModuleType("resources_rc")


class _Settings:
    def __init__(self, **values):
        self.settings = dict(values)
        self.writes: list[tuple[str, str]] = []

    def writeSetting(self, key, value):
        self.settings[key] = value
        self.writes.append((key, value))


class _General:
    def __init__(self, combo):
        self.qualityPresetComboBox = combo


class _Handles:
    def __init__(self, combo):
        self.general = _General(combo)


class _Parent:
    def __init__(self, combo):
        self.page_handles = _Handles(combo)


def _combo(items: list[tuple[str, str]]) -> QComboBox:
    combo = QComboBox()
    for text, key in items:
        combo.addItem(text)
        combo.setItemData(combo.count() - 1, key)
    combo.setCurrentIndex(0)
    return combo


@pytest.fixture()
def no_tensor_cores(monkeypatch):
    from src.ui import GPUDetect

    monkeypatch.setattr(GPUDetect, "int8_body_available", lambda: (False, "no int8 tensor cores"))


@pytest.fixture()
def logged(monkeypatch):
    from src.ui import SettingsTab

    lines: list[str] = []
    monkeypatch.setattr(SettingsTab, "log", lambda message, *a, **k: lines.append(str(message)))
    return lines


@pytest.mark.parametrize("labels", [
    ("Best quality", "Fast"),
    ("Лучшее качество", "Быстрое"),      # `ru`
], ids=["en", "ru"])
def test_fast_is_removed_by_key_under_any_language(labels, no_tensor_cores, logged, qapp):
    from src.ui.SettingsTab import hide_int8_preset_if_unsupported

    combo = _combo([(labels[0], "best"), (labels[1], "fast")])
    settings = _Settings(quality_preset="fast")

    hide_int8_preset_if_unsupported(_Parent(combo), settings)

    assert combo.count() == 1
    assert combo.itemData(0) == "best"
    assert combo.currentData() == "best", "the visible control must show the value that was stored"
    assert settings.settings["quality_preset"] == "best"
    assert any("hidden" in line for line in logged)


def test_the_log_does_not_claim_a_removal_that_did_not_happen(no_tensor_cores, logged, qapp):
    from src.ui.SettingsTab import hide_int8_preset_if_unsupported

    combo = _combo([("Best quality", "best")])          # this card never offered Fast
    settings = _Settings(quality_preset="fast")

    hide_int8_preset_if_unsupported(_Parent(combo), settings)

    assert not any("hidden" in line for line in logged), logged
    assert any("not offered" in line for line in logged), logged
    # A value the combo does not offer cannot stay stored.
    assert settings.settings["quality_preset"] == "best"
