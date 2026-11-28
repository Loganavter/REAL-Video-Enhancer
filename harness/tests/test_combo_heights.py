"""One height for every combobox in the shipped shell, at every UI factor.

Owner, 2026-11-28: *"make every combobox's open button one height in the app"*. Measured on the built shell at 1.0
before this fix: twenty comboboxes, TWO heights - fourteen at 33 px and six at 30 px. The six were the General tab's
model/backend fields, which that page capped with its settings-row literal (30), and `ComboBox.BASE_HEIGHT` is 33 for
all of them: they rendered 3 px BELOW their own `minimumSizeHint().height()` and their chevron read squashed.

The pins below are the contract:

* every ``ComboBox`` / ``ScrollableComboBox`` in the real shell has ONE height, equal to the toolkit's design height
  resolved through ``scaled_px`` at the factor in force (33 / 41 / 50 / 66 at 1.0 / 1.25 / 1.5 / 2.0);
* no combo is shorter than its own `minimumSizeHint().height()` - the clipping guard, and the exact assertion that is
  red on the 2026-11-28 tree at 1.0 (six at 30 against a 33 px minimum);
* the six fields the General tab used to compress are named with their expected height, so a regression there is
  localised to one page instead of surfacing as "some combo is off";
* the whole shell is walked page by page (nav stack, and the Settings tab host's three tabs revealed in turn), because
  a hidden `QStackedWidget` page is 80-100 px wide and carries placeholder geometry rather than the page's layout.

``ScrollableComboBox`` is the toolkit's second combo family (it states the same 33 in its constructor). The app builds
none today - measured 2026-11-28, zero instances in the shell - so the expected set below is twenty ``ComboBox``
names; if one is ever hosted, add it to ``EXPECTED_COMBOS`` and this file holds it to the same one height.
"""

from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("sli_ui_toolkit")

from PySide6.QtWidgets import QMainWindow, QWidget  # noqa: E402

from sli_ui_toolkit.ui.managers.ui_scale import UiScale, scaled_px  # noqa: E402
from sli_ui_toolkit.ui.widgets.comboboxes.scrollable_combobox import ScrollableComboBox  # noqa: E402
from sli_ui_toolkit.widgets import ComboBox  # noqa: E402

if "resources_rc" not in sys.modules:
    sys.modules["resources_rc"] = types.ModuleType("resources_rc")

from src.ui.pages.handles import PAGE_NAMES  # noqa: E402

#: The factors the settings pages support: the shipped 1.0, the Windows 1.25/1.5, and the accessibility 2.0.
FACTORS = (1.0, 1.25, 1.5, 2.0)

#: The design window the other layout gates measure (a 500 px column inside it).
WINDOW_SIZE = (1163, 900)

#: The six fields the General tab compressed to its 30 px row at 1.0. Named here on purpose: this is the regression
#: the owner reported, and a red line in this test must say which page lost the height.
PREVIOUSLY_COMPRESSED = (
    "backendComboBox",
    "interpolateModelComboBox",
    "decompressModelComboBox",
    "denoiseModelComboBox",
    "deblurModelComboBox",
    "upscaleModelComboBox",
)

#: Every combo the shipped shell builds, across the nav pages and the Settings tab host's three tabs.
EXPECTED_COMBOS = frozenset(PREVIOUSLY_COMPRESSED) | frozenset({
    "encoder",
    "video_encoder_speed",
    "video_quality",
    "video_container",
    "video_pixel_format",
    "language",
    "font_family",
    "precision",
    "tensorrt_optimization_level",
    "scene_change_detection_method",
    "pytorch_backend",
    "pytorch_version",
    "tileSizeComboBox",
    "qualityPresetComboBox",
})


@pytest.fixture()
def restore_ui_scale():
    """The factor is a process-wide singleton: put it back for every test that moves it."""
    scale = UiScale.get_instance()
    previous = scale.factor()
    try:
        yield scale
    finally:
        scale.set_factor(previous)


def _settle(qapp, turns: int = 8) -> None:
    for _ in range(turns):
        qapp.processEvents()


def _collect(page: QWidget, found: dict[str, QWidget]) -> None:
    for child in page.findChildren(QWidget):
        if isinstance(child, (ComboBox, ScrollableComboBox)):
            found.setdefault(child.objectName(), child)


def _shell(qapp, factor: float):
    """Build the real shell at ``factor``, reveal every page, and hand back ``(shell, window, combos)``.

    Every page must be CURRENT while it is read: a page hidden inside a stack keeps the placeholder geometry it had
    when it was last laid out (or never got one), and the Settings tab host's tabs are the same shape one level down.
    """
    from src.ui import toolkit_boot, toolkit_theme
    from src.ui.shell.window import MainWindowShell

    UiScale.get_instance().set_factor(factor)
    toolkit_boot.ensure()
    toolkit_theme.install(qapp)

    window = QMainWindow()
    shell = MainWindowShell()
    shell.build(window)
    window.resize(*WINDOW_SIZE)
    window.show()
    _settle(qapp)

    found: dict[str, QWidget] = {}
    stack = window.page_handles.menu.stackedWidget
    for index, name in enumerate(PAGE_NAMES):
        stack.setCurrentIndex(index)
        _settle(qapp)
        if name == "settings":
            tab_host = shell.settingsPage_page.tabWidget
            for tab in range(tab_host.count()):
                tab_host.setCurrentIndex(tab)
                _settle(qapp)
                _collect(tab_host.widget(tab), found)
        else:
            _collect(stack.widget(index), found)
    return shell, window, found


@pytest.mark.parametrize("factor", FACTORS)
def test_every_combo_has_the_one_toolkit_height(qapp, restore_ui_scale, factor):
    """One height, the toolkit's, at every supported factor - and never below the control's own minimum."""
    shell, window, combos = _shell(qapp, factor)
    try:
        expected = scaled_px(ComboBox.BASE_HEIGHT)
        assert len(combos) == len(EXPECTED_COMBOS), (
            f"the shell builds {len(combos)} comboboxes ({', '.join(sorted(combos))}), expected "
            f"{len(EXPECTED_COMBOS)} - a page gained or lost one; update EXPECTED_COMBOS with the control"
        )
        assert set(combos) == EXPECTED_COMBOS, (
            f"combobox set drifted: missing {sorted(EXPECTED_COMBOS - set(combos))}, "
            f"unexpected {sorted(set(combos) - EXPECTED_COMBOS)}"
        )

        heights = {name: combo.height() for name, combo in combos.items()}
        assert set(heights.values()) == {expected}, (
            f"at factor {factor} the comboboxes carry {sorted(set(heights.values()))} px, not the toolkit's one "
            f"height {expected} px: " + ", ".join(f"{name}={h}" for name, h in sorted(heights.items()))
        )

        for name, combo in combos.items():
            floor = combo.minimumSizeHint().height()
            assert combo.height() >= floor, (
                f"at factor {factor} {name} renders {combo.height()} px inside its own {floor} px minimum - the host "
                "row is still capping the field"
            )
            # Not stretched either: the fixed height is the design height, so a layout cannot hand it the row's slack.
            # (The lower bound is the toolkit's own scale pass, which the pages that never cap a combo leave at 0 at
            # construction; the clipping guard above is what holds the floor.)
            assert combo.maximumHeight() == expected, (
                f"at factor {factor} {name} caps its height at {combo.maximumHeight()} px instead of {expected} px - "
                "the height is not this one number"
            )
    finally:
        window.close()


@pytest.mark.parametrize("factor", (1.0, 1.5, 2.0))
def test_the_general_tabs_six_formerly_squashed_fields_are_named(qapp, restore_ui_scale, factor):
    """The owner's report, localised: the six model/backend fields are exactly the toolkit height, not the 30 px row."""
    shell, window, combos = _shell(qapp, factor)
    try:
        expected = scaled_px(ComboBox.BASE_HEIGHT)
        page = shell.procPage_page.generalSettings_page
        for name in PREVIOUSLY_COMPRESSED:
            combo = getattr(page, name)
            assert combo.height() == expected, (
                f"at factor {factor} the General tab's {name} is {combo.height()} px, not {expected} px: the page "
                "capped it with its own row instead of the toolkit's combobox height"
            )
            assert combo.height() >= combo.minimumSizeHint().height()
    finally:
        window.close()
