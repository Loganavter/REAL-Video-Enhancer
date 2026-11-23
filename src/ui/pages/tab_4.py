"""Page `tab_4` (QWidget), split out of the Designer-generated mainwindow.py.

Hand-maintained since 2026-11-23: the tree is the Designer's, the controls are the toolkit's. The Designer is no longer
a reference in this tree - the fixture, the splitter that emitted these pages and every "matches the Designer" gate
were removed - so this page answers to `harness/tests/test_toolkit_native.py`: the toolkit's controls and type scale,
`layout.addStretch()` instead of a `QSpacerItem`, and no new `name_<N>` object name or wrapper `QWidget` beyond the
counts that gate records.
"""

from __future__ import annotations

from PySide6.QtCore import (QCoreApplication, Qt)
from PySide6.QtWidgets import (QHBoxLayout, QSizePolicy, QSpacerItem, QVBoxLayout,
    QWidget)
import resources_rc
import resources_rc
# The page builds the toolkit's own controls: `Text` where the Designer had a `QLabel`, `CheckBox` where it had a
# `QCheckBox`, `Button` where it had a `QPushButton`. Nothing here needs a Qt-shaped compatibility adapter.
from sli_ui_toolkit.widgets import Button, CheckBox, SurfacePanel, Text
from src.ui.typography import apply, BODY_PT
from src.ui.pages.handles import AppSettingsHandles


def _row_checkbox(parent) -> CheckBox:
    """A toolkit `CheckBox` shaped for an RVE settings row.

    Two statements the row needs and the toolkit control does not carry as constructor arguments:

    * **the row size.** `CheckBox` builds on `Button`'s standalone control size, so an empty box is a fixed 36x36
      square (measured offscreen: `minimumSize() == maximumSize() == 36x36`, hint 24x30). RVE's row box is the
      indicator plus its label - the row's own height decides - so the square is released here.
    * **Qt's signal spelling.** RVE's glue binds `stateChanged(int)` (`src/ui/SettingsTab.py` writes the setting from
      it). The toolkit declares `stateChanged`, but emits it from `setCheckState` only: a plain two-state click or
      `setChecked` emits `toggled(bool)` alone (measured), so without this line a click would never be persisted.

    Both belong in the toolkit's `CheckBox` (a `size=` for a row shape, `stateChanged` on every state change); until
    it grows them they are stated once here instead of once per widget.
    """
    box = CheckBox(parent=parent)
    box.setMinimumSize(0, 0)
    box.setMaximumSize(16777215, 16777215)
    box.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
    box.toggled.connect(lambda checked: box.stateChanged.emit(2 if checked else 0))
    return box


class Tab4Page:
    """`tab_4` - placed by its parent with `addTab`."""

    page_id = "tab_4"

    def build(self):
        """Build the widget tree and return the page root."""
        self.tab_4 = SurfacePanel(background_token="surface.page")
        self.tab_4.setObjectName(u"tab_4")
        self.verticalLayout_23 = QVBoxLayout(self.tab_4)
        self.verticalLayout_23.setObjectName(u"verticalLayout_23")
        self.GUIOptionsContainer = QWidget(self.tab_4)
        self.GUIOptionsContainer.setObjectName(u"GUIOptionsContainer")
        self.verticalLayout_20 = QVBoxLayout(self.GUIOptionsContainer)
        self.verticalLayout_20.setObjectName(u"verticalLayout_20")
        self.verticalLayout_20.setContentsMargins(0, -1, -1, -1)
        self.previewEnabledContainer = QWidget(self.GUIOptionsContainer)
        self.previewEnabledContainer.setObjectName(u"previewEnabledContainer")
        self.horizontalLayout_21 = QHBoxLayout(self.previewEnabledContainer)
        self.horizontalLayout_21.setSpacing(0)
        self.horizontalLayout_21.setObjectName(u"horizontalLayout_21")
        self.horizontalLayout_21.setContentsMargins(0, 0, 0, 0)
        self.label_19 = Text(parent=self.previewEnabledContainer)
        self.label_19.setObjectName(u"label_19")
        apply(self.label_19, BODY_PT)
        self.horizontalLayout_21.addWidget(self.label_19)
        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.horizontalLayout_21.addItem(self.horizontalSpacer_2)
        self.preview_enabled = _row_checkbox(self.previewEnabledContainer)
        self.preview_enabled.setObjectName(u"preview_enabled")
        self.horizontalLayout_21.addWidget(self.preview_enabled)
        self.verticalLayout_20.addWidget(self.previewEnabledContainer)
        self.discordRPCSettingsContainer = QWidget(self.GUIOptionsContainer)
        self.discordRPCSettingsContainer.setObjectName(u"discordRPCSettingsContainer")
        self.horizontalLayout_12 = QHBoxLayout(self.discordRPCSettingsContainer)
        self.horizontalLayout_12.setSpacing(10)
        self.horizontalLayout_12.setObjectName(u"horizontalLayout_12")
        self.horizontalLayout_12.setContentsMargins(0, 0, 0, 0)
        self.label_23 = Text(parent=self.discordRPCSettingsContainer)
        self.label_23.setObjectName(u"label_23")
        apply(self.label_23, BODY_PT)
        self.horizontalLayout_12.addWidget(self.label_23)
        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.horizontalLayout_12.addItem(self.horizontalSpacer_5)
        self.discord_rich_presence = _row_checkbox(self.discordRPCSettingsContainer)
        self.discord_rich_presence.setObjectName(u"discord_rich_presence")
        self.discord_rich_presence.setToolTip(QCoreApplication.translate("MainWindow", u"<html><head/><body><p>Show REAL Video Enhancer in discord.</p></body></html>", None))
        self.horizontalLayout_12.addWidget(self.discord_rich_presence)
        self.verticalLayout_20.addWidget(self.discordRPCSettingsContainer)
        self.widget_29 = QWidget(self.GUIOptionsContainer)
        self.widget_29.setObjectName(u"widget_29")
        self.verticalLayout_35 = QVBoxLayout(self.widget_29)
        self.verticalLayout_35.setObjectName(u"verticalLayout_35")
        self.verticalLayout_35.setContentsMargins(0, 0, 0, 0)
        self.openRVEFolderBtn = Button(parent=self.widget_29, text="", variant="surface",
            content_align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            content_padding=(10, 5, 10, 5), corner_radius=10, size=(0, 0))
        self.openRVEFolderBtn.setObjectName(u"openRVEFolderBtn")
        # QPushButton's own policy (as wide as it needs, exactly as tall as its hint): the toolkit button is happy to
        # fill the cell, which turns a row button into a slab when the page is taller than its contents. `size=(0, 0)`
        # above drops the standalone 36x36 control size; this states the row's share.
        self.openRVEFolderBtn.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.openRVEFolderBtn.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.verticalLayout_35.addWidget(self.openRVEFolderBtn)
        self.verticalLayout_20.addWidget(self.widget_29)
        # Rows, not a stack: the tab page is stretched to the pane, and without a filler the box layout shares that
        # height among the rows - at 1920x1080 the three rows sat 275 px apart with the switch floating in the middle of
        # each band (measured 2026-11-22). The filler takes the slack; a row keeps its own height.
        self._settingsFiller = QSpacerItem(0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.verticalLayout_20.addItem(self._settingsFiller)
        self.verticalLayout_23.addWidget(self.GUIOptionsContainer)
        self._handles = AppSettingsHandles(
            preview_enabled=self.preview_enabled,
            discord_rich_presence=self.discord_rich_presence,
            openRVEFolderBtn=self.openRVEFolderBtn,
        )
        self.retranslate()
        return self.tab_4

    def retranslate(self) -> None:
        self.label_19.setText(QCoreApplication.translate("MainWindow", u"Preview Enabled", None))
        self.preview_enabled.setText("")
        self.label_23.setText(QCoreApplication.translate("MainWindow", u"Discord Rich Presence", None))
        self.discord_rich_presence.setText("")
        self.openRVEFolderBtn.setText(QCoreApplication.translate("MainWindow", u"Open RVE Folder", None))
