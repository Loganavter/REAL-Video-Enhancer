"""Page `tab_2` (QWidget), split out of the Designer-generated mainwindow.py.

Hand-maintained since 2026-11-23: the tree is the Designer's, the controls are the toolkit's. The Designer is no longer
a reference in this tree - the fixture, the splitter that emitted these pages and every "matches the Designer" gate
were removed - so this page answers to `harness/tests/test_toolkit_native.py`: the toolkit's controls and type scale,
`layout.addStretch()` instead of a `QSpacerItem`, and no new `name_<N>` object name or wrapper `QWidget` beyond the
counts that gate records.
"""

from __future__ import annotations

from PySide6.QtCore import (QCoreApplication, QRect, QSize, Qt)
from PySide6.QtWidgets import (QHBoxLayout, QSizePolicy, QSpacerItem, QVBoxLayout,
    QWidget)
import resources_rc
import resources_rc
# The page builds the toolkit's own controls: `Text` where the Designer had a `QLabel`, `CustomLineEdit` for a
# `QLineEdit`, `Button` for a `QPushButton`, `CheckBox` for a `QCheckBox`, `ComboBox` for a `QComboBox` and
# `DoubleSpinBox` for a `QDoubleSpinBox`. Nothing here needs a Qt-shaped compatibility adapter.
from sli_ui_toolkit.ui.managers.ui_scale import scaled_px
from sli_ui_toolkit.widgets import (Button, CheckBox, ComboBox, CustomLineEdit, DoubleSpinBox, SurfacePanel,
    SurfaceScrollArea, Text)
from src.ui.widgets.combo_width import cap_combo_to_row
from src.ui.typography import apply, BODY_PT
from src.ui.pages.handles import OutputSettingsHandles
from src.ui.scale_follower import ScaleFollower
from src.ui import i18n

#: RVE's window QSS scrollbar (`QScrollBar:vertical{width:8}`). A scroll area that RESERVES a bar must reserve this,
#: not the toolkit's default 10: measured on the settings rows, the 2 px difference narrowed every row. The value used
#: to come from the compatibility layer (`STOCK_SCROLLBAR_WIDTH`), which this page no longer imports.
STOCK_SCROLLBAR_WIDTH = 8

#: Qt's spin-box text alignment (`QAbstractSpinBox().alignment()` is AlignLeft|AlignVCenter); the toolkit field
#: defaults to centre, which would move every value to the middle of its box.
SPIN_ALIGNMENT = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter

#: The output path field's MINIMUM height in DESIGN px: the Designer's ``(0, 25)``. A design value, so it is resolved
#: through `scaled_px()` at construction and on every factor change - the toolkit field re-pins its own height on a
#: scale change, and the page's floor has to follow it.
ROW_MIN_HEIGHT_PX = 25

#: The scene-change threshold's design row: the Designer's ``75x30`` value field.
FIELD_WIDTH_PX = 75
ROW_HEIGHT_PX = 30

#: The rows' design gap (10) and the tight gap before the last row (6).
ROW_GAP_PX = 10
TIGHT_GAP_PX = 6


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


class Tab2Page:
    """`tab_2` - placed by its parent with `addTab`."""

    page_id = "tab_2"

    def build(self):
        """Build the widget tree and return the page root."""
        self.tab_2 = SurfacePanel(background_token="surface.page")
        self.tab_2.setObjectName(u"tab_2")
        self.verticalLayout_17 = QVBoxLayout(self.tab_2)
        self.verticalLayout_17.setObjectName(u"verticalLayout_17")
        self.VideoSettingsContainer = QWidget(self.tab_2)
        self.VideoSettingsContainer.setObjectName(u"VideoSettingsContainer")
        self.verticalLayout_19 = QVBoxLayout(self.VideoSettingsContainer)
        self.verticalLayout_19.setSpacing(0)
        self.verticalLayout_19.setObjectName(u"verticalLayout_19")
        self.verticalLayout_19.setContentsMargins(0, 0, 0, 0)
        # The page body scrolls in the toolkit's area: RVE's `QScrollArea{background:transparent}` becomes
        # `surface_token=None` (no token fill, the page's own surface shows through). `setWidgetResizable` is inherited.
        self.scrollArea_3 = SurfaceScrollArea(self.VideoSettingsContainer, surface_token=None, content_inset=1,
            scrollbar_width=STOCK_SCROLLBAR_WIDTH)
        self.scrollArea_3.setObjectName(u"scrollArea_3")
        self.scrollArea_3.setWidgetResizable(True)
        self.scrollAreaWidgetContents_3 = QWidget()
        self.scrollAreaWidgetContents_3.setObjectName(u"scrollAreaWidgetContents_3")
        self.scrollAreaWidgetContents_3.setGeometry(QRect(0, 0, 968, 528))
        self.verticalLayout_25 = QVBoxLayout(self.scrollAreaWidgetContents_3)
        self.verticalLayout_25.setObjectName(u"verticalLayout_25")
        self.outputFileLocationContianer = QWidget(self.scrollAreaWidgetContents_3)
        self.outputFileLocationContianer.setObjectName(u"outputFileLocationContianer")
        self.horizontalLayout_19 = QHBoxLayout(self.outputFileLocationContianer)
        self.horizontalLayout_19.setObjectName(u"horizontalLayout_19")
        self.horizontalLayout_19.setContentsMargins(0, 0, 0, 0)
        self.encoderLabel_3 = Text(parent=self.outputFileLocationContianer)
        self.encoderLabel_3.setObjectName(u"encoderLabel_3")
        apply(self.encoderLabel_3, BODY_PT)
        self.horizontalLayout_19.addWidget(self.encoderLabel_3)
        self.horizontalSpacer_11 = QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.horizontalLayout_19.addItem(self.horizontalSpacer_11)
        self.output_folder_location = CustomLineEdit(self.outputFileLocationContianer)
        self.output_folder_location.setObjectName(u"output_folder_location")
        self.horizontalLayout_19.addWidget(self.output_folder_location)
        self.select_output_folder_location_btn = Button(parent=self.outputFileLocationContianer, text="",
            variant="surface", content_align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            content_padding=(10, 5, 10, 5), corner_radius=10, size=(0, 0))
        self.select_output_folder_location_btn.setObjectName(u"select_output_folder_location_btn")
        self.select_output_folder_location_btn.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.horizontalLayout_19.addWidget(self.select_output_folder_location_btn)
        self.verticalLayout_25.addWidget(self.outputFileLocationContianer)
        self.sceneChangeDetectionSettingContainer = QWidget(self.scrollAreaWidgetContents_3)
        self.sceneChangeDetectionSettingContainer.setObjectName(u"sceneChangeDetectionSettingContainer")
        self.horizontalLayout_17 = QHBoxLayout(self.sceneChangeDetectionSettingContainer)
        self.horizontalLayout_17.setObjectName(u"horizontalLayout_17")
        self.horizontalLayout_17.setContentsMargins(0, 0, 0, 0)
        self.label_20 = Text(parent=self.sceneChangeDetectionSettingContainer)
        self.label_20.setObjectName(u"label_20")
        apply(self.label_20, BODY_PT)
        self.horizontalLayout_17.addWidget(self.label_20)
        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.horizontalLayout_17.addItem(self.horizontalSpacer_3)
        self.scene_change_detection_enabled = _row_checkbox(self.sceneChangeDetectionSettingContainer)
        self.scene_change_detection_enabled.setObjectName(u"scene_change_detection_enabled")
        self.scene_change_detection_enabled.setToolTip(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" font-weight:700;\">Scene Change Detection Enabled:</span></p><p>- Enabled: Attempts to detect when scenes change in a video. </p><p>   When running an interpolation, it will avoide interpolating over these scene changes.</p></body></html>", None))
        self.horizontalLayout_17.addWidget(self.scene_change_detection_enabled)
        self.verticalLayout_25.addWidget(self.sceneChangeDetectionSettingContainer)
        self.widget_12 = QWidget(self.scrollAreaWidgetContents_3)
        self.widget_12.setObjectName(u"widget_12")
        self.horizontalLayout_27 = QHBoxLayout(self.widget_12)
        self.horizontalLayout_27.setObjectName(u"horizontalLayout_27")
        self.horizontalLayout_27.setContentsMargins(0, 0, 0, 0)
        self.label_37 = Text(parent=self.widget_12)
        self.label_37.setObjectName(u"label_37")
        apply(self.label_37, BODY_PT)
        self.horizontalLayout_27.addWidget(self.label_37)
        self.horizontalSpacer_13 = QSpacerItem(305, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.horizontalLayout_27.addItem(self.horizontalSpacer_13)
        self.scene_change_detection_method = ComboBox(self.widget_12)
        self.scene_change_detection_method.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        cap_combo_to_row(self.scene_change_detection_method)
        self.scene_change_detection_method.addItem("")
        self.scene_change_detection_method.addItem("")
        self.scene_change_detection_method.addItem("")
        self.scene_change_detection_method.addItem("")
        self.scene_change_detection_method.setObjectName(u"scene_change_detection_method")
        self.scene_change_detection_method.setToolTip(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" font-weight:700;\">Scene Change Detection Method</span></p><p>- mean: fastest, least accurate</p><p>- mean (segmented): fast, prone to overdetection</p><p>- pyscenedetect: slow, accuate <span style=\" font-weight:700;\">(Recommended)</span></p></body></html>", None))
        self.horizontalLayout_27.addWidget(self.scene_change_detection_method)
        self.verticalLayout_25.addWidget(self.widget_12)
        self.widget_7 = QWidget(self.scrollAreaWidgetContents_3)
        self.widget_7.setObjectName(u"widget_7")
        self.horizontalLayout_14 = QHBoxLayout(self.widget_7)
        self.horizontalLayout_14.setObjectName(u"horizontalLayout_14")
        self.horizontalLayout_14.setContentsMargins(0, 0, 0, 0)
        self.label_30 = Text(parent=self.widget_7)
        self.label_30.setObjectName(u"label_30")
        apply(self.label_30, BODY_PT)
        self.horizontalLayout_14.addWidget(self.label_30)
        self.horizontalSpacer_6 = QSpacerItem(305, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.horizontalLayout_14.addItem(self.horizontalSpacer_6)
        # `single_step=0.1` is the adapter's "Qt clamps and rounds to `decimals`" semantic, stated as the step the field
        # actually shows: the toolkit's `DoubleSpinBox.setValue` SNAPS to `single_step` (its default 1.0 turned RVE's
        # stored `3.5` into `3.0`, which the settings glue then wrote back). At `decimals=1` a 0.1 step is exactly that
        # rounding, so `3.5` stays `3.5` (measured offscreen) while the arrow keys still step.
        self.scene_change_detection_threshold = DoubleSpinBox(self.widget_7, single_step=0.1, decimals=1, show_step_buttons=True,
                                      step_button_width=20, field_height=29,
            alignment=SPIN_ALIGNMENT)
        self.scene_change_detection_threshold.setObjectName(u"scene_change_detection_threshold")
        self.scene_change_detection_threshold.setToolTip(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" font-weight:700;\">Scene Change Detection Threshold</span></p><p> - Lower number: <span style=\" font-weight:700;\">higher</span> chance of detecting scene changes, with risk of <span style=\" font-weight:700;\">overdetection</span>.</p><p> - Higher number: <span style=\" font-weight:700;\">lower </span>chance of detecting scene changes, with risk of <span style=\" font-weight:700;\">underdetection.</span></p><p>   - <span style=\" font-weight:700;\">Recommended: </span>2-3 for real life, 4 for animation.</p></body></html>", None))
        self.scene_change_detection_threshold.setDecimals(1)
        # Qt's `setMinimum`/`setMaximum` are the one range spelling the toolkit field does not answer (`DoubleSpinBox`
        # carries `setRange` only), so the Designer's two statements are the one call they compose to.
        self.scene_change_detection_threshold.setRange(1.000000000000000, 10.000000000000000)
        self.horizontalLayout_14.addWidget(self.scene_change_detection_threshold)
        self.verticalLayout_25.addWidget(self.widget_7)
        self.widget_9 = QWidget(self.scrollAreaWidgetContents_3)
        self.widget_9.setObjectName(u"widget_9")
        self.horizontalLayout_25 = QHBoxLayout(self.widget_9)
        self.horizontalLayout_25.setObjectName(u"horizontalLayout_25")
        self.horizontalLayout_25.setContentsMargins(0, 0, 0, 0)
        self.label_48 = Text(parent=self.widget_9)
        self.label_48.setObjectName(u"label_48")
        apply(self.label_48, BODY_PT)
        self.horizontalLayout_25.addWidget(self.label_48)
        self.horizontalSpacer_10 = QSpacerItem(468, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.horizontalLayout_25.addItem(self.horizontalSpacer_10)
        self.auto_border_cropping = _row_checkbox(self.widget_9)
        self.auto_border_cropping.setObjectName(u"auto_border_cropping")
        self.auto_border_cropping.setToolTip(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" font-weight:700;\">Audo Border Cropping:</span></p><p>- Will automatically remove black bars if they are present throughout a video, can increase performance.</p></body></html>", None))
        self.horizontalLayout_25.addWidget(self.auto_border_cropping)
        self.verticalLayout_25.addWidget(self.widget_9)
        # A settings column is a list of ROWS: the layout's leftover height is not theirs to share. Every container here
        # is a plain `QWidget`, whose default Preferred vertical policy grows, so the box layout handed it the slack and
        # a 33 px row became 168 px tall on a 1920x1080 window - the control floated in the middle of a huge band, and
        # the height changed with the window and with every section another control opened (measured 2026-11-22). The
        # filler takes the slack instead: a row is always its own height.
        self._settingsFiller = QSpacerItem(0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.verticalLayout_25.addItem(self._settingsFiller)
        self.scrollArea_3.setWidget(self.scrollAreaWidgetContents_3)
        self.verticalLayout_19.addWidget(self.scrollArea_3)
        self.verticalLayout_17.addWidget(self.VideoSettingsContainer)
        self._handles = OutputSettingsHandles(
            output_folder_location=self.output_folder_location,
            select_output_folder_location_btn=self.select_output_folder_location_btn,
            scene_change_detection_enabled=self.scene_change_detection_enabled,
            scene_change_detection_method=self.scene_change_detection_method,
            scene_change_detection_threshold=self.scene_change_detection_threshold,
            auto_border_cropping=self.auto_border_cropping,
        )
        # The page pass needs a QObject to walk; the relay is that object and forwards
        # `retranslate(lang)` to this builder (`src/ui/i18n.py`).
        i18n.bind_page(self, self.tab_2)
        self.retranslate()
        # One place states this page's design numbers: construction is its first run, the follower re-runs it on a
        # live factor change.
        self._scale = ScaleFollower(self.tab_2, self._apply_scale)
        return self.tab_2

    def _apply_scale(self) -> None:
        """Resolve this page's design numbers at the factor in force now (never at import)."""
        self.output_folder_location.setMinimumSize(QSize(0, scaled_px(ROW_MIN_HEIGHT_PX)))
        self.scene_change_detection_threshold.setMinimumSize(
            QSize(scaled_px(FIELD_WIDTH_PX), scaled_px(ROW_HEIGHT_PX)))

        gap = scaled_px(ROW_GAP_PX)
        for layout in (self.horizontalLayout_19, self.horizontalLayout_17,
                       self.horizontalLayout_27, self.horizontalLayout_14):
            layout.setSpacing(gap)
        self.horizontalLayout_25.setSpacing(scaled_px(TIGHT_GAP_PX))

    def retranslate(self, language: str | None = None) -> None:
        self.encoderLabel_3.setText(QCoreApplication.translate("MainWindow", u"Output folder directory", None))
        self.select_output_folder_location_btn.setText(QCoreApplication.translate("MainWindow", u"Select Output Folder", None))
        self.label_20.setText(QCoreApplication.translate("MainWindow", u"Scene Change Detection Enabled", None))
        self.scene_change_detection_enabled.setText("")
        self.label_37.setText(QCoreApplication.translate("MainWindow", u"Scene Change Detection Method", None))
        self.scene_change_detection_method.setItemText(0, u"mean")
        self.scene_change_detection_method.setItemText(1, u"mean_segmented")
        self.scene_change_detection_method.setItemText(2, u"sudo_scene_detect")
        self.scene_change_detection_method.setItemText(3, u"pyscenedetect")
        self.label_30.setText(QCoreApplication.translate("MainWindow", u"Scene Change Detection Threshold", None))
        self.label_48.setText(QCoreApplication.translate("MainWindow", u"Auto Border Cropping", None))
        self.auto_border_cropping.setText("")
