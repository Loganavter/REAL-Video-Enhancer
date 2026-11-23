"""Page `tab_3` (QWidget), split out of the Designer-generated mainwindow.py.

Hand-maintained since 2026-11-23: the tree is the Designer's, the controls are the toolkit's. The Designer is no longer
a reference in this tree - the fixture, the splitter that emitted these pages and every "matches the Designer" gate
were removed - so this page answers to `harness/tests/test_toolkit_native.py`: the toolkit's controls and type scale,
`layout.addStretch()` instead of a `QSpacerItem`, and no new `name_<N>` object name or wrapper `QWidget` beyond the
counts that gate records.
"""

from __future__ import annotations

from PySide6.QtCore import (QCoreApplication, QSize, Qt)
from PySide6.QtWidgets import (QHBoxLayout, QSizePolicy, QSpacerItem, QVBoxLayout,
    QWidget)
import resources_rc
import resources_rc
# The page builds the toolkit's own controls: `Text` where the Designer had a `QLabel`, `ComboBox` for a `QComboBox`,
# `CheckBox` for a `QCheckBox` and `Stepper` for a `QSpinBox` (the toolkit `SpinBox` plus the step gutter RVE's
# `::up-button`/`::down-button` rules drew). Nothing here needs a Qt-shaped compatibility adapter.
from sli_ui_toolkit.widgets import CheckBox, ComboBox, Stepper, SurfacePanel, Text
from src.ui.widgets.combo_width import cap_combo_to_row
from src.ui.typography import apply, BODY_PT
from src.ui.pages.handles import RenderSettingsHandles

#: Qt's spin-box text alignment (`QAbstractSpinBox().alignment()` is AlignLeft|AlignVCenter); the toolkit field
#: defaults to centre, which would move every value to the middle of its box.
SPIN_ALIGNMENT = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter


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


class Tab3Page:
    """`tab_3` - placed by its parent with `addTab`."""

    page_id = "tab_3"

    def build(self):
        """Build the widget tree and return the page root."""
        self.tab_3 = SurfacePanel(background_token="surface.page")
        self.tab_3.setObjectName(u"tab_3")
        self.verticalLayout_22 = QVBoxLayout(self.tab_3)
        self.verticalLayout_22.setObjectName(u"verticalLayout_22")
        self.renderSettingsContainer = QWidget(self.tab_3)
        self.renderSettingsContainer.setObjectName(u"renderSettingsContainer")
        self.verticalLayout_12 = QVBoxLayout(self.renderSettingsContainer)
        self.verticalLayout_12.setObjectName(u"verticalLayout_12")
        self.precisionSettingsContainer = QHBoxLayout()
        self.precisionSettingsContainer.setSpacing(10)
        self.precisionSettingsContainer.setObjectName(u"precisionSettingsContainer")
        self.label_3 = Text(parent=self.renderSettingsContainer)
        self.label_3.setObjectName(u"label_3")
        apply(self.label_3, BODY_PT)
        self.precisionSettingsContainer.addWidget(self.label_3)
        self.precision = ComboBox(self.renderSettingsContainer)
        self.precision.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        cap_combo_to_row(self.precision)
        self.precision.addItem("")
        self.precision.addItem("")
        self.precision.addItem("")
        self.precision.setObjectName(u"precision")
        self.precisionSettingsContainer.addWidget(self.precision)
        self.verticalLayout_12.addLayout(self.precisionSettingsContainer)
        self.tensorRTOptLevelSettingsContainer = QHBoxLayout()
        self.tensorRTOptLevelSettingsContainer.setSpacing(10)
        self.tensorRTOptLevelSettingsContainer.setObjectName(u"tensorRTOptLevelSettingsContainer")
        self.label_15 = Text(parent=self.renderSettingsContainer)
        self.label_15.setObjectName(u"label_15")
        apply(self.label_15, BODY_PT)
        self.tensorRTOptLevelSettingsContainer.addWidget(self.label_15)
        self.tensorrt_optimization_level = ComboBox(self.renderSettingsContainer)
        self.tensorrt_optimization_level.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        cap_combo_to_row(self.tensorrt_optimization_level)
        self.tensorrt_optimization_level.addItem("")
        self.tensorrt_optimization_level.addItem("")
        self.tensorrt_optimization_level.addItem("")
        self.tensorrt_optimization_level.addItem("")
        self.tensorrt_optimization_level.addItem("")
        self.tensorrt_optimization_level.setObjectName(u"tensorrt_optimization_level")
        self.tensorRTOptLevelSettingsContainer.addWidget(self.tensorrt_optimization_level)
        self.verticalLayout_12.addLayout(self.tensorRTOptLevelSettingsContainer)
        self.horizontalLayout_67 = QHBoxLayout()
        self.horizontalLayout_67.setSpacing(10)
        self.horizontalLayout_67.setObjectName(u"horizontalLayout_67")
        self.label_88 = Text(parent=self.renderSettingsContainer)
        self.label_88.setObjectName(u"label_88")
        apply(self.label_88, BODY_PT)
        self.horizontalLayout_67.addWidget(self.label_88)
        self.horizontalSpacer_34 = QSpacerItem(488, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.horizontalLayout_67.addItem(self.horizontalSpacer_34)
        self.auto_hdr_mode = _row_checkbox(self.renderSettingsContainer)
        self.auto_hdr_mode.setObjectName(u"auto_hdr_mode")
        sizePolicy12 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        sizePolicy12.setHorizontalStretch(0)
        sizePolicy12.setVerticalStretch(0)
        sizePolicy12.setHeightForWidth(self.auto_hdr_mode.sizePolicy().hasHeightForWidth())
        self.auto_hdr_mode.setSizePolicy(sizePolicy12)
        self.horizontalLayout_67.addWidget(self.auto_hdr_mode)
        self.verticalLayout_12.addLayout(self.horizontalLayout_67)
        self.tensorRTOptLevelSettingsContainer_3 = QHBoxLayout()
        self.tensorRTOptLevelSettingsContainer_3.setSpacing(10)
        self.tensorRTOptLevelSettingsContainer_3.setObjectName(u"tensorRTOptLevelSettingsContainer_3")
        self.label_84 = Text(parent=self.renderSettingsContainer)
        self.label_84.setObjectName(u"label_84")
        apply(self.label_84, BODY_PT)
        self.tensorRTOptLevelSettingsContainer_3.addWidget(self.label_84)
        self.horizontalSpacer_31 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.tensorRTOptLevelSettingsContainer_3.addItem(self.horizontalSpacer_31)
        self.dynamic_tensorrt_engine = _row_checkbox(self.renderSettingsContainer)
        self.dynamic_tensorrt_engine.setObjectName(u"dynamic_tensorrt_engine")
        sizePolicy12.setHeightForWidth(self.dynamic_tensorrt_engine.sizePolicy().hasHeightForWidth())
        self.dynamic_tensorrt_engine.setSizePolicy(sizePolicy12)
        self.tensorRTOptLevelSettingsContainer_3.addWidget(self.dynamic_tensorrt_engine)
        self.verticalLayout_12.addLayout(self.tensorRTOptLevelSettingsContainer_3)
        self.precisionSettingsContainer_2 = QHBoxLayout()
        self.precisionSettingsContainer_2.setSpacing(10)
        self.precisionSettingsContainer_2.setObjectName(u"precisionSettingsContainer_2")
        self.label_22 = Text(parent=self.renderSettingsContainer)
        self.label_22.setObjectName(u"label_22")
        apply(self.label_22, BODY_PT)
        self.precisionSettingsContainer_2.addWidget(self.label_22)
        self.horizontalSpacer_15 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.precisionSettingsContainer_2.addItem(self.horizontalSpacer_15)
        self.uhd_mode = _row_checkbox(self.renderSettingsContainer)
        self.uhd_mode.setObjectName(u"uhd_mode")
        self.precisionSettingsContainer_2.addWidget(self.uhd_mode)
        self.verticalLayout_12.addLayout(self.precisionSettingsContainer_2)
        self.precisionSettingsContainer_4 = QHBoxLayout()
        self.precisionSettingsContainer_4.setSpacing(10)
        self.precisionSettingsContainer_4.setObjectName(u"precisionSettingsContainer_4")
        self.label_45 = Text(parent=self.renderSettingsContainer)
        self.label_45.setObjectName(u"label_45")
        apply(self.label_45, BODY_PT)
        self.precisionSettingsContainer_4.addWidget(self.label_45)
        # `alignment=` is Qt's spin-box alignment (`QAbstractSpinBox` is left-aligned); the toolkit field defaults to
        # centre and would move the value into the middle of its box. `Stepper` states its own row height (32 px) and
        # step gutter (22 px) where the adapter pinned 29 px and 20 px - toolkit metrics, reported in DONE.md.
        self.ncnn_gpu_id = Stepper(self.renderSettingsContainer, alignment=SPIN_ALIGNMENT)
        self.ncnn_gpu_id.setObjectName(u"ncnn_gpu_id")
        self.ncnn_gpu_id.setMinimumSize(QSize(0, 0))
        self.precisionSettingsContainer_4.addWidget(self.ncnn_gpu_id)
        self.verticalLayout_12.addLayout(self.precisionSettingsContainer_4)
        self.precisionSettingsContainer_3 = QHBoxLayout()
        self.precisionSettingsContainer_3.setSpacing(10)
        self.precisionSettingsContainer_3.setObjectName(u"precisionSettingsContainer_3")
        self.label_43 = Text(parent=self.renderSettingsContainer)
        self.label_43.setObjectName(u"label_43")
        apply(self.label_43, BODY_PT)
        self.precisionSettingsContainer_3.addWidget(self.label_43)
        self.pytorch_gpu_id = Stepper(self.renderSettingsContainer, alignment=SPIN_ALIGNMENT)
        self.pytorch_gpu_id.setObjectName(u"pytorch_gpu_id")
        self.pytorch_gpu_id.setMinimumSize(QSize(0, 0))
        self.precisionSettingsContainer_3.addWidget(self.pytorch_gpu_id)
        self.verticalLayout_12.addLayout(self.precisionSettingsContainer_3)
        # Rows are HORIZONTAL sub-layouts here, and the page is stretched to the tab pane: without a filler the box
        # layout shares that height among the rows themselves, so at 1920x1080 six rows were 132 px apart with their
        # controls floating in the middle of each band (measured 2026-11-22). The filler takes the slack.
        self._settingsFiller = QSpacerItem(0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.verticalLayout_12.addItem(self._settingsFiller)
        self.verticalLayout_22.addWidget(self.renderSettingsContainer)
        self.tensorrt_optimization_level.setCurrentIndex(2)
        self._handles = RenderSettingsHandles(
            precision=self.precision,
            tensorrt_optimization_level=self.tensorrt_optimization_level,
            auto_hdr_mode=self.auto_hdr_mode,
            dynamic_tensorrt_engine=self.dynamic_tensorrt_engine,
            uhd_mode=self.uhd_mode,
            ncnn_gpu_id=self.ncnn_gpu_id,
            pytorch_gpu_id=self.pytorch_gpu_id,
        )
        self.retranslate()
        return self.tab_3

    def retranslate(self) -> None:
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Precision", None))
        self.precision.setItemText(0, QCoreApplication.translate("MainWindow", u"auto", None))
        self.precision.setItemText(1, QCoreApplication.translate("MainWindow", u"float16", None))
        self.precision.setItemText(2, QCoreApplication.translate("MainWindow", u"float32", None))
        self.label_15.setText(QCoreApplication.translate("MainWindow", u"TensorRT Optimization Level", None))
        self.tensorrt_optimization_level.setItemText(0, QCoreApplication.translate("MainWindow", u"1", None))
        self.tensorrt_optimization_level.setItemText(1, QCoreApplication.translate("MainWindow", u"2", None))
        self.tensorrt_optimization_level.setItemText(2, QCoreApplication.translate("MainWindow", u"3", None))
        self.tensorrt_optimization_level.setItemText(3, QCoreApplication.translate("MainWindow", u"4", None))
        self.tensorrt_optimization_level.setItemText(4, QCoreApplication.translate("MainWindow", u"5", None))
        self.label_88.setText(QCoreApplication.translate("MainWindow", u"Auto HDR Mode", None))
        self.auto_hdr_mode.setText("")
        self.label_84.setText(QCoreApplication.translate("MainWindow", u"TensorRT Dynamic Engine", None))
        self.dynamic_tensorrt_engine.setText("")
        self.label_22.setText(QCoreApplication.translate("MainWindow", u"Interpolation UHD Mode", None))
        self.uhd_mode.setText("")
        self.label_45.setText(QCoreApplication.translate("MainWindow", u"NCNN GPU ID", None))
        self.label_43.setText(QCoreApplication.translate("MainWindow", u"PyTorch GPU ID", None))
