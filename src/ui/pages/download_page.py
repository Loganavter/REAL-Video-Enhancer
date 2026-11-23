"""Page `downloadPage` (QWidget), split out of the Designer-generated mainwindow.py.

Hand-maintained since 2026-11-23: the tree is the Designer's, the controls are the toolkit's. The Designer is no longer
a reference in this tree - the fixture, the splitter that emitted these pages and every "matches the Designer" gate
were removed - so this page answers to `harness/tests/test_toolkit_native.py`: the toolkit's controls and type scale,
`layout.addStretch()` instead of a `QSpacerItem`, and no new `name_<N>` object name or wrapper `QWidget` beyond the
counts that gate records.
"""

from __future__ import annotations

from PySide6.QtCore import (QCoreApplication, QRect, QSize, Qt)
from PySide6.QtGui import (QColor, QIcon)
from PySide6.QtWidgets import (QGridLayout, QHBoxLayout, QSizePolicy, QSpacerItem,
    QVBoxLayout, QWidget)
import resources_rc
import resources_rc
# The page builds the toolkit's own controls: `Text` where the Designer had a `QLabel`, `Button` for a `QPushButton`
# and `ComboBox` for a `QComboBox`. Nothing here needs a Qt-shaped compatibility adapter.
from sli_ui_toolkit.widgets import Button, ComboBox, OverlayScrollArea, SurfacePanel, Text
from src.ui.widgets.combo_width import cap_combo_to_row
from src.ui.typography import apply, BODY_PT, DISPLAY_PT
from src.ui.pages.handles import DownloadPageHandles

# Layer B3 (2026-11-21): the two widget classes this page no longer builds stock. `OverlayScrollArea` is the
# `QScrollArea` counterpart whose surface is transparent - the page's rule is `QScrollArea{background:transparent}`,
# and `SurfaceScrollArea` always paints a token plate (`surface.background` #2c313c or, with `surface_token=None`,
# the palette's #1f232a; measured offscreen). `ImageLabel` is the pixmap counterpart, Qt-shaped already
# (`setPixmap`/`setScaledContents`/`setText`), and paint-identical to `QLabel` at the Designer's default alignment.


class DownloadPagePage:
    """`downloadPage` - placed by its parent with `addWidget`."""

    page_id = "downloadPage"

    def build(self):
        """Build the widget tree and return the page root."""
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy5 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        # splitter: skipped 'sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())' (['MainWindow'] live outside this page)
        # splitter: skipped 'sizePolicy5.setHeightForWidth(self.processContainer.sizePolicy().hasHeightForWidth())' (['processContainer'] live outside this page)
        # splitter: skipped 'sizePolicy5.setHeightForWidth(self.interpolationMultiplierSpinBox.sizePolicy().hasHeightForWidth())' (['interpolationMultiplierSpinBox'] live outside this page)
        # splitter: skipped 'sizePolicy5.setHeightForWidth(self.upscaleScaleSpinBox.sizePolicy().hasHeightForWidth())' (['upscaleScaleSpinBox'] live outside this page)
        # splitter: skipped 'sizePolicy5.setHeightForWidth(self.addToRenderQueueButton.sizePolicy().hasHeightForWidth())' (['addToRenderQueueButton'] live outside this page)
        # splitter: skipped 'sizePolicy5.setHeightForWidth(self.widget_26.sizePolicy().hasHeightForWidth())' (['widget_26'] live outside this page)
        # splitter: skipped 'sizePolicy.setHeightForWidth(self.renderQueueListWidget.sizePolicy().hasHeightForWidth())' (['renderQueueListWidget'] live outside this page)
        # splitter: skipped 'sizePolicy5.setHeightForWidth(self.OutputFilesListWidget.sizePolicy().hasHeightForWidth())' (['OutputFilesListWidget'] live outside this page)
        # Layer C: the page body is the toolkit's token-painted container. Every rule this page carried styled a widget
        # that is now a toolkit control (QLineEdit/QPushButton/QComboBox ignore QSS) or the background itself, so the
        # whole sheet is gone and the colour comes from `surface.page` (#16191d).
        self.downloadPage = SurfacePanel(background_token="surface.page")
        self.downloadPage.setObjectName(u"downloadPage")
        self.gridLayout_5 = QGridLayout(self.downloadPage)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.widget_10 = QWidget(self.downloadPage)
        self.widget_10.setObjectName(u"widget_10")
        self.widget_10.setMinimumSize(QSize(0, 0))
        self.gridLayout_8 = QGridLayout(self.widget_10)
        self.gridLayout_8.setObjectName(u"gridLayout_8")
        self.encoderLabel_4 = Text(parent=self.widget_10)
        self.encoderLabel_4.setObjectName(u"encoderLabel_4")
        apply(self.encoderLabel_4, BODY_PT)
        self.gridLayout_8.addWidget(self.encoderLabel_4, 0, 0, 1, 1)
        self.horizontalSpacer_12 = QSpacerItem(20, 20, QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Minimum)
        self.gridLayout_8.addItem(self.horizontalSpacer_12, 0, 1, 1, 1)
        self.gridLayout_10 = QGridLayout()
        self.gridLayout_10.setObjectName(u"gridLayout_10")
        self.selectNCNNCustomModel = Button(parent=self.widget_10, text="", variant="surface",
            content_align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            content_padding=(10, 5, 10, 5), corner_radius=10, size=(0, 0))
        self.selectNCNNCustomModel.setObjectName(u"selectNCNNCustomModel")
        self.selectNCNNCustomModel.setEnabled(True)
        # QPushButton's own policy (as wide as it needs, exactly as tall as its hint): the toolkit button is happy to
        # fill the cell, which turns a row button into a slab when the page is taller than its contents. `size=(0, 0)`
        # above drops the standalone 36x36 control size; this states the row's share - for this pair, which the
        # Designer's statements never sized, exactly as the icon buttons below get `sizePolicy5`.
        self.selectNCNNCustomModel.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.gridLayout_10.addWidget(self.selectNCNNCustomModel, 1, 0, 1, 1)
        self.selectPytorchCustomModel = Button(parent=self.widget_10, text="", variant="surface",
            content_align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            content_padding=(10, 5, 10, 5), corner_radius=10, size=(0, 0))
        self.selectPytorchCustomModel.setObjectName(u"selectPytorchCustomModel")
        self.selectPytorchCustomModel.setEnabled(True)
        self.selectPytorchCustomModel.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.gridLayout_10.addWidget(self.selectPytorchCustomModel, 0, 0, 1, 1)
        self.gridLayout_8.addLayout(self.gridLayout_10, 0, 2, 1, 1)
        self.gridLayout_5.addWidget(self.widget_10, 4, 0, 1, 1)
        self.ApplicationUpdateContainer = QWidget(self.downloadPage)
        self.ApplicationUpdateContainer.setObjectName(u"ApplicationUpdateContainer")
        self.verticalLayout_30 = QVBoxLayout(self.ApplicationUpdateContainer)
        self.verticalLayout_30.setObjectName(u"verticalLayout_30")
        self.verticalLayout_30.setContentsMargins(9, 9, 9, 9)
        self.gridLayout_5.addWidget(self.ApplicationUpdateContainer, 0, 0, 1, 1)
        self.backendSelectContainer = QWidget(self.downloadPage)
        self.backendSelectContainer.setObjectName(u"backendSelectContainer")
        self.backendSelectContainer.setStyleSheet(u"")
        self.verticalLayout_11 = QVBoxLayout(self.backendSelectContainer)
        self.verticalLayout_11.setObjectName(u"verticalLayout_11")
        self.widget_31 = QWidget(self.backendSelectContainer)
        self.widget_31.setObjectName(u"widget_31")
        self.horizontalLayout_49 = QHBoxLayout(self.widget_31)
        self.horizontalLayout_49.setObjectName(u"horizontalLayout_49")
        self.uninstallAppBtn = Button(parent=self.widget_31, text="", variant="surface",
            content_align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            content_padding=(10, 5, 10, 5), corner_radius=10, size=(0, 0))
        self.uninstallAppBtn.setObjectName(u"uninstallAppBtn")
        sizePolicy5.setHeightForWidth(self.uninstallAppBtn.sizePolicy().hasHeightForWidth())
        self.uninstallAppBtn.setSizePolicy(sizePolicy5)
        self.uninstallAppBtn.setMaximumSize(QSize(50, 45))
        icon6 = QIcon()
        icon6.addFile(u":/icons/icons/trash-2.svg", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.uninstallAppBtn.setIcon(icon6)
        self.uninstallAppBtn.setIconSize(QSize(30, 30))
        self.horizontalLayout_49.addWidget(self.uninstallAppBtn)
        self.label_67 = Text(parent=self.widget_31)
        self.label_67.setObjectName(u"label_67")
        self.horizontalLayout_49.addWidget(self.label_67)
        self.verticalLayout_11.addWidget(self.widget_31)
        self.label_4 = Text(parent=self.backendSelectContainer)
        self.label_4.setObjectName(u"label_4")
        apply(self.label_4, DISPLAY_PT, underline=True)
        self.verticalLayout_11.addWidget(self.label_4)
        self.low_storage_label = Text(parent=self.backendSelectContainer)
        self.low_storage_label.setObjectName(u"low_storage_label")
        apply(self.low_storage_label, None, bold=True)
        # The stock label was red through this stylesheet; a toolkit label ignores QSS, so the colour is stated on the
        # widget itself (the stylesheet was silently dead - a regression this migration inherited).
        self.low_storage_label.setTextColor(QColor("red"))
        self.verticalLayout_11.addWidget(self.low_storage_label)
        # Layer B3: the stock area was transparent (`QScrollArea{background:transparent}` in the page rule) and its
        # scrollbar persistent; the overlay counterpart keeps the transparent surface (measured #16191d = the page)
        # and draws a 10 px overlay bar in a reserved gutter instead of the stock 14 px bar (viewport 368 vs 362 wide).
        self.scrollArea_2 = OverlayScrollArea(self.backendSelectContainer, content_inset=1, reserve_scrollbar_space=False)
        self.scrollArea_2.setObjectName(u"scrollArea_2")
        sizePolicy13 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)
        sizePolicy13.setHorizontalStretch(0)
        sizePolicy13.setVerticalStretch(0)
        sizePolicy13.setHeightForWidth(self.scrollArea_2.sizePolicy().hasHeightForWidth())
        self.scrollArea_2.setSizePolicy(sizePolicy13)
        self.scrollArea_2.setMaximumSize(QSize(16777215, 16777215))
        self.scrollArea_2.setWidgetResizable(True)
        self.scrollAreaWidgetContents_2 = QWidget()
        self.scrollAreaWidgetContents_2.setObjectName(u"scrollAreaWidgetContents_2")
        self.scrollAreaWidgetContents_2.setGeometry(QRect(0, 0, 928, 302))
        self.verticalLayout_13 = QVBoxLayout(self.scrollAreaWidgetContents_2)
        self.verticalLayout_13.setObjectName(u"verticalLayout_13")
        self.verticalLayout_13.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_46 = QHBoxLayout()
        self.horizontalLayout_46.setObjectName(u"horizontalLayout_46")
        self.gridLayout_9 = QGridLayout()
        self.gridLayout_9.setObjectName(u"gridLayout_9")
        self.label_65 = Text(parent=self.scrollAreaWidgetContents_2)
        self.label_65.setObjectName(u"label_65")
        self.gridLayout_9.addWidget(self.label_65, 1, 0, 1, 1)
        self.label_53 = Text(parent=self.scrollAreaWidgetContents_2)
        self.label_53.setObjectName(u"label_53")
        self.gridLayout_9.addWidget(self.label_53, 3, 0, 1, 1)
        self.pytorch_backend = ComboBox(self.scrollAreaWidgetContents_2)
        self.pytorch_backend.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        cap_combo_to_row(self.pytorch_backend)
        self.pytorch_backend.addItem("")
        self.pytorch_backend.addItem("")
        self.pytorch_backend.addItem("")
        self.pytorch_backend.setObjectName(u"pytorch_backend")
        self.pytorch_backend.setToolTipDuration(0)
        self.pytorch_backend.setToolTip(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" font-weight:700;\">PyTorch Backend</span></p><p>- CUDA: NVIDIA GPUs 10 series and up</p><p>- ROCm: AMD (Linux Only) 7000 series(maybe) and up</p><p>- xpu: Intel: idk fam, dunno if this even works lowkey</p><p>- MPS: Apple Mac M series SOCs only.</p></body></html>", None))
        self.gridLayout_9.addWidget(self.pytorch_backend, 3, 2, 1, 1)
        self.pytorch_version = ComboBox(self.scrollAreaWidgetContents_2)
        self.pytorch_version.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        cap_combo_to_row(self.pytorch_version)
        self.pytorch_version.setObjectName(u"pytorch_version")
        self.pytorch_version.setToolTipDuration(0)
        self.pytorch_version.setToolTip(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" font-weight:700;\">PyTorch Version</span></p><p>- This is the version downloaded</p><p>- 2.6: Supports older CUDA enabled GPUs</p><p>- 2.8: Supports all RTX cards, faster and recomended.</p><p>- 2.9: Experimental.</p></body></html>", None))
        self.gridLayout_9.addWidget(self.pytorch_version, 1, 2, 1, 1)
        self.pytorchBackendInstallerContainer_5 = QWidget(self.scrollAreaWidgetContents_2)
        self.pytorchBackendInstallerContainer_5.setObjectName(u"pytorchBackendInstallerContainer_5")
        self.horizontalLayout_37 = QHBoxLayout(self.pytorchBackendInstallerContainer_5)
        self.horizontalLayout_37.setObjectName(u"horizontalLayout_37")
        self.horizontalLayout_37.setContentsMargins(0, 0, 0, 0)
        self.downloadRecommendedBtn = Button(parent=self.pytorchBackendInstallerContainer_5, text="", variant="surface",
            content_align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            content_padding=(10, 5, 10, 5), corner_radius=10, size=(0, 0))
        self.downloadRecommendedBtn.setObjectName(u"downloadRecommendedBtn")
        sizePolicy5.setHeightForWidth(self.downloadRecommendedBtn.sizePolicy().hasHeightForWidth())
        self.downloadRecommendedBtn.setSizePolicy(sizePolicy5)
        self.downloadRecommendedBtn.setMaximumSize(QSize(50, 45))
        icon7 = QIcon()
        icon7.addFile(u":/icons/icons/download.svg", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.downloadRecommendedBtn.setIcon(icon7)
        self.downloadRecommendedBtn.setIconSize(QSize(30, 30))
        self.horizontalLayout_37.addWidget(self.downloadRecommendedBtn)
        self.label_25 = Text(parent=self.pytorchBackendInstallerContainer_5)
        self.label_25.setObjectName(u"label_25")
        self.horizontalLayout_37.addWidget(self.label_25)
        self.gridLayout_9.addWidget(self.pytorchBackendInstallerContainer_5, 0, 0, 1, 1)
        self.horizontalLayout_46.addLayout(self.gridLayout_9)
        self.verticalLayout_13.addLayout(self.horizontalLayout_46)
        self.pytorchBackendInstallerContainer_4 = QWidget(self.scrollAreaWidgetContents_2)
        self.pytorchBackendInstallerContainer_4.setObjectName(u"pytorchBackendInstallerContainer_4")
        self.horizontalLayout_9 = QHBoxLayout(self.pytorchBackendInstallerContainer_4)
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.horizontalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.downloadNCNNBtn = Button(parent=self.pytorchBackendInstallerContainer_4, text="", variant="surface",
            content_align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            content_padding=(10, 5, 10, 5), corner_radius=10, size=(0, 0))
        self.downloadNCNNBtn.setObjectName(u"downloadNCNNBtn")
        sizePolicy5.setHeightForWidth(self.downloadNCNNBtn.sizePolicy().hasHeightForWidth())
        self.downloadNCNNBtn.setSizePolicy(sizePolicy5)
        self.downloadNCNNBtn.setMaximumSize(QSize(50, 45))
        self.downloadNCNNBtn.setIcon(icon7)
        self.downloadNCNNBtn.setIconSize(QSize(30, 30))
        self.horizontalLayout_9.addWidget(self.downloadNCNNBtn)
        self.uninstallNCNNBtn = Button(parent=self.pytorchBackendInstallerContainer_4, text="", variant="surface",
            content_align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            content_padding=(10, 5, 10, 5), corner_radius=10, size=(0, 0))
        self.uninstallNCNNBtn.setObjectName(u"uninstallNCNNBtn")
        sizePolicy5.setHeightForWidth(self.uninstallNCNNBtn.sizePolicy().hasHeightForWidth())
        self.uninstallNCNNBtn.setSizePolicy(sizePolicy5)
        self.uninstallNCNNBtn.setMaximumSize(QSize(50, 45))
        self.uninstallNCNNBtn.setIcon(icon6)
        self.uninstallNCNNBtn.setIconSize(QSize(30, 30))
        self.horizontalLayout_9.addWidget(self.uninstallNCNNBtn)
        self.label_10 = Text(parent=self.pytorchBackendInstallerContainer_4)
        self.label_10.setObjectName(u"label_10")
        self.horizontalLayout_9.addWidget(self.label_10)
        self.verticalLayout_13.addWidget(self.pytorchBackendInstallerContainer_4)
        self.pytorchBackendInstallerContainer = QWidget(self.scrollAreaWidgetContents_2)
        self.pytorchBackendInstallerContainer.setObjectName(u"pytorchBackendInstallerContainer")
        self.horizontalLayout_6 = QHBoxLayout(self.pytorchBackendInstallerContainer)
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.downloadTorchBtn = Button(parent=self.pytorchBackendInstallerContainer, text="", variant="surface",
            content_align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            content_padding=(10, 5, 10, 5), corner_radius=10, size=(0, 0))
        self.downloadTorchBtn.setObjectName(u"downloadTorchBtn")
        sizePolicy5.setHeightForWidth(self.downloadTorchBtn.sizePolicy().hasHeightForWidth())
        self.downloadTorchBtn.setSizePolicy(sizePolicy5)
        self.downloadTorchBtn.setMaximumSize(QSize(50, 45))
        self.downloadTorchBtn.setIcon(icon7)
        self.downloadTorchBtn.setIconSize(QSize(30, 30))
        self.horizontalLayout_6.addWidget(self.downloadTorchBtn)
        self.uninstallTorchBtn = Button(parent=self.pytorchBackendInstallerContainer, text="", variant="surface",
            content_align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            content_padding=(10, 5, 10, 5), corner_radius=10, size=(0, 0))
        self.uninstallTorchBtn.setObjectName(u"uninstallTorchBtn")
        sizePolicy5.setHeightForWidth(self.uninstallTorchBtn.sizePolicy().hasHeightForWidth())
        self.uninstallTorchBtn.setSizePolicy(sizePolicy5)
        self.uninstallTorchBtn.setMaximumSize(QSize(50, 45))
        self.uninstallTorchBtn.setIcon(icon6)
        self.uninstallTorchBtn.setIconSize(QSize(30, 30))
        self.horizontalLayout_6.addWidget(self.uninstallTorchBtn)
        self.label_6 = Text(parent=self.pytorchBackendInstallerContainer)
        self.label_6.setObjectName(u"label_6")
        sizePolicy.setHeightForWidth(self.label_6.sizePolicy().hasHeightForWidth())
        self.label_6.setSizePolicy(sizePolicy)
        self.horizontalLayout_6.addWidget(self.label_6)
        self.verticalLayout_13.addWidget(self.pytorchBackendInstallerContainer)
        self.pytorchBackendInstallerContainer_2 = QWidget(self.scrollAreaWidgetContents_2)
        self.pytorchBackendInstallerContainer_2.setObjectName(u"pytorchBackendInstallerContainer_2")
        self.horizontalLayout_24 = QHBoxLayout(self.pytorchBackendInstallerContainer_2)
        self.horizontalLayout_24.setObjectName(u"horizontalLayout_24")
        self.horizontalLayout_24.setContentsMargins(0, 0, 0, 0)
        self.downloadTensorRTBtn = Button(parent=self.pytorchBackendInstallerContainer_2, text="", variant="surface",
            content_align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            content_padding=(10, 5, 10, 5), corner_radius=10, size=(0, 0))
        self.downloadTensorRTBtn.setObjectName(u"downloadTensorRTBtn")
        sizePolicy5.setHeightForWidth(self.downloadTensorRTBtn.sizePolicy().hasHeightForWidth())
        self.downloadTensorRTBtn.setSizePolicy(sizePolicy5)
        self.downloadTensorRTBtn.setMaximumSize(QSize(50, 45))
        self.downloadTensorRTBtn.setIcon(icon7)
        self.downloadTensorRTBtn.setIconSize(QSize(30, 30))
        self.horizontalLayout_24.addWidget(self.downloadTensorRTBtn)
        self.uninstallTensorRTBtn = Button(parent=self.pytorchBackendInstallerContainer_2, text="", variant="surface",
            content_align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            content_padding=(10, 5, 10, 5), corner_radius=10, size=(0, 0))
        self.uninstallTensorRTBtn.setObjectName(u"uninstallTensorRTBtn")
        sizePolicy5.setHeightForWidth(self.uninstallTensorRTBtn.sizePolicy().hasHeightForWidth())
        self.uninstallTensorRTBtn.setSizePolicy(sizePolicy5)
        self.uninstallTensorRTBtn.setMaximumSize(QSize(50, 45))
        self.uninstallTensorRTBtn.setIcon(icon6)
        self.uninstallTensorRTBtn.setIconSize(QSize(30, 30))
        self.horizontalLayout_24.addWidget(self.uninstallTensorRTBtn)
        self.label_8 = Text(parent=self.pytorchBackendInstallerContainer_2)
        self.label_8.setObjectName(u"label_8")
        self.horizontalLayout_24.addWidget(self.label_8)
        self.verticalLayout_13.addWidget(self.pytorchBackendInstallerContainer_2)
        self.directMLBackendInstallerContainer = QWidget(self.scrollAreaWidgetContents_2)
        self.directMLBackendInstallerContainer.setObjectName(u"directMLBackendInstallerContainer")
        self.horizontalLayout_8 = QHBoxLayout(self.directMLBackendInstallerContainer)
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.horizontalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.downloadDirectMLBtn = Button(parent=self.directMLBackendInstallerContainer, text="", variant="surface",
            content_align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            content_padding=(10, 5, 10, 5), corner_radius=10, size=(0, 0))
        self.downloadDirectMLBtn.setObjectName(u"downloadDirectMLBtn")
        sizePolicy5.setHeightForWidth(self.downloadDirectMLBtn.sizePolicy().hasHeightForWidth())
        self.downloadDirectMLBtn.setSizePolicy(sizePolicy5)
        self.downloadDirectMLBtn.setMaximumSize(QSize(50, 45))
        self.downloadDirectMLBtn.setIcon(icon7)
        self.downloadDirectMLBtn.setIconSize(QSize(30, 30))
        self.horizontalLayout_8.addWidget(self.downloadDirectMLBtn)
        self.uninstallDirectMLBtn = Button(parent=self.directMLBackendInstallerContainer, text="", variant="surface",
            content_align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            content_padding=(10, 5, 10, 5), corner_radius=10, size=(0, 0))
        self.uninstallDirectMLBtn.setObjectName(u"uninstallDirectMLBtn")
        sizePolicy5.setHeightForWidth(self.uninstallDirectMLBtn.sizePolicy().hasHeightForWidth())
        self.uninstallDirectMLBtn.setSizePolicy(sizePolicy5)
        self.uninstallDirectMLBtn.setMaximumSize(QSize(50, 45))
        self.uninstallDirectMLBtn.setIcon(icon6)
        self.uninstallDirectMLBtn.setIconSize(QSize(30, 30))
        self.horizontalLayout_8.addWidget(self.uninstallDirectMLBtn)
        self.label_28 = Text(parent=self.directMLBackendInstallerContainer)
        self.label_28.setObjectName(u"label_28")
        self.horizontalLayout_8.addWidget(self.label_28)
        self.verticalLayout_13.addWidget(self.directMLBackendInstallerContainer)
        # A settings column is a list of ROWS: the layout's leftover height is not theirs to share. Every container here
        # is a plain `QWidget`, whose default Preferred vertical policy grows, so the box layout handed it the slack and
        # a 33 px row became 168 px tall on a 1920x1080 window - the control floated in the middle of a huge band, and
        # the height changed with the window and with every section another control opened (measured 2026-11-22). The
        # filler takes the slack instead: a row is always its own height.
        self._settingsFiller = QSpacerItem(0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.verticalLayout_13.addItem(self._settingsFiller)
        self.scrollArea_2.setWidget(self.scrollAreaWidgetContents_2)
        self.verticalLayout_11.addWidget(self.scrollArea_2)
        self.gridLayout_5.addWidget(self.backendSelectContainer, 1, 0, 1, 1)
        self.retranslate()
        self._handles = DownloadPageHandles(
            pytorch_version=self.pytorch_version,
            pytorch_backend=self.pytorch_backend,
        )
        return self.downloadPage

    def retranslate(self) -> None:
        self.encoderLabel_4.setText(QCoreApplication.translate("MainWindow", u"Import Custom Upscale Model", None))
        self.selectNCNNCustomModel.setText(QCoreApplication.translate("MainWindow", u"Select Model (NCNN/.bin+.param)", None))
        self.selectPytorchCustomModel.setText(QCoreApplication.translate("MainWindow", u"Select Model (PyTorch/TensorRT/.pth)", None))
        self.uninstallAppBtn.setText("")
        self.label_67.setText(QCoreApplication.translate("MainWindow", u"Uninstall App", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"Backends", None))
        self.low_storage_label.setText(QCoreApplication.translate("MainWindow", u"Your PC does not have enough storage for PyTorch/TensorRT. Please free up at least 15gb.", None))
        # `setText("")` is gone with the stock label: an `ImageLabel` has no text channel and says so on stderr
        # ("ImageLabel ignores setText(''): a pixmap label has no text"); "no text" is already the default.
        self.label_65.setText(QCoreApplication.translate("MainWindow", u"PyTorch Version", None))
        self.label_53.setText(QCoreApplication.translate("MainWindow", u"PyTorch Backend", None))
        self.pytorch_backend.setItemText(0, QCoreApplication.translate("MainWindow", u"CUDA", None))
        self.pytorch_backend.setItemText(1, QCoreApplication.translate("MainWindow", u"ROCm", None))
        self.pytorch_backend.setItemText(2, QCoreApplication.translate("MainWindow", u"xpu", None))
        # same as label_64: an ImageLabel has no text channel, so the Designer's empty `setText` has no work
        self.downloadRecommendedBtn.setText("")
        self.label_25.setText(QCoreApplication.translate("MainWindow", u"Install Recommended Dependencies", None))
        self.downloadNCNNBtn.setText("")
        self.uninstallNCNNBtn.setText("")
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"NCNN Vulkan (All GPUs, slower inference, small download)", None))
        self.downloadTorchBtn.setText("")
        self.uninstallTorchBtn.setText("")
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"PyTorch (NVIDIA/CUDA, AMD/ROCm (Linux Only), INTEL/xpu, MacOS/MPS)", None))
        self.downloadTensorRTBtn.setText("")
        self.uninstallTensorRTBtn.setText("")
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"TensorRT (Nvidia RTX 20 series and up, fastest inference, largest download)", None))
        self.downloadDirectMLBtn.setText("")
        self.uninstallDirectMLBtn.setText("")
        self.label_28.setText(QCoreApplication.translate("MainWindow", u"DirectML - NOT IMPLEMENTED YET (All DirectX12 capable GPUs, faster inference, small download, Windows only)", None))
