"""Page `procPage` (QWidget), split out of the Designer-generated mainwindow.py.

Hand-maintained since 2026-11-23: the tree is the Designer's, the controls are the toolkit's. The Designer is no longer
a reference in this tree - the fixture, the splitter that emitted these pages and every "matches the Designer" gate
were removed - so this page answers to `harness/tests/test_toolkit_native.py`: the toolkit's controls and type scale,
`layout.addStretch()` instead of a `QSpacerItem`, and no new `name_<N>` object name or wrapper `QWidget` beyond the
counts that gate records.
"""

from __future__ import annotations

from PySide6.QtCore import (QCoreApplication, QSize, Qt)
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QGridLayout, QHBoxLayout, QSizePolicy, QVBoxLayout,
    QWidget)
import resources_rc
import resources_rc
from src.ui.pages.general_settings import GeneralSettingsPage
from src.ui.pages.tab_6 import Tab6Page
from src.ui.pages.render_queue_tab import RenderQueueTabPage
from src.ui.pages.advanced_settings import AdvancedSettingsPage
from src.ui.pages.tab_5 import Tab5Page
from src.ui.typography import apply, CAPTION_PT
from src.ui.widgets.tab_strip import TAB_HEIGHT, tab_button, tab_style

# Layer B (2026-11-21): the three widget classes this page no longer builds stock. `TopTabHost` is the tab host, built
# with the stock `QTabWidget`'s strip geometry (the Designer's 22 px strip, no pane radius or inset, a 4 px tab corner);
# `ImageLabel` is the pixmap counterpart of `QLabel` (parent first, and the alignment is stated because Qt's default is
# left+vcenter, not the centre the preview label asks for).
#
# Layer E (page migration): `Text` where the Designer had a `QLabel`, `Button` where it had a `QPushButton`,
# `DoubleSpinBox` for a `QDoubleSpinBox` and `ToastProgressBar` for a `QProgressBar`. Nothing here needs a Qt-shaped
# compatibility adapter.
from sli_ui_toolkit.widgets import (Button, DoubleSpinBox, ImageLabel, MinimalistScrollBar, SurfacePanel, TabStyle,
                                    Text, ToastProgressBar, TopTabHost)

#: Qt's spin-box text alignment (`QAbstractSpinBox().alignment()` is AlignLeft|AlignVCenter); the toolkit field
#: defaults to centre, which would move every value to the middle of its box.
SPIN_ALIGNMENT = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter


class ProcPagePage:
    """`procPage` - placed by its parent with `addWidget`."""

    page_id = "procPage"

    def build(self):
        """Build the widget tree and return the page root."""
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy7 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy7.setHorizontalStretch(0)
        sizePolicy7.setVerticalStretch(0)
        # splitter: skipped 'sizePolicy1.setHeightForWidth(self.mainWindowContainer.sizePolicy().hasHeightForWidth())' (['mainWindowContainer'] live outside this page)
        # splitter: skipped 'sizePolicy7.setHeightForWidth(self.inputFileText.sizePolicy().hasHeightForWidth())' (['inputFileText'] live outside this page)
        # splitter: skipped 'sizePolicy7.setHeightForWidth(self.scrollArea_6.sizePolicy().hasHeightForWidth())' (['scrollArea_6'] live outside this page)
        # splitter: skipped 'sizePolicy1.setHeightForWidth(self.renderQueueTab.sizePolicy().hasHeightForWidth())' (['renderQueueTab'] live outside this page)
        self.procPage = SurfacePanel(background_token="surface.page")
        self.procPage.setObjectName(u"procPage")
        # Layer C: the page body paints itself (`surface.page`); every control rule this sheet carried styled a widget
        # that is now a toolkit control painting from tokens. The one rule kept is the read-only rich-text panel on
        # the General tab, which is still a stock `QTextEdit` (a toolkit row: no rich-text pane with a scroll host).
        # No page stylesheet: every control on this page (and its tabs) paints itself from tokens. The read-only
        # rich-text panes are the toolkit's `RichTextView`, which carries its own plate.
        self.gridLayout_3 = QGridLayout(self.procPage)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.processInfoContainer = SurfacePanel(self.procPage, background_token="dialog.background", radius=10)
        self.processInfoContainer.setObjectName(u"processInfoContainer")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Expanding)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.processInfoContainer.sizePolicy().hasHeightForWidth())
        self.processInfoContainer.setSizePolicy(sizePolicy3)
        # The pane's own surface (a token, not an id rule): `dialog.background` is the #1f232a the rule carried.
        self.verticalLayout_5 = QVBoxLayout(self.processInfoContainer)
        self.verticalLayout_5.setSpacing(9)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_5.setContentsMargins(9, 9, 9, 9)
        self.previewSubContainer = SurfacePanel(self.processInfoContainer, background_token="dialog.background")
        self.previewSubContainer.setObjectName(u"previewSubContainer")
        self.verticalLayout_8 = QVBoxLayout(self.previewSubContainer)
        self.verticalLayout_8.setSpacing(0)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.verticalLayout_8.setContentsMargins(0, 0, 0, 0)
        # The stock label's own sheet was `QLabel{color:#fff; background-color:#1f232a}`: the colour had no subject (a
        # pixmap label has no text channel) and the background is `#previewSubContainer`'s own plate, painted by the
        # container's QSS, so the label keeps no surface of its own and the parent shows through.
        # The pane is the GPU surface (plan-rve-video-preview-widget-2026-11-23): a full-resolution frame with zoom,
        # the strip and its markers, all in one QRhi pass. `RVE_UI_RASTER_PREVIEW=1` keeps the old pixmap label for one
        # A/B - the fallback exists because a first QRhiWidget changes the window's compositing path on this stack.
        import os as _preview_env

        if _preview_env.environ.get("RVE_UI_RASTER_PREVIEW", "0").strip().lower() in ("1", "true", "yes", "on"):
            self.previewLabel = ImageLabel(self.previewSubContainer,
                alignment=Qt.AlignmentFlag.AlignCenter)
            self.previewLabel.setObjectName(u"previewLabel")
            self.previewLabel.setScaledContents(False)
            self.previewLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
            # `RVE_UI_PREVIEW_LABEL=default|opaque|fill|hide|small` cuts THIS label down one property at a time.
            from src.ui.devtools import label_bisect

            label_bisect.cut(self.previewLabel)
            self.preview_label_is_surface = False
        else:
            from src.ui.widgets.video_preview import VideoPreviewSurface

            self.previewLabel = VideoPreviewSurface(self.previewSubContainer)
            self.previewLabel.setObjectName(u"previewLabel")
            # The pane is opaque (an RHI widget cannot alpha-blend with its siblings) and shows this colour wherever no
            # frame is drawn: the same token the plate under it uses, so an empty pane reads as part of the page.
            from sli_ui_toolkit.theme import ThemeManager

            self.previewLabel.set_background_colour(
                ThemeManager.get_instance().try_get_color("dialog.background") or "#24252a")
            self.preview_label_is_surface = True
        sizePolicy4 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.previewLabel.sizePolicy().hasHeightForWidth())
        self.previewLabel.setSizePolicy(sizePolicy4)
        self.verticalLayout_8.addWidget(self.previewLabel)
        self.VideoPreview = QWidget(self.previewSubContainer)
        self.VideoPreview.setObjectName(u"VideoPreview")
        self.VideoPreview.setEnabled(True)
        sizePolicy1.setHeightForWidth(self.VideoPreview.sizePolicy().hasHeightForWidth())
        self.VideoPreview.setSizePolicy(sizePolicy1)
        self.horizontalLayout_26 = QHBoxLayout(self.VideoPreview)
        self.horizontalLayout_26.setObjectName(u"horizontalLayout_26")
        self.verticalLayout_8.addWidget(self.VideoPreview)
        self.verticalLayout_5.addWidget(self.previewSubContainer)
        self.RenderedPreviewControlsContainer = QWidget(self.processInfoContainer)
        self.RenderedPreviewControlsContainer.setObjectName(u"RenderedPreviewControlsContainer")
        self.RenderedPreviewControlsContainer.setEnabled(True)

        self.horizontalLayout_47 = QHBoxLayout(self.RenderedPreviewControlsContainer)
        self.horizontalLayout_47.setObjectName(u"horizontalLayout_47")
        self.widget_30 = QWidget(self.RenderedPreviewControlsContainer)
        self.widget_30.setObjectName(u"widget_30")
        self.verticalLayout_37 = QVBoxLayout(self.widget_30)
        self.verticalLayout_37.setObjectName(u"verticalLayout_37")
        self.label_9 = Text(parent=self.widget_30)
        self.label_9.setObjectName(u"label_9")
        self.verticalLayout_37.addWidget(self.label_9)
        # `single_step=0.1` is the toolkit field's "clamp and round to the display precision" semantic (its `setValue`
        # SNAPS to `single_step`, which at the default 1.0 would turn 3.5 into 3.0); `decimals=2` is what the Designer's
        # stock `QDoubleSpinBox` displayed and `alignment` is Qt's own (the toolkit field defaults to centre).
        self.startTimeSpinBox = DoubleSpinBox(self.widget_30, single_step=0.1, decimals=2, show_step_buttons=True,
                                              step_button_width=20, field_height=29, alignment=SPIN_ALIGNMENT)
        self.startTimeSpinBox.setObjectName(u"startTimeSpinBox")
        self.verticalLayout_37.addWidget(self.startTimeSpinBox)
        self.label_52 = Text(parent=self.widget_30)
        self.label_52.setObjectName(u"label_52")
        self.verticalLayout_37.addWidget(self.label_52)
        self.endTimeSpinBox = DoubleSpinBox(self.widget_30, single_step=0.1, decimals=2, show_step_buttons=True,
                                            step_button_width=20, field_height=29, alignment=SPIN_ALIGNMENT)
        self.endTimeSpinBox.setObjectName(u"endTimeSpinBox")
        # Qt's `setMinimum` is the one range spelling the toolkit field does not answer (`DoubleSpinBox` carries
        # `setRange` only), so the Designer's minimum plus Qt's own default maximum (99.99) are one call.
        self.endTimeSpinBox.setRange(1.000000000000000, 99.990000000000000)
        self.endTimeSpinBox.setValue(1.000000000000000)
        self.verticalLayout_37.addWidget(self.endTimeSpinBox)
        self.widget_14 = QWidget(self.widget_30)
        self.widget_14.setObjectName(u"widget_14")
        self.horizontalLayout_48 = QHBoxLayout(self.widget_14)
        self.horizontalLayout_48.setObjectName(u"horizontalLayout_48")
        self.horizontalLayout_48.setContentsMargins(0, 0, 0, 0)
        self.renderPreviewBtn = Button(parent=self.widget_14, text="", variant="surface",
            content_align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            content_padding=(10, 5, 10, 5), corner_radius=10, size=(0, 0))
        self.renderPreviewBtn.setObjectName(u"renderPreviewBtn")
        apply(self.renderPreviewBtn, None)
        # QPushButton's own policy (as wide as it needs, exactly as tall as its hint): the toolkit button is happy to
        # fill the cell, which turns a row button into a slab when the column is taller than its contents.
        self.renderPreviewBtn.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.renderPreviewBtn.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.horizontalLayout_48.addWidget(self.renderPreviewBtn)
        self.verticalLayout_37.addWidget(self.widget_14)
        # The scrub bar is the toolkit's scrollbar, not a stock `QScrollBar`: same value API (`value`/`setValue`/
        # `setRange`/`setMaximum`/`valueChanged`/`setEnabled` - `MinimalistScrollBar` subclasses `QScrollBar`), but the
        # thumb is painted from the theme (`accent` while dragging, a translucent white handle otherwise) instead of
        # from the window's deleted `QScrollBar::handle` rule. Orientation is the constructor's first argument, which
        # replaces the Designer's `setOrientation(Horizontal)` statement.
        self.timeInVideoScrollBar = MinimalistScrollBar(Qt.Orientation.Horizontal, self.widget_30)
        self.timeInVideoScrollBar.setObjectName(u"timeInVideoScrollBar")
        self.verticalLayout_37.addWidget(self.timeInVideoScrollBar)
        self.horizontalLayout_47.addWidget(self.widget_30)
        self.verticalLayout_5.addWidget(self.RenderedPreviewControlsContainer)
        self.verticalLayout_34 = QVBoxLayout()
        self.verticalLayout_34.setObjectName(u"verticalLayout_34")
        self.ETA = Text(parent=self.processInfoContainer)
        self.ETA.setObjectName(u"ETA")
        apply(self.ETA, CAPTION_PT)

        self.verticalLayout_34.addWidget(self.ETA)
        self.FPS = Text(parent=self.processInfoContainer)
        self.FPS.setObjectName(u"FPS")
        apply(self.FPS, CAPTION_PT)

        self.verticalLayout_34.addWidget(self.FPS)
        self.STATUS = Text(parent=self.processInfoContainer)
        self.STATUS.setObjectName(u"STATUS")
        apply(self.STATUS, CAPTION_PT)

        self.verticalLayout_34.addWidget(self.STATUS)
        self.verticalLayout_5.addLayout(self.verticalLayout_34)
        self.gridLayout_3.addWidget(self.processInfoContainer, 0, 2, 1, 1)
        self.processContainer = QWidget(self.procPage)
        self.processContainer.setObjectName(u"processContainer")
        sizePolicy5 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.processContainer.sizePolicy().hasHeightForWidth())
        self.processContainer.setSizePolicy(sizePolicy5)
        self.processContainer.setMaximumSize(QSize(500, 16777215))
        # No stylesheet: a disabled toolkit control takes its colour from its token. The rule only tinted the two stock
        # queue lists, which carry their own look until the toolkit grows a compact list.
        self.verticalLayout_7 = QVBoxLayout(self.processContainer)
        self.verticalLayout_7.setSpacing(0)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.verticalLayout_7.setContentsMargins(0, 10, 0, 0)
        self.bottomMenuSubContainer = SurfacePanel(self.processContainer, background_token="dialog.background")
        self.bottomMenuSubContainer.setObjectName(u"bottomMenuSubContainer")
        sizePolicy6 = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Preferred)
        sizePolicy6.setHorizontalStretch(0)
        sizePolicy6.setVerticalStretch(0)
        sizePolicy6.setHeightForWidth(self.bottomMenuSubContainer.sizePolicy().hasHeightForWidth())
        self.bottomMenuSubContainer.setSizePolicy(sizePolicy6)
        self.bottomMenuSubContainer.setMinimumSize(QSize(0, 0))
        self.bottomMenuSubContainer.setMaximumSize(QSize(500, 16777215))

        self.verticalLayout_21 = QVBoxLayout(self.bottomMenuSubContainer)
        self.verticalLayout_21.setSpacing(0)
        self.verticalLayout_21.setObjectName(u"verticalLayout_21")
        self.verticalLayout_21.setContentsMargins(0, 0, 0, 0)
        self.widget_2 = SurfacePanel(self.bottomMenuSubContainer, background_token="surface.page")
        self.widget_2.setObjectName(u"widget_2")
        self.verticalLayout_24 = QVBoxLayout(self.widget_2)
        self.verticalLayout_24.setSpacing(0)
        self.verticalLayout_24.setObjectName(u"verticalLayout_24")
        self.verticalLayout_24.setContentsMargins(0, 0, 0, 0)
        # The original interface's tabs are BUTTONS standing apart from the content: no folder outline, no join cover,
        # a plate on the selected one and a gap under the strip. `TabStyle` is the toolkit's styling surface for that.
        self.processSettingsContainer = TopTabHost(self.widget_2, tab_height=TAB_HEIGHT, pane_radius=0, pane_inset=(10, 5, 10, 5),
            tab_factory=tab_button,
            tab_style=tab_style(),
            chrome_inset=0, tab_radius=4, pane_fill_token=None)
        self.processSettingsContainer.setObjectName(u"processSettingsContainer")
        sizePolicy3.setHeightForWidth(self.processSettingsContainer.sizePolicy().hasHeightForWidth())
        self.processSettingsContainer.setSizePolicy(sizePolicy3)
        self.processSettingsContainer.setMinimumSize(QSize(0, 0))
        self.processSettingsContainer.setMaximumSize(QSize(16777215, 16777215))
        self.generalSettings_page = GeneralSettingsPage()
        self.generalSettings = self.generalSettings_page.build()
        self.processSettingsContainer.addTab(self.generalSettings, "")
        self.tab_6_page = Tab6Page()
        self.tab_6 = self.tab_6_page.build()
        self.processSettingsContainer.addTab(self.tab_6, "")
        self.renderQueueTab_page = RenderQueueTabPage()
        self.renderQueueTab = self.renderQueueTab_page.build()
        self.processSettingsContainer.addTab(self.renderQueueTab, "")
        self.advancedSettings_page = AdvancedSettingsPage()
        self.advancedSettings = self.advancedSettings_page.build()
        self.processSettingsContainer.addTab(self.advancedSettings, "")
        self.tab_5_page = Tab5Page()
        self.tab_5 = self.tab_5_page.build()
        self.processSettingsContainer.addTab(self.tab_5, "")
        self.verticalLayout_24.addWidget(self.processSettingsContainer)
        self.verticalLayout_21.addWidget(self.widget_2)
        self.widget_13 = SurfacePanel(self.bottomMenuSubContainer, background_token="surface.page")
        self.widget_13.setObjectName(u"widget_13")
        self.widget_13.setMaximumSize(QSize(500, 16777215))
        self.horizontalLayout_32 = QHBoxLayout(self.widget_13)
        self.horizontalLayout_32.setSpacing(6)
        self.horizontalLayout_32.setObjectName(u"horizontalLayout_32")
        self.horizontalLayout_32.setContentsMargins(9, 9, 9, 9)
        self.widget_24 = QWidget(self.widget_13)
        self.widget_24.setObjectName(u"widget_24")
        self.horizontalLayout_39 = QHBoxLayout(self.widget_24)
        self.horizontalLayout_39.setSpacing(0)
        self.horizontalLayout_39.setObjectName(u"horizontalLayout_39")
        self.horizontalLayout_39.setContentsMargins(0, 0, 0, 0)
        self.startRenderButton = Button(parent=self.widget_24, text="", variant="surface",
            content_align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            content_padding=(10, 5, 10, 5), corner_radius=10, size=(59, 55))
        self.startRenderButton.setObjectName(u"startRenderButton")
        self.startRenderButton.setEnabled(True)
        self.startRenderButton.setMaximumSize(QSize(59, 55))
        # The Designer pinned this button at 59x55. The toolkit's hint is 60x50 (the icon is 40x45 and its box is square:
        # 45 + 5 padding), so the ceiling alone left it 5 px short and 1 px wide. A minimum states the Designer's number,
        # and with the ceiling above it the button is exactly 59x55 in any row.

        # QPushButton's own policy: as wide as it needs, exactly as tall as its hint - the toolkit button would
        # otherwise fill the cell and grow past the Designer's 55 px ceiling in a taller row.
        self.startRenderButton.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        icon3 = QIcon()
        icon3.addFile(u":/icons/icons/play-circle.svg", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.startRenderButton.setIcon(icon3)
        self.startRenderButton.setIconSize(QSize(40, 45))
        # `setCheckable(False)` states the button's own default and `Button` has no such setter (toggle mode is the
        # constructor's `toggle=`), so the Designer's line is a no-op here; the two state resets below are kept.
        self.startRenderButton.setChecked(False)
        self.startRenderButton.setAutoRepeat(False)
        self.startRenderButton.setAutoExclusive(False)
        self.horizontalLayout_39.addWidget(self.startRenderButton)
        self.horizontalLayout_32.addWidget(self.widget_24)
        self.onRenderButtonsContiainer = QHBoxLayout()
        self.onRenderButtonsContiainer.setObjectName(u"onRenderButtonsContiainer")
        self.pauseRenderButton = Button(parent=self.widget_13, text="", variant="surface",
            content_align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            content_padding=(10, 5, 10, 5), corner_radius=10, size=(0, 0))
        self.pauseRenderButton.setObjectName(u"pauseRenderButton")
        self.pauseRenderButton.setEnabled(True)
        self.pauseRenderButton.setMaximumSize(QSize(59, 55))
        self.pauseRenderButton.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        icon4 = QIcon()
        icon4.addFile(u":/icons/icons/pause-circle.svg", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.pauseRenderButton.setIcon(icon4)
        self.pauseRenderButton.setIconSize(QSize(40, 45))
        self.pauseRenderButton.setChecked(False)
        self.onRenderButtonsContiainer.addWidget(self.pauseRenderButton)
        self.killRenderButton = Button(parent=self.widget_13, text="", variant="surface",
            content_align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            content_padding=(10, 5, 10, 5), corner_radius=10, size=(0, 0))
        self.killRenderButton.setObjectName(u"killRenderButton")
        sizePolicy11 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy11.setHorizontalStretch(0)
        sizePolicy11.setVerticalStretch(0)
        sizePolicy11.setHeightForWidth(self.killRenderButton.sizePolicy().hasHeightForWidth())
        self.killRenderButton.setSizePolicy(sizePolicy11)
        self.killRenderButton.setMaximumSize(QSize(59, 55))
        icon5 = QIcon()
        icon5.addFile(u":/icons/icons/x-circle.svg", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.killRenderButton.setIcon(icon5)
        self.killRenderButton.setIconSize(QSize(40, 45))
        self.onRenderButtonsContiainer.addWidget(self.killRenderButton)
        self.horizontalLayout_32.addLayout(self.onRenderButtonsContiainer)
        # `height=25` is the Designer's bar row; the toolkit toast bar's own default is 6 px. Its track paints from
        # the toolkit's `toast.progress.background` token (the adapter's call-site token was `progress.track.background`).
        self.progressBar = ToastProgressBar(self.widget_13, height=25, track_token="progress.track.background")
        self.progressBar.setObjectName(u"progressBar")
        sizePolicy7.setHeightForWidth(self.progressBar.sizePolicy().hasHeightForWidth())
        self.progressBar.setSizePolicy(sizePolicy7)
        self.progressBar.setValue(0)
        # `setTextVisible(False)` is gone with the stock bar: `ToastProgressBar` paints no label at all, so that state
        # is the one it is always in.
        self.horizontalLayout_32.addWidget(self.progressBar)
        self.verticalLayout_21.addWidget(self.widget_13)
        # The bar itself paints no label, so the numbers live on their own line right under it: percentage, elapsed,
        # remaining and the frame counter (owner, 2026-11-22: "under the huge progress bar there is no way to learn the
        # progress in percent or minutes, how much has passed").
        self.progressDetail = Text(parent=self.bottomMenuSubContainer)
        self.progressDetail.setObjectName(u"progressDetail")
        apply(self.progressDetail, CAPTION_PT)
        self.verticalLayout_21.addWidget(self.progressDetail)
        self.verticalLayout_7.addWidget(self.bottomMenuSubContainer)
        self.gridLayout_3.addWidget(self.processContainer, 0, 1, 1, 1)
        self.processSettingsContainer.setCurrentIndex(0)
        self.retranslate()
        return self.procPage

    def retranslate(self) -> None:
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"Start Time (seconds)", None))
        self.label_52.setText(QCoreApplication.translate("MainWindow", u"End Time (seconds)", None))
        self.renderPreviewBtn.setText(QCoreApplication.translate("MainWindow", u"Render Preview", None))
        self.ETA.setText(QCoreApplication.translate("MainWindow", u"ETA: ", None))
        self.FPS.setText(QCoreApplication.translate("MainWindow", u"FPS:", None))
        self.STATUS.setText(QCoreApplication.translate("MainWindow", u"STATUS:", None))
        self.processSettingsContainer.setTabText(self.processSettingsContainer.indexOf(self.generalSettings), QCoreApplication.translate("MainWindow", u"General", None))
        self.processSettingsContainer.setTabText(self.processSettingsContainer.indexOf(self.tab_6), QCoreApplication.translate("MainWindow", u"Encoder Options", None))
        self.processSettingsContainer.setTabText(self.processSettingsContainer.indexOf(self.renderQueueTab), QCoreApplication.translate("MainWindow", u"Render Queue", None))
        self.processSettingsContainer.setTabText(self.processSettingsContainer.indexOf(self.advancedSettings), QCoreApplication.translate("MainWindow", u"Advanced", None))
        self.processSettingsContainer.setTabText(self.processSettingsContainer.indexOf(self.tab_5), QCoreApplication.translate("MainWindow", u"Logs", None))
        self.startRenderButton.setText("")
        self.pauseRenderButton.setText("")
        self.killRenderButton.setText("")
