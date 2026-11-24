"""Page `tab_6` (QWidget), split out of the Designer-generated mainwindow.py.

Hand-maintained since 2026-11-23: the tree is the Designer's, the controls are the toolkit's. The Designer is no longer
a reference in this tree - the fixture, the splitter that emitted these pages and every "matches the Designer" gate
were removed - so this page answers to `harness/tests/test_toolkit_native.py`: the toolkit's controls and type scale,
`layout.addStretch()` instead of a `QSpacerItem`, and no new `name_<N>` object name or wrapper `QWidget` beyond the
counts that gate records.
"""

from __future__ import annotations

from PySide6.QtCore import (QCoreApplication, QObject, QRect, QSize)
from PySide6.QtWidgets import (QHBoxLayout, QSizePolicy, QSpacerItem, QVBoxLayout,
    QWidget)
import resources_rc
import resources_rc
# The page builds the toolkit's own controls: `Text` where the Designer had a `QLabel`, `ComboBox` where it had a
# `QComboBox`, `CustomLineEdit` where it had a `QLineEdit`. Nothing here needs a Qt-shaped compatibility adapter.
from sli_ui_toolkit.widgets import ComboBox, CustomLineEdit, OverlayScrollArea, SurfacePanel, Text
from sli_ui_toolkit.ui.managers.ui_scale import scaled_px
from src.ui.widgets.combo_width import cap_combo_to_row
from src.ui.typography import apply, BODY_PT, GROUP_TITLE_PT
from src.ui.scale_follower import ScaleFollower
from src.ui.pages.handles import EncoderHandles
from src.ui import i18n

#: RVE's window QSS scrollbar (`QScrollBar:vertical{width:8}`). A scroll area that RESERVES a bar must reserve this,
#: not the toolkit's default 10: measured on the settings rows, the 2 px difference narrowed every row. The value used
#: to come from the compatibility layer (`STOCK_SCROLLBAR_WIDTH`), which this page no longer imports.
STOCK_SCROLLBAR_WIDTH = 8

#: The encoder-options log row in DESIGN px: the header band the group title and the command row live in, and the row
#: itself (the field's line height plus its frame). Both are host geometry, so they take the UI factor; the field's
#: own hint is a toolkit metric and already scales with its font.
ENCODER_HEADER_MAX_HEIGHT_PX = 100
ENCODER_ROW_HEIGHT_PX = 50

#: The encoder-command field's floor: about 30 characters, so the field stays usable in a narrow column and the page
#: never has to honour a floor it cannot fit (`harness/tests/test_page_overflow.py`). It used to be 900 px - wider than
#: the column itself - and the field was drawn over the preview pane (owner, 2026-11-24). A design value: the floor
#: grows with the row it guards, so a raised factor does not leave it narrower than the text it must show.
ENCODER_COMMAND_MIN_WIDTH = 200


class _EncoderCommandTooltip(QObject):
    """Keep the whole encoder command in the field's tooltip: the field itself is only as wide as its column.

    A `QObject` **parented to the field**, not a method on the page. The page classes are plain objects and the shell
    that builds them is discarded after `build()`, so a page method connected to a widget signal is collected with the
    page and the connection stops firing - measured here: the tooltip kept the description and never grew the command.
    The same reason `src/ui/widgets/combo_width.py` wraps its event filter in a `QObject`.
    """

    def __init__(self, field, help_text: str) -> None:
        super().__init__(field)
        self._field = field
        self._help = help_text
        field.textChanged.connect(self.refresh)
        self.refresh()

    def set_help(self, help_text: str) -> None:
        self._help = help_text
        self.refresh()

    def refresh(self) -> None:
        command = self._field.text()
        self._field.setToolTip(f"{self._help}\n\n{command}" if command else self._help)


class Tab6Page:
    """`tab_6` - placed by its parent with `addTab`."""

    page_id = "tab_6"

    def build(self):
        """Build the widget tree and return the page root."""
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy7 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy7.setHorizontalStretch(0)
        sizePolicy7.setVerticalStretch(0)
        # splitter: skipped 'sizePolicy2.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())' (['label'] live outside this page)
        # splitter: skipped 'sizePolicy2.setHeightForWidth(self.scrollArea_4.sizePolicy().hasHeightForWidth())' (['scrollArea_4'] live outside this page)
        # splitter: skipped 'sizePolicy7.setHeightForWidth(self.inputFileText.sizePolicy().hasHeightForWidth())' (['inputFileText'] live outside this page)
        # splitter: skipped 'sizePolicy2.setHeightForWidth(self.scrollAreaWidgetContents_5.sizePolicy().hasHeightForWidth())' (['scrollAreaWidgetContents_5'] live outside this page)
        # splitter: skipped 'sizePolicy2.setHeightForWidth(self.generalContainer.sizePolicy().hasHeightForWidth())' (['generalContainer'] live outside this page)
        # splitter: skipped 'sizePolicy2.setHeightForWidth(self.upscaleContainer.sizePolicy().hasHeightForWidth())' (['upscaleContainer'] live outside this page)
        # splitter: skipped 'sizePolicy2.setHeightForWidth(self.interpolateContainer_2.sizePolicy().hasHeightForWidth())' (['interpolateContainer_2'] live outside this page)
        # splitter: skipped 'sizePolicy7.setHeightForWidth(self.progressBar.sizePolicy().hasHeightForWidth())' (['progressBar'] live outside this page)
        self.tab_6 = SurfacePanel(background_token="surface.page")
        self.tab_6.setObjectName(u"tab_6")
        self.horizontalLayout_43 = QHBoxLayout(self.tab_6)
        self.horizontalLayout_43.setObjectName(u"horizontalLayout_43")
        self.widget_6 = QWidget(self.tab_6)
        self.widget_6.setObjectName(u"widget_6")
        self.verticalLayout_15 = QVBoxLayout(self.widget_6)
        self.verticalLayout_15.setObjectName(u"verticalLayout_15")
        # Layer B: a stock QScrollArea has a StyledPanel frame (content = area - 2 px) and, on this page, no
        # page QSS rule over it, so the stock geometry is `content_inset=1` and no reserved scrollbar gutter.
        self.scrollArea_7 = OverlayScrollArea(self.widget_6, content_inset=1, scrollbar_width=STOCK_SCROLLBAR_WIDTH)
        self.scrollArea_7.setObjectName(u"scrollArea_7")
        self.scrollArea_7.setWidgetResizable(True)
        self.encoderSettings = QWidget()
        self.encoderSettings.setObjectName(u"encoderSettings")
        self.encoderSettings.setGeometry(QRect(0, 0, 453, 581))
        self.verticalLayout_40 = QVBoxLayout(self.encoderSettings)
        self.verticalLayout_40.setObjectName(u"verticalLayout_40")
        self.widget_34 = QWidget(self.encoderSettings)
        self.widget_34.setObjectName(u"widget_34")
        sizePolicy2.setHeightForWidth(self.widget_34.sizePolicy().hasHeightForWidth())
        self.widget_34.setSizePolicy(sizePolicy2)
        self.verticalLayout_36 = QVBoxLayout(self.widget_34)
        self.verticalLayout_36.setSpacing(0)
        self.verticalLayout_36.setObjectName(u"verticalLayout_36")
        self.verticalLayout_36.setContentsMargins(0, 0, 0, 0)
        self.widget_33 = QWidget(self.widget_34)
        self.widget_33.setObjectName(u"widget_33")
        sizePolicy9 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        sizePolicy9.setHorizontalStretch(0)
        sizePolicy9.setVerticalStretch(0)
        sizePolicy9.setHeightForWidth(self.widget_33.sizePolicy().hasHeightForWidth())
        self.widget_33.setSizePolicy(sizePolicy9)
        self.widget_33.setMaximumSize(QSize(16777215, scaled_px(ENCODER_HEADER_MAX_HEIGHT_PX)))
        self.verticalLayout_9 = QVBoxLayout(self.widget_33)
        self.verticalLayout_9.setObjectName(u"verticalLayout_9")
        self.verticalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.label_74 = Text(parent=self.widget_33)
        self.label_74.setObjectName(u"label_74")
        apply(self.label_74, GROUP_TITLE_PT, underline=True)
        self.verticalLayout_9.addWidget(self.label_74)
        self.widget_32 = QWidget(self.widget_33)
        self.widget_32.setObjectName(u"widget_32")
        self.horizontalLayout_51 = QHBoxLayout(self.widget_32)
        self.horizontalLayout_51.setObjectName(u"horizontalLayout_51")
        # same stock geometry as `scrollArea_7` (framed, no page rule, no reserved bar)
        self.scrollArea_6 = OverlayScrollArea(self.widget_32, content_inset=1, reserve_scrollbar_space=False)
        self.scrollArea_6.setObjectName(u"scrollArea_6")
        sizePolicy7.setHeightForWidth(self.scrollArea_6.sizePolicy().hasHeightForWidth())
        self.scrollArea_6.setSizePolicy(sizePolicy7)
        self.scrollArea_6.setMinimumSize(QSize(0, scaled_px(ENCODER_ROW_HEIGHT_PX)))
        self.scrollArea_6.setMaximumSize(QSize(16777215, scaled_px(ENCODER_ROW_HEIGHT_PX)))
        # The overlay area owns both affordances; a native bar would sit on top of its own and steal viewport space.
        self.scrollArea_6.setWidgetResizable(True)
        self.scrollAreaWidgetContents_6 = QWidget()
        self.scrollAreaWidgetContents_6.setObjectName(u"scrollAreaWidgetContents_6")
        self.scrollAreaWidgetContents_6.setGeometry(QRect(0, 0, 918, 48))
        self.horizontalLayout_54 = QHBoxLayout(self.scrollAreaWidgetContents_6)
        self.horizontalLayout_54.setObjectName(u"horizontalLayout_54")
        self.EncoderCommand = CustomLineEdit(self.scrollAreaWidgetContents_6)
        self.EncoderCommand.setObjectName(u"EncoderCommand")
        # A command is a line that can be longer than its column, and this row holds nothing else: the field takes the
        # row's width and scrolls its own text to the caret. A width driven by the text - or a 900 px floor, which is
        # what this page used to state - is a control outside its page: the layout gives it the minimum, and the field
        # paints past the column and under the preview pane (owner, 2026-11-24; the gate is
        # `harness/tests/test_page_overflow.py`). The tooltip carries the whole command, because the field shows only
        # the part that fits.
        self.EncoderCommand.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.EncoderCommand.setMinimumWidth(scaled_px(ENCODER_COMMAND_MIN_WIDTH))
        self._encoder_command_tooltip = _EncoderCommandTooltip(self.EncoderCommand, "")
        self.horizontalLayout_54.addWidget(self.EncoderCommand)
        self.scrollArea_6.setWidget(self.scrollAreaWidgetContents_6)
        self.horizontalLayout_51.addWidget(self.scrollArea_6)
        self.verticalLayout_9.addWidget(self.widget_32)
        self.verticalLayout_36.addWidget(self.widget_33)
        self.verticalLayout_40.addWidget(self.widget_34)
        self.widget_15 = QWidget(self.encoderSettings)
        self.widget_15.setObjectName(u"widget_15")
        self.verticalLayout_26 = QVBoxLayout(self.widget_15)
        self.verticalLayout_26.setObjectName(u"verticalLayout_26")
        self.widget_17 = QWidget(self.widget_15)
        self.widget_17.setObjectName(u"widget_17")
        self.horizontalLayout_15 = QHBoxLayout(self.widget_17)
        self.horizontalLayout_15.setObjectName(u"horizontalLayout_15")
        self.encoderLabel = Text(parent=self.widget_17)
        self.encoderLabel.setObjectName(u"encoderLabel")
        apply(self.encoderLabel, BODY_PT)
        self.horizontalLayout_15.addWidget(self.encoderLabel)
        self.horizontalSpacer_9 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.horizontalLayout_15.addItem(self.horizontalSpacer_9)
        self.encoder = ComboBox(self.widget_17)
        self.encoder.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        cap_combo_to_row(self.encoder)
        self.encoder.addItem("")
        self.encoder.addItem("")
        self.encoder.addItem("")
        self.encoder.addItem("")
        self.encoder.addItem("")
        self.encoder.addItem("")
        self.encoder.addItem("")
        self.encoder.addItem("")
        self.encoder.addItem("")
        self.encoder.addItem("")
        self.encoder.setObjectName(u"encoder")
        self.encoder.setToolTip(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" font-weight:700;\">Encoder:</span></p><p>- Compression algorithm for video, most common is libx264</p><p>- Any encoder without a device label is a CPU encoder.</p><p>- <span style=\" font-weight:700; text-decoration: underline;\">Lossy Encoders:</span></p><p>- libx264 (<span style=\" font-weight:700;\">mkv, mp4, mov, avi</span>)</p><p>- libx265 (<span style=\" font-weight:700;\">mkv, mp4, mov</span>)</p><p>- vp9 (<span style=\" font-weight:700;\">mkv, webm</span>)</p><p>- av1 (<span style=\" font-weight:700;\">mkv, mp4, webm</span>)</p><p>- <span style=\" font-weight:700; text-decoration: underline;\">Lossless Encoders:</span></p><p>- prores (<span style=\" font-weight:700;\">mkv, mov</span>)</p><p>- ffv1 (<span style=\" font-weight:700;\">mkv, avi</span>)</p></body></html>", None))
        self.encoder.setMinimumSize(QSize(0, 0))
        self.horizontalLayout_15.addWidget(self.encoder)
        self.verticalLayout_26.addWidget(self.widget_17)
        self.widget_16 = QWidget(self.widget_15)
        self.widget_16.setObjectName(u"widget_16")
        self.horizontalLayout_36 = QHBoxLayout(self.widget_16)
        self.horizontalLayout_36.setObjectName(u"horizontalLayout_36")
        self.encoderLabel_2 = Text(parent=self.widget_16)
        self.encoderLabel_2.setObjectName(u"encoderLabel_2")
        apply(self.encoderLabel_2, BODY_PT)
        self.horizontalLayout_36.addWidget(self.encoderLabel_2)
        self.horizontalSpacer_22 = QSpacerItem(237, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.horizontalLayout_36.addItem(self.horizontalSpacer_22)
        self.video_encoder_speed = ComboBox(self.widget_16)
        self.video_encoder_speed.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        cap_combo_to_row(self.video_encoder_speed)
        self.video_encoder_speed.addItem("")
        self.video_encoder_speed.addItem("")
        self.video_encoder_speed.addItem("")
        self.video_encoder_speed.addItem("")
        self.video_encoder_speed.addItem("")
        self.video_encoder_speed.setObjectName(u"video_encoder_speed")
        self.video_encoder_speed.setToolTip(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" font-weight:700;\">Encoder Speed:</span></p><p> - Speed at which the encoder processes the video, <span style=\" font-weight:700; text-decoration: underline;\">Slower</span> processing time means <span style=\" font-weight:700; text-decoration: underline;\">Better results for the file size</span><span style=\" text-decoration: underline;\">.</span></p></body></html>", None))
        self.video_encoder_speed.setMinimumSize(QSize(0, 0))
        self.horizontalLayout_36.addWidget(self.video_encoder_speed)
        self.verticalLayout_26.addWidget(self.widget_16)
        self.widget_18 = QWidget(self.widget_15)
        self.widget_18.setObjectName(u"widget_18")
        self.horizontalLayout_33 = QHBoxLayout(self.widget_18)
        self.horizontalLayout_33.setObjectName(u"horizontalLayout_33")
        self.encoderLabel_5 = Text(parent=self.widget_18)
        self.encoderLabel_5.setObjectName(u"encoderLabel_5")
        apply(self.encoderLabel_5, BODY_PT)
        self.horizontalLayout_33.addWidget(self.encoderLabel_5)
        self.horizontalSpacer_17 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.horizontalLayout_33.addItem(self.horizontalSpacer_17)
        self.video_quality = ComboBox(self.widget_18)
        self.video_quality.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        cap_combo_to_row(self.video_quality)
        self.video_quality.addItem("")
        self.video_quality.addItem("")
        self.video_quality.addItem("")
        self.video_quality.addItem("")
        self.video_quality.addItem("")
        self.video_quality.addItem("")
        self.video_quality.setObjectName(u"video_quality")
        self.video_quality.setMinimumSize(QSize(0, 0))
        self.horizontalLayout_33.addWidget(self.video_quality)
        self.verticalLayout_26.addWidget(self.widget_18)
        self.widget_27 = QWidget(self.widget_15)
        self.widget_27.setObjectName(u"widget_27")
        self.horizontalLayout_44 = QHBoxLayout(self.widget_27)
        self.horizontalLayout_44.setObjectName(u"horizontalLayout_44")
        self.label_49 = Text(parent=self.widget_27)
        self.label_49.setObjectName(u"label_49")
        apply(self.label_49, BODY_PT)
        self.horizontalLayout_44.addWidget(self.label_49)
        self.horizontalSpacer_19 = QSpacerItem(437, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.horizontalLayout_44.addItem(self.horizontalSpacer_19)
        self.video_container = ComboBox(self.widget_27)
        self.video_container.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        cap_combo_to_row(self.video_container)
        self.video_container.addItem("")
        self.video_container.addItem("")
        self.video_container.addItem("")
        self.video_container.addItem("")
        self.video_container.addItem("")
        self.video_container.setObjectName(u"video_container")
        self.video_container.setToolTip(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" font-weight:700;\">Video Container</span></p><p>- Changes the video container in the default output file generated container.</p><p>- <span style=\" font-weight:700;\">mkv </span>is <span style=\" font-weight:700;\">recomended</span> due to its extensive support for different formats.</p><p><span style=\" font-weight:700;\">WARNING: Changing this </span><span style=\" font-weight:700; text-decoration: underline;\">MAY</span><span style=\" font-weight:700;\"> break encoder compadibility. </span></p><p><span style=\" font-weight:700;\">Please look at the audio and video encoder tooltips for compadibility.</span></p></body></html>", None))
        self.horizontalLayout_44.addWidget(self.video_container)
        self.verticalLayout_26.addWidget(self.widget_27)
        self.widget_21 = QWidget(self.widget_15)
        self.widget_21.setObjectName(u"widget_21")
        self.horizontalLayout_35 = QHBoxLayout(self.widget_21)
        self.horizontalLayout_35.setObjectName(u"horizontalLayout_35")
        self.encoderLabel_7 = Text(parent=self.widget_21)
        self.encoderLabel_7.setObjectName(u"encoderLabel_7")
        apply(self.encoderLabel_7, BODY_PT)
        self.horizontalLayout_35.addWidget(self.encoderLabel_7)
        self.horizontalSpacer_20 = QSpacerItem(469, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.horizontalLayout_35.addItem(self.horizontalSpacer_20)
        self.video_pixel_format = ComboBox(self.widget_21)
        self.video_pixel_format.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        cap_combo_to_row(self.video_pixel_format)
        self.video_pixel_format.addItem("")
        self.video_pixel_format.addItem("")
        self.video_pixel_format.addItem("")
        self.video_pixel_format.addItem("")
        self.video_pixel_format.addItem("")
        self.video_pixel_format.addItem("")
        self.video_pixel_format.setObjectName(u"video_pixel_format")
        self.video_pixel_format.setToolTip(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" font-weight:700;\">Pixel Format</span></p><p>- D<span style=\" font-family:'Droid Sans Mono','monospace','monospace'; font-size:9pt; color:#cccccc;\">efines how color information is stored for each pixel in a video frame.</span></p><p><span style=\" font-family:'Droid Sans Mono','monospace','monospace'; font-size:9pt; color:#cccccc;\">- yuv420p: Most common, removes the most information. </span>(<span style=\" font-weight:700;\">mkv, mp4, mov, avi, webm</span>)</p><p><span style=\" font-family:'Droid Sans Mono','monospace','monospace'; font-size:9pt; color:#cccccc;\">- yuv422p: In between 420p and 444p. </span>(<span style=\" font-weight:700;\">mkv, mov, mp4</span>)</p><p><span style=\" font-family:'Droid Sans Mono','monospace','monospace'; font-size:9pt; color:#cccccc;\">- yuv444p: Keeps the most information. </span>(<span style=\" font-weight:700;\">mkv, mov, mp4</span>)<br/></p><p>- <span style=\" font-family:'Droid Sans Mono','monospace','monospace'; font-size:9pt; colo"
                        "r:#cccccc;\">yuv 420/422/444 p10le: 10 bit version of each. (MKV and MP4 containers only) </span>(<span style=\" font-weight:700;\">mkv, mov, mp4</span>)</p></body></html>", None))
        self.video_pixel_format.setMinimumSize(QSize(0, 0))
        self.horizontalLayout_35.addWidget(self.video_pixel_format)
        self.verticalLayout_26.addWidget(self.widget_21)
        self.verticalLayout_40.addWidget(self.widget_15)
        self.verticalSpacer_2 = QSpacerItem(20, 50000, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.verticalLayout_40.addItem(self.verticalSpacer_2)
        self.scrollArea_7.setWidget(self.encoderSettings)
        self.verticalLayout_15.addWidget(self.scrollArea_7)
        self.horizontalLayout_43.addWidget(self.widget_6)
        self._handles = EncoderHandles(
            encoder=self.encoder,
            EncoderCommand=self.EncoderCommand,
            video_container=self.video_container,
            video_pixel_format=self.video_pixel_format,
            video_quality=self.video_quality,
            video_encoder_speed=self.video_encoder_speed,
            encoderSettings=self.encoderSettings,
        )
        # The page pass needs a QObject to walk; the relay is that object and forwards
        # `retranslate(lang)` to this builder (`src/ui/i18n.py`).
        i18n.bind_page(self, self.tab_6)
        self.retranslate()
        # The log row is host geometry: re-stated at the factor in force now and on every later change, so the header
        # band and the command row grow with the text inside them instead of clipping it.
        self._log_row_scale = ScaleFollower(self.tab_6, self._apply_log_row_geometry)
        return self.tab_6

    def _apply_log_row_geometry(self) -> None:
        """State the encoder log row's boxes at the factor in force now (design px through `scaled_px`).

        The header band is the group title plus the command row; the row's height is the field's line height plus its
        frame; the field's floor is stated with it so a raised factor does not leave the floor narrower than the
        text it must show. All three are host geometry, so all three take the factor once.
        """
        header = scaled_px(ENCODER_HEADER_MAX_HEIGHT_PX)
        row = scaled_px(ENCODER_ROW_HEIGHT_PX)
        self.widget_33.setMaximumSize(QSize(16777215, header))
        self.scrollArea_6.setMinimumSize(QSize(0, row))
        self.scrollArea_6.setMaximumSize(QSize(16777215, row))
        self.EncoderCommand.setMinimumWidth(scaled_px(ENCODER_COMMAND_MIN_WIDTH))

    def retranslate(self, language: str | None = None) -> None:
        self._encoder_command_tooltip.set_help(QCoreApplication.translate(
            "MainWindow",
            u"Encoder command: the encoder settings passed to FFMpeg, can be manually tweaked. "
            u"The field shows the part that fits; the whole command is in this tooltip.",
            None))
        self.label_74.setText(QCoreApplication.translate("MainWindow", u"Encoder Settings", None))
        self.encoderLabel.setText(QCoreApplication.translate("MainWindow", u"Video Encoder", None))
        self.encoder.setItemText(0, u"libx264")
        self.encoder.setItemText(1, u"libx265")
        self.encoder.setItemText(2, u"vp9")
        self.encoder.setItemText(3, u"av1")
        self.encoder.setItemText(4, u"prores")
        self.encoder.setItemText(5, u"ffv1")
        self.encoder.setItemText(6, u"utvideo")
        self.encoder.setItemText(7, u"x264_nvenc")
        self.encoder.setItemText(8, u"x265_nvenc")
        self.encoder.setItemText(9, u"av1_nvenc (40 series and up)")
        self.encoderLabel_2.setText(QCoreApplication.translate("MainWindow", u"Video Encoder Speed", None))
        self.video_encoder_speed.setItemText(0, QCoreApplication.translate("MainWindow", u"fastest", None))
        self.video_encoder_speed.setItemText(1, QCoreApplication.translate("MainWindow", u"fast", None))
        self.video_encoder_speed.setItemText(2, QCoreApplication.translate("MainWindow", u"medium", None))
        self.video_encoder_speed.setItemText(3, QCoreApplication.translate("MainWindow", u"slow", None))
        self.video_encoder_speed.setItemText(4, QCoreApplication.translate("MainWindow", u"placebo", None))
        self.video_encoder_speed.setItemData(0, 'fastest')
        self.video_encoder_speed.setItemData(1, 'fast')
        self.video_encoder_speed.setItemData(2, 'medium')
        self.video_encoder_speed.setItemData(3, 'slow')
        self.video_encoder_speed.setItemData(4, 'placebo')
        self.encoderLabel_5.setText(QCoreApplication.translate("MainWindow", u"Video Quality", None))
        self.video_quality.setItemText(0, QCoreApplication.translate("MainWindow", u"Lossless", None))
        self.video_quality.setItemText(1, QCoreApplication.translate("MainWindow", u"Ultra", None))
        self.video_quality.setItemText(2, QCoreApplication.translate("MainWindow", u"Very_High", None))
        self.video_quality.setItemText(3, QCoreApplication.translate("MainWindow", u"High", None))
        self.video_quality.setItemText(4, QCoreApplication.translate("MainWindow", u"Medium", None))
        self.video_quality.setItemText(5, QCoreApplication.translate("MainWindow", u"Low", None))
        # The VALUE the builder reads back; the visible text is a catalogue string.
        self.video_quality.setItemData(0, 'Lossless')
        self.video_quality.setItemData(1, 'Ultra')
        self.video_quality.setItemData(2, 'Very_High')
        self.video_quality.setItemData(3, 'High')
        self.video_quality.setItemData(4, 'Medium')
        self.video_quality.setItemData(5, 'Low')
        self.label_49.setText(QCoreApplication.translate("MainWindow", u"Video Container", None))
        self.video_container.setItemText(0, u"mkv")
        self.video_container.setItemText(1, u"mp4")
        self.video_container.setItemText(2, u"mov")
        self.video_container.setItemText(3, u"webm")
        self.video_container.setItemText(4, u"avi")
        self.encoderLabel_7.setText(QCoreApplication.translate("MainWindow", u"Video Pixel Format", None))
        self.video_pixel_format.setItemText(0, u"yuv420p")
        self.video_pixel_format.setItemText(1, u"yuv422p")
        self.video_pixel_format.setItemText(2, u"yuv444p")
        self.video_pixel_format.setItemText(3, u"yuv420p (10 bit)")
        self.video_pixel_format.setItemText(4, u"yuv422p (10 bit)")
        self.video_pixel_format.setItemText(5, u"yuv444p (10 bit)")
