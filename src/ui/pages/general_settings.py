"""Page `generalSettings` (QWidget), split out of the Designer-generated mainwindow.py.

Hand-maintained since 2026-11-23: the tree is the Designer's, the controls are the toolkit's. The Designer is no longer
a reference in this tree - the fixture, the splitter that emitted these pages and every "matches the Designer" gate
were removed - so this page answers to `harness/tests/test_toolkit_native.py`: the toolkit's controls and type scale,
`layout.addStretch()` instead of a `QSpacerItem`, and no new `name_<N>` object name or wrapper `QWidget` beyond the
counts that gate records.
"""

from __future__ import annotations

from PySide6.QtCore import (QCoreApplication, QRect, QSize, Qt)
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QAbstractScrollArea, QHBoxLayout, QSizePolicy,
    QSpacerItem, QTextEdit, QVBoxLayout, QWidget)
import resources_rc
import resources_rc
# The page builds the toolkit's own controls: `Text` where the Designer had a `QLabel`, `CustomLineEdit` for a
# `QLineEdit`, `Button` for a `QPushButton`, `CheckBox` for a `QCheckBox`, `ComboBox` for a `QComboBox` and
# `DoubleSpinBox` for a `QDoubleSpinBox`. Nothing here needs a Qt-shaped compatibility adapter.
from sli_ui_toolkit.ui.managers.ui_scale import UiScale, scaled_px
from sli_ui_toolkit.widgets import (Button, CheckBox, ComboBox, CustomLineEdit, DoubleSpinBox, RichTextView,
    SurfacePanel, SurfaceScrollArea, Text)
from src.ui.widgets.combo_width import cap_combo_to_row
from src.ui.typography import apply, BODY_PT
from src.ui.pages.handles import GeneralSettingsHandles
from src.ui.scale_follower import ScaleFollower
from src.ui import i18n

#: RVE's control text is 12 PIXELS (the Designer's sizing). The info pane's stock editor was 12 px, and
#: `setFontPointSize` would convert through the screen DPI and render 16 px - the value used to come from the
#: compatibility layer (`RVE_CONTROL_FONT_PX`), which this page no longer imports.
RVE_CONTROL_FONT_PX = 12

#: RVE's window QSS scrollbar (`QScrollBar:vertical{width:8}`). A scroll area that RESERVES a bar must reserve this,
#: not the toolkit's default 10: measured on these settings rows, the 2 px difference narrowed every row.
STOCK_SCROLLBAR_WIDTH = 8

#: The Designer's `QFrame` separator is a 3 px sunken line. The toolkit's `Divider` is a button-region spec, not a
#: widget, so the line is a token-painted panel of that height (`SEPARATOR_HEIGHT` in the layer this page dropped).
#: DESIGN px: a painted line still grows with the factor, or it turns into a hairline against a 45 px row.
SEPARATOR_HEIGHT = 3

#: The settings row's height in DESIGN px (owner, 2026-11-24): the rows, the path fields, the combo boxes and the value
#: spin boxes all state this one row. At 1.0 it is the Designer's 30; 45 at 1.5 and 60 at 2.0 come from `scaled_px()`
#: at apply time, never from an import-time snapshot. The old constant was an absolute pin, which is what clipped a
#: 12 pt label's line at 1.5 (wave 2 left the page body scrollable but the rows still 30 px).
ROW_HEIGHT_PX = 30

#: A value spin box's width in DESIGN px; its height is `ROW_HEIGHT_PX` (the Designer's 75x30).
FIELD_WIDTH_PX = 75

#: The page body's gap and inset in DESIGN px: the Designer's 11 between rows and around the body.
PAGE_GAP_PX = 11

#: A settings section's inner inset in DESIGN px (the Designer's 9 around the interpolation/upscale groups).
SECTION_INSET_PX = 9

#: The upscale section's tighter design gap (the Designer's 6 between its two rows).
SECTION_GAP_PX = 6

#: The path fields' design width ceilings (the Designer's 1000/10000). A ceiling, not the row height, but it is part
#: of the same size statement and scales with it.
INPUT_PATH_MAX_WIDTH_PX = 1000
OUTPUT_PATH_MAX_WIDTH_PX = 10000

#: The render-queue action icon's design px.
ACTION_ICON_PX = 32

#: The General tab's information pane, in DESIGN points: the Designer's 11 pt body and 10 pt spans. Qt's rich-text
#: engine treats an inline `font-size` as ABSOLUTE, so the page must state these at DESIGN x factor itself - leaving
#: `10pt` in the markup kept the body at 10 pt at every factor (measured: 172 px of ink at 1.0, 1.5 and 2.0).
INFO_BODY_PT = 11
INFO_TEXT_PT = 10

#: The pane's nine rows, in the Designer's order. These are catalogue KEYS, not the text on screen: the document is
#: painted from `i18n.translate(key)` at the language in force, so a switch rewrites the names and nothing else.
INFO_LABELS = ("FPS:", "Resolution:", "Frame Count:", "Encoder:", "Container:", "Color Space:",
               "Pixel Format:", "Is HDR:", "Bit Depth:")


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


def _design_pt(design_pt: float, factor: float) -> str:
    """A design point size at the factor in force, as the CSS string Qt's rich-text engine wants.

    ``:g`` keeps 1.0's ``10pt``/``11pt`` byte-identical to the Designer's markup, and writes ``15pt``/``16.5pt`` at 1.5
    without a trailing ``.0``.
    """
    return f"{design_pt * factor:g}pt"


def _info_html(factor: float, language: str | None = None) -> str:
    """The General tab's information body at ``factor``, its nine row names in ``language``.

    Every size is stated as DESIGN x factor. The alternative - removing the inline sizes and letting the document font
    rule - would use `RichTextView`'s 12 px body at 1.0, while the visible text today is the spans' 10 pt (13.3 px):
    that is a 1.0 change, and 1.0 is this wave's identity gate. Building the markup keeps the Designer's pixel at 1.0
    and makes the body grow with the factor instead of standing still on an absolute 10 pt.

    The names go through the catalogue (`src/ui/i18n.py`, keyed by the English source string) rather than being baked
    in: this document is the pane's whole content, so its own text is the only thing a language switch can rewrite.
    ``language=None`` is the language in force, which is what the first paint uses.
    """
    span_pt = _design_pt(INFO_TEXT_PT, factor)
    body_pt = _design_pt(INFO_BODY_PT, factor)
    rows = "\n".join(
        '<p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; '
        f'text-indent:0px;"><span style=" font-family:\'Sans Serif\'; font-size:{span_pt};">'
        f'{i18n.translate(label, language)}</span></p>'
        for label in INFO_LABELS
    )
    return (
        '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
        '<html><head><meta name="qrichtext" content="1" /><meta charset="utf-8" /><style type="text/css">\n'
        'p, li { white-space: pre-wrap; }\n'
        'hr { height: 1px; border-width: 0; }\n'
        'li.unchecked::marker { content: "\\2610"; }\n'
        'li.checked::marker { content: "\\2612"; }\n'
        f'</style></head><body style=" font-family:\'Adwaita Sans\'; font-size:{body_pt}; font-weight:400; '
        f'font-style:normal;">\n{rows}</body></html>'
    )


def _info_name_variants(label: str) -> tuple[str, ...]:
    """Every spelling a row name can carry on screen: the English key plus each language's translation.

    A second switch has to recognise the spelling the first one painted, so the rewrite reads the catalogue rather
    than assuming the pane is still in English.
    """
    variants = [label]
    for table in i18n.TRANSLATIONS.values():
        translated = table.get(label)
        if translated and translated not in variants:
            variants.append(translated)
    return tuple(variants)


def _relabel_info_line(line: str, language: str | None) -> str:
    """``line`` with its leading row name in ``language``; everything after the name is returned unchanged."""
    for label in INFO_LABELS:
        for variant in _info_name_variants(label):
            if line.startswith(variant):
                return i18n.translate(label, language) + line[len(variant):]
    return line


class GeneralSettingsPage:
    """`generalSettings` - placed by its parent with `addTab`."""

    page_id = "generalSettings"

    def build(self):
        """Build the widget tree and return the page root."""
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Expanding)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy5 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        # splitter: skipped 'sizePolicy2.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())' (['label'] live outside this page)
        # splitter: skipped 'sizePolicy3.setHeightForWidth(self.processInfoContainer.sizePolicy().hasHeightForWidth())' (['processInfoContainer'] live outside this page)
        # splitter: skipped 'sizePolicy5.setHeightForWidth(self.processContainer.sizePolicy().hasHeightForWidth())' (['processContainer'] live outside this page)
        # splitter: skipped 'sizePolicy3.setHeightForWidth(self.processSettingsContainer.sizePolicy().hasHeightForWidth())' (['processSettingsContainer'] live outside this page)
        # splitter: skipped 'sizePolicy2.setHeightForWidth(self.widget_34.sizePolicy().hasHeightForWidth())' (['widget_34'] live outside this page)
        # splitter: skipped 'sizePolicy5.setHeightForWidth(self.OutputFilesListWidget.sizePolicy().hasHeightForWidth())' (['OutputFilesListWidget'] live outside this page)
        # splitter: skipped 'sizePolicy3.setHeightForWidth(self.advancedSettings.sizePolicy().hasHeightForWidth())' (['advancedSettings'] live outside this page)
        # splitter: skipped 'sizePolicy3.setHeightForWidth(self.scrollArea_5.sizePolicy().hasHeightForWidth())' (['scrollArea_5'] live outside this page)
        # splitter: skipped 'sizePolicy2.setHeightForWidth(self.scrollAreaWidgetContents_5.sizePolicy().hasHeightForWidth())' (['scrollAreaWidgetContents_5'] live outside this page)
        # splitter: skipped 'sizePolicy2.setHeightForWidth(self.generalContainer.sizePolicy().hasHeightForWidth())' (['generalContainer'] live outside this page)
        # splitter: skipped 'sizePolicy2.setHeightForWidth(self.upscaleContainer.sizePolicy().hasHeightForWidth())' (['upscaleContainer'] live outside this page)
        # splitter: skipped 'sizePolicy2.setHeightForWidth(self.interpolateContainer_2.sizePolicy().hasHeightForWidth())' (['interpolateContainer_2'] live outside this page)
        # splitter: skipped 'sizePolicy5.setHeightForWidth(self.uninstallAppBtn.sizePolicy().hasHeightForWidth())' (['uninstallAppBtn'] live outside this page)
        # splitter: skipped 'sizePolicy5.setHeightForWidth(self.downloadRecommendedBtn.sizePolicy().hasHeightForWidth())' (['downloadRecommendedBtn'] live outside this page)
        # splitter: skipped 'sizePolicy5.setHeightForWidth(self.downloadNCNNBtn.sizePolicy().hasHeightForWidth())' (['downloadNCNNBtn'] live outside this page)
        # splitter: skipped 'sizePolicy5.setHeightForWidth(self.uninstallNCNNBtn.sizePolicy().hasHeightForWidth())' (['uninstallNCNNBtn'] live outside this page)
        # splitter: skipped 'sizePolicy5.setHeightForWidth(self.downloadTorchBtn.sizePolicy().hasHeightForWidth())' (['downloadTorchBtn'] live outside this page)
        # splitter: skipped 'sizePolicy5.setHeightForWidth(self.uninstallTorchBtn.sizePolicy().hasHeightForWidth())' (['uninstallTorchBtn'] live outside this page)
        # splitter: skipped 'sizePolicy5.setHeightForWidth(self.downloadTensorRTBtn.sizePolicy().hasHeightForWidth())' (['downloadTensorRTBtn'] live outside this page)
        # splitter: skipped 'sizePolicy5.setHeightForWidth(self.uninstallTensorRTBtn.sizePolicy().hasHeightForWidth())' (['uninstallTensorRTBtn'] live outside this page)
        # splitter: skipped 'sizePolicy5.setHeightForWidth(self.downloadDirectMLBtn.sizePolicy().hasHeightForWidth())' (['downloadDirectMLBtn'] live outside this page)
        # splitter: skipped 'sizePolicy5.setHeightForWidth(self.uninstallDirectMLBtn.sizePolicy().hasHeightForWidth())' (['uninstallDirectMLBtn'] live outside this page)
        self.generalSettings = SurfacePanel(background_token="surface.page")
        self.generalSettings.setObjectName(u"generalSettings")
        sizePolicy3.setHeightForWidth(self.generalSettings.sizePolicy().hasHeightForWidth())
        self.generalSettings.setSizePolicy(sizePolicy3)
        self.generalSettings.setMaximumSize(QSize(16777215, 16777215))
        self.verticalLayout_10 = QVBoxLayout(self.generalSettings)
        self.verticalLayout_10.setSpacing(0)
        self.verticalLayout_10.setObjectName(u"verticalLayout_10")
        self.verticalLayout_10.setContentsMargins(0, 0, 0, 0)
        self.widget_5 = QWidget(self.generalSettings)
        self.widget_5.setObjectName(u"widget_5")
        sizePolicy3.setHeightForWidth(self.widget_5.sizePolicy().hasHeightForWidth())
        self.widget_5.setSizePolicy(sizePolicy3)
        self.horizontalLayout_13 = QHBoxLayout(self.widget_5)
        self.horizontalLayout_13.setObjectName(u"horizontalLayout_13")
        self.horizontalLayout_13.setContentsMargins(0, 0, 0, 0)
        self.scrollArea_4 = SurfaceScrollArea(self.widget_5, surface_token=None, content_inset=1,
            scrollbar_width=STOCK_SCROLLBAR_WIDTH)
        self.scrollArea_4.setObjectName(u"scrollArea_4")
        sizePolicy2.setHeightForWidth(self.scrollArea_4.sizePolicy().hasHeightForWidth())
        self.scrollArea_4.setSizePolicy(sizePolicy2)
        self.scrollArea_4.setMaximumSize(QSize(16777215, 16777215))
        apply(self.scrollArea_4, BODY_PT)
        # No native scrollbar policy on a toolkit scroll area: it owns an overlay bar that appears on overflow, and a
        # native bar on top of it took another 14 px out of the viewport (measured: 456 px where the stock viewport was
        # 470, which narrowed every row of this page).
        self.scrollArea_4.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        self.scrollArea_4.setWidgetResizable(True)
        self.scrollAreaWidgetContents_4 = QWidget()
        self.scrollAreaWidgetContents_4.setObjectName(u"scrollAreaWidgetContents_4")
        self.scrollAreaWidgetContents_4.setGeometry(QRect(0, 0, 470, 1040))
        self.verticalLayout_32 = QVBoxLayout(self.scrollAreaWidgetContents_4)
        self.verticalLayout_32.setObjectName(u"verticalLayout_32")
        self.inputFileContainer = QWidget(self.scrollAreaWidgetContents_4)
        self.inputFileContainer.setObjectName(u"inputFileContainer")
        self.horizontalLayout_4 = QHBoxLayout(self.inputFileContainer)
        self.horizontalLayout_4.setSpacing(4)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        # The label is passed HERE, not only in `retranslate`: the toolkit's empty-label transition
        # (`_apply_text_geometry`) raises the minimum height to 32 and unbounds the maximum, which would drop the
        # Designer's 30 px row (the adapter restored those pins after every `setText`). A button born with its text
        # takes no such pass, and the row height is applied by `_apply_scale` at the end of `build` (and again on
        # every `UiScale.scale_changed`), so this is the size the layout sees.
        self.inputFileSelectButton = Button(parent=self.inputFileContainer,
            text=QCoreApplication.translate("MainWindow", u"Select Input File", None), variant="surface",
            content_align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            content_padding=(10, 5, 10, 5), corner_radius=10, size=(0, 0))
        self.inputFileSelectButton.setObjectName(u"inputFileSelectButton")
        apply(self.inputFileSelectButton, BODY_PT)
        # QPushButton's own policy (as wide as it needs, exactly as tall as its hint): the toolkit button is happy to
        # fill the cell, which turns a row button into a slab when the page is taller than its contents. `size=(0, 0)`
        # above drops the standalone 36x36 control size; this states the row's share. The Designer's `setStyleSheet("")`
        # states no rule - a toolkit button paints itself - so it is dropped rather than kept as dead text.
        self.inputFileSelectButton.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.horizontalLayout_4.addWidget(self.inputFileSelectButton)
        self.batchSelectButton = Button(parent=self.inputFileContainer,
            text=QCoreApplication.translate("MainWindow", u"Batch Select", None), variant="surface",
            content_align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            content_padding=(10, 5, 10, 5), corner_radius=10, size=(0, 0))
        self.batchSelectButton.setObjectName(u"batchSelectButton")
        apply(self.batchSelectButton, BODY_PT)
        self.batchSelectButton.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.horizontalLayout_4.addWidget(self.batchSelectButton)
        self.verticalLayout_32.addWidget(self.inputFileContainer)
        self.inputFileText = CustomLineEdit(self.scrollAreaWidgetContents_4)
        self.inputFileText.setObjectName(u"inputFileText")
        sizePolicy7 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy7.setHorizontalStretch(0)
        sizePolicy7.setVerticalStretch(0)
        sizePolicy7.setHeightForWidth(self.inputFileText.sizePolicy().hasHeightForWidth())
        self.inputFileText.setSizePolicy(sizePolicy7)
        self.inputFileText.setReadOnly(False)
        self.verticalLayout_32.addWidget(self.inputFileText)
        self.outputFileContainer = QWidget(self.scrollAreaWidgetContents_4)
        self.outputFileContainer.setObjectName(u"outputFileContainer")
        self.horizontalLayout_5 = QHBoxLayout(self.outputFileContainer)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.outputFileSelectButton = Button(parent=self.outputFileContainer,
            text=QCoreApplication.translate("MainWindow", u"Select Output Folder", None), variant="surface",
            content_align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            content_padding=(10, 5, 10, 5), corner_radius=10, size=(0, 0))
        self.outputFileSelectButton.setObjectName(u"outputFileSelectButton")
        self.outputFileSelectButton.setEnabled(False)
        apply(self.outputFileSelectButton, BODY_PT)
        self.outputFileSelectButton.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.horizontalLayout_5.addWidget(self.outputFileSelectButton)
        self.openOutputFolderButton = Button(parent=self.outputFileContainer,
            text=QCoreApplication.translate("MainWindow", u"Open Output Folder", None), variant="surface",
            content_align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            content_padding=(10, 5, 10, 5), corner_radius=10, size=(0, 0))
        self.openOutputFolderButton.setObjectName(u"openOutputFolderButton")
        self.openOutputFolderButton.setEnabled(False)
        apply(self.openOutputFolderButton, BODY_PT)
        self.openOutputFolderButton.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.horizontalLayout_5.addWidget(self.openOutputFolderButton)
        self.verticalLayout_32.addWidget(self.outputFileContainer)
        self.outputFileText = CustomLineEdit(self.scrollAreaWidgetContents_4)
        self.outputFileText.setObjectName(u"outputFileText")
        self.outputFileText.setEnabled(False)
        # `Expanding`, not the Designer's `MinimumExpanding`: the latter has no ShrinkFlag, so `qSmartMinSize` takes
        # the field's `sizeHint` - its TEXT width - as the row's MINIMUM, and a long output path then sets the whole
        # General tab's minimum width (measured 2026-11-24: a 464 px path made the content 486 px inside a 470 px
        # viewport, clipping every field at the column's edge after a page switch). `Expanding` keeps the field
        # filling its row while letting it shrink, which is what the input field next door already does.
        sizePolicy8 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy8.setHorizontalStretch(0)
        sizePolicy8.setVerticalStretch(0)
        sizePolicy8.setHeightForWidth(self.outputFileText.sizePolicy().hasHeightForWidth())
        self.outputFileText.setSizePolicy(sizePolicy8)
        self.verticalLayout_32.addWidget(self.outputFileText)
        # A separator is one painted pixel of the Designer's 3 px sunken `QFrame`. The toolkit's `Divider` is a
        # button-region spec, not a widget, so the line is a token-painted panel of `SEPARATOR_HEIGHT` (the deleted
        # rule painted the frame white, which is `dialog.border`).
        self.line = SurfacePanel(self.scrollAreaWidgetContents_4, background_token="dialog.border")
        self.line.setObjectName(u"line")
        self.line.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.verticalLayout_32.addWidget(self.line)
        self.widget_3 = QWidget(self.scrollAreaWidgetContents_4)
        self.widget_3.setObjectName(u"widget_3")
        self.horizontalLayout_11 = QHBoxLayout(self.widget_3)
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.horizontalLayout_11.setContentsMargins(0, 0, 0, 0)
        self.label_12 = Text(parent=self.widget_3)
        self.label_12.setObjectName(u"label_12")
        apply(self.label_12, BODY_PT)
        self.horizontalLayout_11.addWidget(self.label_12)
        self.backendComboBox = ComboBox(self.widget_3)
        self.backendComboBox.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        cap_combo_to_row(self.backendComboBox)
        self.backendComboBox.setObjectName(u"backendComboBox")
        sizePolicy8.setHeightForWidth(self.backendComboBox.sizePolicy().hasHeightForWidth())
        self.backendComboBox.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.backendComboBox.setMinimumSize(QSize(0, 0))
        self.horizontalLayout_11.addWidget(self.backendComboBox)
        self.verticalLayout_32.addWidget(self.widget_3)
        self.widget_35 = QWidget(self.scrollAreaWidgetContents_4)
        self.widget_35.setObjectName(u"widget_35")
        self.horizontalLayout_7 = QHBoxLayout(self.widget_35)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.horizontalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.widget_37 = QWidget(self.widget_35)
        self.widget_37.setObjectName(u"widget_37")
        self.horizontalLayout_56 = QHBoxLayout(self.widget_37)
        self.horizontalLayout_56.setObjectName(u"horizontalLayout_56")
        self.horizontalLayout_56.setContentsMargins(0, 0, 0, 0)
        self.label_76 = Text(parent=self.widget_37)
        self.label_76.setObjectName(u"label_76")
        apply(self.label_76, BODY_PT)
        self.horizontalLayout_56.addWidget(self.label_76)
        self.horizontalSpacer_21 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.horizontalLayout_56.addItem(self.horizontalSpacer_21)
        # A two-state box: the Designer's `setTristate(False)` below states exactly what the toolkit box is.
        self.interpolateCheckBox = _row_checkbox(self.widget_37)
        self.interpolateCheckBox.setObjectName(u"interpolateCheckBox")
        self.interpolateCheckBox.setTristate(False)
        self.horizontalLayout_56.addWidget(self.interpolateCheckBox)
        self.horizontalLayout_7.addWidget(self.widget_37)
        self.verticalLayout_32.addWidget(self.widget_35)
        self.interpolationContainer = QWidget(self.scrollAreaWidgetContents_4)
        self.interpolationContainer.setObjectName(u"interpolationContainer")
        self.interpolationContainer.setEnabled(True)
        self.verticalLayout_38 = QVBoxLayout(self.interpolationContainer)
        self.verticalLayout_38.setObjectName(u"verticalLayout_38")
        self.widget_28 = QWidget(self.interpolationContainer)
        self.widget_28.setObjectName(u"widget_28")
        self.horizontalLayout_45 = QHBoxLayout(self.widget_28)
        self.horizontalLayout_45.setObjectName(u"horizontalLayout_45")
        self.horizontalLayout_45.setContentsMargins(0, 0, 0, 0)
        self.label_14 = Text(parent=self.widget_28)
        self.label_14.setObjectName(u"label_14")
        apply(self.label_14, BODY_PT)
        self.horizontalLayout_45.addWidget(self.label_14)
        self.interpolateModelComboBox = ComboBox(self.widget_28)
        self.interpolateModelComboBox.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        cap_combo_to_row(self.interpolateModelComboBox)
        self.interpolateModelComboBox.setObjectName(u"interpolateModelComboBox")
        sizePolicy8.setHeightForWidth(self.interpolateModelComboBox.sizePolicy().hasHeightForWidth())
        self.interpolateModelComboBox.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.interpolateModelComboBox.setMinimumSize(QSize(0, 0))
        self.interpolateModelComboBox.setSizeIncrement(QSize(230, 0))
        self.horizontalLayout_45.addWidget(self.interpolateModelComboBox)
        self.verticalLayout_38.addWidget(self.widget_28)
        self.interpolationContainer2 = QWidget(self.interpolationContainer)
        self.interpolationContainer2.setObjectName(u"interpolationContainer2")
        self.horizontalLayout_55 = QHBoxLayout(self.interpolationContainer2)
        self.horizontalLayout_55.setObjectName(u"horizontalLayout_55")
        self.horizontalLayout_55.setContentsMargins(0, 0, 0, 0)
        self.label_11 = Text(parent=self.interpolationContainer2)
        self.label_11.setObjectName(u"label_11")
        apply(self.label_11, BODY_PT)
        self.horizontalLayout_55.addWidget(self.label_11)
        # `single_step=0.1` is the adapter's "Qt clamps and rounds to `decimals`" semantic, stated as the step the field
        # actually shows: the toolkit's `DoubleSpinBox.setValue` SNAPS to `single_step` (its default 1.0 turns a stored
        # `3.5` into `3.0`). The Designer then pins this box to `decimals(0)`, so the field shows whole multipliers - and
        # the toolkit raises its step to that display quantum, otherwise the ▲/▼ arrows move the value by a tenth the
        # text cannot show and read as dead (owner, 2026-11-22: "the arrows do nothing").
        # `show_step_buttons=True`, like every Designer spin box: the toolkit's DoubleSpinBox defaults the step gutter
        # OFF, so this row rendered as a bare box while its neighbours (scene-change threshold, start/end time) had arrows
        # (owner, 2026-11-22: "the arrows disappeared in the rows where they should be").
        self.interpolationMultiplierSpinBox = DoubleSpinBox(self.interpolationContainer2, single_step=0.1, decimals=1,
            show_step_buttons=True, alignment=SPIN_ALIGNMENT)
        self.interpolationMultiplierSpinBox.setObjectName(u"interpolationMultiplierSpinBox")
        sizePolicy5.setHeightForWidth(self.interpolationMultiplierSpinBox.sizePolicy().hasHeightForWidth())
        self.interpolationMultiplierSpinBox.setSizePolicy(sizePolicy5)
        # No `setButtonSymbols`: the toolkit's gutter is not symbol-mode driven (it draws ▲/▼ whenever it is on), and
        # `UpDownArrows` is the only mode the Designer ever asks for, so there is nothing left to state.
        self.interpolationMultiplierSpinBox.setDecimals(0)
        # Qt's `setMinimum` is the one range spelling the toolkit field does not answer (`DoubleSpinBox` carries
        # `setRange` only): the second bound is the layer's Qt default maximum (99.99), not the toolkit field's 100.0.
        self.interpolationMultiplierSpinBox.setRange(2.000000000000000, 99.99)
        self.interpolationMultiplierSpinBox.setValue(2.000000000000000)
        self.horizontalLayout_55.addWidget(self.interpolationMultiplierSpinBox)
        self.verticalLayout_38.addWidget(self.interpolationContainer2)
        self.sloMoModeContainer = QWidget(self.interpolationContainer)
        self.sloMoModeContainer.setObjectName(u"sloMoModeContainer")
        self.horizontalLayout_31 = QHBoxLayout(self.sloMoModeContainer)
        self.horizontalLayout_31.setObjectName(u"horizontalLayout_31")
        self.horizontalLayout_31.setContentsMargins(0, 0, 0, 0)
        self.label_36 = Text(parent=self.sloMoModeContainer)
        self.label_36.setObjectName(u"label_36")
        apply(self.label_36, BODY_PT)
        self.horizontalLayout_31.addWidget(self.label_36)
        self.horizontalSpacer_16 = QSpacerItem(226, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.horizontalLayout_31.addItem(self.horizontalSpacer_16)
        self.sloMoModeCheckBox = _row_checkbox(self.sloMoModeContainer)
        self.sloMoModeCheckBox.setObjectName(u"sloMoModeCheckBox")
        self.sloMoModeCheckBox.setToolTip(i18n.tooltip(
            'Increase video length instead of framerate when interpolating.',
            'Audio and Subtitles will not be transfered to output video.'
        ))
        self.horizontalLayout_31.addWidget(self.sloMoModeCheckBox)
        self.verticalLayout_38.addWidget(self.sloMoModeContainer)
        self.verticalLayout_32.addWidget(self.interpolationContainer)
        self.decompressModelContainer_2 = QWidget(self.scrollAreaWidgetContents_4)
        self.decompressModelContainer_2.setObjectName(u"decompressModelContainer_2")
        self.horizontalLayout_65 = QHBoxLayout(self.decompressModelContainer_2)
        self.horizontalLayout_65.setObjectName(u"horizontalLayout_65")
        self.horizontalLayout_65.setContentsMargins(0, 0, 0, 0)
        self.label_86 = Text(parent=self.decompressModelContainer_2)
        self.label_86.setObjectName(u"label_86")
        apply(self.label_86, BODY_PT)
        self.horizontalLayout_65.addWidget(self.label_86)
        self.horizontalSpacer_33 = QSpacerItem(324, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.horizontalLayout_65.addItem(self.horizontalSpacer_33)
        self.decompressCheckBox = _row_checkbox(self.decompressModelContainer_2)
        self.decompressCheckBox.setObjectName(u"decompressCheckBox")
        self.horizontalLayout_65.addWidget(self.decompressCheckBox)
        self.verticalLayout_32.addWidget(self.decompressModelContainer_2)
        self.decompressContainer = QWidget(self.scrollAreaWidgetContents_4)
        self.decompressContainer.setObjectName(u"decompressContainer")
        self.horizontalLayout_66 = QHBoxLayout(self.decompressContainer)
        self.horizontalLayout_66.setObjectName(u"horizontalLayout_66")
        self.label_87 = Text(parent=self.decompressContainer)
        self.label_87.setObjectName(u"label_87")
        apply(self.label_87, BODY_PT)
        self.horizontalLayout_66.addWidget(self.label_87)
        self.decompressModelComboBox = ComboBox(self.decompressContainer)
        self.decompressModelComboBox.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        cap_combo_to_row(self.decompressModelComboBox)
        self.decompressModelComboBox.setObjectName(u"decompressModelComboBox")
        sizePolicy8.setHeightForWidth(self.decompressModelComboBox.sizePolicy().hasHeightForWidth())
        self.decompressModelComboBox.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.decompressModelComboBox.setMinimumSize(QSize(0, 0))
        self.horizontalLayout_66.addWidget(self.decompressModelComboBox)
        self.verticalLayout_32.addWidget(self.decompressContainer)
        self.denoiseCheckBoxContainer = QWidget(self.scrollAreaWidgetContents_4)
        self.denoiseCheckBoxContainer.setObjectName(u"denoiseCheckBoxContainer")
        self.horizontalLayout_62 = QHBoxLayout(self.denoiseCheckBoxContainer)
        self.horizontalLayout_62.setObjectName(u"horizontalLayout_62")
        self.horizontalLayout_62.setContentsMargins(0, 0, 0, 0)
        self.label_82 = Text(parent=self.denoiseCheckBoxContainer)
        self.label_82.setObjectName(u"label_82")
        apply(self.label_82, BODY_PT)
        self.horizontalLayout_62.addWidget(self.label_82)
        self.horizontalSpacer_32 = QSpacerItem(337, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.horizontalLayout_62.addItem(self.horizontalSpacer_32)
        self.denoiseCheckBox = _row_checkbox(self.denoiseCheckBoxContainer)
        self.denoiseCheckBox.setObjectName(u"denoiseCheckBox")
        self.horizontalLayout_62.addWidget(self.denoiseCheckBox)
        self.verticalLayout_32.addWidget(self.denoiseCheckBoxContainer)
        self.denoiseContainer = QWidget(self.scrollAreaWidgetContents_4)
        self.denoiseContainer.setObjectName(u"denoiseContainer")
        self.horizontalLayout_64 = QHBoxLayout(self.denoiseContainer)
        self.horizontalLayout_64.setObjectName(u"horizontalLayout_64")
        self.widget_44 = QWidget(self.denoiseContainer)
        self.widget_44.setObjectName(u"widget_44")
        self.horizontalLayout_63 = QHBoxLayout(self.widget_44)
        self.horizontalLayout_63.setObjectName(u"horizontalLayout_63")
        self.horizontalLayout_63.setContentsMargins(0, 0, 0, 0)
        self.label_83 = Text(parent=self.widget_44)
        self.label_83.setObjectName(u"label_83")
        apply(self.label_83, BODY_PT)
        self.horizontalLayout_63.addWidget(self.label_83)
        self.denoiseModelComboBox = ComboBox(self.widget_44)
        self.denoiseModelComboBox.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        cap_combo_to_row(self.denoiseModelComboBox)
        self.denoiseModelComboBox.setObjectName(u"denoiseModelComboBox")
        sizePolicy8.setHeightForWidth(self.denoiseModelComboBox.sizePolicy().hasHeightForWidth())
        self.denoiseModelComboBox.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.denoiseModelComboBox.setMinimumSize(QSize(0, 0))
        self.horizontalLayout_63.addWidget(self.denoiseModelComboBox)
        self.horizontalLayout_64.addWidget(self.widget_44)
        self.verticalLayout_32.addWidget(self.denoiseContainer)
        self.deblurCheckBoxContainer = QWidget(self.scrollAreaWidgetContents_4)
        self.deblurCheckBoxContainer.setObjectName(u"deblurCheckBoxContainer")
        self.horizontalLayout_59 = QHBoxLayout(self.deblurCheckBoxContainer)
        self.horizontalLayout_59.setObjectName(u"horizontalLayout_59")
        self.horizontalLayout_59.setContentsMargins(0, 0, 0, 0)
        self.label_78 = Text(parent=self.deblurCheckBoxContainer)
        self.label_78.setObjectName(u"label_78")
        apply(self.label_78, BODY_PT)
        self.horizontalLayout_59.addWidget(self.label_78)
        self.horizontalSpacer_28 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.horizontalLayout_59.addItem(self.horizontalSpacer_28)
        self.deblurCheckBox = _row_checkbox(self.deblurCheckBoxContainer)
        self.deblurCheckBox.setObjectName(u"deblurCheckBox")
        self.horizontalLayout_59.addWidget(self.deblurCheckBox)
        self.verticalLayout_32.addWidget(self.deblurCheckBoxContainer)
        self.deblurContainer = QWidget(self.scrollAreaWidgetContents_4)
        self.deblurContainer.setObjectName(u"deblurContainer")
        self.verticalLayout_41 = QVBoxLayout(self.deblurContainer)
        self.verticalLayout_41.setSpacing(0)
        self.verticalLayout_41.setObjectName(u"verticalLayout_41")
        self.verticalLayout_41.setContentsMargins(0, 0, 0, 0)
        self.widget_43 = QWidget(self.deblurContainer)
        self.widget_43.setObjectName(u"widget_43")
        self.horizontalLayout_60 = QHBoxLayout(self.widget_43)
        self.horizontalLayout_60.setObjectName(u"horizontalLayout_60")
        self.label_79 = Text(parent=self.widget_43)
        self.label_79.setObjectName(u"label_79")
        apply(self.label_79, BODY_PT)
        self.horizontalLayout_60.addWidget(self.label_79)
        self.deblurModelComboBox = ComboBox(self.widget_43)
        self.deblurModelComboBox.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        cap_combo_to_row(self.deblurModelComboBox)
        self.deblurModelComboBox.setObjectName(u"deblurModelComboBox")
        sizePolicy8.setHeightForWidth(self.deblurModelComboBox.sizePolicy().hasHeightForWidth())
        self.deblurModelComboBox.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.deblurModelComboBox.setMinimumSize(QSize(0, 0))
        self.horizontalLayout_60.addWidget(self.deblurModelComboBox)
        self.verticalLayout_41.addWidget(self.widget_43)
        self.verticalLayout_32.addWidget(self.deblurContainer)
        self.widget_38 = QWidget(self.scrollAreaWidgetContents_4)
        self.widget_38.setObjectName(u"widget_38")
        self.horizontalLayout_58 = QHBoxLayout(self.widget_38)
        self.horizontalLayout_58.setObjectName(u"horizontalLayout_58")
        self.horizontalLayout_58.setContentsMargins(0, 0, 0, 0)
        self.widget_39 = QWidget(self.widget_38)
        self.widget_39.setObjectName(u"widget_39")
        self.horizontalLayout_57 = QHBoxLayout(self.widget_39)
        self.horizontalLayout_57.setObjectName(u"horizontalLayout_57")
        self.horizontalLayout_57.setContentsMargins(0, 0, 0, 0)
        self.label_77 = Text(parent=self.widget_39)
        self.label_77.setObjectName(u"label_77")
        apply(self.label_77, BODY_PT)
        self.horizontalLayout_57.addWidget(self.label_77)
        self.horizontalSpacer_27 = QSpacerItem(40, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.horizontalLayout_57.addItem(self.horizontalSpacer_27)
        self.upscaleCheckBox = _row_checkbox(self.widget_39)
        self.upscaleCheckBox.setObjectName(u"upscaleCheckBox")
        self.horizontalLayout_57.addWidget(self.upscaleCheckBox)
        self.horizontalLayout_58.addWidget(self.widget_39)
        self.verticalLayout_32.addWidget(self.widget_38, 0, Qt.AlignmentFlag.AlignTop)
        self.generalUpscaleContainer = QWidget(self.scrollAreaWidgetContents_4)
        self.generalUpscaleContainer.setObjectName(u"generalUpscaleContainer")
        self.verticalLayout_39 = QVBoxLayout(self.generalUpscaleContainer)
        self.verticalLayout_39.setObjectName(u"verticalLayout_39")
        self.widget = QWidget(self.generalUpscaleContainer)
        self.widget.setObjectName(u"widget")
        self.horizontalLayout = QHBoxLayout(self.widget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.label_7 = Text(parent=self.widget)
        self.label_7.setObjectName(u"label_7")
        apply(self.label_7, BODY_PT)
        self.horizontalLayout.addWidget(self.label_7)
        self.upscaleModelComboBox = ComboBox(self.widget)
        self.upscaleModelComboBox.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        cap_combo_to_row(self.upscaleModelComboBox)
        self.upscaleModelComboBox.setObjectName(u"upscaleModelComboBox")
        sizePolicy8.setHeightForWidth(self.upscaleModelComboBox.sizePolicy().hasHeightForWidth())
        self.upscaleModelComboBox.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.upscaleModelComboBox.setMinimumSize(QSize(0, 0))
        self.horizontalLayout.addWidget(self.upscaleModelComboBox)
        self.verticalLayout_39.addWidget(self.widget)
        self.widget_42 = QWidget(self.generalUpscaleContainer)
        self.widget_42.setObjectName(u"widget_42")
        self.horizontalLayout_50 = QHBoxLayout(self.widget_42)
        self.horizontalLayout_50.setObjectName(u"horizontalLayout_50")
        self.horizontalLayout_50.setContentsMargins(0, 0, 0, 0)
        self.label_72 = Text(parent=self.widget_42)
        self.label_72.setObjectName(u"label_72")
        apply(self.label_72, BODY_PT)
        self.horizontalLayout_50.addWidget(self.label_72)
        # Same `single_step`/`decimals` statement as the interpolation multiplier: the toolkit field snaps `setValue`
        # to its step, the Designer pins this box to whole scale factors (`decimals(0)`), and the toolkit lifts the step
        # to that quantum so one ▲ press is one visible scale factor.
        self.upscaleScaleSpinBox = DoubleSpinBox(self.widget_42, single_step=0.1, decimals=1, show_step_buttons=True,
            alignment=SPIN_ALIGNMENT)
        self.upscaleScaleSpinBox.setObjectName(u"upscaleScaleSpinBox")
        self.upscaleScaleSpinBox.setToolTip(i18n.tooltip(
            'Upscale Scale:',
            '- Custom output scale no matter the model. Will run AI on video frame once, and scale the image to match the multiplier. Useful for 4x models.'
        ))
        sizePolicy5.setHeightForWidth(self.upscaleScaleSpinBox.sizePolicy().hasHeightForWidth())
        self.upscaleScaleSpinBox.setSizePolicy(sizePolicy5)
        # No `setButtonSymbols`: the gutter is not symbol-mode driven (see the interpolation multiplier above).
        self.upscaleScaleSpinBox.setDecimals(0)
        # Qt's two range statements are the one call the toolkit field answers (`setRange`).
        self.upscaleScaleSpinBox.setRange(1.000000000000000, 4.000000000000000)
        self.upscaleScaleSpinBox.setValue(2.000000000000000)
        self.horizontalLayout_50.addWidget(self.upscaleScaleSpinBox)
        self.verticalLayout_39.addWidget(self.widget_42)
        self.verticalLayout_32.addWidget(self.generalUpscaleContainer, 0, Qt.AlignmentFlag.AlignTop)
        self.qualityPresetContainer = QWidget(self.scrollAreaWidgetContents_4)
        self.qualityPresetContainer.setObjectName(u"qualityPresetContainer")
        self.verticalLayout_qualityPreset = QVBoxLayout(self.qualityPresetContainer)
        self.verticalLayout_qualityPreset.setSpacing(2)
        self.verticalLayout_qualityPreset.setObjectName(u"verticalLayout_qualityPreset")
        self.qualityPresetRow = QWidget(self.qualityPresetContainer)
        self.qualityPresetRow.setObjectName(u"qualityPresetRow")
        self.horizontalLayout_qualityPresetRow = QHBoxLayout(self.qualityPresetRow)
        self.horizontalLayout_qualityPresetRow.setObjectName(u"horizontalLayout_qualityPresetRow")
        self.horizontalLayout_qualityPresetRow.setContentsMargins(0, 0, 0, 0)
        self.qualityPresetLabel = Text(parent=self.qualityPresetRow)
        self.qualityPresetLabel.setObjectName(u"qualityPresetLabel")
        apply(self.qualityPresetLabel, BODY_PT)
        self.horizontalLayout_qualityPresetRow.addWidget(self.qualityPresetLabel)
        self.horizontalSpacer_qualityPreset = QSpacerItem(237, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.horizontalLayout_qualityPresetRow.addItem(self.horizontalSpacer_qualityPreset)
        self.qualityPresetComboBox = ComboBox(self.qualityPresetRow)
        self.qualityPresetComboBox.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        cap_combo_to_row(self.qualityPresetComboBox)
        self.qualityPresetComboBox.addItem("")
        self.qualityPresetComboBox.addItem("")
        self.qualityPresetComboBox.setObjectName(u"qualityPresetComboBox")
        self.qualityPresetComboBox.setMinimumSize(QSize(0, 0))
        self.horizontalLayout_qualityPresetRow.addWidget(self.qualityPresetComboBox)
        self.verticalLayout_qualityPreset.addWidget(self.qualityPresetRow)
        self.frameReuse = _row_checkbox(self.qualityPresetContainer)
        self.frameReuse.setObjectName(u"frameReuse")
        self.frameReuse.setChecked(True)
        self.verticalLayout_qualityPreset.addWidget(self.frameReuse)
        self.celReuse = _row_checkbox(self.qualityPresetContainer)
        self.celReuse.setObjectName(u"celReuse")
        self.celReuse.setChecked(False)
        self.verticalLayout_qualityPreset.addWidget(self.celReuse)
        self.verticalLayout_32.addWidget(self.qualityPresetContainer)
        self.widget_25 = QWidget(self.scrollAreaWidgetContents_4)
        self.widget_25.setObjectName(u"widget_25")
        self.horizontalLayout_40 = QHBoxLayout(self.widget_25)
        self.horizontalLayout_40.setSpacing(0)
        self.horizontalLayout_40.setObjectName(u"horizontalLayout_40")
        self.horizontalLayout_40.setContentsMargins(0, 0, 0, 0)
        self.addToRenderQueueButton = Button(parent=self.widget_25,
            text=QCoreApplication.translate("MainWindow", u" Add to Render Queue", None), variant="surface",
            content_align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            content_padding=(10, 5, 10, 5), corner_radius=10, size=(200, 35))
        self.addToRenderQueueButton.setObjectName(u"addToRenderQueueButton")
        sizePolicy5.setHeightForWidth(self.addToRenderQueueButton.sizePolicy().hasHeightForWidth())
        self.addToRenderQueueButton.setSizePolicy(sizePolicy5)
        # The Designer pinned 200x35; the toolkit's hint at RVE's 12 px control font is 187x33 (text + 10/10 padding +
        # 5/5), so the row handed it 2-5 px less. The design size is the constructor's `size=(200, 35)` now: the toolkit
        # re-resolves it on `scale_changed` (a page `setMaximumSize(200, 35)` would have capped the scaled button at the
        # 1.0 number and clipped it at 1.5).

        # No stylesheet: a toolkit button paints itself; the deleted rule only repeated its own alignment.
        icon2 = QIcon()
        icon2.addFile(u":/icons/icons/plus-circle.svg", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.addToRenderQueueButton.setIcon(icon2)
        self.horizontalLayout_40.addWidget(self.addToRenderQueueButton)
        self.verticalLayout_32.addWidget(self.widget_25)
        self.widget_26 = QWidget(self.scrollAreaWidgetContents_4)
        self.widget_26.setObjectName(u"widget_26")
        sizePolicy5.setHeightForWidth(self.widget_26.sizePolicy().hasHeightForWidth())
        self.widget_26.setSizePolicy(sizePolicy5)
        # No vertical ceiling: the Designer's 200 px cap sat ABOVE the panel's 190 px floor, so the panel could never
        # grow into its own content - at 2.0 the video-info body needs 228 px and the last field line ended up under
        # the scroll area's edge fade (owner, 2026-11-24: "the fade is aggressive, the text fits"). The height is the
        # content's now (`RichTextView(fit_content=True)`), and the panel's own 190 px floor still keeps an empty one
        # from collapsing.
        self.horizontalLayout_22 = QHBoxLayout(self.widget_26)
        self.horizontalLayout_22.setObjectName(u"horizontalLayout_22")
        # A read-only rich-text panel: the toolkit's `RichTextView` takes the host's `setText`/`setFontPointSize` calls
        # and scrolls its own content (the plate is `surface.list`, the colour the deleted page rule carried).
        # `fit_content`: this pane's body is a record of fields, so it asks for the height that shows all of them
        # instead of sitting on the 190 px floor below. Measured 2026-11-24: at 2.0 the body needed 208 px, the floor
        # kept 190, and the last field line was hidden under the scroll area's edge fade (owner: "the fade is
        # aggressive, the text fits"). The floor still keeps an EMPTY panel from collapsing.
        self.inputVideoInfoTextEdit = RichTextView(self.widget_26, padding=(10, 5, 10, 5), fit_content=True)
        # The stock editor's text is 12 PIXELS (the Designer's sizing); `setFontPointSize` would convert through the
        # screen DPI and render 16 px, which grew every line of this panel.
        self.inputVideoInfoTextEdit.setTextPixelSize(RVE_CONTROL_FONT_PX)
        self.inputVideoInfoTextEdit.setObjectName(u"inputVideoInfoTextEdit")
        self.inputVideoInfoTextEdit.setEnabled(False)
        sizePolicy3.setHeightForWidth(self.inputVideoInfoTextEdit.sizePolicy().hasHeightForWidth())
        self.inputVideoInfoTextEdit.setSizePolicy(sizePolicy3)
        # The Designer had a 190 px MINIMUM under a 180 px MAXIMUM here: Qt clamps to the maximum, the content needed
        # 2 px more than it had, and the toolkit drew its 'there is more below' fade over text that already fitted
        # (owner, 2026-11-24: 'the text fits but the fade is there'). The contradictory ceiling is gone - the default
        # maximum applies and the floor keeps the Designer's 190 so an empty panel does not collapse. The fade itself
        # now also requires a ramp's worth of remaining scroll, so a 2 px rounding can never advertise more text.
        self.inputVideoInfoTextEdit.setMinimumSize(QSize(0, 190))
        self.inputVideoInfoTextEdit.setAccessibleDescription(u"")
        # No stylesheet: the pane paints its own text and plate; the deleted rule only forced white on a disabled
        # editor, which is the colour this pane's token already gives it.
        self.inputVideoInfoTextEdit.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.inputVideoInfoTextEdit.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.inputVideoInfoTextEdit.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        self.inputVideoInfoTextEdit.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.inputVideoInfoTextEdit.setReadOnly(True)
        self.horizontalLayout_22.addWidget(self.inputVideoInfoTextEdit)
        self.verticalLayout_32.addWidget(self.widget_26)
        # The list is a column of fixed rows; the layout's leftover height is NOT theirs to share. Without a spacer the
        # box layout hands the slack to the rows (each `QWidget` defaults to a Preferred vertical policy, which grows),
        # so a row was 47 px when the page had room and 30 px the moment a settings section took that room: every row
        # in the column changed height together and the tab looked like it was breathing (measured 2026-11-22 on a
        # 1920x1080 window: `inputFileContainer` 47 -> 30 px on one checkbox click, layout hint 779 -> 910). The spacer
        # takes the slack instead, so a row is always its own height and opening a section only inserts that section.
        self._settingsFiller = QSpacerItem(0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.verticalLayout_32.addItem(self._settingsFiller)
        self.scrollArea_4.setWidget(self.scrollAreaWidgetContents_4)
        self.horizontalLayout_13.addWidget(self.scrollArea_4)
        self.verticalLayout_10.addWidget(self.widget_5)
        self.interpolateModelComboBox.setCurrentIndex(-1)
        # The glue's typed edge over this tab; every field is required, so a widget that stops being built fails
        # here (TypeError) instead of surfacing as an AttributeError or a silent `getattr(..., None)` at click time.
        self._handles = GeneralSettingsHandles(
            inputFileSelectButton=self.inputFileSelectButton,
            batchSelectButton=self.batchSelectButton,
            inputFileText=self.inputFileText,
            outputFileText=self.outputFileText,
            outputFileSelectButton=self.outputFileSelectButton,
            openOutputFolderButton=self.openOutputFolderButton,
            backendComboBox=self.backendComboBox,
            interpolateCheckBox=self.interpolateCheckBox,
            interpolateModelComboBox=self.interpolateModelComboBox,
            interpolationMultiplierSpinBox=self.interpolationMultiplierSpinBox,
            sloMoModeCheckBox=self.sloMoModeCheckBox,
            decompressCheckBox=self.decompressCheckBox,
            decompressModelComboBox=self.decompressModelComboBox,
            denoiseCheckBox=self.denoiseCheckBox,
            denoiseModelComboBox=self.denoiseModelComboBox,
            deblurCheckBox=self.deblurCheckBox,
            deblurModelComboBox=self.deblurModelComboBox,
            upscaleCheckBox=self.upscaleCheckBox,
            upscaleModelComboBox=self.upscaleModelComboBox,
            upscaleScaleSpinBox=self.upscaleScaleSpinBox,
            qualityPresetComboBox=self.qualityPresetComboBox,
            frameReuse=self.frameReuse,
            celReuse=self.celReuse,
            addToRenderQueueButton=self.addToRenderQueueButton,
            generalSettings=self.generalSettings,
            inputVideoInfoTextEdit=self.inputVideoInfoTextEdit,
            widget_5=self.widget_5,
            scrollArea_4=self.scrollArea_4,
            scrollAreaWidgetContents_4=self.scrollAreaWidgetContents_4,
            outputFileContainer=self.outputFileContainer,
            interpolationContainer=self.interpolationContainer,
            decompressContainer=self.decompressContainer,
            denoiseContainer=self.denoiseContainer,
            deblurCheckBoxContainer=self.deblurCheckBoxContainer,
            deblurContainer=self.deblurContainer,
            generalUpscaleContainer=self.generalUpscaleContainer,
            qualityPresetContainer=self.qualityPresetContainer,
        )
        # The page pass needs a QObject to walk; the relay is that object and forwards
        # `retranslate(lang)` to this builder (`src/ui/i18n.py`).
        i18n.bind_page(self, self.generalSettings)
        self.retranslate()
        # Every design number in this page is stated once in `_apply_scale`: the first run is construction, and the
        # follower re-runs it on a live factor change. After `retranslate` so the labels are final.
        self._scale = ScaleFollower(self.generalSettings, self._apply_scale)
        return self.generalSettings

    def _render_info_pane(self, language: str | None = None) -> None:
        """Paint the pane's nine-row placeholder at the factor and language in force."""
        self.inputVideoInfoTextEdit.setHtml(_info_html(UiScale.get_instance().factor(), language))
        # What the placeholder reads as, so a switch can tell it apart from a loaded video's rows (below).
        self._info_placeholder_text = self.inputVideoInfoTextEdit.text()

    def _retranslate_info_pane(self, language: str | None = None) -> None:
        """Repaint the pane's nine row names in ``language``, keeping a loaded video's values.

        The pane has two writers: this page paints the nine-row placeholder, and `MainWindow.updateVideoGUIText`
        replaces it with the loaded file's values - the same nine names, each followed by its values. A switch is
        therefore one of two rewrites. The placeholder is rebuilt from the catalogue (one document, same spans, same
        factor). A loaded body is relabelled line by line, because re-rendering the placeholder over it would erase
        the video's numbers - the regression this method exists to avoid.
        """
        pane = self.inputVideoInfoTextEdit
        current = pane.text()
        if not current or current == getattr(self, "_info_placeholder_text", None):
            self._render_info_pane(language)
            return
        relabelled = "\n".join(_relabel_info_line(line, language) for line in current.split("\n"))
        if relabelled != current:
            pane.setText(relabelled)

    def _apply_scale(self) -> None:
        """Resolve this page's design numbers at the factor in force now.

        The rows, the value fields, the separator and the design gaps are the Designer's numbers in DESIGN px; each is
        resolved through `scaled_px()` HERE (at construction and on every `UiScale.scale_changed`), never at import,
        where it would freeze the factor the page happened to be built with. 1.0 resolves to the same numbers the page
        stated before this wave, so the look does not move when the feature is off.
        """
        row = scaled_px(ROW_HEIGHT_PX)
        field_width = scaled_px(FIELD_WIDTH_PX)
        gap = scaled_px(PAGE_GAP_PX)
        inset = scaled_px(SECTION_INSET_PX)
        section_gap = scaled_px(SECTION_GAP_PX)

        for button in (self.inputFileSelectButton, self.batchSelectButton,
                       self.outputFileSelectButton, self.openOutputFolderButton):
            button.setMinimumSize(QSize(0, row))
            button.setMaximumSize(QSize(16777215, row))
        for edit, max_width in ((self.inputFileText, INPUT_PATH_MAX_WIDTH_PX),
                                (self.outputFileText, OUTPUT_PATH_MAX_WIDTH_PX)):
            edit.setMinimumSize(QSize(0, row))
            edit.setMaximumSize(QSize(scaled_px(max_width), row))
        for combo in (self.backendComboBox, self.interpolateModelComboBox, self.decompressModelComboBox,
                      self.denoiseModelComboBox, self.deblurModelComboBox, self.upscaleModelComboBox):
            combo.setMaximumHeight(row)   # width follows the content, not the Designer's column
        for spin in (self.interpolationMultiplierSpinBox, self.upscaleScaleSpinBox):
            spin.setMinimumSize(QSize(field_width, row))
            spin.setMaximumSize(QSize(field_width, row))

        self.line.setFixedHeight(scaled_px(SEPARATOR_HEIGHT))
        icon = scaled_px(ACTION_ICON_PX)
        self.addToRenderQueueButton.setIconSize(QSize(icon, icon))

        self.verticalLayout_32.setSpacing(gap)
        self.verticalLayout_32.setContentsMargins(gap, gap, gap, gap)
        self.verticalLayout_38.setContentsMargins(inset, inset, inset, inset)
        self.horizontalLayout_60.setContentsMargins(inset, inset, inset, inset)
        self.verticalLayout_39.setSpacing(section_gap)
        self.verticalLayout_39.setContentsMargins(inset, inset, inset, inset)
        self.horizontalLayout_50.setSpacing(section_gap)

        self._render_info_pane(i18n.current_language())

    def retranslate(self, language: str | None = None) -> None:
        self.inputFileSelectButton.setText(QCoreApplication.translate("MainWindow", u"Select Input File", None))
        self.batchSelectButton.setText(QCoreApplication.translate("MainWindow", u"Batch Select", None))
        self.outputFileSelectButton.setText(QCoreApplication.translate("MainWindow", u"Select Output Folder", None))
        self.openOutputFolderButton.setText(QCoreApplication.translate("MainWindow", u"Open Output Folder", None))
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"Backend", None))
        self.label_76.setText(QCoreApplication.translate("MainWindow", u"Interpolate", None))
        self.interpolateCheckBox.setText("")
        self.label_14.setText(QCoreApplication.translate("MainWindow", u"Interpolate Model", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"Interpolation Multiplier", None))
        self.label_36.setText(QCoreApplication.translate("MainWindow", u"SloMo Mode", None))
        self.sloMoModeCheckBox.setText("")
        self.label_86.setText(QCoreApplication.translate("MainWindow", u"Decompress", None))
        self.decompressCheckBox.setText("")
        self.label_87.setText(QCoreApplication.translate("MainWindow", u"Decompress Model", None))
        self.label_82.setText(QCoreApplication.translate("MainWindow", u"Denoise", None))
        self.denoiseCheckBox.setText("")
        self.label_83.setText(QCoreApplication.translate("MainWindow", u"Denoise Model", None))
        self.label_78.setText(QCoreApplication.translate("MainWindow", u"Deblur", None))
        self.deblurCheckBox.setText("")
        self.label_79.setText(QCoreApplication.translate("MainWindow", u"Deblur Model", None))
        self.label_77.setText(QCoreApplication.translate("MainWindow", u"Upscale", None))
        self.upscaleCheckBox.setText("")
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"Upscale Model", None))
        self.label_72.setText(QCoreApplication.translate("MainWindow", u"Upscale Scale", None))
        self.qualityPresetLabel.setText(QCoreApplication.translate("MainWindow", u"Quality Preset", None))
        self.qualityPresetComboBox.setItemText(0, QCoreApplication.translate("MainWindow", u"Best quality", None))
        self.qualityPresetComboBox.setItemText(1, QCoreApplication.translate("MainWindow", u"Fast", None))
        # VALUE for the backend (`--quality_preset`); the text is a catalogue label.
        self.qualityPresetComboBox.setItemData(0, "best")
        self.qualityPresetComboBox.setItemData(1, "fast")
        self.frameReuse.setText(QCoreApplication.translate("MainWindow", u"Reuse unchanged frames", None))
        self.celReuse.setText(QCoreApplication.translate("MainWindow", u"By cel (ignore grain, dust and scan wobble)", None))
        self.addToRenderQueueButton.setText(QCoreApplication.translate("MainWindow", u" Add to Render Queue", None))
        # Last, and through the pane's own two-writer rule: a language switch must repaint the nine names without
        # erasing the values `updateVideoGUIText` puts after them.
        self._retranslate_info_pane(language)
