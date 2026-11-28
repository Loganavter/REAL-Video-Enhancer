"""Page `generalSettings` (QWidget), split out of the Designer-generated mainwindow.py.

Hand-maintained since 2026-11-23: the tree is the Designer's, the controls are the toolkit's. The Designer is no longer
a reference in this tree - the fixture, the splitter that emitted these pages and every "matches the Designer" gate
were removed - so this page answers to `harness/tests/test_toolkit_native.py`: the toolkit's controls and type scale,
`layout.addStretch()` instead of a `QSpacerItem`, and no new `name_<N>` object name or wrapper `QWidget` beyond the
counts that gate records.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import (QCoreApplication, QRect, QSize, Qt)
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QAbstractScrollArea, QGridLayout, QHBoxLayout, QSizePolicy,
    QSpacerItem, QVBoxLayout, QWidget)
import resources_rc
# The page builds the toolkit's own controls: `Text` where the Designer had a `QLabel`, `CustomLineEdit` for a
# `QLineEdit`, `Button` for a `QPushButton`, `CheckBox` for a `QCheckBox`, `ComboBox` for a `QComboBox` and
# `DoubleSpinBox` for a `QDoubleSpinBox`. Nothing here needs a Qt-shaped compatibility adapter.
from sli_ui_toolkit.ui.managers.ui_scale import scaled_px
from sli_ui_toolkit.widgets import (Button, CheckBox, ComboBox, CustomLineEdit, DoubleSpinBox, ProgressBar,
    SurfacePanel, SurfaceScrollArea, Text, TextListWidget)
from src.ui.widgets.combo_width import cap_combo_to_row
from src.ui.typography import apply, BODY_PT, CAPTION_PT, GROUP_TITLE_PT
from src.ui.pages.handles import GeneralSettingsHandles
from src.ui.scale_follower import ScaleFollower
from src.ui import i18n

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

#: The pane's nine rows, in the Designer's order. These are catalogue KEYS, not the text on screen: every name is
#: painted from `i18n.translate(key)` at build and retranslate time, and the VALUES are stored beside them, so a
#: language switch rebuilds the block from the data instead of rewriting painted lines.
INFO_LABELS = ("FPS:", "Resolution:", "Frame Count:", "Encoder:", "Container:", "Color Space:",
               "Pixel Format:", "Is HDR:", "Bit Depth:")

#: The two facts the Encoder cell can report, named by the settings key each is read from. `encoder` is the row's
#: setting and is the truth while the app generates the command line; `encoder_command` is the command the user typed,
#: which is what EXECUTES once it exists (`src/ui/SettingsTab.py::publishEncoderSource`).
ENCODER_SOURCE_ROW = "encoder"
ENCODER_SOURCE_COMMAND = "encoder_command"

#: The pane's values are aligned digits that must not jitter while a run counts, so they keep one monospace face.
INFO_MONO_FAMILY = "Fira Mono"

#: The pane's own DESIGN geometry (the chosen mockup, variant A). The column's WIDTH is the page's (never stated
#: here); these are only the insets the three blocks breathe with.
INFO_CARD_RADIUS_PX = 6
INFO_BLOCK_GAP_PX = 8
INFO_CARD_INSET_PX = 12
INFO_CELL_INSET_PX = 7
INFO_HAIRLINE_PX = 1
INFO_COLUMN_GAP_PX = 12
INFO_RUN_GAP_PX = 7
INFO_TAG_RADIUS_PX = 3
INFO_TAG_INSET_X_PX = 6
INFO_TAG_INSET_Y_PX = 3
INFO_PILL_RADIUS_PX = 9
INFO_BAR_HEIGHT_PX = 6

#: The compact Render Queue card under the run card: FOUR rows of the SAME rows the Render Queue tab paints, then the
#: list's own scrollbar. `INFO_QUEUE_ROW_HEIGHT_PX` is the tab's number (`render_queue_tab.py` builds its list with
#: `row_height=20`) so the two views read alike at one scale. The card's height is derived, never stated: four row
#: heights plus the list's own padding plus the card's insets (`_queue_rows_height`), so a scale change moves both.
INFO_QUEUE_ROWS = 4
INFO_QUEUE_ROW_HEIGHT_PX = 20
INFO_QUEUE_PIN_SIZE_PX = 16
INFO_QUEUE_PIN_ICON_PX = 12

#: The card's two sentences, as catalogue KEYS (translated at build/retranslate time, never stored translated).
INFO_QUEUE_EMPTY = "The render queue is empty."
INFO_QUEUE_PIN_TOOLTIP = "Open the Render Queue tab"

#: The pipeline ribbon's own DESIGN geometry (variant A), read off the mockup's own rules: `::before{margin-bottom:8px}`
#: under the rail, `.s-val{margin-top:4px}`, `.s-sub{margin-top:2px}`, and the 12 px column the chevron sits in.
INFO_RAIL_HEIGHT_PX = 2
INFO_RAIL_GAP_PX = 8
INFO_VALUE_GAP_PX = 4
INFO_SUB_GAP_PX = 2
INFO_ARROW_COLUMN_PX = 12
INFO_FOOTER_GAP_PX = 12
INFO_FOOTER_KV_GAP_PX = 6

#: The five stages in the order the renderer runs them, each with the share of the ribbon's width it takes. The upscale
#: stage carries the longest value (a model name plus its scale), so it is the widest - the mockup's `2.2fr`. The
#: names are catalogue KEYS (`i18n.translate`), never painted text.
#: (stage key, column share). The shares follow the LABEL width, not the stage's importance: at the shipped 485 px
#: card, four 12 px chevrons leave ~437 px, and an equal split gave "Interpolate" 70 px - it elided to "Interpo…" on
#: the live window. Measured caption widths: Interpolate ~62 px, Decompress ~72, Denoise/Deblur ~55, and Upscale
#: needs room for a model file name, which elides by design.
INFO_STAGE_ORDER = (("Interpolate", 13), ("Decompress", 13), ("Denoise", 9), ("Deblur", 9), ("Upscale", 14))

#: The catalogue key a stored quality preset paints itself from: `settings['quality_preset']` ("best"/"fast") is the
#: VALUE the backend is handed; the two keys are the labels the Quality Preset row shows.
INFO_PRESET_LABELS = {"best": "Best quality", "fast": "Fast"}

#: The footer's three facts, in the mockup's order. The keys are catalogue entries.
INFO_FOOTER_LABELS = ("Backend", "Preset", "Reuse")

#: The pane's one sentence for the idle state (the mockup's). The plan is true with no video loaded; the three filled
#: blocks are not, so this line is what the reader gets instead of a labelled skeleton.
INFO_IDLE_NOTE = "No video loaded. Nothing will run until a stage is on and an input file is chosen."


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


def _as_int(text: str) -> int:
    """``text`` as a frame count, or 0 when it is not one (the pane never trusts a formatted string to be a number)."""
    try:
        return max(0, int(float(str(text).strip())))
    except (TypeError, ValueError):
        return 0


def _upscale_tag(resolution_in: str, resolution_out: str) -> int | None:
    """The whole upscale factor between two ``WxH`` strings, or ``None`` when they do not state one.

    The tag is the one fact the headline adds: the resolution is the panel's reason to exist, and "2x upscale" is the
    multiplier the output size carries. A non-integer ratio (or a source size that is not one) states no tag rather
    than a rounded lie.
    """
    try:
        in_w, in_h = (int(part) for part in str(resolution_in).lower().split("x", 1))
        out_w, out_h = (int(part) for part in str(resolution_out).lower().split("x", 1))
    except (TypeError, ValueError):
        return None
    if in_w <= 0 or in_h <= 0 or out_w <= 0 or out_h <= 0:
        return None
    if out_w % in_w or out_h % in_h:
        return None
    width_ratio, height_ratio = out_w // in_w, out_h // in_h
    if width_ratio != height_ratio or width_ratio <= 1:
        return None
    return width_ratio


@dataclass(frozen=True)
class PipelineStage:
    """One stage of the configured pipeline, as the ribbon states it.

    `key` is the catalogue key the name is painted from (`"Interpolate"`, `"Upscale"`, ... - the same words the General
    tab's rows use). `model` is the model the user selected - the text the combo shows, which
    `getCurrentRenderOptions` maps through `getModels` to the file the backend is handed - and `factor` is the
    multiplier the render applies (interpolation times, the requested upscale scale). Both stay empty/``None`` when the
    stage does not run, so a stage that was switched off cannot leave its old parameter on screen.
    """

    key: str
    enabled: bool = False
    model: str = ""
    factor: int | None = None


@dataclass(frozen=True)
class PipelineFacts:
    """The configured pipeline: the stages in render order, then the settings-backed footer facts.

    Frozen and comparable on purpose - `ProcessTab` holds the last facts it handed over and only repaints the pane when
    this value changes, so the 100 ms tick costs one tuple compare on an idle second. Every field is a fact the render
    command actually reads: a stage's switch and parameter, `render_options.backend`, `settings['quality_preset']`,
    `settings['temporal_reuse']` and `settings['temporal_reuse_statistic']`.
    """

    stages: tuple[PipelineStage, ...] = ()
    backend: str = ""
    preset: str = ""
    reuse: bool = False
    by_cel: bool = False


class GeneralSettingsPage:
    """`generalSettings` - placed by its parent with `addTab`."""

    page_id = "generalSettings"

    #: The user-owned command line the pane must report instead of the Encoder row's setting ("" = the row is the
    #: fact). Declared at class level so a pane that has not rendered anything yet answers `encoder_source()` sanely.
    _encoder_command = ""
    #: The language the pane was last painted in: a late `set_encoder_command` re-renders in the SAME one instead of
    #: guessing from the app's current language (they differ for the moment a language switch is in flight).
    _info_language: str | None = None

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
        # No `setGeometry`: the Designer's 470x1040 preview rect is overwritten by the resizable area on the first
        # layout (measured after settle: 470x1025 at 1.0, 961x1952 at 2.0 - the width is the viewport's, the height is
        # the rows' own). A pinned rect here is dead text, not a design length the interface honours
        # (`raw-chrome-px` in `harness/scripts/ui_gates.py`).
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
        # The Designer's `setSizeIncrement(QSize(230, 0))` is gone: `QWidget.sizeIncrement` only governs the steps of
        # a user's interactive resize of a TOP-LEVEL window, and this combo is laid out by its row at every factor
        # (measured 230x0 at 1.0 and 2.0 before this line was removed, with the combo's geometry unchanged either
        # way). It is not a design length the interface honours, so it is retired rather than scaled.
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
            content_padding=(10, 5, 10, 5), corner_radius=10, size=(0, 35))
        self.addToRenderQueueButton.setObjectName(u"addToRenderQueueButton")
        sizePolicy5.setHeightForWidth(self.addToRenderQueueButton.sizePolicy().hasHeightForWidth())
        self.addToRenderQueueButton.setSizePolicy(sizePolicy5)
        # The Designer pinned 200x35, and 200 was NARROWER than this button's own text (owner, 2026-11-28: the label
        # rendered "Add to Render Que…"). A button whose text is known must take its width from that text, not from a
        # literal: `size=(0, 35)` means "width from the content, height fixed" (the toolkit's own convention - the zoom
        # buttons use `size=(0, 30)`), so the label is never elided at any factor.

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
        # The video-info pane: the pipeline plan (a card that speaks with no clip loaded) over three stacked toolkit
        # blocks (transition headline, facts grid, run card) that size themselves. No ceiling and no floor here - the
        # pane's vertical policy is Fixed, so its height is the content's
        # (the old 190 px floor made a short body always reach the same position; owner, 2026-11-26).
        self.horizontalLayout_22 = QHBoxLayout(self.widget_26)
        self.horizontalLayout_22.setObjectName(u"horizontalLayout_22")
        self.horizontalLayout_22.setContentsMargins(0, 0, 0, 0)
        self._build_video_info_pane()
        self.horizontalLayout_22.addWidget(self.videoInfoPane)
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

    # -- video-info pane (variant A: transition headline, facts grid, run card) ----------------------

    def _build_video_info_pane(self) -> None:
        """Build the pane once: the pipeline plan (a card of its own), then the three filled blocks.

        Everything here is a toolkit control (`SurfacePanel`, `Text`, `ProgressBar`) in plain box layouts - no QSS, no
        custom paint, no hand-drawn rectangle. The states are built together and toggled by `set_video_info` /
        `set_pipeline`, so a language switch or a scale change only re-states text and design numbers, never rebuilds
        the tree.

        The PLAN card is separate from the three filled blocks because the two answer different questions: the pipeline
        is what the user configured (true before any video exists - the mockup's idle state), while the headline, the
        facts table and the run card describe a loaded clip and a run.
        """
        self._video_values: dict[str, tuple[str, str]] | None = None
        self._pipeline: PipelineFacts | None = None
        self._run_frame = 0
        self._run_total = 0
        self._run_percent = 0.0
        self._run_eta = ""
        self._run_fps = ""
        self._run_state = "Idle"
        self._run_elapsed = "0:00:00"

        # The pane host paints nothing: the cards below carry the plates, and the page already paints `surface.page`.
        self.videoInfoPane = SurfacePanel(self.widget_26, background_token=None)
        self.videoInfoPane.setObjectName("videoInfoPane")
        self.videoInfoPane.setAccessibleDescription(u"")
        info_policy = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed)
        info_policy.setHeightForWidth(False)
        self.videoInfoPane.setSizePolicy(info_policy)
        self.videoInfoLayout = QVBoxLayout(self.videoInfoPane)
        self.videoInfoLayout.setContentsMargins(0, 0, 0, 0)
        self.videoInfoLayout.setSpacing(scaled_px(INFO_BLOCK_GAP_PX))

        # -- the plan: the pipeline ribbon (variant A), a card of its own above the fill ------------------
        self.infoPipelineCard = SurfacePanel(self.videoInfoPane, background_token="surface.list",
                                             radius=scaled_px(INFO_CARD_RADIUS_PX))
        self.infoPipelineCard.setObjectName("videoInfoPipelineCard")
        pipeline_layout = QVBoxLayout(self.infoPipelineCard)
        pipeline_layout.setContentsMargins(0, 0, 0, 0)
        pipeline_layout.setSpacing(0)

        ribbon_host = QWidget(self.infoPipelineCard)
        # A GRID, not a row of per-stage boxes: the mockup's rhythm is three DIFFERENT gaps (rail->name 8, name->value 4,
        # value->sub 2), and a grid states them once as row heights instead of a spacer per stage (`QSpacerItem` may not
        # come back to this page - `harness/tests/test_toolkit_native.py::test_no_new_spacer_items`). Every stage shares
        # the same rows, so two stage names can never sit at different heights.
        self.infoRibbonLayout = QGridLayout(ribbon_host)
        self.infoRibbonLayout.setContentsMargins(0, 0, 0, 0)
        self.infoRibbonLayout.setHorizontalSpacing(0)
        self.infoRibbonLayout.setVerticalSpacing(0)
        #: stage key -> its four painted parts, so a repaint only re-states text and tokens.
        self.infoStages: dict[str, dict] = {}
        #: the chevrons between the stages, kept only to re-size their column with the factor.
        self.infoArrows: list[Text] = []
        for index, (stage_key, share) in enumerate(INFO_STAGE_ORDER):
            rail = SurfacePanel(ribbon_host, background_token="separator.color")
            rail.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            name = Text(parent=ribbon_host, elide=True)
            name.setObjectName(f"videoInfoStage{stage_key}")
            apply(name, CAPTION_PT)
            value = Text(parent=ribbon_host, elide=True)
            apply(value, CAPTION_PT, family=INFO_MONO_FAMILY)
            sub = Text(parent=ribbon_host, elide=True, color_token="label.text.muted")
            apply(sub, CAPTION_PT, family=INFO_MONO_FAMILY)
            # `Ignored` horizontally: a model name is long and a stage is ~80 px, so the label takes what the row gives
            # it and elides - its own text must never become the page's minimum width (`test_general_tab_fit`).
            narrow = QSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            for widget in (name, value, sub):
                widget.setSizePolicy(narrow)
            # Column 2i holds the stage's four lines, column 2i+1 the chevron that separates it from the next stage.
            column = 2 * index
            self.infoRibbonLayout.addWidget(rail, 0, column)
            self.infoRibbonLayout.addWidget(name, 2, column)
            self.infoRibbonLayout.addWidget(value, 4, column)
            self.infoRibbonLayout.addWidget(sub, 6, column)
            self.infoRibbonLayout.setColumnStretch(column, share)
            self.infoStages[stage_key] = {"rail": rail, "name": name, "value": value, "sub": sub}
            if index < len(INFO_STAGE_ORDER) - 1:
                arrow = Text(parent=ribbon_host, text=u"\u203a", color_token="label.text.muted",
                             alignment=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
                apply(arrow, CAPTION_PT)
                # Its own narrow column, as the mockup's `12px` grid column between two stages.
                arrow.setFixedWidth(scaled_px(INFO_ARROW_COLUMN_PX))
                self.infoArrows.append(arrow)
                self.infoRibbonLayout.setColumnMinimumWidth(column + 1, scaled_px(INFO_ARROW_COLUMN_PX))
                self.infoRibbonLayout.addWidget(arrow, 2, column + 1)
        pipeline_layout.addWidget(ribbon_host)

        self.infoPipelineRule = SurfacePanel(self.infoPipelineCard, background_token="separator.color")
        self.infoPipelineRule.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        pipeline_layout.addWidget(self.infoPipelineRule)

        foot_host = QWidget(self.infoPipelineCard)
        self.infoFooterLayout = QHBoxLayout(foot_host)
        #: catalogue key -> (its label, its value, the pair's host widget) - the host is what hides with the fact.
        self.infoFooterValues: dict[str, tuple[Text, Text, QWidget]] = {}
        for label_key in INFO_FOOTER_LABELS:
            pair = QWidget(foot_host)
            pair_layout = QHBoxLayout(pair)
            pair_layout.setContentsMargins(0, 0, 0, 0)
            label = Text(parent=pair, color_token="label.text.muted")
            apply(label, CAPTION_PT)
            value = Text(parent=pair, elide=True)
            apply(value, CAPTION_PT, family=INFO_MONO_FAMILY)
            pair_layout.addWidget(label)
            pair_layout.addWidget(value)
            self.infoFooterLayout.addWidget(pair)
            self.infoFooterValues[label_key] = (label, value, pair)
        self.infoFooterLayout.addStretch(1)
        pipeline_layout.addWidget(foot_host)
        self.videoInfoLayout.addWidget(self.infoPipelineCard)

        # -- the filled state: three stacked blocks -------------------------------------------------
        self.infoFilled = QWidget(self.videoInfoPane)
        self.infoFilled.setObjectName("videoInfoFilled")
        filled_layout = QVBoxLayout(self.infoFilled)
        filled_layout.setContentsMargins(0, 0, 0, 0)
        filled_layout.setSpacing(scaled_px(INFO_BLOCK_GAP_PX))

        # Block 1 + 2 share one card: the headline, a hairline, then two columns x three rows of facts.
        self.infoFactsCard = SurfacePanel(self.infoFilled, background_token="surface.list",
                                          radius=scaled_px(INFO_CARD_RADIUS_PX))
        self.infoFactsCard.setObjectName("videoInfoFactsCard")
        facts_card_layout = QVBoxLayout(self.infoFactsCard)
        facts_card_layout.setContentsMargins(0, 0, 0, 0)
        facts_card_layout.setSpacing(0)

        transition = QWidget(self.infoFactsCard)
        self.infoTransitionLayout = QHBoxLayout(transition)
        self.infoTransitionLayout.setSpacing(scaled_px(INFO_COLUMN_GAP_PX))
        self.infoTransitionLayout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        self.infoSizeIn = Text(parent=transition)
        apply(self.infoSizeIn, GROUP_TITLE_PT, bold=True, family=INFO_MONO_FAMILY)
        self.infoSizeOut = Text(parent=transition)
        apply(self.infoSizeOut, GROUP_TITLE_PT, bold=True, family=INFO_MONO_FAMILY)
        self.infoMetaIn = Text(parent=transition, color_token="label.text.muted")
        apply(self.infoMetaIn, CAPTION_PT)
        self.infoMetaOut = Text(parent=transition, color_token="label.text.muted")
        apply(self.infoMetaOut, CAPTION_PT)
        # Each side is its own column (size over its meta line); `addStretch` inside the column is not needed because
        # both sides take an equal share of the row and the tag is pushed to the right by the second share.
        for size_label, meta_label in ((self.infoSizeIn, self.infoMetaIn), (self.infoSizeOut, self.infoMetaOut)):
            side = QWidget(transition)
            side_layout = QVBoxLayout(side)
            side_layout.setContentsMargins(0, 0, 0, 0)
            side_layout.setSpacing(1)
            side_layout.addWidget(size_label)
            side_layout.addWidget(meta_label)
            self.infoTransitionLayout.addWidget(side, 1)
            if size_label is self.infoSizeIn:
                self.infoArrow = Text(parent=transition, text=u"\u2192", color_token="label.text.muted")
                apply(self.infoArrow, GROUP_TITLE_PT, family=INFO_MONO_FAMILY)
                self.infoTransitionLayout.addWidget(self.infoArrow, 0,
                                                    Qt.AlignmentFlag.AlignVCenter)
        self.infoTagPlate = SurfacePanel(transition, background_token="list_item.background.focused",
                                         radius=scaled_px(INFO_TAG_RADIUS_PX))
        self.infoTagPlate.setObjectName("videoInfoTagPlate")
        self.infoTagLayout = QHBoxLayout(self.infoTagPlate)
        self.infoTagLayout.setSpacing(0)
        self.infoTag = Text(parent=self.infoTagPlate, color_token="accent")
        apply(self.infoTag, CAPTION_PT, family=INFO_MONO_FAMILY)
        self.infoTagLayout.addWidget(self.infoTag)
        self.infoTransitionLayout.addWidget(self.infoTagPlate, 0, Qt.AlignmentFlag.AlignVCenter)
        facts_card_layout.addWidget(transition)

        self.infoHairlines: list[SurfacePanel] = []
        self.infoColumnHairlines: list[SurfacePanel] = []
        self.infoCellLayouts: list[QHBoxLayout] = []
        self.infoFactLabels: dict[str, Text] = {}
        self.infoFactValues: dict[str, Text] = {}

        def add_hairline() -> None:
            hairline = SurfacePanel(self.infoFactsCard, background_token="separator.color")
            hairline.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            facts_card_layout.addWidget(hairline)
            self.infoHairlines.append(hairline)

        add_hairline()
        fact_pairs = (("Encoder:", "Container:"), ("Color Space:", "Pixel Format:"),
                      ("Bit Depth:", "Is HDR:"))
        for index, pair in enumerate(fact_pairs):
            fact_row = QWidget(self.infoFactsCard)
            fact_row_layout = QHBoxLayout(fact_row)
            fact_row_layout.setContentsMargins(0, 0, 0, 0)
            fact_row_layout.setSpacing(0)
            for column, key in enumerate(pair):
                if column:
                    column_hairline = SurfacePanel(fact_row, background_token="separator.color")
                    column_hairline.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
                    fact_row_layout.addWidget(column_hairline)
                    self.infoColumnHairlines.append(column_hairline)
                cell = QWidget(fact_row)
                cell_layout = QHBoxLayout(cell)
                cell_layout.setSpacing(8)
                label = Text(parent=cell, color_token="label.text.muted")
                apply(label, CAPTION_PT)
                value = Text(parent=cell, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                apply(value, CAPTION_PT, family=INFO_MONO_FAMILY)
                cell_layout.addWidget(label)
                cell_layout.addStretch(1)
                cell_layout.addWidget(value)
                self.infoCellLayouts.append(cell_layout)
                self.infoFactLabels[key] = label
                self.infoFactValues[key] = value
                fact_row_layout.addWidget(cell, 1)
            facts_card_layout.addWidget(fact_row)
            if index < len(fact_pairs) - 1:
                add_hairline()
        filled_layout.addWidget(self.infoFactsCard)

        # Block 3: the run card - the counters, the toolkit bar, then fps/status/elapsed.
        self.infoRunCard = SurfacePanel(self.infoFilled, background_token="surface.list",
                                        radius=scaled_px(INFO_CARD_RADIUS_PX))
        self.infoRunCard.setObjectName("videoInfoRunCard")
        self.infoRunLayout = QVBoxLayout(self.infoRunCard)
        self.infoRunLayout.setSpacing(scaled_px(INFO_RUN_GAP_PX))

        run_line = QWidget(self.infoRunCard)
        run_line_layout = QHBoxLayout(run_line)
        run_line_layout.setContentsMargins(0, 0, 0, 0)
        run_line_layout.setSpacing(10)
        self.infoFrameLine = Text(parent=run_line)
        apply(self.infoFrameLine, CAPTION_PT, family=INFO_MONO_FAMILY)
        self.infoRemainingLine = Text(parent=run_line, color_token="label.text.muted",
                                      alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        # Named for the control surface and for the typography test: this is the caption-scale text the shell builds
        # (the transport row's own caption line was removed 2026-11-28, so the run card is the surviving example).
        self.infoRemainingLine.setObjectName("videoInfoRemaining")
        apply(self.infoRemainingLine, CAPTION_PT, family=INFO_MONO_FAMILY)
        run_line_layout.addWidget(self.infoFrameLine)
        run_line_layout.addStretch(1)
        run_line_layout.addWidget(self.infoRemainingLine)
        self.infoRunLayout.addWidget(run_line)

        self.infoProgress = ProgressBar(self.infoRunCard, height=INFO_BAR_HEIGHT_PX)
        self.infoProgress.setObjectName("videoInfoProgress")
        self.infoRunLayout.addWidget(self.infoProgress)

        run_foot = QWidget(self.infoRunCard)
        run_foot_layout = QHBoxLayout(run_foot)
        run_foot_layout.setContentsMargins(0, 0, 0, 0)
        run_foot_layout.setSpacing(10)
        self.infoFpsPlate = SurfacePanel(run_foot, background_token="list_item.background.focused",
                                         radius=scaled_px(INFO_PILL_RADIUS_PX))
        self.infoFpsPlate.setObjectName("videoInfoFpsPlate")
        self.infoFpsLayout = QHBoxLayout(self.infoFpsPlate)
        self.infoFpsLayout.setSpacing(0)
        self.infoFps = Text(parent=self.infoFpsPlate, color_token="accent")
        apply(self.infoFps, CAPTION_PT, family=INFO_MONO_FAMILY)
        self.infoFpsLayout.addWidget(self.infoFps)
        self.infoStatus = Text(parent=run_foot, color_token="label.text.muted")
        apply(self.infoStatus, CAPTION_PT)
        self.infoElapsed = Text(parent=run_foot, color_token="label.text.muted",
                                alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        apply(self.infoElapsed, CAPTION_PT, family=INFO_MONO_FAMILY)
        run_foot_layout.addWidget(self.infoFpsPlate)
        run_foot_layout.addWidget(self.infoStatus)
        run_foot_layout.addStretch(1)
        run_foot_layout.addWidget(self.infoElapsed)
        self.infoRunLayout.addWidget(run_foot)
        filled_layout.addWidget(self.infoRunCard)

        self.videoInfoLayout.addWidget(self.infoFilled)

        # -- the compact Render Queue card: the tab's rows, four of them, under the run card -----------------
        # The SAME list: `TextListWidget` with the tab's painter and tooltip (`src/ui/queue_rows.py`), bound by the
        # glue to the same `RenderQueue` model - a second, smaller view of one queue, never a second renderer. It is a
        # sibling of `infoFilled` (not a child) because a queue exists with no clip loaded, and the card must say so
        # there instead of disappearing behind the pane's placeholder rule.
        self._queue_preview_names: list[str] | None = None
        self.infoQueueCard = SurfacePanel(self.videoInfoPane, background_token="surface.list",
                                          radius=scaled_px(INFO_CARD_RADIUS_PX))
        self.infoQueueCard.setObjectName("videoInfoQueueCard")
        self.infoQueueLayout = QVBoxLayout(self.infoQueueCard)
        self.infoQueueLayout.setSpacing(0)
        # The pin's band is a LAYOUT, not a wrapper widget: a bare `QWidget` here would be the Designer's wrapper
        # level the page may not grow (`test_no_new_bare_wrapper_widgets`). It sits at the card's top-right, above
        # the four rows, so no row's percentage is ever painted under it.
        self.infoQueueHeaderLayout = QHBoxLayout()
        self.infoQueueHeaderLayout.setContentsMargins(0, 0, 0, 0)
        self.infoQueueHeaderLayout.addStretch(1)
        pin_icon = QIcon()
        pin_icon.addFile(u":/icons/icons/map-pin.svg", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.infoQueuePin = Button(pin_icon, size=(INFO_QUEUE_PIN_SIZE_PX, INFO_QUEUE_PIN_SIZE_PX),
                                   icon_size=INFO_QUEUE_PIN_ICON_PX, parent=self.infoQueueCard)
        self.infoQueuePin.setObjectName("videoInfoQueuePin")
        self.infoQueueHeaderLayout.addWidget(self.infoQueuePin)
        self.infoQueueLayout.addLayout(self.infoQueueHeaderLayout)
        self.infoQueueList = TextListWidget(self.infoQueueCard, row_height=INFO_QUEUE_ROW_HEIGHT_PX, radius=10,
                                            surface_token="dialog.background")
        self.infoQueueList.setObjectName("videoInfoQueueList")
        self.infoQueueList.setSelectionMode("single")
        self.infoQueueList.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.infoQueueLayout.addWidget(self.infoQueueList)
        self.infoQueueEmpty = Text(parent=self.infoQueueCard, color_token="label.text.muted", word_wrap=True)
        self.infoQueueEmpty.setObjectName("videoInfoQueueEmpty")
        apply(self.infoQueueEmpty, CAPTION_PT)
        self.infoQueueLayout.addWidget(self.infoQueueEmpty)
        self.videoInfoLayout.addWidget(self.infoQueueCard)
        # The card opens in its EMPTY state: one muted line, no four-row hole. The first glue sync (or the first
        # item) replaces it through `set_queue_preview_names`, the card's only content writer.
        self.set_queue_preview_names([])

        # The idle line under the plan (the mockup's): shown only while the plan is up and no clip is loaded. Built here
        # so it is a sibling of the cards, and `setVisible` is the only thing a re-render does to it.
        self.infoIdleNote = Text(parent=self.videoInfoPane, color_token="label.text.muted", word_wrap=True)
        self.infoIdleNote.setObjectName("videoInfoIdleNote")
        # `Ignored` horizontally: `word_wrap=True` alone wrapped nothing, because the widget's own size hint was the whole
        # sentence - it took that width, the pane clipped it, and the reader got "…an input file is chos" on the live
        # window (2026-11-28). A word-wrapped label must be told it does NOT get a vote on the row's width.
        self.infoIdleNote.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        apply(self.infoIdleNote, CAPTION_PT)
        self.infoIdleNote.setVisible(False)
        self.videoInfoLayout.addWidget(self.infoIdleNote)

    def set_pipeline(self, facts: PipelineFacts | None) -> None:
        """State the configured pipeline - the ribbon's ONE input, fed by `ProcessTab` from the settings the run reads.

        `None` means "nothing states a plan yet" (no render surface to read it from) and hides the card. A later
        language switch or scale change repaints from the stored facts, so the labels follow the catalogue while the
        pipeline itself does not move - the same rule the three filled blocks already follow.
        """
        if facts == self._pipeline:
            return
        self._pipeline = facts
        self._render_video_info(self._info_language or i18n.current_language())

    def set_video_info(self, values: dict[str, tuple[str, str]] | None) -> None:
        """Fill the pane. `values` is keyed by the `INFO_LABELS` keys ("FPS:", "Resolution:", ...) and each value is
        (input, output); None restores the nine-row placeholder. A language switch re-renders from the stored values."""
        previous = self._video_values
        if values is None:
            self._video_values = None
            self._reset_run_status()
        else:
            self._video_values = {
                key: (str(pair[0]), str(pair[1])) if isinstance(pair, (tuple, list)) and len(pair) >= 2
                else (str(pair), "")
                for key, pair in values.items()
            }
            total = _as_int(self._video_values.get("Frame Count:", ("", ""))[1])
            if previous is None or total != self._run_total:
                self._run_total = total
                self._run_frame = 0
                self._run_percent = 0.0
        self._render_video_info()

    def video_info_text(self) -> str:
        """The nine rows as the plain text the control probe reads ("FPS: 24.0 -> 24.0" per line), or "" when empty."""
        if self._video_values is None:
            return ""
        _, encoder_value = self.encoder_source()
        lines = []
        for key in INFO_LABELS:
            value_in, value_out = self._video_values.get(key, ("", ""))
            if key == "Encoder:":
                # The probe reads what the pane reports - one source for both (F5), not the row's key when the
                # user's command is what runs.
                value_out = encoder_value
            lines.append(f"{key} {value_in} -> {value_out}".rstrip())
        return "\n".join(lines)

    def set_encoder_command(self, command: str = "") -> None:
        """Declare which encoder fact will run: `""` = the Encoder row's setting, else the user's command line.

        `MainWindow.updateVideoGUIText` paints the row from `settings['encoder']`, and that is the truth only while
        the app generates the command. Once the user has typed one, the field is what executes (`SettingsTab` stops
        regenerating it) and the row's key is not - the pane then reported an encoder the run would not use
        (settings audit 2026-11-27, F5). `src/ui/SettingsTab.py::publishEncoderSource` owns the ownership flag and
        hands it here; the pane names the setting it read in the cell's label and tooltip.
        """
        command = (command or "").strip()
        if command == self._encoder_command:
            return
        self._encoder_command = command
        if self._video_values is not None:
            self._render_video_info(self._info_language)

    def encoder_source(self) -> tuple[str, str]:
        """`(the settings key read, the value reported)` - the Encoder cell's one source of truth.

        `("encoder", <row value>)` while the app generates the command line, `("encoder_command", <the command>)` while
        the user owns it. The label, the value, the tooltip and `video_info_text()` all read this pair, so the pane
        cannot state one encoder in its cell and another in the probe's text.
        """
        if self._encoder_command:
            return (ENCODER_SOURCE_COMMAND, self._encoder_command)
        row = (self._video_values or {}).get("Encoder:", ("", ""))[1]
        return (ENCODER_SOURCE_ROW, str(row))

    def set_run_status(self, *, frame: int | None = None, total: int | None = None,
                       percent: float | None = None, eta: str | None = None, fps: str | None = None,
                       status: str | None = None, elapsed: str | None = None) -> None:
        """State the run card's counters (`frame`, `total`, `percent`, `eta`, `fps`, `status`, `elapsed`).

        The pane is the block's owner; the process tick that knows these numbers is outside this page, so it hands
        them over here and the pane paints what it was given. Omitted arguments keep their current value.
        """
        if frame is not None:
            self._run_frame = max(0, int(frame))
        if total is not None:
            self._run_total = max(0, int(total))
        if percent is not None:
            self._run_percent = float(percent)
        if eta is not None:
            self._run_eta = str(eta)
        if fps is not None:
            self._run_fps = str(fps)
        if status is not None:
            self._run_state = str(status)
        if elapsed is not None:
            self._run_elapsed = str(elapsed)
        self._render_video_info()

    def reset_run_status(self) -> None:
        """Return the run card to its idle state - the tick's "no render is live" branch.

        `_reset_run_status` clears the stored counters; without a repaint the last job's numbers would stay frozen on
        screen after the render ends. Nothing is painted while the video-info placeholder is up (there is no card to
        paint), which also keeps an idle page from re-stating a hidden tree on every tick.
        """
        self._reset_run_status()
        if self._video_values is not None:
            self._render_run_status()

    def _reset_run_status(self) -> None:
        self._run_frame = 0
        self._run_total = 0
        self._run_percent = 0.0
        self._run_eta = ""
        self._run_fps = ""
        self._run_state = "Idle"
        self._run_elapsed = "0:00:00"

    # -- the compact queue card (four rows, a pin) ----------------------------------------------------

    def _queue_rows_height(self) -> int:
        """The list's fixed height: FOUR rows at ITS row height plus ITS own padding.

        Derived from the widget, not stated: `content_height()` is `count * row_height + padding`, so subtracting the
        rows gives back exactly the padding the toolkit resolved through `scaled_px`. The card therefore holds four
        rows and neither a pixel more nor a pixel less, at any factor, and the sixth row scrolls inside it.
        """
        panel = self.infoQueueList
        padding = panel.content_height() - panel.count() * panel.row_height()
        return INFO_QUEUE_ROWS * panel.row_height() + padding

    def queue_preview_scroll_maximum(self) -> int:
        """The compact list's vertical scroll range - 0 while its four rows hold every item.

        Reads the toolkit's own overlay area (the list builds exactly one `QAbstractScrollArea` inside itself); the
        test asserts that more items than fit scroll rather than grow the card.
        """
        area = self.infoQueueList.findChild(QAbstractScrollArea)
        return int(area.verticalScrollBar().maximum()) if area is not None else 0

    def set_queue_preview_names(self, names) -> None:
        """Mirror the render queue's rows into the compact card - the card's ONE content writer.

        The glue calls this with the model's names (same list the Render Queue tab shows); it never parses a row's
        text back into state. An unchanged queue is a no-op, so a progress tick repaints rows
        (`refresh_queue_preview_rows`) without resetting the scroll position, and a language switch changes the one
        sentence, never the list object.
        """
        names = [str(name) for name in (names or ())]
        if names == self._queue_preview_names:
            return
        self._queue_preview_names = names
        if names:
            self.infoQueueList.setItems(names)
            self.infoQueueEmpty.setVisible(False)
            self.infoQueueList.setVisible(True)
            return
        self.infoQueueList.clear()
        self.infoQueueList.setVisible(False)
        self.infoQueueEmpty.setVisible(True)
        self._render_queue_preview_text(self._info_language)

    def refresh_queue_preview_rows(self) -> None:
        """Repaint the card's rows: the painter reads the model, so a state change is an update, never a rebuild."""
        self.infoQueueList.update()

    def _render_queue_preview_text(self, language: str | None = None) -> None:
        """The card's one sentence, from the catalogue - re-stated on every pane pass, including a language switch."""
        self.infoQueueEmpty.setText(i18n.translate(INFO_QUEUE_EMPTY, language))

    @staticmethod
    def _fact_name(key: str, language: str | None) -> str:
        """A facts-grid name from the catalogue, without the row's trailing colon (the mockup's cell labels)."""
        return i18n.translate(key, language).rstrip(":").strip()

    @staticmethod
    def _meta_line(fps: str, frames: str, language: str | None) -> str:
        """`24 fps · 493 frames` under one side of the headline, from that side's own values."""
        parts = []
        if fps:
            parts.append(f"{fps} {i18n.translate('fps', language)}")
        if frames:
            parts.append(f"{frames} {i18n.translate('frames', language)}")
        return " · ".join(parts)

    def _render_video_info(self, language: str | None = None) -> None:
        """Repaint the pane from the STORED values at ``language`` - the plan alone when there are none.

        This is the only writer of the pane's text. A language switch calls it again, so the row names follow the
        catalogue while every value stays byte-identical (the regression the old line-rewriting relabel guarded).
        """
        self._info_language = i18n.current_language() if language is None else language
        language = self._info_language
        self._render_pipeline(language)
        # Rebuilt on every pass so it is already correct when a clip is dropped and the note comes back.
        self.infoIdleNote.setText(i18n.translate(INFO_IDLE_NOTE, language))
        # The queue card's one sentence follows the same rule (a language switch must not touch its list).
        self._render_queue_preview_text(language)
        if self._video_values is None:
            # Nothing to describe: the three filled blocks are invisible rather than a labelled skeleton (owner,
            # 2026-11-26: "зачем мы это рисуем пока не выбрано ни одно видео? Какой-то рудимент"). The host paints
            # nothing of its own, so a hidden card collapses the block completely. The PLAN stays: it is what the user
            # configured, and with no clip loaded it is the whole answer.
            self.infoFilled.setVisible(False)
            self.infoIdleNote.setVisible(self._pipeline is not None)
            return

        self.infoIdleNote.setVisible(False)
        self.infoFilled.setVisible(True)
        get = self._video_values.get
        resolution_in, resolution_out = get("Resolution:", ("", ""))
        fps_in, fps_out = get("FPS:", ("", ""))
        frames_in, frames_out = get("Frame Count:", ("", ""))
        self.infoSizeIn.setText(resolution_in or u"\u2014")
        self.infoSizeOut.setText(resolution_out or u"\u2014")
        self.infoMetaIn.setText(self._meta_line(fps_in, frames_in, language))
        self.infoMetaOut.setText(self._meta_line(fps_out, frames_out, language))
        source_key, encoder_value = self.encoder_source()
        for key, widget in self.infoFactLabels.items():
            name = self._fact_name(key, language)
            if key == "Encoder:" and source_key == ENCODER_SOURCE_COMMAND:
                # Name the setting the cell read, so the pane says WHICH of the two encoder facts it reports: the
                # user's command line, not the row above it (F5).
                name = f"{name} ({ENCODER_SOURCE_COMMAND})"
            widget.setText(name)
        for key, widget in self.infoFactValues.items():
            if key == "Encoder:":
                widget.setText(encoder_value or u"\u2014")
            else:
                widget.setText(get(key, ("", ""))[1] or u"\u2014")
        encoder_cell = self.infoFactValues.get("Encoder:")
        if encoder_cell is not None:
            encoder_cell.setToolTip(
                f"{source_key}: {encoder_value}" if encoder_value else source_key)

        factor = _upscale_tag(resolution_in, resolution_out)
        self.infoTag.setText(f"{factor}x {i18n.translate('upscale', language)}" if factor else "")
        self.infoTagPlate.setVisible(bool(factor))
        self._render_run_status(language)

    def _render_pipeline(self, language: str | None = None) -> None:
        """Paint the ribbon and its footer from the stored facts - the only writer of the plan's text.

        Every stage the plan does not name is drawn in its OFF form (muted, `off`), because the ribbon's job is to show
        the whole path: a reader must see that Decompress sits between interpolation and denoise even when it does not
        run. An enabled stage states its own parameter - and only its own, so switching one off cannot leave its old
        model or factor on screen.
        """
        facts = self._pipeline
        self.infoPipelineCard.setVisible(facts is not None)
        if facts is None:
            return

        by_key = {stage.key: stage for stage in facts.stages}
        for key, widgets in self.infoStages.items():
            stage = by_key.get(key) or PipelineStage(key=key)
            widgets["name"].setText(i18n.translate(key, language))
            widgets["rail"].set_background_token("accent" if stage.enabled else "separator.color")
            if stage.enabled:
                value, sub = self._stage_state(stage, language)
            else:
                value, sub = i18n.translate("off", language), ""
            widgets["value"].setText(value)
            widgets["sub"].setText(sub)
            widgets["sub"].setVisible(bool(sub))
            for widget in (widgets["name"], widgets["value"]):
                widget.setColorToken(None if stage.enabled else "label.text.muted")

        for label_key, (label, value_widget, host) in self.infoFooterValues.items():
            label.setText(i18n.translate(label_key, language))
            text, visible = self._footer_state(facts, label_key, language)
            value_widget.setText(text)
            host.setVisible(visible)

    @staticmethod
    def _stage_state(stage: PipelineStage, language: str | None) -> tuple[str, str]:
        """The `(value, sub)` an ENABLED stage states: its parameter, never a placeholder.

        A model and a factor on one stage is the upscale case (`model` over `scale ×N`); a model alone is deblur/denoise/
        decompress; a factor alone is interpolation (`×2`); neither is a switch with no parameter.
        """
        if stage.model and stage.factor is not None:
            return (stage.model, f"{i18n.translate('scale', language)} ×{stage.factor}")
        if stage.model:
            return (stage.model, "")
        if stage.factor is not None:
            return (f"×{stage.factor}", "")
        return (i18n.translate("on", language), "")

    @staticmethod
    def _footer_state(facts: PipelineFacts, key: str, language: str | None) -> tuple[str, bool]:
        """`(value, visible)` for one footer fact - a fact that is not on/selected is not painted at all."""
        if key == "Backend":
            return (facts.backend, bool(facts.backend))
        if key == "Preset":
            label = INFO_PRESET_LABELS.get(facts.preset, facts.preset)
            return (i18n.translate(label, language) if label else "", bool(label))
        # "Reuse": the switch, plus `by cel` only while that is the statistic the render compares on (the command's
        # own rule: `--temporal_reuse_statistic coarse` is passed only while reuse is on).
        value = i18n.translate("on", language)
        if facts.by_cel:
            value = f"{value} · {i18n.translate('by cel', language)}"
        return (value, facts.reuse)

    def _render_run_status(self, language: str | None = None) -> None:
        """Paint the run card: frame counter and ETA on one line, the bar, then fps/status/elapsed."""
        total = self._run_total or _as_int(self._video_values.get("Frame Count:", ("", ""))[1]
                                           if self._video_values else "")
        frame = min(self._run_frame, total) if total else self._run_frame
        total_text = str(total) if total else u"\u2014"
        self.infoFrameLine.setText(i18n.translate("frame ", language) + f"{frame} / {total_text}")
        eta = self._run_eta or u"\u2014"
        self.infoRemainingLine.setText(f"{self._run_percent:g}% · {eta}" + i18n.translate(" left", language))
        self.infoProgress.setRange(0, max(1, total))
        self.infoProgress.setValue(frame if total else 0)
        fps = self._run_fps or u"\u2014"
        self.infoFps.setText(f"{fps} {i18n.translate('fps', language)}")
        self.infoStatus.setText(i18n.translate(self._run_state, language))
        self.infoElapsed.setText(self._run_elapsed + i18n.translate(" elapsed", language))

    def _apply_info_scale(self) -> None:
        """Resolve the pane's design geometry at the factor in force (called from `_apply_scale`)."""
        gap = scaled_px(INFO_BLOCK_GAP_PX)
        inset = scaled_px(INFO_CARD_INSET_PX)
        cell_inset = scaled_px(INFO_CELL_INSET_PX)
        hairline = max(1, scaled_px(INFO_HAIRLINE_PX))
        radius = scaled_px(INFO_CARD_RADIUS_PX)
        tag_x = scaled_px(INFO_TAG_INSET_X_PX)
        tag_y = scaled_px(INFO_TAG_INSET_Y_PX)

        self.videoInfoLayout.setSpacing(gap)
        for card in (self.infoFactsCard, self.infoRunCard, self.infoPipelineCard):
            card.set_radius(radius)
        self.infoTransitionLayout.setContentsMargins(inset, inset, inset, inset)
        self.infoTransitionLayout.setSpacing(scaled_px(INFO_COLUMN_GAP_PX))
        self.infoTagLayout.setContentsMargins(tag_x, tag_y, tag_x, tag_y)
        self.infoTagPlate.set_radius(scaled_px(INFO_TAG_RADIUS_PX))
        self.infoFpsLayout.setContentsMargins(tag_x, tag_y, tag_x, tag_y)
        self.infoFpsPlate.set_radius(scaled_px(INFO_PILL_RADIUS_PX))
        for cell_layout in self.infoCellLayouts:
            cell_layout.setContentsMargins(inset, cell_inset, inset, cell_inset)
        for line in self.infoHairlines:
            line.setFixedHeight(hairline)
        for line in self.infoColumnHairlines:
            line.setFixedWidth(hairline)
        self.infoRunLayout.setContentsMargins(inset, inset, inset, inset)
        self.infoRunLayout.setSpacing(scaled_px(INFO_RUN_GAP_PX))

        # The queue card: the four-row body is derived from the list's own row height, and the pin's band is a layout
        # row above it, so a scale change moves the whole card by re-running this one pass.
        self.infoQueueCard.set_radius(radius)
        self.infoQueueLayout.setContentsMargins(inset, inset, inset, inset)
        self.infoQueueLayout.setSpacing(scaled_px(INFO_VALUE_GAP_PX))
        self.infoQueueHeaderLayout.setSpacing(scaled_px(INFO_VALUE_GAP_PX))
        self.infoQueueList.setFixedHeight(self._queue_rows_height())

        # The plan's ribbon: the rail is a painted 2 px line like every other hairline on this pane, and the three
        # gaps inside a stage are the mockup's own (rail->name 8, name->value 4, value->sub 2) - grid rows 1/3/5.
        rail = max(1, scaled_px(INFO_RAIL_HEIGHT_PX))
        self.infoRibbonLayout.setContentsMargins(inset, inset, inset, 0)
        for row, height in ((1, INFO_RAIL_GAP_PX), (3, INFO_VALUE_GAP_PX), (5, INFO_SUB_GAP_PX)):
            self.infoRibbonLayout.setRowMinimumHeight(row, max(0, scaled_px(height)))
        for widgets in self.infoStages.values():
            widgets["rail"].setFixedHeight(rail)
        for arrow in self.infoArrows:
            arrow.setFixedWidth(scaled_px(INFO_ARROW_COLUMN_PX))
        for column in range(1, 2 * len(INFO_STAGE_ORDER), 2):
            self.infoRibbonLayout.setColumnMinimumWidth(column, scaled_px(INFO_ARROW_COLUMN_PX))
        self.infoPipelineRule.setFixedHeight(hairline)
        self.infoFooterLayout.setContentsMargins(inset, cell_inset, inset, cell_inset)
        self.infoFooterLayout.setSpacing(scaled_px(INFO_FOOTER_GAP_PX))
        for pair in self.infoFooterValues.values():
            pair[2].layout().setSpacing(scaled_px(INFO_FOOTER_KV_GAP_PX))

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

        self._apply_info_scale()
        # Re-state the pane at the factor just resolved: the texts are unchanged, but the fragments (units, frames,
        # elapsed) are rebuilt so nothing is left frozen from the previous language/scale pass.
        self._render_video_info(i18n.current_language())

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
        self.sloMoModeCheckBox.setToolTip(i18n.tooltip(
            'Increase video length instead of framerate when interpolating.',
            'Audio and Subtitles will not be transfered to output video.'
        ))
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
        self.upscaleScaleSpinBox.setToolTip(i18n.tooltip(
            'Upscale Scale:',
            '- Custom output scale no matter the model. Will run AI on video frame once, and scale the image to match the multiplier. Useful for 4x models.'
        ))
        self.qualityPresetLabel.setText(QCoreApplication.translate("MainWindow", u"Quality Preset", None))
        self.qualityPresetComboBox.setItemText(0, QCoreApplication.translate("MainWindow", u"Best quality", None))
        self.qualityPresetComboBox.setItemText(1, QCoreApplication.translate("MainWindow", u"Fast", None))
        # VALUE for the backend (`--quality_preset`); the text is a catalogue label.
        self.qualityPresetComboBox.setItemData(0, "best")
        self.qualityPresetComboBox.setItemData(1, "fast")
        self.frameReuse.setText(QCoreApplication.translate("MainWindow", u"Reuse unchanged frames", None))
        self.celReuse.setText(QCoreApplication.translate("MainWindow", u"By cel (ignore grain, dust and scan wobble)", None))
        self.addToRenderQueueButton.setText(QCoreApplication.translate("MainWindow", u" Add to Render Queue", None))
        # The queue card's pin states one catalogue key; the four-row list below it is data, not text.
        self.infoQueuePin.setToolTip(i18n.tooltip(INFO_QUEUE_PIN_TOOLTIP))
        # Last: the pane re-renders from its STORED values, so the nine row names follow the catalogue while the
        # values a loaded video put there stay byte-identical.
        self._render_video_info(language)
