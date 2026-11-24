"""Page `homePage` (QWidget), split out of the Designer-generated mainwindow.py.

Hand-maintained since 2026-11-23: the tree is the Designer's, the controls are the toolkit's. The Designer is no longer
a reference in this tree - the fixture, the splitter that emitted these pages and every "matches the Designer" gate
were removed - so this page answers to `harness/tests/test_toolkit_native.py`: the toolkit's controls and type scale,
`layout.addStretch()` instead of a `QSpacerItem`, and no new `name_<N>` object name or wrapper `QWidget` beyond the
counts that gate records.
"""

from __future__ import annotations

from PySide6.QtCore import (QCoreApplication, QSize, Qt)
from PySide6.QtGui import (QIcon, QPixmap)
from PySide6.QtWidgets import (QGridLayout, QHBoxLayout, QSizePolicy, QSpacerItem,
    QWidget)
import resources_rc
import resources_rc
from src.ui.typography import apply, DISPLAY_PT, GROUP_TITLE_PT

# Layer B3 (2026-11-21): the logo is the page's only stock pixmap label, and `ImageLabel` is its toolkit counterpart
# (Qt-shaped already: `setPixmap`/`setScaledContents`/`setText`). The Designer's `QLabel` paints a pixmap-only label at
# `AlignLeft|AlignVCenter`; `ImageLabel` defaults to centre, so the alignment is stated to keep the logo where it is
# (measured offscreen: identical pixels at this alignment, mean 0.00 against the stock label).
#
# The page builds the toolkit's own controls: `Text` where the Designer had a `QLabel`, `Button` where it had a
# `QPushButton`. Nothing here needs a Qt-shaped compatibility adapter.
from sli_ui_toolkit.widgets import Button, ImageLabel, RichTextView, SurfacePanel, Text
from sli_ui_toolkit.ui.managers.ui_scale import scaled_px

from src.ui.scale_follower import ScaleFollower
from src.ui.pages.handles import HomePageHandles
from src.ui import i18n

#: The two icon buttons' design box: the width the text beside them leaves, and the glyph the toolkit paints. Same
#: split as the download page - the box is host geometry and takes the factor here; the glyph is a design value the
#: toolkit scales once at paint time.
HOME_ICON_MAX_WIDTH_PX = 55
HOME_ICON_PX = 35

#: The buttons' content inset: design px the toolkit scales at use time, named once instead of repeated per button.
HOME_ICON_CONTENT_PADDING = (10, 5, 10, 5)


class HomePagePage:
    """`homePage` - placed by its parent with `addWidget`."""

    page_id = "homePage"

    def build(self):
        """Build the widget tree and return the page root."""
        # The two icon buttons' box: width only (the height side is `QWIDGETSIZE_MAX` because the row decides it).
        home_icon_box = QSize(scaled_px(HOME_ICON_MAX_WIDTH_PX), 16777215)
        home_icon_size = QSize(HOME_ICON_PX, HOME_ICON_PX)
        self.homePage = SurfacePanel(background_token="surface.page")
        self.homePage.setObjectName(u"homePage")
        # Layer C: the page body paints itself; its buttons are toolkit plates. The two read-only panes are the
        # toolkit's `RichTextView` (token plate, 10 px radius, padding 10/5) - the colour and padding the deleted
        # `QTextEdit` rule carried.
        self.gridLayout_2 = QGridLayout(self.homePage)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.horizontalLayout_28 = QHBoxLayout()
        self.horizontalLayout_28.setObjectName(u"horizontalLayout_28")
        self.widget_11 = QWidget(self.homePage)
        self.widget_11.setObjectName(u"widget_11")
        self.horizontalLayout_29 = QHBoxLayout(self.widget_11)
        self.horizontalLayout_29.setObjectName(u"horizontalLayout_29")
        self.horizontalSpacer_14 = QSpacerItem(466, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.horizontalLayout_29.addItem(self.horizontalSpacer_14)
        self.label_13 = Text(parent=self.widget_11)
        self.label_13.setObjectName(u"label_13")
        self.horizontalLayout_29.addWidget(self.label_13)
        self.kofiBtn = Button(parent=self.widget_11, text="", variant="surface",
            content_align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            content_padding=HOME_ICON_CONTENT_PADDING, corner_radius=10, size=(0, 0))
        self.kofiBtn.setObjectName(u"kofiBtn")
        # QPushButton's own policy (as wide as it needs, exactly as tall as its hint): the toolkit button is happy to
        # fill the cell, which turns an icon button into a slab when the row is taller than its contents. `size=(0, 0)`
        # above drops the standalone 36x36 control size; this states the row's share.
        self.kofiBtn.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.kofiBtn.setMaximumSize(home_icon_box)
        icon = QIcon()
        icon.addFile(u":/icons/icons/coffee.svg", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.kofiBtn.setIcon(icon)
        self.kofiBtn.setIconSize(home_icon_size)
        # `setCheckable(False)` states the button's own default, and `Button` has no such setter (toggle mode is the
        # constructor's `toggle=`), so the Designer's line is a no-op here.
        self.horizontalLayout_29.addWidget(self.kofiBtn)
        self.label_16 = Text(parent=self.widget_11)
        self.label_16.setObjectName(u"label_16")
        self.horizontalLayout_29.addWidget(self.label_16)
        self.githubBtn = Button(parent=self.widget_11, text="", variant="surface",
            content_align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            content_padding=HOME_ICON_CONTENT_PADDING, corner_radius=10, size=(0, 0))
        self.githubBtn.setObjectName(u"githubBtn")
        self.githubBtn.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.githubBtn.setMaximumSize(home_icon_box)
        icon1 = QIcon()
        icon1.addFile(u":/icons/icons/github.svg", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.githubBtn.setIcon(icon1)
        self.githubBtn.setIconSize(home_icon_size)
        self.horizontalLayout_29.addWidget(self.githubBtn)
        self.horizontalLayout_28.addWidget(self.widget_11)
        self.gridLayout_2.addLayout(self.horizontalLayout_28, 4, 0, 1, 1)
        self.widget_8 = QWidget(self.homePage)
        self.widget_8.setObjectName(u"widget_8")
        self.gridLayout_7 = QGridLayout(self.widget_8)
        self.gridLayout_7.setObjectName(u"gridLayout_7")
        self.label_2 = ImageLabel(self.widget_8,
            alignment=Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setPixmap(QPixmap(u":/icons/icons/logo-v2.svg"))
        self.label_2.setScaledContents(False)
        self.gridLayout_7.addWidget(self.label_2, 0, 0, 1, 1)
        self.label = Text(parent=self.widget_8)
        self.label.setObjectName(u"label")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        apply(self.label, DISPLAY_PT, family="Fira Mono")
        # `Text` re-derives its own size policy on every style pass (Fixed vertically unless it word-wraps), so the
        # Designer's Preferred/Expanding is stated after `apply`, the pass that would otherwise drop it.
        self.label.setSizePolicy(sizePolicy2)
        self.gridLayout_7.addWidget(self.label, 0, 1, 1, 1)
        self.gridLayout_2.addWidget(self.widget_8, 1, 0, 1, 1)
        self.horizontalLayout_30 = QHBoxLayout()
        self.horizontalLayout_30.setObjectName(u"horizontalLayout_30")
        self.changeLogText = RichTextView(self.homePage, link_token="link")
        self.changeLogText.setObjectName(u"changeLogText")
        self.changeLogText.setReadOnly(True)
        self.horizontalLayout_30.addWidget(self.changeLogText)
        self.systemInfoText = RichTextView(self.homePage, link_token="link")
        self.systemInfoText.setObjectName(u"systemInfoText")
        apply(self.systemInfoText, GROUP_TITLE_PT)
        self.systemInfoText.setReadOnly(True)
        self.horizontalLayout_30.addWidget(self.systemInfoText)
        self.gridLayout_2.addLayout(self.horizontalLayout_30, 2, 0, 1, 1)
        # The glue's typed edge over this page; every field is required, so a widget that stops being built fails here
        # (TypeError) instead of surfacing as an AttributeError or a silent `getattr(..., None)` at fetch time.
        self._handles = HomePageHandles(
            changeLogText=self.changeLogText,
            systemInfoText=self.systemInfoText,
            githubBtn=self.githubBtn,
            kofiBtn=self.kofiBtn,
        )
        # The page pass needs a QObject to walk; the relay is that object and forwards
        # `retranslate(lang)` to this builder (`src/ui/i18n.py`).
        i18n.bind_page(self, self.homePage)
        self.retranslate()
        # The two icon boxes are one design width: re-stated at the factor in force now and on every later change, so
        # a live `scale:<factor>` moves them with the glyph the toolkit paints.
        self._icon_button_scale = ScaleFollower(self.homePage, self._apply_icon_button_geometry)
        return self.homePage

    def _apply_icon_button_geometry(self) -> None:
        """State the two icon buttons' max width at the factor in force now.

        Width only: the row owns the height, and `QWIDGETSIZE_MAX` is Qt's "unbounded". The icon itself stays a design
        value - the toolkit multiplies it once when it paints (see `HOME_ICON_PX`).
        """
        box = QSize(scaled_px(HOME_ICON_MAX_WIDTH_PX), 16777215)
        icon = QSize(HOME_ICON_PX, HOME_ICON_PX)
        for button in (self.kofiBtn, self.githubBtn):
            button.setMaximumSize(box)
            button.setIconSize(icon)

    def retranslate(self, language: str | None = None) -> None:
        self.label_13.setText(QCoreApplication.translate("MainWindow", u"Support Me!", None))
        self.kofiBtn.setToolTip(QCoreApplication.translate("MainWindow", u"Support Me!", None))
        self.kofiBtn.setStatusTip(QCoreApplication.translate("MainWindow", u"\"Support Me!\"", None))
        self.kofiBtn.setText("")
        self.label_16.setText(QCoreApplication.translate("MainWindow", u"GitHub Page", None))
        self.githubBtn.setText("")
        # `setText("")` is gone with the stock label: an ImageLabel has no text channel ("no text" is its default)
        # The product name, not a label to translate.
        self.label.setText("REAL Video Enhancer")
        self.changeLogText.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"li.unchecked::marker { content: \"\\2610\"; }\n"
"li.checked::marker { content: \"\\2612\"; }\n"
"</style></head><body style=\" font-family:'Adwaita Sans'; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:'Cantarell';\"><br /></p></body></html>", None))
