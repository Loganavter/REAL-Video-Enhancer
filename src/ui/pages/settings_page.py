"""Page `settingsPage` (QWidget), split out of the Designer-generated mainwindow.py.

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
# The page body is a toolkit scroll area and its tab strip the toolkit's tab host: a stock `QScrollArea`/`QTabWidget`
# paints through the widget style (so the page QSS styled them) while these paint themselves and read theme tokens.
# `Button` is the toolkit's `QPushButton`; the page imports no compatibility adapter.
from sli_ui_toolkit.widgets import Button, SurfacePanel, SurfaceScrollArea, TopTabHost
from src.ui.pages.proc_page import TabButtonFactory, TabStripHost, overflow_style
from src.ui.widgets.tab_strip import TAB_HEIGHT, tab_style
from src.ui.pages.handles import SettingsPageHandles
from src.ui import i18n
from src.ui.pages.tab_2 import Tab2Page
from src.ui.pages.tab_3 import Tab3Page
from src.ui.pages.tab_4 import Tab4Page

#: RVE's window QSS scrollbar (`QScrollBar:vertical{width:8}`). A scroll area that RESERVES a bar must reserve this, not
#: the toolkit's default 10: measured on the settings rows, the 2 px difference narrowed every row.
STOCK_SCROLLBAR_WIDTH = 8


class SettingsPagePage:
    """`settingsPage` - placed by its parent with `addWidget`."""

    page_id = "settingsPage"

    def build(self):
        """Build the widget tree and return the page root."""
        # Layer C: no stylesheet. The page body is the toolkit's token-painted container; every rule this sheet carried
        # (the page colour, QLineEdit/QSpinBox and its arrow buttons/QTextEdit/QPushButton/QScrollArea, and the QTabBar
        # block) styled a widget that is now a toolkit control painting from tokens, or the background itself.
        self.settingsPage = SurfacePanel(background_token="surface.page")
        self.settingsPage.setObjectName(u"settingsPage")
        self.verticalLayout_14 = QVBoxLayout(self.settingsPage)
        self.verticalLayout_14.setObjectName(u"verticalLayout_14")
        self.scrollArea = SurfaceScrollArea(self.settingsPage, surface_token=None, content_inset=1,
            scrollbar_width=STOCK_SCROLLBAR_WIDTH)
        self.scrollArea.setObjectName(u"scrollArea")
        self.scrollArea.setMaximumSize(QSize(16777215, 112312))
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(u"scrollAreaWidgetContents")
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 1026, 652))
        self.verticalLayout_18 = QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout_18.setObjectName(u"verticalLayout_18")
        # RVE's `QTabBar::tab` rule, stated as the host's own constants instead of QSS the painter would ignore: the
        # Designer's tab is `padding: 6px` around a 12 pt line, i.e. 29 px tall, and its pane is a square transparent
        # box (`QTabWidget::pane { background: transparent }`, no radius), so `pane_radius=0` instead of the toolkit's
        # rounded 10 px pane (which would also inset the content by 10 px a side).
        # The same strip as the process page: one style and one button factory live in `src/ui/widgets/tab_strip.py`,
        # so the two cannot drift into two looks again. The overflow row's policy and the strip's host half live with
        # the process page's strip (`overflow_style`, `TabStripHost`) for the same reason - one answer for both.
        strip_style = overflow_style(tab_style())
        strip_buttons = TabButtonFactory()
        self.tabWidget = TopTabHost(self.scrollAreaWidgetContents, tab_height=TAB_HEIGHT, pane_radius=0,
                                   pane_inset=(10, 5, 10, 5), chrome_inset=0, tab_radius=8,
                                   pane_fill_token=None, tab_factory=strip_buttons, tab_style=strip_style)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tab_2_page = Tab2Page()
        self.tab_2 = self.tab_2_page.build()
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3_page = Tab3Page()
        self.tab_3 = self.tab_3_page.build()
        self.tabWidget.addTab(self.tab_3, "")
        self.tab_4_page = Tab4Page()
        self.tab_4 = self.tab_4_page.build()
        self.tabWidget.addTab(self.tab_4, "")
        self.verticalLayout_18.addWidget(self.tabWidget)
        self.widget_4 = QWidget(self.scrollAreaWidgetContents)
        self.widget_4.setObjectName(u"widget_4")
        self.horizontalLayout_20 = QHBoxLayout(self.widget_4)
        self.horizontalLayout_20.setObjectName(u"horizontalLayout_20")
        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.horizontalLayout_20.addItem(self.horizontalSpacer_4)
        self.resetSettingsBtn = Button(parent=self.widget_4, text="", variant="surface",
            content_align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            content_padding=(10, 5, 10, 5), corner_radius=10, size=(0, 0))
        self.resetSettingsBtn.setObjectName(u"resetSettingsBtn")
        # QPushButton's own policy: as wide as it needs, exactly as tall as its hint - the toolkit button would
        # otherwise take the whole cell next to the spacer.
        self.resetSettingsBtn.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.horizontalLayout_20.addWidget(self.resetSettingsBtn)
        self.verticalLayout_18.addWidget(self.widget_4)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.verticalLayout_14.addWidget(self.scrollArea)
        self.tabWidget.setCurrentIndex(0)
        self._tab_strip = TabStripHost(self.tabWidget, strip_style, strip_buttons)
        self._handles = SettingsPageHandles(resetSettingsBtn=self.resetSettingsBtn)
        # The page pass needs a QObject to walk; the relay is that object and forwards
        # `retranslate(lang)` to this builder (`src/ui/i18n.py`).
        i18n.bind_page(self, self.settingsPage)
        self.retranslate()
        return self.settingsPage

    def retranslate(self, language: str | None = None) -> None:
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), QCoreApplication.translate("MainWindow", u"Output Settings", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), QCoreApplication.translate("MainWindow", u"Render Settings", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), QCoreApplication.translate("MainWindow", u"RVE Settings", None))
        # The strip measures the real labels, not the placeholders it was built with.
        self._tab_strip.refresh()
        self.resetSettingsBtn.setText(QCoreApplication.translate("MainWindow", u"Reset Settings", None))
