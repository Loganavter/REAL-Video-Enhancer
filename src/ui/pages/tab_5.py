"""Page `tab_5` (QWidget), split out of the Designer-generated mainwindow.py.

Hand-maintained since 2026-11-23: the tree is the Designer's, the controls are the toolkit's. The Designer is no longer
a reference in this tree - the fixture, the splitter that emitted these pages and every "matches the Designer" gate
were removed - so this page answers to `harness/tests/test_toolkit_native.py`: the toolkit's controls and type scale,
`layout.addStretch()` instead of a `QSpacerItem`, and no new `name_<N>` object name or wrapper `QWidget` beyond the
counts that gate records.
"""

from __future__ import annotations

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import (QHBoxLayout, QSizePolicy)
import resources_rc

from sli_ui_toolkit.widgets import SurfacePanel, ConsoleView
import resources_rc
from src.ui.typography import apply, SMALL_PT
from src.ui.pages.handles import LogsHandles
from src.ui import i18n


class Tab5Page:
    """`tab_5` - placed by its parent with `addTab`."""

    page_id = "tab_5"

    def build(self):
        """Build the widget tree and return the page root."""
        self.tab_5 = SurfacePanel(background_token="surface.page")
        self.tab_5.setObjectName(u"tab_5")
        self.horizontalLayout_10 = QHBoxLayout(self.tab_5)
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        # Layer B: a log pane is the toolkit's painted console. ConsoleView carries the three QTextEdit calls the glue
        # makes - setPlainText, setHtml (its plain text) and verticalScrollBar().setValue(maximum()) - which is why this
        # widget could finally leave stock (src/ui/ProcessTab.py:481-485).
        self.renderOutput = ConsoleView(self.tab_5)
        self.renderOutput.setObjectName(u"renderOutput")
        # An empty document at build time (never in `retranslate`, which used to wipe the user's log on a switch).
        self.renderOutput.setHtml("")

        sizePolicy10 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        sizePolicy10.setHorizontalStretch(0)
        sizePolicy10.setVerticalStretch(0)
        sizePolicy10.setHeightForWidth(self.renderOutput.sizePolicy().hasHeightForWidth())
        self.renderOutput.setSizePolicy(sizePolicy10)
        apply(self.renderOutput, SMALL_PT)
        # Content that continues past the pane's edge fades instead of being cut by a hard line, and the log keeps
        # itself pinned to the bottom while the user has not scrolled away (owner, 2026-11-22). Both live in the console:
        # the fade is one painted gradient, the follow is the console's own scroll state.
        self.renderOutput.set_edge_fade_enabled(True)
        self.renderOutput.set_max_lines(4000)
        self.horizontalLayout_10.addWidget(self.renderOutput)
        self._handles = LogsHandles(renderOutput=self.renderOutput)
        # The page pass needs a QObject to walk; the relay is that object and forwards
        # `retranslate(lang)` to this builder (`src/ui/i18n.py`).
        i18n.bind_page(self, self.tab_5)
        self.retranslate()
        return self.tab_5

    def retranslate(self, language: str | None = None) -> None:
        # The pane holds RUNTIME text (the render log), not a label: it has nothing to translate, and the empty
        # document this method used to set through the catalogue wiped the user's log on every language switch.
        pass

