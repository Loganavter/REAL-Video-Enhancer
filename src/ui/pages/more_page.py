"""Page `morePage` (QWidget), split out of the Designer-generated mainwindow.py.

Hand-maintained since 2026-11-23: the tree is the Designer's, the controls are the toolkit's. The Designer is no longer
a reference in this tree - the fixture, the splitter that emitted these pages and every "matches the Designer" gate
were removed - so this page answers to `harness/tests/test_toolkit_native.py`: the toolkit's controls and type scale,
`layout.addStretch()` instead of a `QSpacerItem`, and no new `name_<N>` object name or wrapper `QWidget` beyond the
counts that gate records.
"""

from __future__ import annotations

from PySide6.QtWidgets import QGridLayout
import resources_rc
import resources_rc

from sli_ui_toolkit.widgets import SurfacePanel


class MorePagePage:
    """`morePage` - placed by its parent with `addWidget`."""

    page_id = "morePage"

    def build(self):
        """Build the widget tree and return the page root."""
        # Layer C: the page body is the toolkit's token-painted container, not a widget with a stylesheet. The colour is
        # `surface.page` (#16191d in RVE's palette) - the same value the deleted rule carried.
        self.morePage = SurfacePanel(background_token="surface.page")
        self.morePage.setObjectName(u"morePage")
        self.gridLayout_6 = QGridLayout(self.morePage)
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        return self.morePage
