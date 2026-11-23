# The 8-vs-6 px scrollbar width, as this gate can see it: a bar pinned to 6 px while the window QSS states 8
# (`QScrollBar:vertical{max-width:8px}`). Two pixels come off every row that carries a bar.
# provenance: NONE
# provenance-note: the measured 8-vs-6 drift belongs to the toolkit's own bar sweep (`TabStyle`/reserve row), whose
#   checkout is outside this cell's jail; no RVE module ever pinned 6 px in history (`git log -S "width:6"` and
#   `-S scrollbar_width=6` are empty). This fixture is authored to the shape ui_gates can refuse, not claimed as
#   a historical line: the honest provenance is the gap, so it says NONE.
#@rule: raw-chrome-px
#@findings: 1

class Page:
    def build(self):
        self.bar = QScrollBar(Qt.Orientation.Vertical, self.widget)
        self.bar.setFixedWidth(6)      # the stock QSS bar is 8 px
