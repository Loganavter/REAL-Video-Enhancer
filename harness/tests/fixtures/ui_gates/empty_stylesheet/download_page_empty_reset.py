# The Designer's `setStyleSheet("")` reset kept after the split: it states no rule, so it can never change a pixel,
# and it reads like theming that does something.
# provenance: src/ui/pages/download_page.py:126-130 @ 9bbcd776ecd9
# provenance-note: the same call is still recorded in ui_gates.KNOWN today - this cut is where it came from.
#@rule: empty-stylesheet
#@findings: 1

self.backendSelectContainer = QWidget(self.downloadPage)
self.backendSelectContainer.setObjectName(u"backendSelectContainer")
self.backendSelectContainer.setStyleSheet(u"")
self.verticalLayout_11 = QVBoxLayout(self.backendSelectContainer)
self.verticalLayout_11.setObjectName(u"verticalLayout_11")
