# The Designer's red storage warning, still styled through QSS after the label moved onto the toolkit.
# The label is a `QtLabel` (an adapter over the toolkit's `Text`), so the rule is a no-op: the painter owns the
# pixels and a stock-QSS selector can never reach it.
# provenance: src/ui/pages/download_page.py:157-165 @ 542b55379cc3
# provenance-note: cut at the parent of f4efbab3 (W5: the seven bespoke popups move onto toolkit dialogs); the
#   label was still an adapter there. The migration later dropped the adapter and the rule with it.
#@rule: qss-on-adapter
#@findings: 1

self.low_storage_label = QtLabel(self.backendSelectContainer)
self.low_storage_label.setObjectName(u"low_storage_label")
font8 = QFont()
font8.setBold(True)
self.low_storage_label.setFont(font8)
self.low_storage_label.setStyleSheet(u"QLabel {\n"
"color: red;\n"
"}")
self.verticalLayout_11.addWidget(self.low_storage_label)
