# The hardcoded combo width: the General tab pinned every combo to the Designer's column and let the row stretch it,
# so a 139 px hint rendered at 230-397 px. The width pin is this gate's finding; the height cap is legitimate.
# provenance: src/ui/pages/general_settings.py:236-245 @ dfcf06e56e17
# provenance-note: cut at the parent of e063afb9 ("Combos: width follows the content, height keeps the Designer
#   cap"), which retired `setMaximumSize(QSize(230, 30))`/`(250, 30)`) and re-keyed this gate's KNOWN list 88 -> 82.
#@rule: raw-chrome-px
#@findings: 1

self.label_12.setObjectName(u"label_12")
apply(self.label_12, BODY_PT)
self.horizontalLayout_11.addWidget(self.label_12)
self.backendComboBox = ComboBox(self.widget_3)
self.backendComboBox.setObjectName(u"backendComboBox")
sizePolicy8.setHeightForWidth(self.backendComboBox.sizePolicy().hasHeightForWidth())
self.backendComboBox.setSizePolicy(sizePolicy8)
self.backendComboBox.setMinimumSize(QSize(0, 0))
self.backendComboBox.setMaximumSize(QSize(230, 30))
self.horizontalLayout_11.addWidget(self.backendComboBox)
