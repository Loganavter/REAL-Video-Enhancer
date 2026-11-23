# The tab-strip overlay bar: the process strip was a stock `QTabWidget`, and the page then had to decide bar
# visibility by hand (`hide when maximum is zero`), which is why a strip that fitted could still paint a bar.
# The fix is the toolkit host (`TopTabHost(tab_factory=..., tab_style=TabStyle(scroll_policy=...))`), where the
# bar follows Qt's own ScrollBarPolicy.
# provenance: src/ui/pages/proc_page.py:345-354 @ 9bbcd776ecd9
# provenance-note: W3 split the Designer's compiled output into 13 page modules; this is the strip it produced.
#   The stock widget was replaced during the toolkit migration and cannot come back without this gate failing.
#@rule: stock-widget
#@findings: 1

self.processSettingsContainer = QTabWidget(self.widget_2)
self.processSettingsContainer.setObjectName(u"processSettingsContainer")
sizePolicy3.setHeightForWidth(self.processSettingsContainer.sizePolicy().hasHeightForWidth())
self.processSettingsContainer.setSizePolicy(sizePolicy3)
self.processSettingsContainer.setMinimumSize(QSize(0, 0))
self.processSettingsContainer.setMaximumSize(QSize(16777215, 16777215))
self.generalSettings_page = GeneralSettingsPage()
self.generalSettings = self.generalSettings_page.build()
self.processSettingsContainer.addTab(self.generalSettings, "")
self.tab_6_page = Tab6Page()
