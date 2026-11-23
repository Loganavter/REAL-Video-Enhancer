# A theme colour read while the module is imported: a literal hex bypasses the token table, and `get_color` runs
# before the theme that is installed later, so neither can follow a theme change.
# provenance: NONE
# provenance-note: no RVE page ever carried a module-level `get_color`/literal `QColor` (`git log -S` over
#   src/ui is empty); the rule started green and stays green, so this fixture is authored to its shape. The
#   order-dependence it guards is the toolkit's scrollbar sweep, cited in the rule's own docstring.
#@rule: theme-read-at-build
#@findings: 2

PAGE_BACKGROUND = QColor("#1f232a")
ACCENT = ThemeManager.get_instance().get_color("accent")
