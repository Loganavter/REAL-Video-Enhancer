# A backend module importing the UI: the edge that turns a headless render process into a GUI process. It cannot
# even resolve there (`backend/src` is the `src` package on the backend's own path).
# provenance: NONE
# provenance-note: measured green in this tree's whole history (`git log -S "from src.ui"`, `-S "from src import
#   ui"` and `-S "import src.ui"` over backend/ are empty), and no rule may grow it. Authored to the shape.
#@rule: backend-imports-ui
#@findings: 2

"""A headless renderer that asks the UI for a page: the layering this gate refuses."""

import src.ui.shell.window
from src.ui.pages.settings_page import SettingsPagePage


def build(window_name):
    return SettingsPagePage(), src.ui.shell.window
