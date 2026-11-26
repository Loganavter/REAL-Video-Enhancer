# A mechanical method split left a body indented after `return True` (1f3878c3): `_open_source_decoder` answered
# "ready" without creating anything, `_source_decoder` stayed None, and the compare's source half was never fed while
# 1538 tests stayed green. The rule exists because a dead branch reads as a working one to every test that mocks the
# seam it lives behind.
# provenance: src/ui/ProcessTab.py:3331-3341 @ 1f3878c3
#@check: unreachable code
#@findings: 1

def _open_source_decoder(self) -> bool:
    """Open the compare's source decoder (own thread, pane-boxed, content time). True when one is ready.

    Split out of `_on_original_requested` so the FIRST switch to a compare mode does not pay for it: opening a 1440p
    file costs a few hundred milliseconds on top of the first seek, which the owner felt as "пролаг больше секунды"
    when switching to side by side (2026-11-26). `_warm_source_decoder` opens it while the preview is settling.
    """
    if getattr(self, "_source_decoder", None) is not None:
        return True
        from PySide6.QtCore import QThread
