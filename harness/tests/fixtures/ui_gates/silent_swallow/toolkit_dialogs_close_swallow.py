# A toast close that swallows any exception with no comment and no log: the failure leaves no trace, so the next
# bug report about a toast that did not close has nothing to grep.
# provenance: src/ui/dialogs/toolkit_dialogs.py:381-389 @ 79520cbdf5b3
# provenance-note: cut at the parent of 2d57fc96 (the shell loses its last stylesheet), where the file still
#   carried this handler; three of its siblings are recorded in ui_gates.KNOWN today.
#@rule: silent-swallow
#@findings: 1

def close(self) -> bool:
    toast_id = getattr(self, "_toast_id", None)
    if toast_id is not None:
        self._toast_id = None
        try:
            self._manager.close_toast(toast_id)
        except Exception:
            pass
    return super().close()
