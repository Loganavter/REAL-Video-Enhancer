"""The preview's three view modes: which exist, which of them require a pair, and the one place a mode is set.

Owns: the mode POLICY of this tab. `single` is ONE surface - the result, with the LMB compare flash over it on demand.
`sbs` (result right, original left) and `split` (one picture, the original left of a divider) are a PAIR: both halves
of one moment, or the last matched pair held behind them (`preview/pair.py` is the one writer of that decision).
`pair_required(tab)` is the question a caller asks instead of comparing a mode string itself.

May not touch: the pair writer, the renderer, the register, the decoder, or the mode state - the PANE owns `_view_mode`
and its own `set_view_mode` does the geometry, the re-feed and the standing-source ask. This module forwards, it does
not re-implement. No widget is constructed and no signal is connected: the tab keeps its wiring and its state.

Reads from the tab: `page_handles.proc.previewLabel`, and the pane's public `set_view_mode(mode)` / `view_mode()`.
"""

from __future__ import annotations

from ..state import VIEW_MODES

__all__ = ["PAIR_MODES", "pair_required", "set_view_mode"]

#: The modes that show two halves of one moment. `single` shows one surface (plus the LMB flash), so it needs no pair.
PAIR_MODES = ("sbs", "split")


def _pane(tab):
    """The pane this tab is showing, or None: an object with `set_view_mode`/`view_mode` when it can compare."""
    return getattr(getattr(getattr(tab, "page_handles", None), "proc", None), "previewLabel", None)


def set_view_mode(tab, mode: str) -> None:
    """Give the pane a view mode; `single`, `sbs` and `split` are the only ones, and the first is not a pair.

    All three modes paint from the SAME two pane APIs (`set_result`/`set_source`); what differs is what the pane's one
    writer may hand the renderer. `single` forwards a result and, on an LMB hold, the flash picture the host decoded -
    no pair is formed, so a source that never arrives costs nothing. `sbs`/`split` paint both halves of one moment or
    hold the last matched pair, and a source that never arrives is finally covered by the pair writer's one hard limit
    (`MAX_PAIR_STALE_S`) - never by a lone result on the first frame.

    Entering a compare mode also makes the pane ask the host for the source at the playhead (`originalRequested`, which
    the tab routes through its existing compare path). No second ask is started here: two asks for one moment is the
    decode cost the register exists to remove, and the pane's own ask is the one that names the live playhead.
    """
    if mode not in VIEW_MODES:
        # The pane refuses an unknown mode by contract (`video_preview.py`), and the mode policy is this module's: say
        # the same thing in the same place instead of half-changing the pane and then failing on the renderer.
        raise ValueError(f"unknown view mode {mode!r}: expected one of {', '.join(VIEW_MODES)}")
    pane = _pane(tab)
    if pane is None:
        return
    setter = getattr(pane, "set_view_mode", None)
    if not callable(setter):
        return
    try:
        setter(str(mode))
    except Exception as exc:
        # The log helper lives in the app's `Util`, which drags the runtime's optional imports (`cpuinfo`, `requests`):
        # it is imported where it is used so this module stays importable as plain policy. Same line, same stream as the
        # host's own ("preview: view mode ... failed"), because a mode the pane refused must be legible in `log.txt`.
        from ....Util import log

        log(f"preview: view mode {mode} failed ({exc})")


def pair_required(tab) -> bool:
    """Whether the pane's current mode paints a PAIR (`sbs`/`split`) or one surface (`single`).

    Asked of the PANE, because the pane is where the mode lives: a tab-side copy of the mode would be a second answer
    to "what is on screen" (`PREVIEWER_RULES.md` §1, §7). A host with no compare pane at all (`proc_page.py` falls
    back to an `ImageLabel` on a GPU-less box) has no modes and therefore no pair requirement.
    """
    pane = _pane(tab)
    mode_of = getattr(pane, "view_mode", None)
    if not callable(mode_of):
        return False
    try:
        mode = mode_of()
    except Exception:
        return False
    return str(mode) in PAIR_MODES
