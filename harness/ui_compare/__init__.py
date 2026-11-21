"""One comparison model for UI builds: pages, and the widgets inside them.

Why this package exists: the first version of the harness was three scripts with three partial views - a page table of
pixel shares, a structural diff of the tree, and a per-widget crop tool bolted on after a button that was visibly wrong
scored "1.2% of the page". None of them could answer the only question a migration actually raises: *does this control
look and sit like the one it replaced?*

The model here:

* a **page** is a screenshot plus a dump (JSON) of its widget tree, taken in one walk;
* a **widget** is identified by its page plus its object name (`settings_page/tab_2/output_folder_location`), which
  both builds share because the Designer named everything and the split preserved the names;
* a widget delta carries every axis the eye uses: **geometry**, **text**, **state**, **visibility**, **icon presence**,
  **font size**, **palette colours** and the **share of its own pixels** that differ.

Page-level pixel shares stay - they are how a whole-screen regression is caught - but they are a summary, never the
verdict. The verdict is the widget table.
"""

from __future__ import annotations

from .dumps import current_page_root, flatten, load, parse_aliases, prune_under
from .metrics import abs_rect, ink_share
from .report import format_page_table, format_widget_table
from .widgets import page_deltas, widget_deltas

__all__ = [
    "current_page_root",
    "abs_rect",
    "flatten",
    "format_page_table",
    "format_widget_table",
    "ink_share",
    "load",
    "page_deltas",
    "parse_aliases",
    "prune_under",
    "widget_deltas",
]
