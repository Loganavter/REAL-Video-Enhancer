"""Reading page dumps and pairing them: one keying rule, used by every table.

A dump is `{"geometry_space": "parent-local", "root": <node>}`, one file per page, written by
`src/ui/layout_dump.py`. A node carries class, object name, parent-local geometry, visibility, enabled, layout, text,
current index/value, checked, and now icon presence/size and the colours a widget actually paints.

The pairing key is the path of object names down the tree (`page/container/widget`). Every widget in RVE is named by
the Designer, and the migration preserved those names, so two builds of the same page produce the same keys. Where a
name repeats (Qt's private children), a positional suffix keeps them apart.
"""

from __future__ import annotations

import json


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def parse_aliases(pairs: list[str]) -> dict[str, str]:
    """`--class-alias New=Old`: a deliberate class substitution, so paths still pair and the swap is visible."""
    aliases: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--class-alias expects New=Old, got {pair!r}")
        new, old = pair.split("=", 1)
        aliases[new.strip()] = old.strip()
    return aliases


def _key_name(node: dict, index: int, aliases: dict[str, str] | None) -> str:
    name = node.get("name") or ""
    if name:
        return name
    cls = (aliases or {}).get(node.get("class", ""), node.get("class", "?"))
    return f"~{cls}[{index}]"


def flatten(tree: dict, aliases: dict[str, str] | None = None, prefix: str = "") -> dict[str, dict]:
    """`{path -> node}` with `path` built from object names, so two builds of one page share keys."""
    flat: dict[str, dict] = {}

    def walk(node: dict, path: str, index: int) -> None:
        key = f"{path}/{_key_name(node, index, aliases)}" if path else _key_name(node, index, aliases)
        base = key
        suffix = 2
        while key in flat:                 # Qt's private children repeat names; position keeps them apart
            key = f"{base}#{suffix}"
            suffix += 1
        flat[key] = node
        for child_index, child in enumerate(node.get("children", [])):
            walk(child, key, child_index)

    walk(tree.get("root", {}), prefix, 0)
    return flat


def prune_under(tree: dict, names: set[str]) -> int:
    """Drop the children of substituted widgets: a toolkit combo has no QListView inside, and that is not a defect.

    Returns how many subtrees were pruned so a table can say it happened instead of hiding it.
    """
    pruned = 0

    def walk(node: dict) -> None:
        nonlocal pruned
        if node.get("name") in names:
            if node.get("children"):
                node.pop("children", None)
                pruned += 1
            return
        for child in node.get("children", []):
            walk(child)

    walk(tree.get("root", {}))
    return pruned


def current_page_root(tree: dict) -> str | None:
    """Name of the page the dump was taken with - every dump contains the WHOLE window.

    Without this, the comparison pairs widgets of the *other* pages too (a dump of `nav0` also holds `nav2`'s tree), and
    a page that was current in one build but not the other reads as dozens of moved, 100%-different containers.
    """
    root = tree.get("root", {})

    def find(node: dict, name: str) -> dict | None:
        if node.get("name") == name:
            return node
        for child in node.get("children", []):
            found = find(child, name)
            if found is not None:
                return found
        return None

    stack = find(root, "stackedWidget")
    if stack is None:
        return None
    index = stack.get("currentIndex", 0)
    children = stack.get("children", [])
    # QStackedWidget keeps its pages as direct QWidget children (its private frame is not a QWidget child in this dump).
    pages = [child for child in children if child.get("class") != "QScrollBar"]
    if not (0 <= index < len(pages)):
        return None
    page = pages[index]
    tabs = find(page, "tabWidget") or find(page, "processSettingsContainer")
    if tabs is not None and tabs.get("currentIndex", -1) >= 0:
        tab_pages = [child for child in tabs.get("children", [])
                     if child.get("class") not in {"QScrollBar", "QTabBar", "QStackedWidget"}]
        tab_index = tabs.get("currentIndex", 0)
        if 0 <= tab_index < len(tab_pages):
            return tab_pages[tab_index].get("name") or page.get("name")
    return page.get("name")
