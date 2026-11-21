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


#: Toolkit containers that replace a stock one and change the path to the pages inside them.
#: `TopTabHost` puts its pages under `TopTabPane/TopTabStack`, where `QTabWidget` had `qt_tabwidget_stackedwidget`;
#: without this rewrite every widget inside a migrated tab stops pairing with its stock twin (measured: the settings
#: page's `nav3-tab0` kept 0 of 315 paired widgets, so the widget table said "no widgets compared").
_PATH_REWRITES = (
    ("/TopTabPane/TopTabStack/", "/qt_tabwidget_stackedwidget/"),
)


def _canonical(path: str) -> str:
    for new, stock in _PATH_REWRITES:
        if new in path:
            return path.replace(new, stock)
    return path


def flatten(tree: dict, aliases: dict[str, str] | None = None, prefix: str = "") -> dict[str, dict]:
    """`{path -> node}` with `path` built from object names, so two builds of one page share keys."""
    flat: dict[str, dict] = {}

    def walk(node: dict, path: str, index: int) -> None:
        key = f"{path}/{_key_name(node, index, aliases)}" if path else _key_name(node, index, aliases)
        key = _canonical(key)
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
    """Name of the page that was current when the dump was written - every dump contains the WHOLE window.

    Resolved by VISIBILITY, not by `currentIndex`: a stack hides every page but one, and an index is only meaningful
    when it was written by the same walker version that wrote the names. A baseline captured with an older walker had
    its indices shifted by one, which made a dump of one page report another page's widgets (measured: "Advanced"
    listed the Home page's buttons).
    """

    def find(node: dict, name: str) -> dict | None:
        if node.get("name") == name:
            return node
        for child in node.get("children", []):
            found = find(child, name)
            if found is not None:
                return found
        return None

    def visible_page(container: dict | None) -> str | None:
        if container is None:
            return None
        pages = [child for child in container.get("children", []) if child.get("class") != "QScrollBar"]
        for page in pages:
            if page.get("visible"):
                return page.get("name")
        index = container.get("currentIndex", 0)
        return pages[index].get("name") if 0 <= index < len(pages) else None

    root = tree.get("root", {})
    stack = find(root, "stackedWidget")
    page_name = visible_page(stack)
    if page_name is None:
        return None
    page = find(root, page_name) or {}
    tabs = find(page, "tabWidget") or find(page, "processSettingsContainer")
    if tabs is not None:
        # The toolkit's `TopTabHost` keeps its pages in a `QStackedWidget` named `TopTabStack` next to the tab strip;
        # without this branch the strip itself (`TopTabBar`, the first visible child) would be reported as the page,
        # and every widget comparison against a stock dump paired nothing (measured after a tab-host migration).
        holder = find(tabs, "qt_tabwidget_stackedwidget") or find(tabs, "TopTabStack") or tabs
        tab_name = visible_page(holder)
        if tab_name:
            return tab_name
    return page_name
