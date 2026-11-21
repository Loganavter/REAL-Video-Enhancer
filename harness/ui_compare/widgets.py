"""Pairing two builds of a page into widget deltas - every axis the eye uses, per widget.

A delta is not "differs / does not differ": the table has to say *which* axis, because the fix differs. A moved widget,
a widget whose icon disappeared, a widget whose font shrank and a widget whose plate colour is wrong are four different
regressions, and a share-of-pixels number alone cannot tell them apart (it says "42%" for all four).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .dumps import flatten
from .metrics import abs_rect, box_of, ink_share


@dataclass
class WidgetDelta:
    page: str
    path: str
    name: str
    cls: str
    stock_rect: tuple[int, int, int, int] | None = None
    new_rect: tuple[int, int, int, int] | None = None
    moved: tuple[int, int, int, int] | None = None
    text: tuple[str, str] | None = None
    state: tuple[dict, dict] | None = None
    visibility: tuple[bool, bool] | None = None
    enabled: tuple[bool, bool] | None = None
    icon: tuple[bool, bool] | None = None
    font_px: tuple[int, int] | None = None
    colours: dict[str, tuple[str, str]] = field(default_factory=dict)
    ink_pct: float | None = None
    pruned: bool = False

    @property
    def content_changed(self) -> bool:
        return bool(self.text or self.icon or self.font_px or self.colours or self.state or self.visibility
                    or self.enabled)

    @property
    def severity(self) -> float:
        """Ordering for the table: content regressions first, then the worst ink, then movement."""
        base = 1000.0 if self.content_changed else 0.0
        return base + (self.ink_pct or 0.0) + (min(20.0, sum(abs(v) for v in (self.moved or ()))) / 10.0)

    def summary(self) -> str:
        parts = []
        if self.moved and any(self.moved):
            parts.append(f"moved {self.moved}")
        if self.text:
            parts.append(f"text {self.text[0]!r}->{self.text[1]!r}")
        if self.icon:
            parts.append(f"icon {self.icon[0]}->{self.icon[1]}")
        if self.font_px:
            parts.append(f"font {self.font_px[0]}->{self.font_px[1]}px")
        for key, (a, b) in self.colours.items():
            parts.append(f"{key} {a}->{b}")
        if self.state:
            parts.append(f"state {self.state[0]}->{self.state[1]}")
        if self.visibility:
            parts.append(f"visible {self.visibility[0]}->{self.visibility[1]}")
        if self.enabled:
            parts.append(f"enabled {self.enabled[0]}->{self.enabled[1]}")
        return "; ".join(parts)


def _state(node: dict) -> dict:
    return {key: node[key] for key in ("currentIndex", "value", "checked") if key in node}


def _colours(node: dict) -> dict[str, str]:
    # Only roles a control actually paints from; comparing all nine produced noise on widgets that never read them.
    return {key: value for key, value in (node.get("colors") or {}).items()
            if key in {"Button", "Base", "Window", "Text"}}


def widget_deltas(page: str, stock_tree: dict, new_tree: dict, stock_img, new_img,
                  aliases: dict[str, str] | None = None, ignore_text: list[str] | None = None) -> list[WidgetDelta]:
    ignore_text = ignore_text or []
    left = flatten(stock_tree, aliases)
    right = flatten(new_tree, aliases)
    rows: list[WidgetDelta] = []
    for path in sorted(set(left) & set(right)):
        a, b = left[path], right[path]
        row = WidgetDelta(page=page, path=path, name=path.split("/")[-1], cls=a.get("class", ""))
        rect_a, rect_b = abs_rect(a), abs_rect(b)
        row.stock_rect, row.new_rect = rect_a, rect_b
        row.moved = tuple(y - x for x, y in zip(rect_a, rect_b))
        if a.get("visible") != b.get("visible"):
            row.visibility = (bool(a.get("visible")), bool(b.get("visible")))
        if a.get("enabled") != b.get("enabled"):
            row.enabled = (bool(a.get("enabled")), bool(b.get("enabled")))
        if a.get("text", "") != b.get("text", "") and not any(p in path for p in ignore_text):
            row.text = (a.get("text", ""), b.get("text", ""))
        if _state(a) != _state(b):
            row.state = (_state(a), _state(b))
        if bool(a.get("icon")) != bool(b.get("icon")):
            row.icon = (bool(a.get("icon")), bool(b.get("icon")))
        if a.get("font_px") and b.get("font_px") and a["font_px"] != b["font_px"]:
            row.font_px = (a["font_px"], b["font_px"])
        colours_a, colours_b = _colours(a), _colours(b)
        for key in sorted(set(colours_a) & set(colours_b)):
            if colours_a[key] != colours_b[key]:
                row.colours[key] = (colours_a[key], colours_b[key])
        if stock_img is not None and new_img is not None:
            row.ink_pct = ink_share(box_of(stock_img, rect_a), box_of(new_img, rect_a))
        rows.append(row)
    return rows


def page_deltas(page: str, stock_tree: dict, new_tree: dict, aliases: dict[str, str] | None = None) -> dict:
    """Tree-level counts for one page: what the structural gate prints."""
    left = flatten(stock_tree, aliases)
    right = flatten(new_tree, aliases)
    result = {
        "page": page,
        "nodes": (len(left), len(right)),
        "missing": sorted(set(left) - set(right)),
        "extra": sorted(set(right) - set(left)),
    }
    differences = {"geometry": 0, "text": 0, "visibility": 0, "state": 0, "icon": 0}
    for path in set(left) & set(right):
        a, b = left[path], right[path]
        if a.get("geometry") != b.get("geometry"):
            differences["geometry"] += 1
        if a.get("text", "") != b.get("text", ""):
            differences["text"] += 1
        if a.get("visible") != b.get("visible"):
            differences["visibility"] += 1
        if _state(a) != _state(b):
            differences["state"] += 1
        if bool(a.get("icon")) != bool(b.get("icon")):
            differences["icon"] += 1
    result.update(differences)
    return result
