"""Pairing two builds of a page into widget deltas - every axis the eye uses, per widget.

A delta is not "differs / does not differ": the table has to say *which* axis, because the fix differs. A moved widget,
a widget whose icon disappeared, a widget whose font shrank and a widget whose plate colour is wrong are four different
regressions, and a share-of-pixels number alone cannot tell them apart (it says "42%" for all four).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .dumps import flatten_with_origins
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
        """Ordering for the table.

        What the eye notices is ink and lost content (text, icon, font, state). Palette roles are reported but weigh
        little: on this app they are a side effect of the page QSS (each widget class ends up with its own Base/Button
        values), so a role difference with identical pixels is information, not a regression - measured by flipping the
        palette and watching the table stay identical.
        """
        content = 1000.0 if (self.text or self.icon or self.font_px or self.state or self.visibility
                             or self.enabled) else 0.0
        palette = 10.0 if self.colours else 0.0
        return content + palette + (self.ink_pct or 0.0) + (min(20.0, sum(abs(v) for v in (self.moved or ()))) / 10.0)

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
    left, left_origin = flatten_with_origins(stock_tree, aliases)
    right, right_origin = flatten_with_origins(new_tree, aliases)
    rows: list[WidgetDelta] = []
    for path in sorted(set(left) & set(right)):
        a, b = left[path], right[path]
        # The dump stores PARENT-LOCAL geometry, so a widget's screen rect is the sum of its ancestors' offsets. Reading
        # it as absolute put every crop in the wrong place: a container mid-page reported 100 % ink between two builds
        # whose centre pixel is identical, and a pane reported its text "clipped" while the label was intact.
        rect_a = abs_rect(a, *left_origin.get(path, (0, 0)))
        rect_b = abs_rect(b, *right_origin.get(path, (0, 0)))
        # A collapsed widget (0-4 px in either axis) has no ink to compare; cropping it produced "100% different"
        # rows for things like a 33x0 button the Designer never laid out.
        if rect_a[2] < 4 or rect_a[3] < 4:
            continue
        # A widget hidden in BOTH builds is not on the screen this dump was taken of: its geometry is whatever the
        # construction left behind (a page that was never shown keeps its build-time size), and comparing those numbers
        # reports defects nobody can see. Measured: five icon buttons read 36x36 against the stock 50x30 on a page that
        # was hidden while dumping a DIFFERENT tab, and that reading sent a fix chasing a size no screenshot contains.
        if not (a.get("visible") or b.get("visible")):
            continue
        row = WidgetDelta(page=page, path=path, name=path.split("/")[-1], cls=a.get("class", ""))
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
            stock_box = box_of(stock_img, rect_a)
            new_box = box_of(new_img, rect_b)
            if stock_box is None or new_box is None:
                # The widget is outside the captured window (scrolled out or below the fold). `ink_share` reports 100 %
                # for a missing crop, which made every below-the-fold container the "worst" row; a widget that is not on
                # screen is not evidence about the paint.
                continue
            row.ink_pct = ink_share(stock_box, new_box)
        rows.append(row)
    return rows


def page_deltas(page: str, stock_tree: dict, new_tree: dict, aliases: dict[str, str] | None = None) -> dict:
    """Tree-level counts for one page: what the structural gate prints."""
    left, left_origin = flatten_with_origins(stock_tree, aliases)
    right, right_origin = flatten_with_origins(new_tree, aliases)
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
