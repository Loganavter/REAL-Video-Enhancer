"""The two tables: pages (did the screen change?) and widgets (which control, and how?)."""

from __future__ import annotations


def format_widget_table(rows, limit: int = 25, budget: float = 2.0) -> str:
    """`page / widget / class / size / moved / ink % / what changed` - the verdict table."""
    if not rows:
        return "no widgets compared"
    width = max(len(row.name) for row in rows)
    lines = [f"{'page'.ljust(13)}  {'widget'.ljust(width)}  {'class'.ljust(14)}  {'size':>11}  {'ink %':>6}  what changed"]
    for row in rows[:limit]:
        rect = row.stock_rect or (0, 0, 0, 0)
        size = f"{rect[2]}x{rect[3]}"
        ink = "-" if row.ink_pct is None else f"{row.ink_pct:6.2f}"
        lines.append(f"{row.page[:-5].ljust(13)}  {row.name.ljust(width)}  {row.cls[:14].ljust(14)}  "
                     f"{size:>11}  {ink}  {row.summary()}")
    over = [row for row in rows if row.content_changed or (row.ink_pct or 0.0) > budget]
    worst = over[0].name if over else "-"
    lines.append("")
    lines.append(f"{len(rows) - len(over)} of {len(rows)} widgets inside the budget "
                 f"(content identical and ink <= {budget}%); worst: {worst}")
    return "\n".join(lines)


def format_page_table(rows) -> str:
    """`page / mean / >8 %` - the whole-screen summary, never the verdict."""
    if not rows:
        return "no pages compared"
    width = max(len(row["page"]) for row in rows)
    lines = [f"{'page'.ljust(width)}  {'mean':>7}  {'>8 %':>7}  {'max':>4}"]
    for row in rows:
        lines.append(f"{row['page'].ljust(width)}  {row['mean']:7.2f}  {row['over8_pct']:7.3f}  {row['max']:4d}")
    return "\n".join(lines)
