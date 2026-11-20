"""Verdicts: the lifetime of a conclusion, and the refusal of a stale one.

A `Record` answers "what was measured". A claim answers "what do we conclude". Neither answers "is this
conclusion still the current one" - and that gap produced the worst failure of this investigation: four
verdicts from the 15th were assembled into a wall that our own measurements had already replaced.

So a verdict carries `status` (`current` / `superseded` / `refuted`), a mandatory `scope` (the
configuration the statement is true of), a `date`, and - when it is no longer current - the id of what
replaced it. Two rules, enforced here and not in prose:

1. **A supersession must resolve to something current.** A chain that ends in a superseded verdict, or
   loops, is a registry that cannot be read, and is refused at load.
2. **Citing a non-current verdict is refused, not warned about.** A claim that rests on
   `trt-int8-two-pass` after `int8-triton-block` replaced it is the "int8 is blocked for weeks"
   error, mechanically detected.

`scope` is the other half: "per-channel PReLU cannot join an int8 epilogue" is true of a TRT build and
false of our own kernel. Without `scope` those are one sentence, and it travels anywhere.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

STATUSES = ("current", "superseded", "refuted")
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass
class Verdict:
    id: str
    status: str
    scope: str
    date: str
    statement: str = ""
    evidence: str = ""
    superseded_by: str = ""
    note: str = ""
    open_question: str = ""

    @property
    def is_current(self) -> bool:
        return self.status == "current"


def load(data: list[dict]) -> list[Verdict]:
    out = []
    for row in data:
        out.append(Verdict(**{k: (row.get(k) or "") for k in
                              ("id", "status", "scope", "statement", "evidence", "superseded_by",
                               "note", "open_question")},
                           date=str(row.get("date", ""))))
    return out


def resolve(verdicts: list[Verdict], vid: str, hops: int = 10) -> Verdict | None:
    by_id = {v.id: v for v in verdicts}
    seen: set[str] = set()
    cur = by_id.get(vid)
    while cur is not None and cur.superseded_by and hops > 0:
        if cur.id in seen:
            return None                      # cycle
        seen.add(cur.id)
        cur = by_id.get(cur.superseded_by)
        hops -= 1
    return cur


def check(verdicts: list[Verdict]) -> list[str]:
    """Registry-level errors. Empty list = readable."""
    problems: list[str] = []
    ids = [v.id for v in verdicts]
    for dup in sorted({i for i in ids if ids.count(i) > 1}):
        problems.append(f"{dup}: duplicate verdict id")
    by_id = {v.id: v for v in verdicts}
    for v in verdicts:
        if not v.id:
            problems.append("a verdict without an id")
            continue
        if v.status not in STATUSES:
            problems.append(f"{v.id}: status {v.status!r} not one of {STATUSES}")
        if not v.scope.strip():
            problems.append(f"{v.id}: no scope - a statement that does not say what it is true of "
                            f"travels to configurations it was never measured on")
        if not DATE.match(v.date):
            problems.append(f"{v.id}: date {v.date!r} is not YYYY-MM-DD - a supersession chain "
                            f"without dates cannot say which measurement is newer")
        if not v.statement.strip():
            problems.append(f"{v.id}: empty statement (use note for commentary)")
        if v.status != "current":
            if not v.superseded_by:
                problems.append(f"{v.id}: status {v.status} but no superseded_by - a conclusion that "
                                f"is no longer current must name what replaced it")
                continue
            target = by_id.get(v.superseded_by)
            if target is None:
                problems.append(f"{v.id}: superseded_by {v.superseded_by!r} is not a verdict")
                continue
            end = resolve(verdicts, v.superseded_by)
            if end is None:
                problems.append(f"{v.id}: supersession chain loops")
            elif not end.is_current:
                problems.append(f"{v.id}: supersession chain ends at {end.id} which is "
                                f"{end.status} - follow it further")
    return problems


def refuses(verdicts: list[Verdict], cited: list[str]) -> list[str]:
    """Which of the cited verdict ids may not be used as evidence, and why."""
    by_id = {v.id: v for v in verdicts}
    out = []
    for vid in cited:
        v = by_id.get(vid)
        if v is None:
            out.append(f"{vid}: unknown verdict")
            continue
        if not v.is_current:
            end = resolve(verdicts, vid)
            out.append(f"{vid}: status {v.status}; current verdict is "
                       f"{(end.id if end else 'unresolvable')}")
    return out


def current(verdicts: list[Verdict]) -> list[Verdict]:
    return [v for v in verdicts if v.is_current]
