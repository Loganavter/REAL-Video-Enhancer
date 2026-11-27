"""Every `path:line` in REPORT.md must still carry the string it quotes. This says which ones do not.

Why a checker and not a proof-read: the emitter matrix in ``REPORT.md`` is 60-odd locators written against one revision
of the backend, and the backend keeps growing above them. A row that quotes the right line at the wrong offset reads
exactly like a correct one - the string is there, the number is plausible - so nothing short of opening the file at that
line catches it. Measured 2026-11-27: all emitter locators were off by 0-300+ lines, ``harness/README.md`` claimed 22
test files where the tree holds 142, and the ledger's generated head claimed 1525 records where the files hold 1569.

What counts as a locator:

* a table row whose first cell is a line spec (``83/85``, ``303``, ``423-430``) under a heading that names the file
  (``### `backend/src/RenderVideo.py` ...``);
* any backticked ``path:line`` in a table cell or in prose.

The anchor is the quoted string in the same row (or sentence), not the number. Placeholders (``N``, ``M``, ``WxH``,
``<name>``, ``...``) are wildcards, so ``phase: video probe N ms (WxH, frames, fps)`` resolves to the ``log(f"phase:
video probe ...`` call even though the code interpolates and splits the literal across two source lines. A match inside
a docstring or a comment is not an emitter: ``BORDERS`` appears in ``format_borders_line``'s docstring and again in its
return value, and only the second one prints.

Fail-closed rules:

* a locator whose file is missing is an ERROR (not a pass);
* a locator whose anchor is quoted nowhere in the file is an ERROR;
* an empty corpus is exit 2, so a parser regression cannot look green;
* prose without a string anchor is verified against the identifier it names (``ProcessTab.renderToPipeThread`` ->
  its ``def``); a locator with neither is listed as unchecked, never silently passed.

Usage:
  python3 harness/scripts/check_report_locators.py                 # fail on any stale locator
  python3 harness/scripts/check_report_locators.py --list          # also list unchecked + matched source
  python3 harness/scripts/check_report_locators.py --tolerance 3   # allow +/-3 lines, report the offset
  python3 harness/scripts/check_report_locators.py --fix           # rewrite the numbers from the anchors
  python3 harness/scripts/check_report_locators.py --json
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DOC = "REPORT.md"
DEFAULT_README = "harness/README.md"
DEFAULT_LEDGER = "harness/docs/RESEARCH-LEDGER.md"
TEST_DIR = "harness/tests"
RECORDS_GLOB = "harness/docs/wsp/verified/records/*.jsonl"
CLAIMS_GLOB = "harness/docs/claims-*.txt"
VERDICTS = "harness/docs/wsp/verdicts.yaml"

#: How many words of source may sit between two anchor tokens. A doc anchor elides an interpolation and keeps the
#: words around it; three source lines of a split f-string stay inside this budget.
MAX_GAP_WORDS = 14
#: Lines a single emitter may span (adjacent literals concatenate at compile time).
WINDOW_WIDTHS = (1, 2, 3)

_SPEC = r"\d+(?:[/-]\d+)*"
_HEADING = re.compile(r"^#{2,4}\s+(.*)$")
_BACKTICK = re.compile(r"`([^`]*)`")
_SECTION_PATH = re.compile(r"`([^`]+\.py)`")
_TABLE_SPEC = re.compile(rf"^{_SPEC}$")
_EXPLICIT = re.compile(rf"^(?P<path>[^\s`]+\.py):(?P<spec>{_SPEC})$")
_IDENTIFIER = re.compile(r"`([A-Za-z_][\w.]*)`")
_SIGNATURE = re.compile(r"^[A-Za-z_][\w.]*\([^)]*\)$")
#: Placeholders the doc uses for a value the code interpolates. Removing them leaves the literal words.
_PLACEHOLDER = re.compile(r"<[^>]*>|\bN\b|\bM\b|\bWxH\b|\bh:mm:ss\b|\b\d+\b|\.\.\.|…")
_WORD = re.compile(r"[a-z0-9_]+")
#: `int8: start/built/refused/...` is a shorthand list, not one line: each alternative is its own anchor.
_ALTERNATIVES = re.compile(r"^(?P<prefix>.*?: )?(?P<run>[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)+)$")


@dataclass(frozen=True)
class Locator:
    """One `path:line` in the doc, with the strings that must be at it."""

    doc_line: int
    path: str
    spec: str
    spec_at: tuple[int, int]
    numbers: tuple[int, ...] = ()
    span: tuple[int, int] | None = None
    anchors: tuple[str, ...] = ()
    identifier: str | None = None
    cell: str = ""

    @property
    def anchorable(self) -> bool:
        return bool(self.anchors or self.identifier)

    @property
    def declared(self) -> tuple[int, ...]:
        if self.span is not None:
            return tuple(range(self.span[0], self.span[1] + 1))
        return self.numbers


@dataclass(frozen=True)
class Hit:
    line: int
    tier: str          # "exact" | "prefix"
    anchor: int        # index into Locator.anchors
    text: str          # the source line, trimmed, for the report


@dataclass(frozen=True)
class Result:
    locator: Locator
    status: str        # ok | stale | unmatched | missing-file
    found: tuple[int, ...] = ()
    offsets: tuple[int, ...] = ()
    tiers: tuple[str, ...] = ()
    detail: str = ""

    @property
    def failed(self) -> bool:
        return self.status != "ok"


# ----------------------------------------------------------------------------------------------------------------------------------
# parsing


def _spec_numbers(spec: str) -> tuple[tuple[int, ...], tuple[int, int] | None]:
    if "-" in spec:
        a, b = (int(x) for x in spec.split("-", 1))
        return ((), (min(a, b), max(a, b)))
    return (tuple(int(x) for x in spec.split("/")), None)


def _anchor_spans(row: str, spec_at: tuple[int, int]) -> tuple[str, ...]:
    """Quoted strings in a row that are emitted text, in order. Not identifiers, paths, signatures or history."""
    out: list[str] = []
    for m in _BACKTICK.finditer(row):
        span = m.group(1).strip()
        if not span or not any(ch.isalpha() for ch in span):
            continue
        if ".py" in span:
            continue
        if _SIGNATURE.match(span):
            continue
        if not any(ch.isspace() for ch in span):
            continue
        if span in {"\\r", '"""'}:
            continue
        if row[max(0, m.start() - 4):m.start()].rstrip().endswith("was"):
            continue          # `(was `DEBUG after hotReload`)` names the string this one replaced
        if m.start() >= spec_at[0] and m.end() <= spec_at[1]:
            continue
        if span not in out:
            out.append(span)
    return tuple(out)


def _identifier_before(text: str, at: int) -> str | None:
    """The last backticked identifier before `at` - the name prose uses for the line it cites."""
    best = None
    for m in _IDENTIFIER.finditer(text, 0, at):
        name = m.group(1)
        if name.endswith(".py") or name == "log.txt":
            continue
        best = name
    return best


def parse_locators(text: str) -> list[Locator]:
    """Every line locator in the document, in reading order. Pure: no filesystem access."""
    lines = text.splitlines()
    out: list[Locator] = []
    section: str | None = None

    def paragraph_at(index: int) -> tuple[str, int]:
        start = index
        while start > 0 and lines[start - 1].strip():
            start -= 1
        end = index
        while end + 1 < len(lines) and lines[end + 1].strip():
            end += 1
        prefix = "\n".join(lines[start:index])
        return "\n".join(lines[start:end + 1]), len(prefix) + (1 if prefix else 0)

    for index, raw in enumerate(lines):
        heading = _HEADING.match(raw)
        if heading:
            found = _SECTION_PATH.search(heading.group(1))
            section = found.group(1) if found else None
            continue
        stripped = raw.strip()
        if not stripped.startswith("|"):
            # prose locator: `path:line`
            for m in _BACKTICK.finditer(raw):
                explicit = _EXPLICIT.match(m.group(1).strip())
                if not explicit:
                    continue
                numbers, span = _spec_numbers(explicit.group("spec"))
                anchors = _anchor_spans(raw, m.span(1))
                identifier = None
                if not anchors:
                    # the name the sentence gives it if that name is in the same sentence, else the paragraph's last
                    sentence_start = max(raw.rfind(". ", 0, m.start()), raw.rfind("; ", 0, m.start())) + 2
                    identifier = _identifier_before(raw[sentence_start:m.start()],
                                                    len(raw[sentence_start:m.start()]))
                    if identifier is None:
                        paragraph, offset = paragraph_at(index)
                        before = paragraph[:offset + m.start()]
                        identifier = _identifier_before(before, len(before))
                out.append(Locator(
                    doc_line=index + 1, path=explicit.group("path"), spec=explicit.group("spec"),
                    spec_at=(m.start(1) + len(explicit.group("path")) + 1, m.end(1)),
                    numbers=numbers, span=span, anchors=anchors, identifier=identifier, cell=stripped))
            continue

        if set(stripped) <= set("|-: "):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if not cells:
            continue
        first = cells[0]
        explicit = _EXPLICIT.match(first.strip("`"))
        if explicit:
            path, spec = explicit.group("path"), explicit.group("spec")
            tick = raw.find("`")
            at = (tick + 1 + len(path) + 1, tick + 1 + len(path) + 1 + len(spec))
        elif section and _TABLE_SPEC.match(first):
            path, spec = section, first
            at = (raw.find(first), raw.find(first) + len(first))
        else:
            continue
        numbers, span = _spec_numbers(spec)
        out.append(Locator(
            doc_line=index + 1, path=path, spec=spec, spec_at=at, numbers=numbers, span=span,
            anchors=_anchor_spans(raw, at), cell=stripped))
    return out


# ----------------------------------------------------------------------------------------------------------------------------------
# matching


def anchor_words(anchor: str) -> tuple[list[str], int]:
    """Literal words of an anchor and how many of them precede its first placeholder.

    ``phase: video probe N ms (WxH, frames, fps)`` -> ``(["phase","video","probe","ms","frames","fps"], 3)``.
    """
    words: list[str] = []
    prefix = 0
    first = True
    for part in _PLACEHOLDER.split(anchor):
        for word in _WORD.findall(part.lower()):
            if len(word) < 2 or word.isdigit():
                continue
            words.append(word)
            if first:
                prefix += 1
        if first and _PLACEHOLDER.search(anchor):
            first = False
    if not words:
        return [], 0
    if prefix == len(words):        # no placeholder at all: the whole anchor is the prefix
        prefix = len(words)
    return words, prefix


def _line_words(line: str) -> list[str]:
    return [w for w in _WORD.findall(line.lower()) if len(w) > 1 or w.isdigit()]


def _match(words: list[str], gap: int, flat: list[tuple[str, int, str]]) -> tuple[int, str] | None:
    """Window position where `words` appear in order with at most `gap` words between them.

    The reported line is the one carrying the most of the anchor, not the window's first line: a ``log(f"render loop:
    ...`` call one line under ``self._phases.first("render-loop")`` must resolve to the call, not to the guard.
    """
    for start in range(len(flat)):
        if flat[start][0] != words[0]:
            continue
        pos = start
        used = [start]
        ok = True
        for want in words[1:]:
            nxt = pos + 1
            while nxt < len(flat) and nxt - pos - 1 <= gap and flat[nxt][0] != want:
                nxt += 1
            if nxt >= len(flat) or nxt - pos - 1 > gap or flat[nxt][0] != want:
                ok = False
                break
            pos = nxt
            used.append(pos)
        if ok:
            counts: dict[int, int] = {}
            for position in used:
                line = flat[position][1]
                counts[line] = counts.get(line, 0) + 1
            line = max(counts, key=lambda ln: (counts[ln], -ln))
            text = next(entry[2] for entry in flat if entry[1] == line)
            return line, text
    return None


def _prose_lines(source: str) -> set[int]:
    """1-based lines inside comments or docstrings: prose that describes an emitter without being one."""
    prose = set()
    for i, line in enumerate(source.splitlines(), 1):
        if line.lstrip().startswith("#"):
            prose.add(i)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return prose
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            for i in range(first.lineno, (first.end_lineno or first.lineno) + 1):
                prose.add(i)
    return prose


def _find(anchor: str, lines: list[str], prose: set[int]) -> list[Hit]:
    """Every non-prose source line that carries the anchor, best tier first."""
    if _ALTERNATIVES.match(anchor.strip()):
        m = _ALTERNATIVES.match(anchor.strip())
        prefix = m.group("prefix") or ""
        hits: list[Hit] = []
        for part in m.group("run").split("/"):
            hits.extend(_find(prefix + part, lines, prose))
        return hits

    words, prefix_len = anchor_words(anchor)
    if not words:
        return []
    if len(words) <= 2:
        # A one- or two-word anchor carries its punctuation: `reuse: ...` must find `reuse:`, not the variable
        # `reuse = (`; `ERROR: ...` must find the print, not the `def log_error` above it.
        head = _PLACEHOLDER.split(re.split(r"\.\.\.|…", anchor)[0])[0].strip()
        if head:
            return [Hit(i, "exact", 0, line.strip()[:160])
                    for i, line in enumerate(lines, 1)
                    if i not in prose and head in line]
    tiers = [("exact", words)]
    if 2 <= prefix_len < len(words):
        tiers.append(("prefix", words[:prefix_len]))

    hits: list[Hit] = []
    seen: set[tuple[int, str]] = set()
    for tier, want in tiers:
        gap = MAX_GAP_WORDS if tier == "exact" else MAX_GAP_WORDS * 2
        for width in WINDOW_WIDTHS:
            for start in range(0, len(lines) - width + 1):
                first = start + 1
                if any(ln in prose for ln in range(first, first + width)):
                    continue
                flat: list[tuple[str, int, str]] = []
                for offset, line in enumerate(lines[start:start + width]):
                    for word in _line_words(line):
                        flat.append((word, first + offset, line.strip()))
                found = _match(want, gap, flat)
                if found is None:
                    continue
                key = (found[0], tier)
                if key in seen:
                    continue
                seen.add(key)
                hits.append(Hit(line=found[0], tier=tier, anchor=0, text=found[1][:160]))
    return hits


def _resolve_identifier(locator: Locator, lines: list[str], tolerance: int = 0) -> Result:
    name = locator.identifier.split(".")[-1]
    pattern = re.compile(rf"^\s*(?:async\s+)?(?:def|class)\s+{re.escape(name)}\b")
    for i, line in enumerate(lines, 1):
        if pattern.match(line):
            offsets = tuple(i - n for n in locator.declared)
            return Result(locator, "ok" if all(abs(o) <= tolerance for o in offsets) else "stale",
                          found=(i,), offsets=offsets, tiers=("def",), detail=f"def {name}")
    return Result(locator, "unmatched", detail=f"no def/class {name} in {locator.path}")


def resolve(root: Path, locator: Locator, tolerance: int = 0) -> Result:
    """Answer one locator against the tree. A missing file or a string quoted nowhere is a failure."""
    path = root / locator.path
    if not path.is_file():
        return Result(locator, "missing-file", detail=f"no such file: {locator.path}")
    source = path.read_text(encoding="utf-8", errors="replace")
    lines = source.splitlines()

    if locator.identifier and not locator.anchors:
        return _resolve_identifier(locator, lines, tolerance)

    prose = _prose_lines(source)
    per_anchor: dict[int, list[Hit]] = {}
    for index, anchor in enumerate(locator.anchors):
        per_anchor[index] = [Hit(h.line, h.tier, index, h.text) for h in _find(anchor, lines, prose)]
    if not any(per_anchor.values()):
        quoted = " / ".join(locator.anchors)
        return Result(locator, "unmatched", detail=f"quoted nowhere: {quoted}")

    if locator.span is not None:
        start, end = locator.span
        inside = sorted({h.line for hits in per_anchor.values() for h in hits if start <= h.line <= end})
        if inside:
            return Result(locator, "ok", found=tuple(inside), offsets=tuple(0 for _ in inside), tiers=("exact",))
        nearest = min((abs(h.line - start), h.line) for hits in per_anchor.values() for h in hits)[1]
        return Result(locator, "stale", found=(nearest,), offsets=(nearest - start,), tiers=("exact",),
                      detail=f"range {start}-{end}")

    # One line number per anchor site; assign each declared number to a distinct hit, nearest first.
    numbers = list(locator.numbers)
    all_hits = [hit for hits in per_anchor.values() for hit in hits]
    chosen: dict[int, Hit] = {}
    used: set[int] = set()
    for n in numbers:
        # An exact anchor beats a prefix hit, but only within 40 lines: a family description (`reuse frame=...`)
        # is the honest anchor at its own site, and a far exact hit must not drag the number across the file.
        candidates = sorted(all_hits, key=lambda h: (abs(h.line - n) + (0 if h.tier == "exact" else 40), h.line))
        pick = next((h for h in candidates if h.line not in used), candidates[0])
        chosen[n] = pick
        used.add(pick.line)
    if len(chosen) < len(numbers):
        missing = [str(n) for n in numbers if n not in chosen]
        return Result(locator, "unmatched", detail=f"no hit for line(s) {', '.join(missing)}")

    covered = {h.anchor for h in chosen.values()}
    offsets = tuple(chosen[n].line - n for n in numbers)
    status = "ok" if all(abs(o) <= tolerance for o in offsets) else "stale"
    detail = "" if len(covered) == len(locator.anchors) else "one anchor not matched by any declared line"
    if detail:
        status = "unmatched"
    return Result(locator, status, found=tuple(chosen[n].line for n in numbers), offsets=offsets,
                  tiers=tuple(chosen[n].tier for n in numbers), detail=detail)


def check(root: Path, text: str, tolerance: int = 0) -> list[Result]:
    """Resolve every anchorable locator; unanchored ones are reported by `unchecked`, not failed."""
    return [resolve(root, loc, tolerance) for loc in parse_locators(text) if loc.anchorable]


def unchecked(text: str) -> list[Locator]:
    return [loc for loc in parse_locators(text) if not loc.anchorable]


# ----------------------------------------------------------------------------------------------------------------------------------
# regeneration


def regenerate(text: str, root: Path, tolerance: int = 0) -> tuple[str, list[tuple[int, str, str]]]:
    """Rewrite every locator's numbers from the strings it quotes. Returns (new text, changes)."""
    lines = text.splitlines(keepends=True)
    changes: list[tuple[int, str, str]] = []
    for result in check(root, text, tolerance=0):
        loc = result.locator
        # A range whose anchor is inside it is a correct range (a block of emitters): keep its shape. Only a range
        # that no longer contains its string is rewritten, and then to the exact line that carries it.
        if result.status == "ok" or not result.found or result.status == "unmatched":
            continue
        new = "/".join(str(n) for n in sorted(set(result.found)))
        if new == loc.spec:
            continue
        line = lines[loc.doc_line - 1]
        start, end = loc.spec_at
        lines[loc.doc_line - 1] = line[:start] + new + line[end:]
        changes.append((loc.doc_line, loc.spec, new))
    return "".join(lines), changes


# ----------------------------------------------------------------------------------------------------------------------------------
# machine counts


@dataclass(frozen=True)
class Claim:
    doc: str
    what: str
    claimed: str
    actual: str
    at: tuple[int, int]

    @property
    def ok(self) -> bool:
        return self.claimed == self.actual


def count_test_files(root: Path) -> tuple[int, int]:
    """(files in the suite, test modules). "Files" is every `*.py` in the suite (conftest included); "modules" is
    `test_*.py`, the pair the README states as "N files / M files that exist today"."""
    tests = root / TEST_DIR
    files = sorted(p for p in tests.glob("*.py"))
    modules = [p for p in files if p.name.startswith("test_")]
    return len(files), len(modules)


def count_records(root: Path) -> int:
    return sum(1 for path in sorted(glob.glob(str(root / RECORDS_GLOB)))
               for line in open(path, encoding="utf-8") if line.strip())


_CLAIM = re.compile(r'@claim\s+(\S+)\s+([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s+(\S+)\s+"([^"]*)"'
                    r'(?:\s+(measured|derived|assumed))?')


def count_claims(root: Path) -> int:
    return sum(len(_CLAIM.findall(line)) for path in sorted(glob.glob(str(root / CLAIMS_GLOB)))
               for line in open(path, encoding="utf-8"))


def count_verdicts(root: Path) -> tuple[int, dict[str, int]] | None:
    try:
        import yaml
    except ImportError:         # the app interpreter has torch and no PyYAML; the locator half must still run
        return None
    data = yaml.safe_load((root / VERDICTS).read_text(encoding="utf-8")) or []
    counts: dict[str, int] = {}
    for row in data:
        status = str(row.get("status", "?"))
        counts[status] = counts.get(status, 0) + 1
    return len(data), counts


def check_counts(root: Path, readme: str, ledger: str) -> list[Claim]:
    """Machine counts stated in prose, compared with the filesystem. A vanished claim is an error, not a pass."""
    files, modules = count_test_files(root)
    records, claims = count_records(root), count_claims(root)
    verdicts = count_verdicts(root)
    out: list[Claim] = []

    for pattern, what, actual in (
        (r"the suite:\s+(\d+)\s+files", "README test files", files),
        (r"The\s+(\d+)\s+files that exist today", "README test modules", modules),
    ):
        m = re.search(pattern, readme)
        if not m:
            out.append(Claim("harness/README.md", what + " (claim missing)", "-", str(actual), (0, 0)))
            continue
        out.append(Claim("harness/README.md", what, m.group(1), str(actual), m.span(1)))

    m = re.search(r"Index:\s+(\d+)\s+records,\s+(\d+)\s+claims,\s+(\d+)\s+verdicts"
                  r"\s+\((\d+)\s+current,\s+(\d+)\s+superseded,\s+(\d+)\s+refuted\)", ledger)
    if not m:
        out.append(Claim(DEFAULT_LEDGER, "ledger index (claim missing)", "-", "-", (0, 0)))
        return out
    actuals = [str(records), str(claims)]
    if verdicts is None:
        # No PyYAML (the app interpreter): the two file counts are still facts, the registry counts are not readable.
        for i, (claimed, actual, name) in enumerate(zip(m.groups()[:2], actuals[:2], ("ledger records", "ledger claims"))):
            out.append(Claim(DEFAULT_LEDGER, name, claimed, actual, m.span(i + 1)))
        return out
    actuals += [str(verdicts[0]), str(verdicts[1].get("current", 0)), str(verdicts[1].get("superseded", 0)),
                str(verdicts[1].get("refuted", 0))]
    names = ["ledger records", "ledger claims", "ledger verdicts", "ledger current", "ledger superseded",
             "ledger refuted"]
    for i, (claimed, actual, name) in enumerate(zip(m.groups(), actuals, names)):
        out.append(Claim(DEFAULT_LEDGER, name, claimed, actual, m.span(i + 1)))
    return out


def fix_counts(root: Path, readme: str, ledger: str) -> tuple[str, str, list[tuple[str, str, str]]]:
    """Rewrite the prose counts that disagree with the files. Returns (readme, ledger, changes)."""
    changes: list[tuple[str, str, str]] = []

    def apply(text: str, claims: list[Claim], name: str) -> str:
        for claim in sorted((c for c in claims if not c.ok and c.at != (0, 0) and c.actual != "?"),
                            key=lambda c: c.at[0], reverse=True):
            text = text[:claim.at[0]] + claim.actual + text[claim.at[1]:]
            changes.append((name, claim.what, f"{claim.claimed} -> {claim.actual}"))
        return text

    readme = apply(readme, check_counts(root, readme, ""), DEFAULT_README)
    ledger = apply(ledger, check_counts(root, "", ledger), DEFAULT_LEDGER)
    return readme, ledger, changes


# ----------------------------------------------------------------------------------------------------------------------------------
# CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(_ROOT))
    ap.add_argument("--doc", default=DEFAULT_DOC)
    ap.add_argument("--readme", default=DEFAULT_README)
    ap.add_argument("--ledger", default=DEFAULT_LEDGER)
    ap.add_argument("--tolerance", type=int, default=0, help="allow the string within +/- N lines, and report it")
    ap.add_argument("--fix", action="store_true", help="rewrite the locators and counts from the files")
    ap.add_argument("--list", action="store_true", help="list unchecked locators and matched source lines")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    doc_path = Path(args.doc) if Path(args.doc).is_absolute() else root / args.doc
    if not doc_path.is_file():
        print(f"error: no document at {doc_path}", file=sys.stderr)
        return 2
    text = doc_path.read_text(encoding="utf-8")
    readme = (root / args.readme).read_text(encoding="utf-8") if (root / args.readme).is_file() else ""
    ledger = (root / args.ledger).read_text(encoding="utf-8") if (root / args.ledger).is_file() else ""

    if args.fix:
        text, changes = regenerate(text, root, args.tolerance)
        count_changes: list[tuple[str, str, str]] = []
        if doc_path.resolve() == (root / DEFAULT_DOC).resolve():
            # the counts live in two other files, so only a fix of the real report may rewrite them
            readme, ledger, count_changes = fix_counts(root, readme, ledger)
        if changes:
            doc_path.write_text(text, encoding="utf-8")
        for name, text_now in ((args.readme, readme), (args.ledger, ledger)):
            if text_now and count_changes:
                (root / name).write_text(text_now, encoding="utf-8")
        if not args.quiet:
            for line, old, new in changes:
                print(f"fix {doc_path.name}:{line}: {old} -> {new}")
            for name, what, how in count_changes:
                print(f"fix {name}: {what}: {how}")
            if not changes and not count_changes:
                print("fix: nothing to rewrite")

    results = check(root, text, args.tolerance)
    locators = parse_locators(text)
    unchecked_rows = [loc for loc in locators if not loc.anchorable]
    claims = check_counts(root, readme, ledger)

    if not results:
        print("error: empty corpus - no `path:line` locator with a quoted string in the document", file=sys.stderr)
        return 2

    bad = [r for r in results if r.failed]
    bad_claims = [c for c in claims if not c.ok]

    if args.json:
        print(json.dumps({
            "locators": len(locators), "checked": len(results), "unchecked": len(unchecked_rows),
            "failures": [{"doc_line": r.locator.doc_line, "path": r.locator.path, "spec": r.locator.spec,
                          "status": r.status, "found": r.found, "offsets": r.offsets, "detail": r.detail}
                         for r in bad],
            "counts": [{"doc": c.doc, "what": c.what, "claimed": c.claimed, "actual": c.actual,
                        "ok": c.ok} for c in claims],
        }, indent=2))
    elif not args.quiet:
        print(f"locators: {len(locators)} ({len(results)} anchored, {len(unchecked_rows)} unchecked) in {doc_path.name}")
        for result in results:
            if result.failed or args.list:
                print(f"  {doc_path.name}:{result.locator.doc_line} {result.locator.path}:{result.locator.spec}"
                      f"  [{result.status}]" + (f" found {result.found} offsets {result.offsets}" if result.failed else
                                                 (f" -> {result.found}" if result.found else "")))
                if result.detail:
                    print(f"      {result.detail}")
        for claim in claims:
            if not claim.ok:
                print(f"  {claim.doc}: {claim.what}: claims {claim.claimed}, files hold {claim.actual}")
        if args.list:
            for loc in unchecked_rows:
                print(f"  unchecked {doc_path.name}:{loc.doc_line} {loc.path}:{loc.spec} (no string anchor)")
        if not bad and not bad_claims:
            print(f"ok: every quoted string is at its line; counts agree with the files")

    return 1 if bad or bad_claims else 0


if __name__ == "__main__":
    raise SystemExit(main())
