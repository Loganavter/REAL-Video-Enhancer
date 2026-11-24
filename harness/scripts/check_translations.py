"""Two questions about the catalogue, asked separately - because they fail differently.

    python3 harness/scripts/check_translations.py            # both, against the baseline
    python3 harness/scripts/check_translations.py --json     # the same as data
    python3 harness/scripts/check_translations.py --update   # rewrite the baseline (refuses growth)

1. **parity** - every locale's table has the same key set, and no value is empty. With one translated language today
   (`ru`) this is cheap, but the question becomes real the moment a second locale appears, and it is the one that
   catches a half-copied table.
2. **code keys** - every string a widget asks the catalogue for EXISTS there. Parity alone was green in BlinkTable while
   the UI showed English, because a widget requested `common.switch.switch_on` and only `..._off` existed
   (`tests/devtools/test_check_translations_code_keys.py` in that repo). Here the equivalent is a `translate(...)` call
   whose English source is missing from `TRANSLATIONS`.

Baseline semantics (a ratchet, like `harness/scripts/ui_gates.py`): the file lists the gaps accepted today, one
English source per line. A NEW missing key fails the run; a baseline entry that is no longer missing is a warning that
the file should shrink - `--update` rewrites it and refuses to grow. The baseline is a record of debt, not permission.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DEFAULT_BASELINE = Path(__file__).with_name("translations-baseline.txt")
#: The roots the app's own strings live in. `harness/` is included on purpose: its scripts print for humans too.
SOURCE_ROOTS = ("src", "harness", "REAL-Video-Enhancer.py")

#: Calls whose argument is a catalogue key. `translate(context, source, ...)` is Qt's; `tr(source)` is the app's.
_TRANSLATE_ATTRS = ("translate", "tr")


def is_label(key: str) -> bool:
    """A key worth translating: it carries a LETTER. `"128"`, `"4"` and `"—"` are values (a bitrate, a level, a dash)
    and a catalogue entry for them would be a bug waiting for a translator to "fix" it into the data path."""
    return any(ch.isalpha() for ch in key)


def _string_literal(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def collect_code_keys(root: Path = _ROOT, roots=SOURCE_ROOTS) -> dict[str, set[str]]:
    """``{source string: {files}}`` for every catalogue lookup in the tree.

    Only STRING LITERALS are keys: a formatted key (`translate(f"...")`) cannot be resolved statically, and the gate
    says so by leaving it out rather than guessing.
    """
    found: dict[str, set[str]] = {}
    targets: list[Path] = []
    for entry in roots:
        path = root / entry
        if path.is_file():
            targets.append(path)
        elif path.is_dir():
            targets.extend(sorted(p for p in path.rglob("*.py") if "__pycache__" not in p.parts))
    for path in targets:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # a tree the gate cannot read is not a translation failure
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = getattr(func, "attr", None) or getattr(func, "id", None)
            if name not in _TRANSLATE_ATTRS:
                continue
            # Qt's `QCoreApplication.translate(context, source, ...)` and the app's own
            # `i18n.translate(source, language=None)` are DIFFERENT signatures: reading the wrong argument made the
            # gate report "ru"/"en" as missing keys because a TEST calls `i18n.translate(key, "ru")`.
            receiver = getattr(func, "value", None)
            receiver_name = getattr(receiver, "id", None) or getattr(receiver, "attr", None)
            if getattr(func, "attr", None) == "translate" and receiver_name in ("QCoreApplication", "QApplication"):
                key = _string_literal(node.args[1]) if len(node.args) > 1 else None
            else:
                key = _string_literal(node.args[0]) if node.args else None
            if key is None or not is_label(key):
                continue
            found.setdefault(key, set()).add(str(path.relative_to(root)))
    return found


def catalogue() -> dict[str, dict[str, str]]:
    from src.ui.i18n import TRANSLATIONS

    return TRANSLATIONS


def parity_problems(table: dict[str, dict[str, str]]) -> list[str]:
    """Every locale carries the same keys, and no value is blank."""
    if not table:
        return ["the catalogue has no locale at all"]
    locales = sorted(table)
    reference = locales[0]
    problems: list[str] = []
    for locale in locales[1:]:
        only_ref = sorted(set(table[reference]) - set(table[locale]))
        only_loc = sorted(set(table[locale]) - set(table[reference]))
        if only_ref:
            problems.append(f"{locale} is missing {len(only_ref)} key(s) that {reference} has, e.g. {only_ref[:3]}")
        if only_loc:
            problems.append(f"{locale} has {len(only_loc)} key(s) that {reference} does not, e.g. {only_loc[:3]}")
    for locale, entries in table.items():
        blank = sorted(k for k, v in entries.items() if not str(v).strip())
        if blank:
            problems.append(f"{locale} has {len(blank)} blank value(s), e.g. {blank[:3]}")
    return problems


def read_baseline(path: Path) -> list[str]:
    """One key per line, JSON-encoded - an HTML tooltip key contains newlines and raw storage would split it."""
    if not path.is_file():
        return []
    keys: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            keys.append(json.loads(stripped))
        except ValueError:
            keys.append(stripped)  # a hand-written line stays readable
    return keys


def report(root: Path = _ROOT, baseline: Path = DEFAULT_BASELINE) -> dict:
    table = catalogue()
    codes = collect_code_keys(root)
    translated = set(table.get("ru", {}))
    missing = sorted(key for key in codes if key not in translated)
    known = set(read_baseline(baseline))
    fresh = [key for key in missing if key not in known]
    stale = sorted(known - set(missing))
    return {
        "locales": sorted(table),
        "keys": {locale: len(entries) for locale, entries in table.items()},
        "code_keys": len(codes),
        "missing": missing,
        "fresh": fresh,
        "stale": stale,
        "parity": parity_problems(table),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--root", default=str(_ROOT))
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--update", action="store_true", help="rewrite the baseline from the tree (refuses growth)")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    baseline = Path(args.baseline)
    data = report(root, baseline)

    if args.update:
        previous = set(read_baseline(baseline))
        grown = [key for key in data["missing"] if key not in previous]
        if previous and grown:
            print(f"refusing to grow the baseline: {len(grown)} new key(s), e.g. {grown[:3]}", file=sys.stderr)
            return 2
        header = ("# Accepted translation gaps, one English source per line. A NEW gap fails the run; this file may\n"
                  "# only shrink (`--update` rewrites it and refuses growth).\n")
        baseline.write_text(header + "".join(json.dumps(key, ensure_ascii=False) + "\n" for key in data["missing"]),
                            encoding="utf-8")
        print(f"baseline written: {len(data['missing'])} gap(s)")
        return 0

    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=1))
    else:
        print(f"catalogue: {data['locales']} keys={data['keys']}")
        print(f"code keys: {data['code_keys']} | missing: {len(data['missing'])} | new: {len(data['fresh'])}"
              f" | stale baseline: {len(data['stale'])}")
        if data["parity"]:
            print("parity:", *data["parity"], sep="\n  ")
        if data["fresh"]:
            print("NEW missing keys (a call site asks for a key nobody translated):")
            for key in data["fresh"][:10]:
                print(f"  {key!r}")
        if data["stale"]:
            print(f"stale baseline entries ({len(data['stale'])}): run --update to shrink the file")
    return 1 if (data["fresh"] or data["parity"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
