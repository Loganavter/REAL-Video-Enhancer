"""The REPORT.md locator gate refuses a number that no longer carries its string.

Written the same way as BlinkTable's checker tests: every case builds its own document (or mutates a copy in memory)
and runs the REAL checker over the REAL tree, so a regression that turns the parser into a no-op - or that starts
accepting any line - fails here instead of passing green.

Red-before, measured on this checkout: at the commit before the regeneration the first test below failed on 33
locators (offsets 1-1723 lines), `harness/README.md` claimed 22 test files where the tree holds 142, and the ledger's
head claimed 1525 records where the files hold 1569. Those three are pinned as `test_*_old_numbers_*` cases.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_report_locators.py"
_ROOT = Path(__file__).resolve().parents[2]


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_report_locators_under_test", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module          # dataclasses resolves the module by name
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def checker():
    return _load_checker()


def _report(checker) -> str:
    return (_ROOT / checker.DEFAULT_DOC).read_text(encoding="utf-8")


# ----------------------------------------------------------------------------------------------- the real document


def test_every_anchored_locator_in_REPORT_md_is_at_its_string(checker):
    results = checker.check(_ROOT, _report(checker))
    assert len(results) >= 40, "the parser lost locators - a silent no-op is not a pass"
    bad = [(r.locator.doc_line, r.locator.path, r.locator.spec, r.status, r.offsets) for r in results if r.failed]
    assert bad == []


def test_the_document_still_has_unanchored_rows_and_they_are_reported_not_passed(checker):
    """A table row with no quoted string cannot be verified; it must be listed, never counted as green."""
    text = _report(checker)
    unchecked = checker.unchecked(text)
    assert len(unchecked) > 0
    assert all(not loc.anchors and not loc.identifier for loc in unchecked)


def test_a_docstring_or_comment_is_not_an_emitter(checker):
    """`BORDERS` appears in `format_borders_line`'s docstring and in its return: only the return prints."""
    text = _report(checker)
    locator = next(loc for loc in checker.parse_locators(text)
                   if loc.doc_line == 40 and "RenderVideo.py" in loc.path)
    result = checker.resolve(_ROOT, locator)
    assert result.status == "ok"
    source = (_ROOT / locator.path).read_text(encoding="utf-8").splitlines()
    # The found lines must be the EMITTERS (a `return`/an append carrying the string), never the docstring that also
    # quotes it. Absolute numbers are the merge's business: this file is edited by other lanes, and a test that pins
    # `89` is a datum asserting a conclusion - it broke the first time two branches met.
    assert result.found, f"resolved to nothing: {result.detail}"
    found_text = [source[line_no - 1] for line_no in result.found]
    assert any(line.lstrip().startswith("return") for line in found_text), (
        f"the emitter's return must be among the found lines: {found_text}")
    for line in found_text:
        assert "BORDERS" in line or "borders" in line.lower(), (
            f"a found line does not carry the anchor: {line!r}")
        assert not line.lstrip().startswith("#"), f"a comment is not an emitter: {line!r}"


def test_placeholders_and_split_literals_are_matched(checker):
    """`phase: video probe N ms (WxH, frames, fps)` is one f-string split over two source lines."""
    text = _report(checker)
    locator = next(loc for loc in checker.parse_locators(text) if "video probe" in " ".join(loc.anchors))
    result = checker.resolve(_ROOT, locator)
    assert result.status == "ok"
    assert result.found, "the split literal matched nothing"
    assert len(result.found) == len(locator.numbers), (
        f"one line per number in the doc: {result.found} vs {locator.numbers}")


def test_identifier_prose_locator_resolves_to_its_def(checker):
    """`ProcessTab.renderToPipeThread` is cited by name, with no quoted string: it must land on the `def`."""
    text = _report(checker)
    locator = next(loc for loc in checker.parse_locators(text) if loc.identifier == "ProcessTab.renderToPipeThread")
    result = checker.resolve(_ROOT, locator)
    assert result.status == "ok"
    source = (_ROOT / locator.path).read_text(encoding="utf-8").splitlines()
    assert source[result.found[0] - 1].strip().startswith("def renderToPipeThread")


# ----------------------------------------------------------------------------------------------- moving one line


def _shift_one(text: str, doc_line: int, delta: int) -> str:
    rows = text.splitlines(keepends=True)
    row = rows[doc_line - 1]
    numbers = __import__("re").search(r"\|\s*(\d[\d/-]*)\s*\|", row)
    assert numbers, f"no line spec on row {doc_line}: {row!r}"
    spec = numbers.group(1)
    head = spec.split("/")[0]
    rows[doc_line - 1] = row[:numbers.start(1)] + str(int(head) + delta) + spec[len(head):] + row[numbers.end(1):]
    return "".join(rows)


def test_one_line_off_is_caught_and_the_offset_is_reported(checker):
    text = _report(checker)
    locator = next(loc for loc in checker.parse_locators(text) if "video probe" in " ".join(loc.anchors))
    shifted = _shift_one(text, locator.doc_line, 1)
    results = {r.locator.doc_line: r for r in checker.check(_ROOT, shifted)}
    result = results[locator.doc_line]
    assert result.status == "stale"
    assert result.offsets == (-1,), result
    assert result.found == tuple(locator.numbers), result


def test_the_cli_fails_on_a_shifted_copy_and_fix_restores_it(checker, tmp_path):
    text = _report(checker)
    locator = next(loc for loc in checker.parse_locators(text) if "video probe" in " ".join(loc.anchors))
    copy = tmp_path / "REPORT.md"
    copy.write_text(_shift_one(text, locator.doc_line, 3), encoding="utf-8")

    assert checker.main(["--root", str(_ROOT), "--doc", str(copy), "--quiet"]) == 1
    assert checker.main(["--root", str(_ROOT), "--doc", str(copy), "--fix", "--quiet"]) == 0
    assert f"| {locator.numbers[0]} |" in copy.read_text(encoding="utf-8")
    assert checker.main(["--root", str(_ROOT), "--doc", str(copy), "--quiet"]) == 0


# ----------------------------------------------------------------------------------------------- fail-closed


def test_a_missing_file_is_an_error_not_a_pass(checker, tmp_path):
    doc = tmp_path / "doc.md"
    doc.write_text("### `backend/src/NoSuchModule.py`\n\n| line | what |\n|---|---|\n| 12 | `phase: gone` |\n",
                   encoding="utf-8")
    results = checker.check(_ROOT, doc.read_text(encoding="utf-8"))
    assert [r.status for r in results] == ["missing-file"]
    assert checker.main(["--root", str(_ROOT), "--doc", str(doc), "--quiet"]) == 1


def test_a_string_quoted_nowhere_is_an_error(checker, tmp_path):
    doc = tmp_path / "doc.md"
    doc.write_text("### `backend/src/RenderVideo.py`\n\n| line | what |\n|---|---|\n"
                   "| 325 | `this string is not in the file` |\n", encoding="utf-8")
    results = checker.check(_ROOT, doc.read_text(encoding="utf-8"))
    assert [r.status for r in results] == ["unmatched"]
    assert checker.main(["--root", str(_ROOT), "--doc", str(doc), "--quiet"]) == 1


def test_an_empty_corpus_exits_two(checker, tmp_path):
    doc = tmp_path / "doc.md"
    doc.write_text("# Nothing to check here\n\nNo locators, no quoted strings.\n", encoding="utf-8")
    assert checker.parse_locators(doc.read_text(encoding="utf-8")) == []
    assert checker.main(["--root", str(_ROOT), "--doc", str(doc), "--quiet"]) == 2


# ----------------------------------------------------------------------------------------------- the machine counts


def test_the_real_counts_agree_with_the_files(checker):
    readme = (_ROOT / "harness/README.md").read_text(encoding="utf-8")
    ledger = (_ROOT / checker.DEFAULT_LEDGER).read_text(encoding="utf-8")
    claims = checker.check_counts(_ROOT, readme, ledger)
    assert claims, "no count claim was found - the checker went blind"
    assert [(c.what, c.claimed, c.actual) for c in claims if not c.ok] == []


def test_the_old_file_count_is_caught(checker):
    """The defect this gate was built for: `22 files` where the tree holds more than a hundred."""
    readme = (_ROOT / "harness/README.md").read_text(encoding="utf-8")
    files_now, modules_now = checker.count_test_files(_ROOT)
    old = readme.replace(f"the suite: {files_now} files", "the suite: 22 files")
    claims = checker.check_counts(_ROOT, old, "")
    files = next(c for c in claims if c.what == "README test files")
    assert not files.ok and files.claimed == "22" and files.actual == str(files_now)
    assert files_now > 100


def test_the_old_record_count_is_caught(checker):
    """The defect this gate was built for: `1525 records` where the files hold 1569."""
    ledger = (_ROOT / checker.DEFAULT_LEDGER).read_text(encoding="utf-8")
    old = ledger.replace("Index: 1569 records", "Index: 1525 records")
    claims = checker.check_counts(_ROOT, "", old)
    records = next(c for c in claims if c.what == "ledger records")
    assert not records.ok and records.claimed == "1525" and records.actual == "1569"


def test_counts_are_rewritten_from_the_files(checker, tmp_path):
    readme = tmp_path / "README.md"
    ledger = tmp_path / "LEDGER.md"
    readme.write_text("tests/ the suite: 22 files\n", encoding="utf-8")
    ledger.write_text("Index: 1525 records, 213 claims, 39 verdicts (23 current, 15 superseded, 1 refuted)\n",
                      encoding="utf-8")
    real_readme, real_ledger, changes = checker.fix_counts(
        _ROOT, readme.read_text(encoding="utf-8"), ledger.read_text(encoding="utf-8"))
    files_now, _ = checker.count_test_files(_ROOT)
    assert f"the suite: {files_now} files" in real_readme
    assert "Index: 1569 records" in real_ledger
    assert len(changes) == 2
