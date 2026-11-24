"""The translation gate answers two different questions, and each has its own falsifiable fixture.

Modelled on BlinkTable's `tests/devtools/test_check_translations_code_keys.py`: every test builds a throwaway tree and
runs the REAL checker over it, so a regression that turns the scanner into a silent no-op fails here instead of passing
green. The two questions:

* **parity** - the locales carry the same key set and no value is blank (a half-copied table);
* **code keys** - the string a widget asks for exists in the table at all. BlinkTable needed this one because parity was
  green while the UI showed English (`common.switch.switch_on` requested, only `..._off` translated).

The baseline is a ratchet: a NEW gap fails, an entry that is no longer a gap only warns, and `--update` refuses to grow
the file. The last test runs the checker over the REAL tree, which is what makes this suite the gate.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_translations.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_translations_under_test", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def checker():
    return _load_checker()


def _tree(tmp_path: Path, code: str) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "page.py").write_text(code, encoding="utf-8")
    return tmp_path


QT_CALL = 'from PySide6.QtCore import QCoreApplication\nlabel = QCoreApplication.translate("MainWindow", u"%s", None)\n'


# -- question 2: the key a widget asks for exists --------------------------------------------------


def test_a_missing_key_is_reported(checker, tmp_path):
    root = _tree(tmp_path, QT_CALL % "Not in the table")
    keys = checker.collect_code_keys(root, roots=("src",))
    assert "Not in the table" in keys
    assert keys["Not in the table"] == {"src/page.py"}


def test_a_value_without_letters_is_not_a_translation_key(checker, tmp_path):
    """`translate("128")` is a bitrate somebody wrapped by mistake; a catalogue entry for it would be a trap."""
    root = _tree(tmp_path, QT_CALL % "128")
    assert checker.collect_code_keys(root, roots=("src",)) == {}


def test_the_two_translate_signatures_are_told_apart(checker, tmp_path):
    """`QCoreApplication.translate(context, source)` vs the app's `i18n.translate(source, language)`.

    Reading the wrong argument reported "ru"/"en" as missing keys, because a test calls
    `i18n.translate(key, "ru")` - the language is the SECOND argument there, not the key.
    """
    code = (
        'from PySide6.QtCore import QCoreApplication\n'
        'from src.ui import i18n\n'
        'a = QCoreApplication.translate("MainWindow", u"Qt source", None)\n'
        'b = i18n.translate("App source")\n'
        'c = i18n.translate("App source", "ru")\n'
    )
    keys = checker.collect_code_keys(_tree(tmp_path, code), roots=("src",))
    assert set(keys) == {"Qt source", "App source"}


def test_a_new_gap_fails_against_the_baseline_and_an_old_one_does_not(checker, tmp_path, monkeypatch, capsys):
    root = _tree(tmp_path, QT_CALL % "Brand new gap")
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("", encoding="utf-8")
    assert checker.main(["--root", str(root), "--baseline", str(baseline)]) == 1, (
        "a key nobody translated must fail the run, not be silently absent")
    baseline.write_text(json.dumps("Brand new gap") + "\n", encoding="utf-8")
    assert checker.main(["--root", str(root), "--baseline", str(baseline)]) == 0


def test_the_baseline_refuses_to_grow(checker, tmp_path):
    root = _tree(tmp_path, QT_CALL % "A gap")
    baseline = tmp_path / "baseline.txt"
    baseline.write_text(json.dumps("Some other gap") + "\n", encoding="utf-8")
    assert checker.main(["--root", str(root), "--baseline", str(baseline), "--update"]) == 2, (
        "--update may shrink the debt, never grow it")
    assert "Some other gap" in baseline.read_text(encoding="utf-8"), "a refused update must not rewrite the file"


def test_a_multiline_key_survives_the_baseline_round_trip(checker, tmp_path):
    """An HTML tooltip key contains newlines; storing it raw split one entry into five."""
    key = "<html><body><p>line one</p>\n<p>line two</p></body></html>"
    root = _tree(tmp_path, QT_CALL % key.replace('"', '\\"').replace("\n", "\\n"))
    baseline = tmp_path / "baseline.txt"
    assert checker.main(["--root", str(root), "--baseline", str(baseline), "--update"]) == 0
    assert checker.read_baseline(baseline) == [key]
    assert checker.main(["--root", str(root), "--baseline", str(baseline)]) == 0


# -- question 1: the locales agree with each other -------------------------------------------------


def test_parity_names_a_locale_that_lags(checker):
    """The report must say which locale LAGS, whichever locale happens to be the reference."""
    problems = checker.parity_problems({"ru": {"a": "А", "b": "Б"}, "de": {"a": "A"}})
    assert any("1 key" in p and "de" in p and "ru" in p for p in problems), (
        f"the report must name both locales and the size of the disagreement: {problems}")


def test_parity_names_a_blank_value(checker):
    problems = checker.parity_problems({"ru": {"a": "А", "b": " "}})
    assert any("blank" in p for p in problems), problems


# -- the real tree is the gate ---------------------------------------------------------------------


def test_the_real_tree_passes_its_own_baseline(checker):
    """0 new gaps on the shipped tree: this is the assertion a future wave must not break."""
    data = checker.report()
    assert data["parity"] == [], data["parity"]
    assert data["fresh"] == [], f"new untranslated keys: {data['fresh'][:3]}"
    assert data["stale"] == [], "the baseline lists gaps that no longer exist - run --update to shrink it"
