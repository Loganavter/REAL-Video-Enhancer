"""Every gate proves it can be red: one fixture per rule, cut from the history that carried the violation.

`test_ui_gates.py` and `test_ui_diff.py` prove the *checks* on synthetic modules and captures. This file proves the
*gates*: each rule is run through its own entry point over a fixture it MUST fail on, and then the same gate is run
over the current tree and must exit zero. A rule with no fixture, a fixture that stopped being red, and a gate that
cannot be green are three different failures, and each has its own test here.

The fixtures live under `harness/tests/fixtures/` and carry a header: `provenance:` names the commit and the lines the
body was cut from, or says `NONE`; `@findings` is how many findings the check must report. A history claim is verified
against `git show`, so an authored body that cites a commit is refused - see `fixtures/README.md` for the contract.

Nothing here needs Qt, a display, cv2 or torch: the gates are AST walks and a page-delta table. The ui_diff CLI case
reads two small PNGs (Pillow/numpy) because that is how the page metric gets its rows.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from harness import astcheck
from harness.scripts import ui_diff, ui_gates

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures"

#: `#@rule: qss-on-adapter` and `#@findings: 1` - the machine-readable half of the header.
_DIRECTIVE_RE = re.compile(r"^#\s*@(?P<key>[a-z]+):\s*(?P<value>.+)$")

#: `# provenance: path/to/file.py:12-34 @ 0123abcd` - the toolkit's fixture-header shape, one commit and one range.
_PROVENANCE_RE = re.compile(r"^(?P<path>.+):(?P<start>\d+)-(?P<end>\d+) @ (?P<sha>[0-9a-f]{7,40})$")

_FREE_TEXT_RE = re.compile(r"^#\s*(?P<key>provenance|provenance-note):\s*(?P<value>.+)$")

PY_FIXTURES = sorted(p for p in FIXTURES.rglob("*.py") if p.is_file())


# --- the header contract -----------------------------------------------------------------------------------

def split_fixture(path: Path) -> tuple[dict[str, str], str]:
    """`(header, body)` for one fixture: the leading comment block, a blank line, then the code.

    The header carries free prose lines and the directives; the body is everything after the separator, which is what
    a provenance claim is checked against.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    header: dict[str, str] = {}
    index = 0
    while index < len(lines) and lines[index].startswith("#"):
        directive = _DIRECTIVE_RE.match(lines[index])
        free = _FREE_TEXT_RE.match(lines[index])
        if directive:
            header[directive["key"]] = directive["value"].strip()
        elif free:
            header[free["key"]] = free["value"].strip()
        index += 1
    assert index < len(lines) and not lines[index].strip(), f"{path}: the header must end with one blank line"
    body = "\n".join(lines[index + 1:])
    assert body.strip(), f"{path}: a fixture without a body proves nothing"
    return header, body.rstrip("\n") + "\n"


def _undent(chunk: str) -> str:
    """Remove the slice's own indentation, taken from its first non-blank line.

    `textwrap.dedent` cannot do this for a cut: a Designer-style `setStyleSheet(u"..." \\n "color: red;\\n")` continues
    its string at column 0, so the common prefix would be empty. The first line's indent is the block's indent.
    """
    first = next((line for line in chunk.splitlines() if line.strip()), "")
    prefix = first[:len(first) - len(first.lstrip())]
    return "\n".join(line[len(prefix):] if prefix and line.startswith(prefix) else line
                     for line in chunk.splitlines())


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def history_slice(provenance: str) -> str:
    """The body `provenance` claims, read back from the commit it names."""
    match = _PROVENANCE_RE.match(provenance)
    assert match, f"not a provenance header: {provenance!r} (use `<path>:<start>-<end> @ <sha>` or `NONE`)"
    sha, path = match["sha"], match["path"]
    start, end = int(match["start"]), int(match["end"])
    assert _git("cat-file", "-e", f"{sha}^{{commit}}").returncode == 0, f"{sha} is not a commit in this clone"
    shown = _git("show", f"{sha}:{path}")
    assert shown.returncode == 0, f"{path} does not exist at {sha}: {shown.stderr.strip()}"
    lines = shown.stdout.splitlines()
    assert 1 <= start <= end <= len(lines), f"{path} at {sha} has {len(lines)} lines, not {start}-{end}"
    return _undent("\n".join(lines[start - 1:end])).rstrip("\n") + "\n"


def verify_fixture(path: Path) -> dict[str, str]:
    """The header, after proving a history claim byte-for-byte and accepting `NONE` on its own terms."""
    header, body = split_fixture(path)
    provenance = header.get("provenance")
    assert provenance, f"{path}: every fixture states `provenance:` - history or NONE"
    if provenance == "NONE":
        assert header.get("provenance-note"), f"{path}: `NONE` must say why there is no history to cite"
    else:
        assert body == history_slice(provenance), (
            f"{path}: the body is not the lines {provenance} names - an authored fixture may not claim history")
    return header


def _ui_gate_fixtures() -> list[Path]:
    return [p for p in PY_FIXTURES if p.parent.parent.name == "ui_gates"]


def _astcheck_fixtures() -> list[Path]:
    return [p for p in PY_FIXTURES if p.parent.parent.name == "astcheck"]


def _id(path: Path) -> str:
    return f"{path.parent.name}/{path.stem}"


# --- completeness: every rule has a fixture, in the directory its header names ---------------------------------

def test_every_ui_gate_rule_has_a_fixture() -> None:
    """A rule with no fixture is a gate whose red state nobody has seen; it fails here by name."""
    covered = {verify_fixture(path)["rule"] for path in _ui_gate_fixtures()}
    assert covered == set(ui_gates.CHECKS), f"rules without a fixture: {sorted(set(ui_gates.CHECKS) - covered)}"


def test_every_astcheck_rule_has_a_fixture() -> None:
    covered = {verify_fixture(path)["check"] for path in _astcheck_fixtures()}
    assert covered == {"dead store", "used before assignment"}


def test_the_ui_diff_ratchet_has_a_fixture() -> None:
    data = json.loads((FIXTURES / "ui_diff" / "page_budget_growth_2026_11_21.json").read_text(encoding="utf-8"))
    assert data["provenance"].startswith("harness/docs/wsp/verified/logs/ui-page-diff-2026-11-21.json")
    assert data["provenance_note"], "a data fixture states where it came from too"
    sha = data["provenance"].rsplit("@ ", 1)[1]
    assert _git("cat-file", "-e", f"{sha}^{{commit}}").returncode == 0, f"{sha} is not a commit in this clone"
    assert data["rows"] and data["refused"] > 0


@pytest.mark.parametrize("fixture", PY_FIXTURES, ids=_id)
def test_every_fixture_is_parsable_and_carries_its_header(fixture: Path) -> None:
    header = verify_fixture(fixture)
    assert "findings" in header and int(header["findings"]) >= 1
    compile(fixture.read_text(encoding="utf-8"), str(fixture), "exec")


@pytest.mark.parametrize("fixture", _ui_gate_fixtures(), ids=_id)
def test_a_ui_gate_fixture_sits_in_its_rules_directory(fixture: Path) -> None:
    rule = verify_fixture(fixture)["rule"]
    assert rule in ui_gates.CHECKS
    assert fixture.parent.name == rule.replace("-", "_"), f"{fixture}: directory and @rule disagree"


@pytest.mark.parametrize("fixture", _astcheck_fixtures(), ids=_id)
def test_an_astcheck_fixture_sits_in_its_rules_directory(fixture: Path) -> None:
    check = verify_fixture(fixture)["check"]
    assert fixture.parent.name == check.replace(" ", "_"), f"{fixture}: directory and @check disagree"


def test_an_authored_fixture_claiming_history_is_refused(tmp_path: Path) -> None:
    """The other direction of the provenance contract: citing a commit is not a way to look historical."""
    fake = tmp_path / "authored.py"
    fake.write_text(
        "# provenance: src/ui/pages/download_page.py:157-165 @ 542b55379cc3\n"
        "#@rule: qss-on-adapter\n"
        "#@findings: 1\n"
        "\n"
        "self.label = QtLabel(self.parent)\n"
        "self.label.setStyleSheet(u'QLabel{color:red;}')\n",
        encoding="utf-8")
    with pytest.raises(AssertionError, match="not the lines"):
        verify_fixture(fake)


def test_a_missing_sha_is_refused(tmp_path: Path) -> None:
    fake = tmp_path / "stale.py"
    fake.write_text(
        "# provenance: src/ui/pages/download_page.py:157-165 @ deadbeefdead\n"
        "#@rule: qss-on-adapter\n"
        "#@findings: 1\n"
        "\nx = 1\n",
        encoding="utf-8")
    with pytest.raises(AssertionError, match="not a commit"):
        verify_fixture(fake)


def test_a_none_fixture_must_say_why() -> None:
    """`NONE` is accepted without pretending it is history - but only with the reason written down."""
    none_headers = [verify_fixture(path) for path in PY_FIXTURES
                    if split_fixture(path)[0]["provenance"] == "NONE"]
    assert none_headers, "the authored fixtures are part of the contract, not an exception to it"
    assert all(header["provenance-note"] for header in none_headers)
    assert len(none_headers) < len(PY_FIXTURES), "at least one fixture must prove the history branch as well"


@pytest.mark.parametrize("fixture", PY_FIXTURES, ids=_id)
def test_a_fixture_cannot_enter_a_gate_scope(fixture: Path) -> None:
    """Fixtures are evidence, not debt: if one sat under `src/ui/pages`, the ratchet would count it as code."""
    relative = fixture.resolve().relative_to(ROOT).as_posix()
    for rule, roots in ui_gates.SCOPES.items():
        assert not any(relative == root or relative.startswith(f"{root}/") for root in roots), (rule, relative)


# --- red: the gate's own exit code over its fixture --------------------------------------------------------

@pytest.mark.parametrize("fixture", _ui_gate_fixtures(), ids=_id)
def test_a_ui_gate_exits_non_zero_on_its_fixture(fixture: Path, monkeypatch: pytest.MonkeyPatch,
                                                 capsys: pytest.CaptureFixture[str]) -> None:
    """Run the real CLI over the fixture: a new finding, a non-zero exit, and the rule by name in the table."""
    header = verify_fixture(fixture)
    monkeypatch.setitem(ui_gates.SCOPES, header["rule"], (str(fixture),))
    assert ui_gates.main() == 1
    out = capsys.readouterr().out
    assert "NEW" in out and header["rule"] in out
    findings = ui_gates.CHECKS[header["rule"]]([fixture])
    assert len(findings) == int(header["findings"]), [str(finding) for finding in findings]


@pytest.mark.parametrize("fixture", _astcheck_fixtures(), ids=_id)
def test_astcheck_exits_non_zero_on_its_fixture(fixture: Path, capsys: pytest.CaptureFixture[str]) -> None:
    header = verify_fixture(fixture)
    assert astcheck.main([str(fixture)]) == 1
    out = capsys.readouterr().out
    assert header["check"] in out
    assert len(astcheck.check_paths([str(fixture)])) == int(header["findings"])


def test_ui_diff_refuses_the_recorded_page_growth() -> None:
    """The 2026-11-21 numbers through today's per-page table: four pages still over, the Process rows inside 6.0."""
    data = json.loads((FIXTURES / "ui_diff" / "page_budget_growth_2026_11_21.json").read_text(encoding="utf-8"))
    grown = ui_diff.over_budget(data["rows"], ui_diff.PAGE_BASELINE, 0.05)
    # The fixture records four rows over the table of 2026-11-21. One of them is the Home page at 0.168 %, and its
    # entry was raised to 0.5 % on 2026-11-24 for a DELIBERATE look change (the bundled Fira Mono heading; the bound is
    # measured, see `PAGE_BASELINE`). So that row is inside its budget now and the refusals drop to three - the other
    # pages are untouched, which is the point of a per-page table.
    assert len(grown) == data["refused"] - 1
    assert ("nav3-tab0.png", 3.137, 0.05) in grown
    assert not [row for row in grown if row[0] == "nav0.png"], (
        "the Home page's 0.168 % sits inside its recorded 0.5 % - the raised entry must be the only reason")
    assert not [row for row in grown if row[0] in ("nav2-tab0.png", "nav2.png")], \
        "the Process page measured 0.393 and its entry is 6.0 - the per-page number is the whole point"
    table, refused = ui_diff.record_baseline(data["rows"], ui_diff.PAGE_BASELINE, 0.05)
    assert table == dict(ui_diff.PAGE_BASELINE), "a refused recording must not rewrite the table"
    assert refused == grown, "--record-baseline names the same growth compare refuses"


def test_ui_diff_exits_non_zero_on_a_grown_page_and_zero_on_an_unchanged_tree(tmp_path: Path,
                                                                            capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI: the page the history measured at 3.137 % is red with one painted block, and green unchanged."""
    pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    page = json.loads((FIXTURES / "ui_diff" / "page_budget_growth_2026_11_21.json").read_text())["page"]

    def captures(root: Path, painted: bool) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        dump = {"geometry_space": "parent-local",
                "root": {"class": "QMainWindow", "name": "MainWindow", "geometry": [0, 0, 64, 64],
                         "visible": True, "enabled": True,
                         "children": [{"class": "QWidget", "name": "body", "geometry": [4, 4, 20, 20],
                                       "visible": True, "enabled": True}]}}
        (root / page.replace(".png", ".json")).write_text(json.dumps(dump), encoding="utf-8")
        image = Image.new("RGB", (64, 64), (30, 30, 30))
        if painted:
            for x in range(44, 64):
                for y in range(44, 64):
                    image.putpixel((x, y), (220, 220, 220))
        image.save(root / page)
        return root

    stock = captures(tmp_path / "stock", painted=False)
    grown = captures(tmp_path / "grown", painted=True)
    assert ui_diff.main(["compare", "--stock-dumps", str(stock), "--new-dumps", str(grown),
                         "--views", "pages"]) == 1
    assert f"over: {page}" in capsys.readouterr().out
    assert ui_diff.main(["compare", "--stock-dumps", str(stock), "--new-dumps", str(stock),
                         "--views", "pages"]) == 0
    assert "inside" in capsys.readouterr().out


# --- green: the same gates over the current tree ------------------------------------------------------------

def test_ui_gates_exit_zero_on_the_current_tree(capsys: pytest.CaptureFixture[str]) -> None:
    assert ui_gates.main() == 0
    out = capsys.readouterr().out
    assert "no new violations" in out
    assert "NEW" not in out


def test_astcheck_is_clean_on_the_harness_core(capsys: pytest.CaptureFixture[str]) -> None:
    """The documented invocation (`harness/README.md`): the harness's own modules, dead stores and use-before-assign."""
    files = sorted(str(path) for path in (ROOT / "harness").glob("*.py"))
    assert files
    assert astcheck.main(files) == 0
    assert "no dead stores, no use-before-assignment" in capsys.readouterr().out


def test_the_committed_evidence_run_is_inside_the_committed_table() -> None:
    """`ui_diff` is green on the run its table was recorded from: the 2026-11-21 shell capture measured 0.00 on all
    thirteen pages, and every one of them is inside the entry the table now records."""
    log = json.loads((ROOT / "harness/docs/wsp/verified/logs/ui-shell-pixels-2026-11-21.json").read_text())
    rows = [{"page": page, "over8_pct": body["over8_pct"]} for page, body in log["pages"].items()]
    assert ui_diff.over_budget(rows, ui_diff.PAGE_BASELINE, 0.05) == []
    assert {row["page"] for row in rows} == set(ui_diff.PAGE_BASELINE)
