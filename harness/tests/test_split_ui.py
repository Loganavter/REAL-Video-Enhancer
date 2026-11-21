"""The splitter's contract: 13 pages, nothing lost, nothing shared, modules that compile.

`harness/scripts/split_ui.py` moves the Designer's statements into page modules. The move is only safe if (a) every
statement lands exactly once, (b) no page reads a widget another page owns, and (c) the modules it writes are valid
Python. The visual identity of each page is checked separately, offscreen with PySide6
(`harness/scripts/ui_pages_identity.py`); this test is the half that runs everywhere.
"""

from __future__ import annotations

import importlib.util
import os
import py_compile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPLIT = ROOT / "harness" / "scripts" / "split_ui.py"
# `mainwindow.py` was the Designer's compiled output; it was deleted with the .ui in W3, so the fixture keeps the
# splitter's contract testable (and the split auditable) without the generated file living in the tree.
DESIGNER = ROOT / "mainwindow.py"
if not DESIGNER.exists():
    DESIGNER = ROOT / "harness" / "fixtures" / "designer_mainwindow.py"


def _module():
    spec = importlib.util.spec_from_file_location("split_ui", SPLIT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _partition():
    tool = _module()
    lines, setup, retranslate, locals_defined = tool.load(str(DESIGNER))
    pages, shell = tool.build_partition(setup)
    return tool, setup, pages, shell


def test_thirteen_pages_and_no_statement_shared() -> None:
    _tool, setup, pages, shell = _partition()
    assert len(pages) == 13, [page.name for page in pages]
    owners: dict[int, str] = {}
    for page in pages:
        for index in page.own:
            assert index not in owners, f"statement {index} emitted twice ({owners[index]} and {page.name})"
            owners[index] = page.name
    for index in shell:
        assert index not in owners, f"shell statement {index} also belongs to {owners[index]}"
    placements = {page.place for page in pages}
    covered = set(owners) | set(shell) | placements
    missing = sorted(set(range(len(setup))) - covered)
    assert not missing, f"statements that no builder would emit: {missing[:10]}"


def test_no_page_reads_another_pages_widgets() -> None:
    tool, setup, pages, shell = _partition()
    tool._migrate_page_defaults(pages, shell, setup)
    for page in pages:
        external = tool.external_references(page, setup)
        assert not external, f"{page.name} reaches outside itself: {external}"


def test_nested_pages_are_children_of_their_container() -> None:
    _tool, _setup, pages, _shell = _partition()
    by_name = {page.name: page for page in pages}
    assert {child.name for child in by_name["procPage"].children} == {
        "generalSettings", "tab_6", "renderQueueTab", "advancedSettings", "tab_5"}
    assert {child.name for child in by_name["settingsPage"].children} == {"tab_2", "tab_3", "tab_4"}
    roots = sorted(page.name for page in pages if page.parent == "stackedWidget")
    assert roots == ["downloadPage", "homePage", "morePage", "procPage", "settingsPage"]


def test_generated_modules_compile() -> None:
    tool, setup, pages, shell = _partition()
    tool._migrate_page_defaults(pages, shell, setup)
    with tempfile.TemporaryDirectory() as tmp:
        header = tool._header(DESIGNER.read_text(encoding="utf-8"))
        for page in pages:
            text = tool.module_for(page, setup, [], {}, header)
            path = os.path.join(tmp, f"{tool._snake(page.name)}.py")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
            py_compile.compile(path, doraise=True)
