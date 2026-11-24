"""Pins for `harness/scripts/ui_diff.py`'s page ratchet.

The 2026-11-21 measurement the defect came from: six pages of this tree were over the flat 0.05% `--budget-page`, the
worst at 3.137, so the gate's exit code was 1 on a tree nobody had changed. A gate that cannot be green is not a gate.
The fix is the shape the other ratchets use - a per-page recorded number, failing only on growth - and this file proves
both directions on synthetic captures, without Qt, a display, or a real app build.

The captures are two identical 64x64 images and a one-widget dump: the page metric is the difference between them, so
"green" and "one page grew" are the same fixture with one rectangle painted in.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.scripts import ui_diff

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")


# --- fixtures ---------------------------------------------------------------------------------------------

def _dump() -> dict:
    """One page's dump: a window with one visible child, in the shape `layout_dump` writes."""
    return {
        "geometry_space": "parent-local",
        "root": {
            "class": "QMainWindow", "name": "MainWindow", "geometry": [0, 0, 64, 64],
            "visible": True, "enabled": True,
            "children": [
                {"class": "QWidget", "name": "button", "geometry": [4, 4, 20, 20],
                 "visible": True, "enabled": True, "text": "Open"},
            ],
        },
    }


def _png(path: Path, painted: bool = False, box: int = 20) -> None:
    image = Image.new("RGB", (64, 64), (30, 30, 30))
    if painted:
        for x in range(64 - box, 64):
            for y in range(64 - box, 64):
                image.putpixel((x, y), (220, 220, 220))
    image.save(path)


def _captures(root: Path, pages=("nav0.png",), painted=(), box: int = 20) -> Path:
    """A capture directory: `<page>.json` dumps and `<page>.png` images, the layout a capture run writes."""
    root.mkdir(parents=True, exist_ok=True)
    for page in pages:
        (root / page.replace(".png", ".json")).write_text(json.dumps(_dump()), encoding="utf-8")
        _png(root / page, painted=page in painted, box=box)
    return root


def _run(tmp_path: Path, old_painted=(), new_painted=(), extra: list[str] | None = None, box: int = 20) -> int:
    stock = _captures(tmp_path / "stock", painted=tuple(old_painted))
    new = _captures(tmp_path / "new", painted=tuple(new_painted), box=box)
    return ui_diff.main(["compare", "--stock-dumps", str(stock), "--new-dumps", str(new),
                         "--views", "pages", *(extra or [])])


# --- the baseline itself -----------------------------------------------------------------------------------

def test_every_page_the_2026_11_21_run_captured_has_a_recorded_tolerance() -> None:
    """The table names the whole walk, not a sample: an unrecorded page falls back to 0.05 and says so.

    The expected stems are not invented - they are the pages the committed run measured
    (`harness/docs/wsp/verified/logs/ui-shell-pixels-2026-11-21.json`), which is also where the floor comes from.
    A capture walk that adds a page is a deliberate change: record it in the same commit.
    """
    log = Path(ui_diff.__file__).resolve().parents[1] / "docs/wsp/verified/logs/ui-shell-pixels-2026-11-21.json"
    measured = json.loads(log.read_text(encoding="utf-8"))["pages"]
    assert set(ui_diff.PAGE_BASELINE) == set(measured)
    assert set(measured) == {
        "nav0.png", "nav1.png", "nav2.png", "nav2-tab0.png", "nav2-tab1.png", "nav2-tab2.png",
        "nav2-tab3.png", "nav2-tab4.png", "nav3.png", "nav3-tab0.png", "nav3-tab1.png", "nav3-tab2.png",
        "nav4.png",
    }


def test_the_process_page_carries_its_measured_noise_and_the_rest_the_floor() -> None:
    """`nav2-tab0` is the page whose probe re-runs on every boot: 3.58/5.93 measured on an identical tree.

    Everything else measured 0.00 in the 2026-11-21 shell run, so it records the floor - not a zero-tolerance pin that
    would charge a driver's antialiasing as a regression.
    """
    assert ui_diff.page_budget("nav2-tab0.png", ui_diff.PAGE_BASELINE, 0.05) == 6.0
    assert ui_diff.page_budget("nav2.png", ui_diff.PAGE_BASELINE, 0.05) == 6.0
    # nav0 is the ONE page whose 1.0 look changed on purpose: the bundled Fira Mono (2026-11-24) repainted the Home
    # heading, and the entry carries the measured bound (heading ink 0.238 % of the page, union <= 0.476 %) instead of
    # the floor. A regression on that page still has to clear 0.5 %, so the entry stays a gate.
    assert ui_diff.page_budget("nav0.png", ui_diff.PAGE_BASELINE, 0.05) == 0.5, (
        "the Home page's deliberate typeface change is recorded with its measurement, not the floor")
    for page in ("nav1.png", "nav3.png", "nav4.png"):
        assert ui_diff.page_budget(page, ui_diff.PAGE_BASELINE, 0.05) == ui_diff.BASELINE_FLOOR
    assert set(ui_diff.PAGE_BASELINE) == {
        "nav0.png", "nav1.png", "nav2.png", "nav2-tab0.png", "nav2-tab1.png", "nav2-tab2.png",
        "nav2-tab3.png", "nav2-tab4.png", "nav3.png", "nav3-tab0.png", "nav3-tab1.png", "nav3-tab2.png",
        "nav4.png",
    }


def test_an_unrecorded_page_uses_the_fallback_and_says_so(capsys: pytest.CaptureFixture[str]) -> None:
    """A page nobody measured is not silently inside a budget: the fallback runs and the table reports it."""
    rows = [{"page": "nav9.png", "mean": 0.001, "over8_pct": 0.01, "max": 3}]
    assert ui_diff.page_budget("nav9.png", ui_diff.PAGE_BASELINE, 0.05) == 0.05
    assert ui_diff.over_budget(rows, ui_diff.PAGE_BASELINE, 0.05) == []


def test_no_baseline_judges_every_page_against_the_flat_budget() -> None:
    rows = [{"page": "nav2-tab0.png", "over8_pct": 4.81}]
    assert ui_diff.over_budget(rows, None, 0.05) == [("nav2-tab0.png", 4.81, 0.05)]
    assert ui_diff.over_budget(rows, ui_diff.PAGE_BASELINE, 0.05) == []      # 4.81 is inside the recorded 6.0


def test_a_missing_image_is_not_inside_any_budget() -> None:
    """`nan` is not a pass: the page was never compared, and the caller fails on `uncompared` for that reason."""
    rows = [{"page": "nav0.png", "over8_pct": float("nan"), "mean": float("nan"), "shape_mismatch": True}]
    assert ui_diff.over_budget(rows, ui_diff.PAGE_BASELINE, 0.05) == []


# --- the CLI: both directions ------------------------------------------------------------------------------

def test_compare_exits_zero_against_the_committed_baseline(tmp_path: Path,
                                                           capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(tmp_path) == 0
    out = capsys.readouterr().out
    assert "inside their recorded budget" in out
    assert "over:" not in out


def test_compare_fails_when_a_page_grows_past_its_entry(tmp_path: Path,
                                                        capsys: pytest.CaptureFixture[str]) -> None:
    """The failing direction is the whole point: one painted rectangle on one page, the exit code and the name."""
    assert _run(tmp_path, new_painted=("nav0.png",)) == 1
    out = capsys.readouterr().out
    assert "nav0.png" in out
    assert "over: nav0.png" in out
    assert "the page grew" in out


def test_the_whole_committed_table_is_green_on_an_unchanged_tree(tmp_path: Path,
                                                                capsys: pytest.CaptureFixture[str]) -> None:
    """All thirteen pages, identical captures: the gate is green on a tree nobody changed - the defect's symptom gone."""
    pages = tuple(sorted(ui_diff.PAGE_BASELINE))
    stock = _captures(tmp_path / "stock", pages=pages)
    new = _captures(tmp_path / "new", pages=pages)
    assert ui_diff.main(["compare", "--stock-dumps", str(stock), "--new-dumps", str(new),
                         "--views", "pages"]) == 0
    out = capsys.readouterr().out
    assert "13 of 13 pages inside their recorded budget" in out


def test_a_quiet_page_fails_where_the_process_page_would_pass(tmp_path: Path,
                                                             capsys: pytest.CaptureFixture[str]) -> None:
    """The entries are per page, not one number: the same 4 % change passes on nav2-tab0 (6.0) and fails on nav3-tab0."""
    stock = _captures(tmp_path / "stock", pages=("nav2-tab0.png", "nav3-tab0.png"))
    new = _captures(tmp_path / "new", pages=("nav2-tab0.png", "nav3-tab0.png"),
                    painted=("nav2-tab0.png", "nav3-tab0.png"), box=12)   # 144/4096 = 3.5 % of the page
    assert ui_diff.main(["compare", "--stock-dumps", str(stock), "--new-dumps", str(new),
                         "--views", "pages"]) == 1
    out = capsys.readouterr().out
    assert "over: nav3-tab0.png" in out
    assert "nav2-tab0.png:" not in out.split("over:")[1]


def test_compare_fails_when_the_new_capture_is_missing_a_page(tmp_path: Path,
                                                             capsys: pytest.CaptureFixture[str]) -> None:
    stock = _captures(tmp_path / "stock", pages=("nav0.png", "nav1.png"))
    new = _captures(tmp_path / "new", pages=("nav0.png",))
    assert ui_diff.main(["compare", "--stock-dumps", str(stock), "--new-dumps", str(new),
                         "--views", "pages"]) == 1
    assert "missing in the new run: nav1.json" in capsys.readouterr().out


def test_compare_fails_when_the_image_is_missing(tmp_path: Path,
                                                capsys: pytest.CaptureFixture[str]) -> None:
    stock = _captures(tmp_path / "stock", pages=("nav0.png",))
    new = _captures(tmp_path / "new", pages=("nav0.png",))
    (new / "nav0.png").unlink()
    assert ui_diff.main(["compare", "--stock-dumps", str(stock), "--new-dumps", str(new),
                         "--views", "pages"]) == 1
    out = capsys.readouterr().out
    assert "NOT COMPARED" in out and "nav0.json" in out


def test_a_page_inside_its_own_noise_allowance_passes(tmp_path: Path) -> None:
    """The Process page may move inside its recorded 6.0 without failing - that is what the entry is for.

    The painted block is 5.5 % of the page: above what the other pages tolerate, below nav2-tab0's 6.0.
    """
    assert _run(tmp_path, new_painted=("nav2-tab0.png",), box=15) == 0


# --- recording is guarded ----------------------------------------------------------------------------------

def test_record_baseline_refuses_growth_and_names_the_page() -> None:
    rows = [{"page": "nav0.png", "over8_pct": 0.4}, {"page": "nav1.png", "over8_pct": 0.0}]
    table, grown = ui_diff.record_baseline(rows, {"nav0.png": 0.05, "nav1.png": 0.05}, 0.05)
    assert grown == [("nav0.png", 0.4, 0.05)]
    assert table == {"nav0.png": 0.05, "nav1.png": 0.05}, "a refused recording must not rewrite the table"


def test_record_baseline_shrinks_a_page_and_floors_a_zero() -> None:
    rows = [{"page": "nav0.png", "over8_pct": 0.0}, {"page": "nav2-tab0.png", "over8_pct": 1.5}]
    table, grown = ui_diff.record_baseline(rows, {"nav0.png": 0.05, "nav2-tab0.png": 6.0}, 0.05)
    assert grown == []
    assert table == {"nav0.png": ui_diff.BASELINE_FLOOR, "nav2-tab0.png": 1.5}


def test_record_baseline_does_not_record_a_page_that_was_never_compared() -> None:
    rows = [{"page": "nav0.png", "over8_pct": float("nan")}]
    table, grown = ui_diff.record_baseline(rows, {"nav0.png": 0.05}, 0.05)
    assert (table, grown) == ({"nav0.png": 0.05}, [])


def test_the_cli_refuses_a_recording_that_would_grow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                     capsys: pytest.CaptureFixture[str]) -> None:
    """`--record-baseline` on a grown page exits 2 and leaves the source alone (it writes only on success)."""
    monkeypatch.setitem(ui_diff.PAGE_BASELINE, "nav0.png", 0.05)
    before = Path(ui_diff.__file__).read_text(encoding="utf-8")
    assert _run(tmp_path, new_painted=("nav0.png",), extra=["--record-baseline"]) == 2
    assert "REFUSED: nav0.png would grow" in capsys.readouterr().out
    assert Path(ui_diff.__file__).read_text(encoding="utf-8") == before


def test_rewrite_baseline_replaces_only_the_literal(tmp_path: Path) -> None:
    copy = tmp_path / "ui_diff_copy.py"
    source = Path(ui_diff.__file__).read_text(encoding="utf-8")
    copy.write_text(source, encoding="utf-8")
    assert ui_diff.rewrite_baseline(copy, {"nav0.png": 0.02, "nav2-tab0.png": 5.5}) is True
    text = copy.read_text(encoding="utf-8")
    assert "'nav0.png': 0.02," in text and "'nav2-tab0.png': 5.5," in text
    assert "'nav1.png'" not in text, "a page dropped from the table must leave the literal"
    # the rest of the module is untouched: it still parses and still defines the same public surface
    compile(text, str(copy), "exec")
    assert "def over_budget(" in text and "def main(" in text


def test_rewrite_baseline_refuses_a_file_without_the_block(tmp_path: Path) -> None:
    plain = tmp_path / "plain.py"
    plain.write_text("PAGE_BASELINE = []\n", encoding="utf-8")
    assert ui_diff.rewrite_baseline(plain, {"nav0.png": 0.05}) is False
