"""The crop record the render prints and the LMB compare reads, pinned at the seam between them.

The compare used to decode the whole source while the render cropped it, so pressing LMB showed a differently
framed picture than the pane - the letterbox read as a scale error rather than as "this is the source". The fix
crosses two processes: the backend prints one ``BORDERS w=.. h=.. x=.. y=..`` line and the front end parses it
out of the render log. Neither half can be wrong on its own without the OTHER half noticing, so what is asserted
here is the agreement, not the halves in isolation.

Both pieces are pure and are executed straight out of their shipped source with ``ast`` (the same technique as
``test_page_handles.py``): this file therefore needs no torch, no cv2 and no PySide6, and it cannot pass against
a stale copy of the format.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend" / "src" / "RenderVideo.py"
GLUE = ROOT / "src" / "ui" / "ProcessTab.py"


def _load(path: Path, name: str, **namespace):
    """The shipped top-level function ``name`` from ``path``, compiled on its own (no module imports run)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            module = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(module)
            scope = dict(namespace)
            exec(compile(module, str(path), "exec"), scope)
            return scope[name]
    raise AssertionError(f"{path.name} no longer defines a top-level {name}()")


#: The shipped functions, exactly as written: the parser is compiled with only `re` in scope, which is all it uses.
format_borders_line = _load(BACKEND, "format_borders_line")
parse_borders_line = _load(GLUE, "parse_borders_line", re=re)
crop_region = _load(GLUE, "crop_region")


# -- the format -------------------------------------------------------------


@pytest.mark.parametrize(
    "box",
    [
        (1920, 1080, 0, 0),        # no crop detected: the box is the full frame
        (1864, 1040, 28, 20),      # a real letterbox, even offsets as `decide()` guarantees
        (1280, 720, 0, 140),
    ],
)
def test_the_parser_reads_exactly_what_the_backend_writes(box):
    line = format_borders_line(*box)
    assert line == f"BORDERS w={box[0]} h={box[1]} x={box[2]} y={box[3]}"
    assert parse_borders_line(line) == box


def test_the_record_is_one_bare_line():
    """The format the front end's parser is strict about: four fields, no prose, nothing before the marker."""
    line = format_borders_line(1920, 1080, 28, 20)
    assert line.splitlines() == ["BORDERS w=1920 h=1080 x=28 y=20"]
    assert line == line.strip()


# -- the read side ----------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "BORDERS w=1864 h=1040 x=28 y=20",
        "BACKEND: BORDERS w=1864 h=1040 x=28 y=20",   # tolerated transport prefix
        "   BORDERS w=1864 h=1040 x=28 y=20   ",      # the log loop strips, the parser must not care
    ],
)
def test_parse_accepts_the_record(line):
    assert parse_borders_line(line) == (1864, 1040, 28, 20)


@pytest.mark.parametrize(
    "line",
    [
        "",
        "   ",
        "BORDERS",
        "BORDERS w=1864 h=1040 x=28",                        # truncated
        "BORDERS w=1864 h=1040 x=28 y=20 note",              # extra field on the line
        "BORDERS w=1864 h=1040 x=28 y=20 (frame 1)",         # trailing prose
        "BACKEND: BORDERS w=1864 h=1040 x=28 y=20 extra",
        "XBORDERS w=1864 h=1040 x=28 y=20",                  # not our marker
        "BORDERS w=1864.0 h=1040 x=28 y=20",                 # the record carries ints
        "BORDERS w=-1864 h=1040 x=28 y=20",
        "BORDERS w=1864 h=1040 x=28 y=",                     # empty value
        "Detected borders: Width,Height:1864x1040, X,Y: 28x20",  # the human line is not the record
        None,
    ],
)
def test_parse_refuses_anything_else(line):
    assert parse_borders_line(line) is None


def test_a_malformed_line_leaves_todays_picture():
    """No box -> no crop -> the compare decodes the whole source, exactly as before this fix."""
    box = parse_borders_line("BORDERS w=1864 h=1040 x=28")
    assert box is None
    assert crop_region(box, 1920, 1080) is None


def test_a_box_that_does_not_fit_the_source_is_refused_whole():
    assert crop_region((1864, 1040, 28, 20), 1920, 1080) == (28, 20, 1864, 1040)
    assert crop_region((1920, 1080, 0, 0), 1920, 1080) == (0, 0, 1920, 1080)
    # an offset past the right/bottom edge, a zero side, a negative side, a short tuple, nothing at all
    assert crop_region((1864, 1040, 58, 20), 1920, 1080) is None
    assert crop_region((1864, 1080, 28, 20), 1920, 1080) is None
    assert crop_region((1864, 0, 28, 20), 1920, 1080) is None
    assert crop_region((1864, 1040, -28, 20), 1920, 1080) is None
    assert crop_region((1864, 1040, 28), 1920, 1080) is None
    assert crop_region(None, 1920, 1080) is None


def test_the_full_frame_box_is_the_same_picture_as_today():
    """'No crop' is a decision, not a missing value: the box covers the frame and the crop is a no-op."""
    box = parse_borders_line(format_borders_line(1920, 1080, 0, 0))
    assert crop_region(box, 1920, 1080) == (0, 0, 1920, 1080)


# -- the two halves are wired where they must be ----------------------------


def _statements(path: Path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _initialiser(path: Path):
    tree = _statements(path)
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    return item
    raise AssertionError(f"{path.name} no longer defines a class __init__")


def _calls(node, name: str) -> list[ast.Call]:
    found = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == name:
            found.append(child)
    return found


def test_report_borders_writes_the_record_and_nothing_else(capsys):
    """The writer itself, executed: what lands on the stream is the record the parser reads."""
    logged = []
    report_borders = _load(BACKEND, "report_borders", format_borders_line=format_borders_line,
                           log=logged.append, sys=sys)
    report_borders(1864, 1040, 28, 20)
    assert capsys.readouterr().err == "BORDERS w=1864 h=1040 x=28 y=20\n"
    assert logged == []


def test_report_borders_says_so_instead_of_inventing_a_box(capsys):
    """An unknown frame size must not become ``BORDERS w=None ...``: the compare then keeps today's picture."""
    logged = []
    report_borders = _load(BACKEND, "report_borders", format_borders_line=format_borders_line,
                           log=logged.append, sys=sys)
    report_borders(None, None, None, None)
    assert capsys.readouterr().err == ""
    assert len(logged) == 1
    assert "no crop to report" in logged[0]


def test_the_backend_prints_the_record_on_every_render():
    """Directly in ``Render.__init__``, not inside ``if border_detect:`` - a full-frame render reports its box too."""
    body = _initialiser(BACKEND).body
    printing = [
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and _calls(statement.value, "report_borders")
    ]
    assert len(printing) == 1, "the render must write the BORDERS record exactly once"
    read = [
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and any(isinstance(target, ast.Attribute) and target.attr == "readBuffer"
                for target in statement.targets)
    ]
    assert read, "Render.__init__ no longer assigns self.readBuffer"
    assert printing[0] < read[0], "the record must be written before the reader is built (i.e. before frame one)"


def test_the_glue_keeps_the_box_with_the_job_and_crops_the_source():
    tree = _statements(GLUE)
    methods = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    methods[item.name] = item
    loop = methods["renderToPipeThread"]
    assert _calls(loop, "parse_borders_line"), "the render-log loop no longer parses the BORDERS record"
    assert any(isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Attribute) and target.attr == "_source_crop"
                       for target in node.targets)
               for node in ast.walk(loop)), "the parsed box is not kept for the job"
    frame = methods["_on_original_frame"]
    assert _calls(frame, "crop_region"), "the decoded source is no longer cropped to the render's box"
    assert any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "set_original"
        for node in ast.walk(frame)
    ), "_on_original_frame no longer hands the picture to the pane"
