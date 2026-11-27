"""One fact, one writer - enforced for the pane's picture state and for the pair's commit counter.

Two of the four infrastructure pieces in `projects/rve/docs/notes/blinktable-infrastructure-2026-11-27.md` §3 (items 3
and 4), turned into gates instead of review notes:

1. **The pane's "which picture am I showing" state has ONE writer per fact.** `VideoPreviewSurface` holds the result
   picture (`_frame`, `_planes_size`, `_pending_frame_obj`), its identity on screen (`_shown`, `_shown_time`,
   `_result_moment`, `_result_frame_seconds`, `_result_generation`) and the source half (`_original`,
   `_original_shown`, `_original_time`, `_original_changed`). Each is decided in exactly one method, and a second
   `self._frame = ...` anywhere in the class has to be a red test rather than a code review (`CODE_PATTERNS.md` §2).
   Facts that already carry two writers today are recorded in `KNOWN_SECOND_WRITERS` (a ratchet: the recorded set is
   exact, so a THIRD writer fails and so does a fix that leaves the stale entry behind).
2. **The pair counter is written where the pair is COMMITTED.** `pair_commits` is incremented in
   `PairWriter._paint`, in the same function body that stores `self._committed = pair` and *after* that store - not in
   `paint()`, which is the decision entry and returns without committing on every hold. A refactor that moves the
   commit without the counter turns red (`CODE_PATTERNS.md` §7; the 2026-11-27 defect: every result held, every paint
   counting a stale pair).

The checkers are pure `(source text) -> problems` functions so the last two tests can feed them deliberately broken
source: a gate that cannot be shown red pins nothing (`CODE_PATTERNS.md` §9). Nothing here imports Qt or the app - this
is a static contract, read off the AST the way `harness/scripts/ui_gates.py` reads the UI tree.
"""

from __future__ import annotations

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
PANE_FILE = ROOT / "src" / "ui" / "widgets" / "video_preview.py"
PAIR_FILE = ROOT / "src" / "ui" / "preview" / "pair.py"
SRC = ROOT / "src"

PANE_CLASS = "VideoPreviewSurface"
PAIR_CLASS = "PairWriter"
PAIR_COMMIT_FUNCTION = "_paint"
COUNTER_CALL = "note_pair_commit"

#: Initial values are declared in `__init__`; a declaration is not a decision about a picture.
DECLARATION_METHOD = "__init__"

#: fact -> the ONE method of `VideoPreviewSurface` allowed to write it (besides `__init__`).
#:
#: The value is the method name; the key is the attribute as written on `self`. Keep the descriptions in the companion
#: `FACT_MEANING` map beside a fact's identifier in the failure message: the test has to name the seam, not a line.
PANE_PICTURE_FACTS = {
    "_frame": "_accept_result",                 # the result picture as a QImage
    "_planes_size": "_accept_result",           # the result picture's size when it arrives as YUV planes
    "_pending_frame_obj": "_accept_result",     # the planes payload handed to the pair writer
    "_shown": "_accept_result",                 # ("frame"|"seq", n) - the result's identity on screen
    "_shown_time": "_accept_result",            # the result's media time - the pane's one clock
    "_result_moment": "_accept_result",         # the accepted result's moment (the refusal baseline)
    "_result_frame_seconds": "_accept_result",  # the pane's learned cadence (source tolerance = half of it)
    "_result_generation": "_accept_result",     # the generation of the result on screen
    "_result_gap": "_accept_result",            # "the half on screen is a gap" (cleared by an acceptance)
    "_original": "_apply_source_picture",       # the source picture as a QImage
    "_original_shown": "_apply_source_picture",  # the source picture's identity
    "_original_time": "_apply_source_picture",  # the source picture's media time
    "_original_changed": "_apply_source_picture",  # "the source picture changed since the last draw line"
}

FACT_MEANING = {
    "_frame": "the result picture the pane is showing",
    "_planes_size": "the result picture's size (YUV path)",
    "_pending_frame_obj": "the planes payload offered to the pair writer",
    "_shown": "the identity of the result picture on screen",
    "_shown_time": "the media time of the result picture on screen (the pane's one clock)",
    "_result_moment": "the accepted result's moment",
    "_result_frame_seconds": "the pane's learned result cadence",
    "_result_generation": "the generation of the result on screen",
    "_result_gap": "whether the result half on screen is a gap",
    "_original": "the source picture the compare is showing",
    "_original_shown": "the identity of the source picture",
    "_original_time": "the media time of the source picture",
    "_original_changed": "the 'source picture changed' flag the draw line consumes",
}

#: Facts that already have more than one writer TODAY, recorded instead of fixed (this row adds enforcement, not
#: behaviour). The recorded set is exact: a new writer joins the failure, and a fix that leaves the entry here also
#: fails ("KNOWN only shrinks", `harness/scripts/ui_gates.py`). Each entry says WHY the second site exists.
KNOWN_SECOND_WRITERS = {
    # The flag is one answer ("the result half on screen is a GAP") set by the refusal and cleared by the acceptance:
    # the two halves of the same API (`_refuse_result` keeps the picture, `_accept_result` replaces it), not two owners.
    "_result_gap": {"_accept_result", "_refuse_result"},
    # A clear drops the source half as well as the result (CODE_PATTERNS §12); the source path is still the one place
    # that DECIDES a source picture.
    "_original": {"_apply_source_picture", "_accept_result"},
    "_original_shown": {"_apply_source_picture", "_accept_result"},
    "_original_time": {"_apply_source_picture", "_accept_result"},
    # Written when a source arrives and consumed when the draw line reports it; two writers because the flag closes a
    # log line rather than naming a picture. It is the only fact here that is not an answer about the picture itself.
    "_original_changed": {"_apply_source_picture", "_on_rhi_draw"},
}

#: Names a pane is bound to outside its own module. A direct attribute store through one of these is a second owner of
#: the pane's picture - the API (`set_result`/`set_source`) is the only way in.
PANE_HANDLES = ("pane", "previewLabel", "preview_surface")


# -- AST helpers ---------------------------------------------------------


def _parse(source: str) -> ast.Module:
    return ast.parse(source)


def _class_node(source: str, class_name: str) -> ast.ClassDef:
    for node in _parse(source).body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"class {class_name} is gone from the module - the gate has nothing to read")


def _self_stores(source: str, class_name: str, facts: set[str]) -> dict[str, set[str]]:
    """fact -> names of the methods that store `self.<fact>` (a store in a nested callback counts to its owner)."""
    found: dict[str, set[str]] = {fact: set() for fact in facts}
    for node in _class_node(source, class_name).body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Attribute) and isinstance(sub.ctx, ast.Store)
                    and sub.attr in found
                    and isinstance(sub.value, ast.Name) and sub.value.id == "self"):
                found[sub.attr].add(node.name)
    return found


def _single_writer_problems(source: str, facts: dict[str, str], known: dict[str, set[str]]) -> list[str]:
    """Every fact must have exactly its declared writer(s); a KNOWN entry must still have exactly its recorded set."""
    stores = _self_stores(source, PANE_CLASS, set(facts) | set(known))
    problems: list[str] = []
    for fact in sorted(set(facts) | set(known)):
        writers = set(stores.get(fact, set())) - {DECLARATION_METHOD}
        meaning = FACT_MEANING.get(fact, "a pane picture fact")
        if fact in known:
            recorded = set(known[fact])
            if not writers:
                problems.append(
                    f"{fact} ({meaning}): the KNOWN entry has no writer any more - delete it from "
                    f"KNOWN_SECOND_WRITERS, the ratchet only shrinks")
            elif writers != recorded:
                added = sorted(writers - recorded)
                gone = sorted(recorded - writers)
                problems.append(
                    f"{fact} ({meaning}): writers are {sorted(writers)} but KNOWN_SECOND_WRITERS records "
                    f"{sorted(recorded)}"
                    + (f"; NEW writer(s): {added}" if added else "")
                    + (f"; no longer written by: {gone}" if gone else ""))
            continue
        owner = facts[fact]
        if not writers:
            problems.append(
                f"{fact} ({meaning}): nothing writes it any more - the fact moved; name its new owner in "
                f"PANE_PICTURE_FACTS")
        elif writers != {owner}:
            extra = sorted(writers - {owner})
            problems.append(
                f"{fact} ({meaning}) has {len(writers)} writers {sorted(writers)}, expected only {owner!r}"
                + (f"; a SECOND writer appeared: {extra} - offer the picture through set_result/set_source instead"
                   if extra else ""))
    return problems


def _external_stores(source: str, facts: set[str]) -> list[str]:
    """Direct `<pane handle>.<fact> = ...` stores, as `attribute (function)` strings."""
    found: list[str] = []

    def receiver_handle(node: ast.expr) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None

    def walk(node: ast.AST, function: str | None) -> None:
        for child in ast.iter_child_nodes(node):
            name = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else function
            if (isinstance(child, ast.Attribute) and isinstance(child.ctx, ast.Store)
                    and child.attr in facts and receiver_handle(child.value) in PANE_HANDLES):
                found.append(f"{child.attr} (in {name or '<module>'})")
            walk(child, name)

    walk(_parse(source), None)
    return found


def _counter_sites(source: str) -> list[tuple[str | None, str | None, int]]:
    """`(class, nearest function, line)` for every call to `telemetry.note_pair_commit(...)`."""
    sites: list[tuple[str | None, str | None, int]] = []

    def walk(node: ast.AST, cls: str | None, function: str | None) -> None:
        for child in ast.iter_child_nodes(node):
            child_cls = child.name if isinstance(child, ast.ClassDef) else cls
            child_fn = (child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                        else function)
            if isinstance(child, ast.Call):
                func = child.func
                if isinstance(func, ast.Attribute) and func.attr == COUNTER_CALL:
                    sites.append((cls, function, child.lineno))
            walk(child, child_cls, child_fn)

    walk(_parse(source), None, None)
    return sites


def _commit_store_line(source: str) -> int | None:
    """The line of `self._committed = ...` in `PairWriter._paint` - the store that makes a pair the picture on screen."""
    method = None
    for node in _class_node(source, PAIR_CLASS).body:
        if isinstance(node, ast.FunctionDef) and node.name == PAIR_COMMIT_FUNCTION:
            method = node
    if method is None:
        return None
    for sub in ast.walk(method):
        if (isinstance(sub, ast.Attribute) and isinstance(sub.ctx, ast.Store)
                and sub.attr == "_committed"
                and isinstance(sub.value, ast.Name) and sub.value.id == "self"):
            return sub.lineno
    return None


def _counter_problems(source: str) -> list[str]:
    """The counter must sit in `_paint`, after `self._committed` is stored, and nowhere else in the file."""
    problems: list[str] = []
    sites = _counter_sites(source)
    if not sites:
        problems.append(f"{COUNTER_CALL} has no call site: the counter exists but nobody writes it "
                        f"(CODE_PATTERNS §7)")
        return problems
    if len(sites) > 1:
        problems.append(f"{COUNTER_CALL} is called {len(sites)} times {sites[1:]}: a pair must be counted once, "
                        f"where it is committed")
    cls, function, line = sites[0]
    if (cls, function) != (PAIR_CLASS, PAIR_COMMIT_FUNCTION):
        problems.append(
            f"{COUNTER_CALL} is called in {cls}.{function}, not in {PAIR_CLASS}.{PAIR_COMMIT_FUNCTION}: the counter "
            f"must ride the commit, not the decision entry (`paint()` counts a hold as a paint)")
    commit = _commit_store_line(source)
    if commit is None:
        problems.append(f"{PAIR_CLASS}.{PAIR_COMMIT_FUNCTION} no longer stores `self._committed` - the commit moved, "
                        f"and this gate can no longer place the counter")
    elif line < commit:
        problems.append(f"{COUNTER_CALL} is called at line {line}, before `self._committed` is stored at line "
                        f"{commit}: an intent is counted, not a commit")
    return problems


# -- the real tree -------------------------------------------------------


def test_every_pane_picture_fact_has_exactly_one_writer():
    problems = _single_writer_problems(PANE_FILE.read_text(encoding="utf-8"),
                                       PANE_PICTURE_FACTS, KNOWN_SECOND_WRITERS)
    assert not problems, ("the pane's picture state must have one writer per fact "
                          "(CODE_PATTERNS §2):\n  " + "\n  ".join(problems))


def test_no_module_outside_the_pane_writes_its_picture_facts():
    facts = set(PANE_PICTURE_FACTS) | set(KNOWN_SECOND_WRITERS)
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        if path == PANE_FILE:
            continue
        for store in _external_stores(path.read_text(encoding="utf-8", errors="replace"), facts):
            offenders.append(f"{path.relative_to(ROOT)}: {store}")
    assert not offenders, ("only the pane's own methods decide which picture it shows; a caller sets it through "
                           "set_result/set_source:\n  " + "\n  ".join(offenders))


def test_the_pair_counter_is_written_in_the_function_that_commits_the_pair():
    problems = _counter_problems(PAIR_FILE.read_text(encoding="utf-8"))
    assert not problems, "the pair commit counter is misplaced:\n  " + "\n  ".join(problems)


# -- the gates themselves can go red ------------------------------------


def test_the_ownership_gate_catches_a_second_writer():
    """A source with a second `self._frame` store must be reported, naming the fact and the second method."""
    source = '''
class VideoPreviewSurface:
    def __init__(self):
        self._frame = None
        self._shown_time = None

    def _accept_result(self, image):
        self._frame = image
        self._shown_time = 1.0

    def set_view_mode(self, mode):
        self._frame = None          # the second writer this gate exists to catch
'''
    problems = _single_writer_problems(source, {"_frame": "_accept_result",
                                                "_shown_time": "_accept_result"}, {})
    assert problems, "a second writer must be red"
    assert any("_frame" in problem and "set_view_mode" in problem for problem in problems), problems


def test_the_ownership_gate_catches_a_ratchet_entry_that_rots():
    """A KNOWN fact fixed to one writer must fail until its entry is deleted; a new third writer must fail too."""
    fixed = '''
class VideoPreviewSurface:
    def __init__(self):
        self._original = None

    def _apply_source_picture(self):
        self._original = 1
'''
    problems = _single_writer_problems(fixed, {"_original": "_apply_source_picture"},
                                       {"_original": {"_apply_source_picture", "_accept_result"}})
    assert any("KNOWN" in problem for problem in problems), problems

    third = '''
class VideoPreviewSurface:
    def __init__(self):
        self._original = None

    def _apply_source_picture(self):
        self._original = 1

    def _accept_result(self):
        self._original = None

    def set_result(self):
        self._original = 2
'''
    problems = _single_writer_problems(third, {"_original": "_apply_source_picture"},
                                       {"_original": {"_apply_source_picture", "_accept_result"}})
    assert any("NEW writer" in problem and "set_result" in problem for problem in problems), problems


def test_the_counter_gate_catches_the_counter_moved_to_the_decision_entry():
    """The 2026-11-27 regression shape: `paint()` counts, `_paint()` commits, a hold reads as a pair."""
    source = '''
class PairWriter:
    def paint(self):
        if not self._pending_result:
            return False
        telemetry.note_pair_commit(moment=None)
        return self._paint()

    def _paint(self):
        pair = object()
        self._committed = pair
        return True
'''
    problems = _counter_problems(source)
    assert problems, "a counter at the decision entry must be red"
    assert any("PairWriter.paint" in problem for problem in problems), problems


def test_the_counter_gate_catches_a_counter_written_before_the_commit():
    source = '''
class PairWriter:
    def paint(self):
        return self._paint()

    def _paint(self):
        telemetry.note_pair_commit(moment=None)
        pair = object()
        self._committed = pair
        return True
'''
    problems = _counter_problems(source)
    assert any("before `self._committed`" in problem for problem in problems), problems
