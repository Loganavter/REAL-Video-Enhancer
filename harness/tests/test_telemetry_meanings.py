"""What a telemetry line MEANS, pinned where a proof reads it.

Four findings of the zero-trust telemetry audit (2026-11-27) share one failure: a line means something a reader does
not expect, and the readers are the live proofs this project ships against. This file pins the corrected contract:

* 5.4 - `preview: draw` is a DELTA log at levels 0/1 (one line per change, not per paint) and one-per-paint only at
  level 2; the line says which it is (`record=`), and the matrix rows read the per-window `stats paints=` counter
  instead of counting draw lines.
* 5.7 - `pool=` used to mean decoded frames and now meant the whole source register; the field is renamed (`held=`)
  and `newest=` says in its own name that it is generation-filtered (`held_newest=`).
* 5.10 - two "since process start" numbers use two different zeros; the dictionary now names both, so a reader
  cannot carry one number into the other's table (the backend one-clock fix belongs to the RenderVideo lane).
* 5.11 - a `key=value` record with no escaping could not carry a reason containing a colon and a space; values are
  quoted and `telemetry.parse` is the documented reader.

No Qt widget is needed for 5.4/5.7/5.11 beyond importing the tab's method; 5.10 is a contract over the reader's dictionary.
"""

from __future__ import annotations

import pathlib
import sys
import types

import pytest

from src.ui.video import telemetry

# The app's optional runtime imports, stubbed only when the interpreter lacks them (the same pattern as
# `test_process_readouts_language.py`): `ProcessTab` is where the compare-fill line is written.
for _name, _attrs in (("cpuinfo", {"get_cpu_info": lambda *args, **kwargs: {}}),
                      ("pypresence", {"Presence": object})):
    try:  # pragma: no cover - the branch depends on the interpreter
        __import__(_name)
    except ModuleNotFoundError:  # pragma: no cover
        _module = types.ModuleType(_name)
        for _attr, _value in _attrs.items():
            setattr(_module, _attr, _value)
        sys.modules[_name] = _module
if "resources_rc" not in sys.modules:          # a test tree has no compiled resource blob
    _resources = types.ModuleType("resources_rc")
    _resources.qInitResources = lambda: None
    sys.modules["resources_rc"] = _resources

_ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    """A clean module per test: counters, writer, clock and the level all reset."""
    lines: list[tuple[str, str]] = []
    now = {"t": 0.0}
    telemetry.set_writer(lambda line, which: lines.append((line, which)))
    telemetry.set_clock(lambda: now["t"])
    telemetry._reset(session=True)
    yield lines, now
    telemetry.set_writer(None)
    telemetry.set_clock(None)


def _level(monkeypatch, value: str | None):
    if value is None:
        monkeypatch.delenv(telemetry.ENV_NAME, raising=False)
    else:
        monkeypatch.setenv(telemetry.ENV_NAME, value)


# --- 5.4: a draw line says whether it is a delta or one-per-paint --------------------------------------------------


def test_a_draw_line_names_its_own_record_kind(monkeypatch, fresh):
    """At level 1 a `draw` line appears on a CHANGE; only at level 2 is it one per paint.

    The reader cannot tell the two apart by counting the lines (`0 draw lines` is the healthy state of a cleared pane
    at level 1), so the line carries the fact itself.
    """
    lines, _ = fresh
    _level(monkeypatch, "1")
    telemetry.event("draw", mode="sbs", half="pair")
    assert telemetry.parse(lines[-1][0])["record"] == "delta", lines[-1][0]

    _level(monkeypatch, "2")
    telemetry.event("draw", mode="sbs", half="pair")
    assert telemetry.parse(lines[-1][0])["record"] == "paint", lines[-1][0]


def test_the_matrix_reads_the_paint_counter_not_the_draw_lines():
    """Row 7 used `grep -c "preview: draw"`; a cleared pane writes ZERO draw lines at level 1 and that is healthy."""
    script = (_ROOT / "harness/scripts/live_matrix.sh").read_text(encoding="utf-8")
    assert 'grep -c "preview: draw"' not in script, "the matrix still counts delta lines as paints"
    assert "window_paints" in script, "no per-window paint counter reader"
    assert "preview: stats" in script and "paints=" in script, script[:200]


def test_the_dictionary_calls_the_draw_log_a_delta():
    doc = (_ROOT / "harness/docs/LIVE-TESTING.md").read_text(encoding="utf-8")
    assert "record=delta" in doc and "record=paint" in doc, "the dictionary does not name the two draw records"
    lower = doc.lower()
    assert "delta record" in lower, "the dictionary does not call the draw log a delta"
    assert "never by counting" in lower, "the dictionary still teaches counting draw lines as paints"


# --- 5.7: the register fields are renamed, and newest says it is filtered -------------------------------------------


class _Register:
    """The three answers `_compare_fill_report` asks a `SourceRegister`."""

    def __init__(self, total: int, block: int, decoder: int, newest: float | None, generation_seen: list[int]):
        self._total = total
        self._counts = {"block": block, "decoder": decoder}
        self._newest = newest
        self._generation_seen = generation_seen

    def __len__(self) -> int:
        return self._total

    def counts(self) -> dict[str, int]:
        return dict(self._counts)

    def newest_moment(self, generation: int | None = None):
        if generation is not None:
            self._generation_seen.append(int(generation))
        return self._newest


class _Tab:
    """Only the attributes `_compare_fill_report` touches."""

    _result_generation = 7
    _compare_playhead = lambda self: 4.25          # noqa: E731 - the method the report reads

    def __init__(self, register):
        self._source_register = register
        # Per instance: the report mutates `at`, and a class-level dict made the second test's window look fresh.
        self._compare_fill_stats = {"asks": 3, "arrivals": 2, "at": 0.0, "target": 1.5, "span": 1}


def _fill_line(fresh) -> dict[str, str]:
    from src.ui.ProcessTab import ProcessTab

    lines, _ = fresh
    generation_seen: list[int] = []
    register = _Register(total=2, block=1, decoder=1, newest=4.0, generation_seen=generation_seen)
    ProcessTab._compare_fill_report(_Tab(register))
    fill = [line for line, _ in lines if line.startswith("preview: compare fill")]
    assert fill, lines
    fields = telemetry.parse(fill[-1])
    assert generation_seen == [7], "the newest moment was not asked for the result's generation"
    return fields


def test_the_register_count_is_not_called_a_pool(fresh):
    """`pool=` meant a ±0.5 s decoder pool before `6c8f54b8` and the whole register after it: an old note cannot quote
    today's number. The field is renamed and its producer stopped printing the dead `pool_min=`."""
    fields = _fill_line(fresh)
    assert "pool" not in fields and "pool_min" not in fields, fields
    assert fields["held"] == "2", fields
    assert fields["held_block"] == "1" and fields["held_decoder"] == "1", fields


def test_newest_says_in_its_name_that_it_is_generation_filtered(fresh):
    """`held=24 held_newest=-` is a legitimate state (an older generation holds them), not an empty register."""
    fields = _fill_line(fresh)
    assert "newest" not in fields, fields
    assert float(fields["held_newest"]) == pytest.approx(4.0), fields


def test_an_older_generation_reads_as_a_dash_and_the_docs_say_so(fresh):
    """A dash is 'nothing from THIS generation', and the dictionary must say it where the reader looks."""
    from src.ui.ProcessTab import ProcessTab

    lines, _ = fresh
    register = _Register(total=24, block=0, decoder=24, newest=None, generation_seen=[])
    ProcessTab._compare_fill_report(_Tab(register))
    fields = telemetry.parse([line for line, _ in lines if line.startswith("preview: compare fill")][-1])
    assert fields["held"] == "24" and fields["held_newest"] == "-", fields
    doc = (_ROOT / "harness/docs/LIVE-TESTING.md").read_text(encoding="utf-8")
    assert "held_newest" in doc and "generation" in doc


# --- 5.10: the reader names the zero (the backend reconciliation is the RenderVideo lane's) --------------------------


def test_the_dictionary_names_which_zero_each_since_number_used():
    """One zero, named. This test first pinned the SPLIT (two zeros, not subtractable, backend lane owns the fix) because
    that is what the code did; the backend lane landed the one-clock change on 2026-11-27, so the contract is now the
    opposite: the dictionary must say the numbers share `Util`'s mark, and must not still describe two origins.
    """
    doc = (_ROOT / "harness/docs/LIVE-TESTING.md").read_text(encoding="utf-8")
    assert "Util.py" in doc, "the process zero is not named"
    assert "SAME zero since 2026-11-27" in doc, "the dictionary does not say the lines share one zero"
    assert "can be subtracted now" in doc, "the dictionary does not say the numbers are comparable"
    assert "cannot be subtracted" not in doc, "the dictionary still describes the old, split clocks"


# --- 5.11: a value with `=`, `:` and a space round-trips through the documented parser ------------------------------


REASON = "queue-decode-failed: boom happened = exit 1"


def test_a_reason_with_a_colon_and_a_space_round_trips(fresh):
    """The audit observed `reason=queue-decode-failed: boom happened` parse back as `queue-decode-failed:`."""
    lines, _ = fresh
    telemetry.event("seek refused", press=1, reason=REASON)
    line = lines[-1][0]
    fields = telemetry.parse(line)
    assert fields["reason"] == REASON, fields
    assert fields["press"] == "1" and fields["_event"] == "seek refused", fields


def test_a_value_cannot_inject_a_second_record(fresh):
    lines, _ = fresh
    telemetry.event("seek refused", press=2, reason="line one\npreview: forged record=x")
    line = lines[-1][0]
    assert "\n" not in line, line
    assert telemetry.parse(line)["reason"] == "line one\\npreview: forged record=x"


def test_the_session_and_snapshot_writers_use_the_same_quoting(monkeypatch, fresh):
    """A path with a space and a state value with a space are records too; the header's `file=` is built by hand."""
    lines, _ = fresh
    telemetry.session_start(file="/tmp/a directory/preview-log.txt", pane_w=500)
    session = telemetry.parse(lines[-1][0])
    assert session["file"] == "/tmp/a directory/preview-log.txt", session

    lines.clear()
    _level(monkeypatch, "1")
    telemetry.note_snapshot(kind="queue file", duration=2.0)      # the first call only starts the window
    telemetry.set_clock(lambda: 5.0)
    assert telemetry.note_snapshot(kind="queue file", duration=2.0) is True
    fields = telemetry.parse(lines[-1][0])
    assert fields["kind"] == "queue file" and fields["duration"] == "2.000", fields


def test_the_dictionary_documents_the_parser_the_readers_use():
    doc = (_ROOT / "harness/docs/LIVE-TESTING.md").read_text(encoding="utf-8")
    assert "telemetry.parse" in doc, "the dictionary still recommends an escaping-free split"
    assert "reason=" in doc and "queue-decode-failed" in doc, "no quoted-value example"
