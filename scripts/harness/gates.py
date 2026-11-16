"""Gates: the rules that turn records into decisions, and the places where a green result is not allowed.

Four gates, each named after the class of retraction it closes:

``aa_gate`` / ``refuse``
    A/A before A/B, and a refusal only relative to a base that passed its own A/A (class 3). The
    expensive error in this project was a false refusal - a correct kernel rejected because a dead
    line had corrupted both arms to identical 2.15278% - and a false refusal is never re-checked.

``below``
    A bar is a record too, and it names what it is a bar *of*. ``peak 7`` against a bar of ``8``
    fails because 8 is the share threshold, not the peak bar (class 7).

``check_claims``
    Claims are checked by key, and the result carries the set of kinds that were *not* checked
    (class 8). ``Scope.finalize`` refuses to be green while an expected kind is uncovered.

``cause``
    A mechanism may be named only with a pair of records that differ in that variable and nothing
    else (class 2). Two records on the same population never support a cause.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .core import Population, Record, RecordError, Store, unit_family

# --------------------------------------------------------------------------------------------------
# bars
# --------------------------------------------------------------------------------------------------
OF_KINDS = ("share", "peak", "frame_time", "engine_time", "aa_delta", "pixel_defect",
            "speedup", "excess", "pass_rate", "slope")


@dataclass(frozen=True)
class Bar:
    name: str
    value: float
    unit: str
    of: str                     # what this bar is a bar OF: see OF_KINDS
    statistic: str = "worst_of"
    min_n: int = 2              # a bar on a sample needs a sample
    note: str = ""


def below(rec: Record, bar: Bar) -> None:
    """Refuse a record that is not comparable with the bar, or is not under it.

    The bar is authoritative and the record has to earn its comparison: same unit (not family), the
    statistic the bar is stated in, a sample at least ``bar.min_n`` wide, a measured value, and an
    ``of`` from the fixed vocabulary. A record whose author writes its own ``of`` to match the bar is
    no longer enough on its own.
    """
    rec.validate()
    if rec.tier == "assumed":
        raise RecordError(f"{rec.id} is assumed - an assumed number is not evidence for a bar")
    if rec.unit != bar.unit:
        hint = " (same family, different scale)" if unit_family(rec.unit) == unit_family(bar.unit) else ""
        raise RecordError(f"{rec.id}: {rec.value:g} {rec.unit} cannot be judged against bar "
                          f"{bar.name} = {bar.value:g} {bar.unit}{hint}")
    if bar.of not in OF_KINDS:
        raise RecordError(f"bar {bar.name}: of={bar.of!r} is not one of {OF_KINDS}")
    if unit_family(rec.unit) != unit_family(bar.unit):
        raise RecordError(f"{rec.id}: {rec.value:g} {rec.unit} cannot be judged against bar "
                          f"{bar.name} = {bar.value:g} {bar.unit}")
    if not bar.of or not rec.extra.get("of"):
        raise RecordError(f"bar {bar.name} does not say what it is a bar of "
                          f"(bar.of={bar.of!r}, record.of={rec.extra.get('of')!r})")
    if bar.of != rec.extra["of"]:
        raise RecordError(f"{rec.id} measures {rec.extra['of']!r} and bar {bar.name} is a bar of "
                          f"{bar.of!r} - a peak is not judged against a share threshold")
    if rec.statistic != bar.statistic:
        raise RecordError(f"{rec.id} is statistic {rec.statistic!r}, bar {bar.name} is "
                          f"{bar.statistic!r} - n=1 must not be certified against a worst-of bar")
    if rec.population.n < bar.min_n:
        raise RecordError(f"{rec.id} rests on n={rec.population.n} and bar {bar.name} needs "
                          f"n>={bar.min_n} - one sample is a smoke test, not a verdict")
    if not rec.value < bar.value:
        raise RecordError(f"{rec.id} = {rec.value:g} {rec.unit} is not below {bar.value:g} "
                          f"({bar.name}) on '{rec.population.claims_text()}'")


# --------------------------------------------------------------------------------------------------
# A/A and refusal
# --------------------------------------------------------------------------------------------------
@dataclass
class AA:
    identical: bool
    max_delta: float
    n: int
    record: Record
    detail: str = ""


def aa_gate(store: Store, a: str, b: str, *, tol: float = 0.0,
            calibration: str = "") -> AA:
    """Two runs of the *unchanged* configuration. If they are not identical, A/B measures the
    harness, not the change."""
    ra, rb = store.get(a), store.get(b)
    for f in ("instrument", "config", "session", "statistic"):
        va, vb = getattr(ra, f), getattr(rb, f)
        if va != vb:
            raise RecordError(f"A/A needs one configuration: {f} differs ({va!r} vs {vb!r})")
    if ra.population.items_sha != rb.population.items_sha:
        raise RecordError("A/A needs the same item set: "
                          f"{ra.population.claims_text()} vs {rb.population.claims_text()}")
    d = abs(ra.value - rb.value)
    ok = d <= tol
    rec = store.emit(Record(
        value=round(d, 9), unit=ra.unit, population=ra.population, instrument="aa_delta",
        config=ra.config, session=ra.session, statistic="max", tier="measured",
        run=ra.run, formula=f"|{a}-{b}|", inputs=(a, b),
        extra={"of": "aa_delta", "calibration": calibration, "identical": ok}))
    return AA(identical=ok, max_delta=d, n=ra.population.n, record=rec,
              detail=f"|{a}-{b}| = {d:g} {ra.unit}")


def refuse(store: Store, claim_id: str, base: str, aa: AA, reason: str) -> None:
    """A refusal relative to a base. Unchecked bases produce false refusals, which are permanent.

    The ``AA`` dataclass is forgeable by a caller (``AA(identical=True, ...)`` with no run behind it),
    so the gate does not trust it: the emitted A/A record must exist in the store and the pair is
    re-derived from the record's own inputs before a refusal is written.
    """
    rec = store.get(aa.record.id)
    if aa.record.id != rec.id or len(rec.inputs) != 2:
        raise RecordError(f"refusing to record {claim_id}: the A/A it cites is not a two-input record "
                          f"in this store")
    fresh = aa_gate(store, rec.inputs[0], rec.inputs[1], calibration=rec.extra.get("calibration", ""))
    if not fresh.identical:
        raise RecordError(f"refusing to record {claim_id}: re-deriving the A/A from {rec.inputs} gives "
                          f"{fresh.detail}, not identity")
    if not aa.identical:
        raise RecordError(f"refusing to record {claim_id}: the base {base} did not pass its own A/A "
                          f"({aa.detail}) - an A/B on a broken base rejects the correct variant")
    if store.get(base).tier != "measured":
        raise RecordError(f"refusing to record {claim_id}: base {base} is "
                          f"{store.get(base).tier}, not measured")
    store.emit(Record(value=-1.0, unit="count", population=store.get(base).population,
                      instrument="refusal", config=store.get(base).config,
                      session=store.get(base).session, statistic="single", tier="measured",
                      run=aa.record.run, note=f"{claim_id}: {reason}"))


# --------------------------------------------------------------------------------------------------
# claims and coverage
# --------------------------------------------------------------------------------------------------
@dataclass
class Claim:
    id: str
    kind: str                       # below | ratio | cause | verdict | equal
    args: tuple[str, ...]
    claimed: float | None = None
    mechanism: str = ""
    bar: Bar | None = None


@dataclass
class Scope:
    """What this run checked, and what it did not. Green is a floor: ``finalize`` refuses a green
    verdict while an expected claim kind is uncovered."""
    checked: list[str] = field(default_factory=list)
    unchecked: list[str] = field(default_factory=list)
    expected: set[str] = field(default_factory=set)

    def cover(self, kind: str) -> None:
        if kind not in self.checked:
            self.checked.append(kind)

    def miss(self, kind: str, why: str) -> None:
        self.unchecked.append(f"{kind}: {why}")

    def report(self) -> str:
        out = [f"checked: {', '.join(self.checked) if self.checked else 'nothing'}"]
        if self.unchecked:
            out.append("NOT checked:")
            out += [f"  - {u}" for u in self.unchecked]
        else:
            out.append("NOT checked: nothing declared - see harness.gates for what it cannot see")
        return "\n".join(out)

    def finalize(self) -> None:
        missing = sorted(self.expected - set(self.checked))
        if missing:
            raise RecordError(
                f"green over an uncovered scope: {missing} expected but not checked "
                f"({len(self.checked)} kinds covered). A pass that does not cover the claim is vacuous")


def _varying(a: Record, b: Record) -> tuple[set[str], bool]:
    """What differs between two records, and whether their populations differ.

    A population difference is never a mechanism: it is the class-1 error (mask area vs mask +
    dilation), so a cause needs the SAME item set and exactly one of config/instrument/statistic.
    """
    fields = ("instrument", "config", "statistic", "session")
    diff = {f for f in fields if getattr(a, f) != getattr(b, f)}
    return diff, a.population.items_sha != b.population.items_sha


def check_claims(store: Store, claims: Iterable[Claim], *, scope: Scope | None = None,
                 expected: Iterable[str] = ("below", "ratio", "verdict")) -> tuple[list[str], Scope]:
    """Check every claim by key. Returns the problems and the scope of what was verified."""
    if scope is not None:
        scope.expected |= set(expected)      # a passed scope must not lose the caller's expectations
        sc = scope
    else:
        sc = Scope(expected=set(expected))
    problems: list[str] = []

    known = {"below", "ratio", "verdict", "cause", "equal"}
    for c in claims:
        if c.kind in known:
            sc.cover(c.kind)          # coverage means the checker looked, not that it passed
        try:
            if c.kind == "below":
                rec = store.get(c.args[0])
                if c.bar is None:
                    raise RecordError(f"{c.id}: @below needs a bar")
                below(rec, c.bar)
                sc.cover("below")

            elif c.kind == "ratio":
                for r in c.args:
                    store.get(r)
                store.ratio(c.args[0], c.args[1], c.claimed)
                sc.cover("ratio")

            elif c.kind == "verdict":
                recs = [store.get(r).validate() for r in c.args]
                if len(recs) < 2:
                    raise RecordError(f"{c.id}: a verdict needs at least two records")
                keys = {(r.instrument, r.config, r.session, r.statistic, r.unit,
                         r.population.items_sha) for r in recs}
                if len(keys) > 1:
                    raise RecordError(f"{c.id}: verdict mixes {len(keys)} (instrument, config, "
                                      f"session, population) keys - each claim may be true and the "
                                      f"verdict is not")
                bad = [(r.id, r.tier) for r in recs if r.tier != "measured"]
                if bad:
                    raise RecordError(f"{c.id}: a verdict is taken over measurements; cites {bad}"
                                      f" - a derived number whose arithmetic was never recomputed, or an "
                                      f"assumed one, cannot carry a decision")
                sc.cover("verdict")

            elif c.kind == "cause":
                a, b = store.get(c.args[0]).validate(), store.get(c.args[1]).validate()
                if a.tier != "measured" or b.tier != "measured":
                    raise RecordError(f"{c.id}: a cause needs two measurements")
                diff, pop_differs = _varying(a, b)
                if pop_differs:
                    raise RecordError(
                        f"{c.id}: '{c.mechanism}' cites two different populations "
                        f"('{a.population.claims_text()}' vs '{b.population.claims_text()}') - a "
                        f"population difference is the error, not the mechanism")
                if len(diff) != 1:
                    raise RecordError(
                        f"{c.id}: '{c.mechanism}' cites two records differing in {sorted(diff) or 'nothing'} "
                        f"- the mechanism was never varied (exactly one field must differ)")
                sc.cover("cause")

            elif c.kind == "equal":
                a, b = store.get(c.args[0]).validate(), store.get(c.args[1]).validate()
                if "assumed" in (a.tier, b.tier):
                    raise RecordError(f"{c.id}: equality of an assumed number is not evidence")
                if a.instrument != b.instrument or a.config != b.config:
                    raise RecordError(f"{c.id}: {a.id} ({a.instrument}/{a.config}) and {b.id} "
                                      f"({b.instrument}/{b.config}) are not the same measurement point")
                if a.value != b.value or a.unit != b.unit:
                    raise RecordError(f"{c.id}: {a.id}={a.value:g}{a.unit} != {b.id}={b.value:g}{b.unit}")
                sc.cover("equal")

            else:
                sc.miss(c.kind, f"claim {c.id}: this checker has no rule for kind {c.kind!r}")
        except RecordError as exc:
            problems.append(f"{c.id}: {exc}")

    for kind in sorted(set(expected) - set(sc.checked)):
        sc.miss(kind, "no claim of this kind in the file - the check did not run")
    return problems, sc


# --------------------------------------------------------------------------------------------------
# config pairing, calibration, graph, suspicious identity
# --------------------------------------------------------------------------------------------------
def require_config_pair(store: Store, config: str, *, quantities: Iterable[str] = ("speed", "accuracy")):
    """A fusion changes the graph and the numbers together. Speed for one config and accuracy for
    another is not a pair, and the three times this project compared them it was wrong."""
    have = {r.extra.get("quantity") for r in store.records.values()
            if r.config == config and r.instrument != "derived"}
    missing = sorted(set(quantities) - have)
    if missing:
        raise RecordError(f"config {config}: no {missing} measured - re-measure the pair after a "
                          f"graph or precision change (have {sorted(q for q in have if q)})")


def shared_calibration(store: Store, ids: Iterable[str], *, expect: str | None = None) -> str:
    """One set of forward passes feeds scales, probes and the tuner. Two consumers on two
    calibrations drift silently."""
    got = {store.get(i).extra.get("calibration") for i in ids}
    if None in got or len(got) != 1:
        raise RecordError(f"consumers cite different calibration sets: {sorted(map(str, got))}")
    cal = got.pop()
    if expect is not None and cal != expect:
        raise RecordError(f"calibration {cal!r} != expected {expect!r}")
    return cal


GRAPH_KEYS = ("total", "fused_mylin", "reformat", "pointwise")


def graph_counts(rec: Record) -> dict:
    g = rec.extra.get("graph")
    if not isinstance(g, dict) or any(k not in g for k in GRAPH_KEYS):
        raise RecordError(f"{rec.id}: a claim after a graph edit must cite layer counts "
                          f"({', '.join(GRAPH_KEYS)}), not only time (got {g!r})")
    return g


def graph_delta(base: Record, new: Record) -> dict:
    """After a graph edit, compare partitioning before comparing time: ``F.pad`` cut the engine in
    two and cost 71% while every layer looked the same."""
    gb, gn = graph_counts(base), graph_counts(new)
    return {k: (gb.get(k), gn.get(k)) for k in ("total", "fused_mylin", "reformat", "pointwise")
            if gb.get(k) != gn.get(k)}


def suspicious_identity(a: Record, b: Record, digits: int = 5) -> str:
    """Two *different* implementations agreeing to five digits is evidence of a shared fault, not of
    agreement. This project read that signal as 'the PReLU is applied the same way'."""
    if a.instrument == b.instrument:
        return ""
    same = round(a.value, digits) == round(b.value, digits)
    return (f"{a.id} and {b.id} agree to {digits} significant digits across different instruments "
            f"({a.instrument} vs {b.instrument}) - look for a shared cause before reporting either") if same else ""
