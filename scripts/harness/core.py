"""A measurement record: the tuple, not the number.

Root cause of the ten retractions in this investigation, in one line: a number was treated as a
scalar, while a number is ``(value, unit, population, instrument, config, session, n, statistic)``.
The harness emits that tuple; prose and claims only ever cite the tuple's id.

The two fields that do the heavy lifting:

``population.items``
    Not a label. The exact item set the statistic was computed on (frame indices, problem ids,
    tile keys), hashed into ``items_sha``. ``mask area`` and ``mask + dilation`` are both "share of
    frame"; they differ here or not at all. A ratio between two records with different
    ``items_sha`` is refused - that is the class-1 failure, mechanically.

``statistic``
    ``worst_of`` / ``median`` / ``mean`` / ``min`` / ``max`` / ``single``. A ``worst_of`` may not be
    divided by a ``median``, because the quotient is not a quantity.

Compatibility for a relation, checked in this order: unit family, instrument, config, session,
population items, statistic. Every failed check names the field that differs - never "incompatible".
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field, asdict, replace
from typing import Any, Iterable

UNITS: dict[str, str] = {
    "ms": "time", "s": "time", "us": "time",
    "TOPS": "rate", "TFLOP/s": "rate", "GFLOP/s": "rate", "GB/s": "rate",
    "GFLOP": "work", "TFLOP": "work", "MAC": "work",
    "%": "share", "pp": "delta",
    "levels": "levels", "x": "factor", "dB": "quality",
    "MiB": "bytes", "GiB": "bytes", "B": "bytes", "MB": "bytes", "GB": "bytes", "KB": "bytes",
    "count": "count", "frames": "count", "ch": "count", "regs": "count",
    "min": "time",                       # minutes; the corpus also uses "min" as a statistic name
}

STATISTICS = ("single", "mean", "median", "min", "max", "worst_of", "p90", "p99", "p99.9", "sum")
TIERS = ("measured", "derived", "assumed")


class RecordError(ValueError):
    """A relation was asked for between records that do not describe the same thing."""


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def unit_family(unit: str) -> str:
    return UNITS.get(unit, unit)


# --------------------------------------------------------------------------------------------------
# population
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Population:
    """The item set a statistic was computed on. The label is for humans and is never compared."""

    kind: str
    items: tuple[str, ...] = ()
    label: str = ""

    def __post_init__(self) -> None:
        if not self.kind:
            raise RecordError("a population needs a kind (what is being counted)")

    @classmethod
    def of(cls, kind: str, items: Iterable[Any], label: str = "") -> "Population":
        return cls(kind=kind, items=tuple(sorted(str(i) for i in items)), label=label)

    @property
    def items_sha(self) -> str:
        return _sha({"kind": self.kind, "items": list(self.items)})[:16]

    @property
    def n(self) -> int:
        return len(self.items)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "items": list(self.items), "items_sha": self.items_sha,
                "n": self.n, "label": self.label}

    def claims_text(self) -> str:
        """The string the legacy auditor compares. Derived, so two equal labels with different item
        sets are no longer equal - which is the whole point."""
        return f"{self.kind} [{self.items_sha}] n={self.n}"


# --------------------------------------------------------------------------------------------------
# record
# --------------------------------------------------------------------------------------------------
@dataclass
class Record:
    value: float
    unit: str
    population: Population
    instrument: str
    config: str
    session: str
    statistic: str = "single"
    tier: str = "measured"
    id: str = ""
    formula: str = ""
    inputs: tuple[str, ...] = ()
    run: str = ""
    artifact: str = ""
    threshold_unit: str = ""      # for bars: the unit of the threshold the value is a share of
    note: str = ""
    extra: dict = field(default_factory=dict)   # calibration id, engine layer counts, anything keyed

    def __post_init__(self) -> None:
        if not isinstance(self.value, float):
            object.__setattr__(self, "value", float(self.value))
        if not math.isfinite(self.value):          # NaN satisfies every comparison, so it cannot exist
            raise RecordError(f"non-finite value {self.value!r}: it would pass every comparison")
        if self.unit not in UNITS:
            raise RecordError(f"unknown unit {self.unit!r} - add it to UNITS, do not invent one")
        if self.statistic not in STATISTICS:
            raise RecordError(f"unknown statistic {self.statistic!r}; one of {STATISTICS}")
        if self.tier not in TIERS:
            raise RecordError(f"unknown tier {self.tier!r}; one of {TIERS}")
        if not self.instrument:
            raise RecordError("a record needs an instrument (where the number was read)")
        if self.tier == "derived" and not (self.formula and self.inputs):
            raise RecordError("a derived record must carry its formula and its input ids")
        if self.tier == "measured" and not (self.run or self.artifact):
            raise RecordError("a measured record must name its run or its artifact")

    @property
    def family(self) -> str:
        return unit_family(self.unit)

    def validate(self) -> "Record":
        """Re-check the invariants before a record is used as evidence.

        Dataclasses are mutable, so a caller can change `tier` or `value` after `emit` and before a
        relation. Re-validating on use makes the mutation a loud error instead of a quiet redefinition.
        """
        if not math.isfinite(self.value):
            raise RecordError(f"{self.id or 'record'}: non-finite value {self.value!r}")
        if self.unit not in UNITS:
            raise RecordError(f"{self.id or 'record'}: unknown unit {self.unit!r}")
        if self.tier not in TIERS:
            raise RecordError(f"{self.id or 'record'}: unknown tier {self.tier!r}")
        if self.tier == "derived" and not (self.formula and self.inputs):
            raise RecordError(f"{self.id or 'record'}: derived without formula+inputs")
        if self.tier == "measured" and not (self.run or self.artifact):
            raise RecordError(f"{self.id or 'record'}: measured with no run and no artifact - provenance "
                              f"was deleted after emit")
        if self.artifact and not self.extra.get("artifact_sha256"):
            raise RecordError(f"{self.id or 'record'} cites artifact {self.artifact!r} with no digest - "
                              f"a record that names a file it never read is the class-5 error")
        return self

    def to_dict(self) -> dict:
        d = asdict(self)
        d["population"] = self.population.to_dict()
        d["family"] = self.family
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Record":
        d = dict(d)
        d.pop("family", None)
        pop = d.pop("population")
        return cls(population=Population(kind=pop["kind"], items=tuple(pop.get("items", ())),
                                         label=pop.get("label", "")),
                   **{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def claims_line(self) -> str:
        pop = self.population.claims_text().replace('"', "'")
        # repr, not :g - the claims file must round-trip to the exact float it describes
        return f'@claim {self.id} {self.value!r} {self.unit} "{pop}" {self.tier}'


def _compare_field(name: str, a: Any, b: Any, allow: bool, why: str) -> None:
    if allow or a == b:
        return
    raise RecordError(f"{name} differs: {a!r} vs {b!r} - {why}")


# --------------------------------------------------------------------------------------------------
# store + relations
# --------------------------------------------------------------------------------------------------
@dataclass
class Store:
    session: str = field(default_factory=lambda: time.strftime("s-%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    records: dict[str, Record] = field(default_factory=dict)
    seq: int = 0

    def emit(self, rec: Record) -> Record:
        if not rec.id:
            self.seq += 1
            rec = replace(rec, id=f"R{self.seq:03d}")
        if rec.id in self.records:
            raise RecordError(f"duplicate record id {rec.id}")
        rec.validate()                      # a record that cannot be evidence is not stored at all
        self.records[rec.id] = rec
        return rec

    def get(self, rid: str) -> Record:
        try:
            return self.records[rid]
        except KeyError:
            raise RecordError(f"unknown record {rid}") from None

    # -- relations --------------------------------------------------------------------------------
    def _relation(self, a_id: str, b_id: str, *, cross_session: bool, cross_population: bool,
                  cross_statistic: bool, why: str) -> tuple[Record, Record]:
        a, b = self.get(a_id).validate(), self.get(b_id).validate()
        if a.unit != b.unit:
            hint = "" if a.family != b.family else " (same family, different scale: 2000 ms / 2 s is not 2x)"
            raise RecordError(f"{why}: units differ - {a_id} is {a.unit}, {b_id} is {b.unit}{hint}"
                              f" - convert into a declared derived record, never inside a relation")
        _compare_field("instrument", a.instrument, b.instrument, False,
                       "a relation needs one measurement point (wall clock vs GPU events is not a ratio)")
        _compare_field("config", a.config, b.config, False,
                       "the configuration changed; the pair must be re-measured on the new one")
        _compare_field("statistic", a.statistic, b.statistic, cross_statistic,
                       "a worst_of and a median are not the same quantity")
        _compare_field("session", a.session, b.session, cross_session,
                       "machine drift between sessions is larger than most effects here")
        if not cross_population and a.population.items_sha != b.population.items_sha:
            raise RecordError(
                f"{why}: populations differ - {a_id} on '{a.population.claims_text()}', "
                f"{b_id} on '{b.population.claims_text()}'")
        return a, b

    def ratio(self, num: str, den: str, claimed: float | None = None, *,
              cross_session: bool = False, cross_population: bool = False,
              cross_statistic: bool = False, instrument: str = "derived", note: str = "") -> Record:
        a, b = self._relation(num, den, cross_session=cross_session,
                              cross_population=cross_population,
                              cross_statistic=cross_statistic, why="ratio")
        if b.value == 0:
            raise RecordError(f"ratio {num}/{den}: zero denominator")
        value = a.value / b.value
        if claimed is not None and abs(value - claimed) > max(0.005, 0.005 * abs(claimed)):
            raise RecordError(f"{num}/{den} = {value:.4f}, claimed {claimed:g}")
        soft = cross_session or cross_statistic or cross_population or "assumed" in (a.tier, b.tier)
        tier = "assumed" if soft else "derived"
        return self.emit(Record(
            value=round(value, 6), unit="x", population=a.population, instrument=instrument,
            config=a.config, session=a.session, statistic=a.statistic, tier=tier,
            formula=f"{num}/{den}", inputs=(num, den), note=note))

    def delta(self, a_id: str, b_id: str, *, unit: str | None = None, **kw) -> Record:
        a, b = self._relation(a_id, b_id, cross_session=kw.pop("cross_session", False),
                              cross_population=kw.pop("cross_population", False),
                              cross_statistic=kw.pop("cross_statistic", False), why="delta")
        out_unit = unit or ("pp" if a.family == "share" and a.unit == "%" else a.unit)
        soft = a.session != b.session or a.statistic != b.statistic or "assumed" in (a.tier, b.tier)
        return self.emit(Record(
            value=round(a.value - b.value, 6), unit=out_unit, population=a.population,
            instrument=kw.pop("instrument", "derived"), config=a.config, session=a.session,
            statistic=a.statistic, tier="assumed" if soft else "derived",
            formula=f"{a_id}-{b_id}", inputs=(a_id, b_id), note=kw.pop("note", "")))

    def derived(self, name: str, value: float, unit: str, formula: str, inputs: Iterable[str], **kw) -> Record:
        """A number computed from records. Its arithmetic is re-checked by ``verify_derived``."""
        refs = tuple(inputs)
        for r in refs:
            self.get(r)
        return self.emit(Record(value=value, unit=unit, population=kw.pop("population", self.get(refs[0]).population),
                                instrument=kw.pop("instrument", "derived"), config=kw.pop("config", self.get(refs[0]).config),
                                session=kw.pop("session", self.get(refs[0]).session),
                                statistic=kw.pop("statistic", self.get(refs[0]).statistic),
                                tier="derived", formula=formula, inputs=refs, **kw))

    # -- verification -----------------------------------------------------------------------------
    def verify_derived(self) -> list[str]:
        """Recompute every derived record from its inputs. Class 8: a derived number used to be free
        text that nothing read; here its arithmetic is checked like a ratio."""
        problems: list[str] = []
        for rid, r in self.records.items():
            if r.tier != "derived":
                continue
            try:
                a = self.get(r.inputs[0])
                b = self.get(r.inputs[1]) if len(r.inputs) > 1 else None
                if not r.inputs:
                    problems.append(f"{rid}: derived record cites no inputs")
                    continue
                if len(r.inputs) < 2:
                    problems.append(f"{rid}: formula {r.formula!r} needs two inputs, has {len(r.inputs)}")
                    continue
                if r.formula == f"{r.inputs[0]}/{r.inputs[1]}":
                    expect = a.value / b.value
                elif r.formula == f"{r.inputs[0]}-{r.inputs[1]}":
                    expect = a.value - b.value
                else:
                    problems.append(f"{rid}: formula {r.formula!r} is not one this verifier can recompute "
                                    f"- an unrecomputable derived number is not a result")
                    continue
                if abs(expect - r.value) > 1e-6:
                    problems.append(f"{rid}: {r.formula} = {expect:g}, recorded {r.value:g}")
            except RecordError as exc:
                problems.append(f"{rid}: {exc}")
        return problems

    # -- io ---------------------------------------------------------------------------------------
    def dump(self, path: str) -> None:
        with open(path, "w") as f:
            for r in self.records.values():
                f.write(json.dumps(r.to_dict()) + "\n")

    @classmethod
    def load(cls, path: str) -> "Store":
        st = cls()
        for lineno, line in enumerate(open(path), 1):
            line = line.strip()
            if not line:
                continue
            rec = Record.from_dict(json.loads(line))
            if rec.id in st.records:
                raise RecordError(f"{path}:{lineno}: duplicate record id {rec.id} - a hand-edited file "
                                  f"cannot silently overwrite a record")
            stated = json.loads(line)["population"].get("items_sha")
            if stated and stated != rec.population.items_sha:
                raise RecordError(f"{path}:{lineno}: {rec.id} claims items_sha {stated} but its items "
                                  f"hash to {rec.population.items_sha} - the file was edited")
            st.records[rec.id] = rec
        return st

    def claims_text(self, ids: Iterable[str] | None = None) -> str:
        out = ["# generated by harness.emit - populations are item-set hashes, not labels"]
        for rid in (ids or self.records):
            out.append(self.get(rid).claims_line())
        return "\n".join(out) + "\n"
