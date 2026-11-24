"""Every record in the evidence registry satisfies the `Record` contract.

Why a gate: the registry is the project's proof, and seven records once carried a *sentence* in `statistic`
("median of 2 interleaved rounds x 20 iterations", "share of pixels >8 levels in any channel, worst frame of 16"). They
were written by scripts that build dictionaries instead of `Record`, so nothing validated them - and a reader could not
tell a `median` from a `worst_of` without prose. The dataset is the thing every bar, verdict and promotion rests on, so
its own invariants are checked here rather than trusted.

The check is `Record.validate()` on every line: a non-finite value, an unknown unit/statistic/tier, a derived record
without formula+inputs, a measured record with neither a run nor an artifact, an artifact without a digest, or a unit
the registry does not know all fail. `test_evidence_artifacts.py` covers the artifact files themselves; this covers the
record shape.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from harness.core import Record  # noqa: E402

RECORDS = ROOT / "harness" / "docs" / "wsp" / "verified" / "records"

#: `to_dict()` adds this derived convenience key; it is not a `Record` field.
_NOT_FIELDS = ("family",)


def _records():
    """Every parsed record with its file and line, for a failure message that points at the line."""
    for path in sorted(RECORDS.glob("*.jsonl")):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                pytest.fail(f"{path.name}:{line_no} is not JSON: {exc}")
            for key in _NOT_FIELDS:
                data.pop(key, None)
            yield path.name, line_no, data


def test_every_record_satisfies_the_contract():
    failures = []
    total = 0
    for name, line_no, data in _records():
        total += 1
        try:
            Record(**{key: value for key, value in data.items() if key in Record.__dataclass_fields__}).validate()
        except Exception as exc:          # RecordError, TypeError from a renamed field, anything
            failures.append(f"{name}:{line_no} id={data.get('id')!r}: {exc}")
    assert total > 1000, f"the registry answered only {total} records - the corpus is missing, not clean"
    assert not failures, f"{len(failures)} record(s) break the contract:\n" + "\n".join(failures[:12])


def test_no_record_carries_a_sentence_in_a_structured_field():
    """`statistic`, `unit` and `tier` are vocabulary, not prose: the prose belongs beside them."""
    offenders = []
    for name, line_no, data in _records():
        for field in ("statistic", "unit", "tier"):
            value = data.get(field)
            if isinstance(value, str) and " " in value.strip():
                offenders.append(f"{name}:{line_no} {field}={value!r}")
    assert not offenders, ("a structured field holds a sentence:\n" + "\n".join(offenders[:12]))
