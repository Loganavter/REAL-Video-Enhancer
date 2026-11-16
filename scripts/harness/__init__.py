"""Evidence harness for this project.

The failure mode it exists for is not "the model computed wrong". It is: a number was treated as a
scalar while it is a tuple ``(value, unit, population, instrument, config, session, n, statistic)``,
and the missing fields were filled from memory. Ten conclusions were withdrawn that way.

Shape of the thing:

* ``core.Record`` - the tuple. ``core.Population`` carries the *item set hash*, not a label, so two
  equal labels over different frames are no longer equal.
* ``run.run`` - a process run returns ``(exit_code, stdout, stderr, parsed)`` and never detaches.
  ``run.probe`` is three-state; ``ResultsTable`` refuses an ``error`` row.
* ``run.parse_ptxas`` / ``parse_engine_inspector`` / ``parse_checkpoint`` - artifacts parsed into
  schemas with explicit ``require`` accessors ("smem" must say static or dynamic).
* ``gates`` - A/A before A/B, refusal only relative to a verified base, a bar that names what it is
  a bar of, claims checked by key, and ``Scope.finalize`` which refuses a green over an uncovered
  scope.
* ``astcheck`` - dead stores and use-before-assignment, the two non-measurement bugs that produced a
  false refusal and a fake verdict.

Division of labour, stated once: the harness emits numbers, the model proposes experiments. Anything
the model prints as a number is, by construction, not evidence.
"""
from .core import Population, Record, RecordError, Store, UNITS, unit_family
from .gates import (AA, Bar, Claim, Scope, aa_gate, below, check_claims, graph_counts,
                    graph_delta, refuse, require_config_pair, shared_calibration, suspicious_identity)
from .run import (Probe, ResultsTable, Run, RunError, Schema, claimable, parse_checkpoint,
                  parse_engine_inspector, parse_ptxas, probe, run)

__all__ = [
    "Population", "Record", "RecordError", "Store", "UNITS", "unit_family",
    "AA", "Bar", "Claim", "Scope", "aa_gate", "below", "check_claims", "graph_counts",
    "graph_delta", "refuse", "require_config_pair", "shared_calibration", "suspicious_identity",
    "Probe", "ResultsTable", "Run", "RunError", "Schema", "claimable", "parse_checkpoint",
    "parse_engine_inspector", "parse_ptxas", "probe", "run",
]
__version__ = "0.1.0"
