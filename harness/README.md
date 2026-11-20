# The harness

Evidence machinery for this repo: every number that leaves a measurement is a **record** (value, unit, population
item-set, instrument, config, session, statistic), prose cites a record's id instead of restating the value, and a
verdict citing an `assumed` record fails mechanically. The package used to live in `scripts/harness/`; it is at the
root now, and the instruments that write records live in `harness/scripts/`.

```
harness/
    cli.py        the commands: check, scope, migrate, db, source-check, astcheck
    core.py       Record / Store / Population: the tuple, the claim text, the item-set hash
    gates.py      the checks a claim has to pass; bars
    run.py        three-state probes and the A/A gate (a refusal is only valid against a base that passed its own)
    db.py         the index over records/claims/verdicts, and the ledger view
    migrate.py    the gate over the legacy claims files
    astcheck.py   dead stores / use-before-assignment in the instruments
    source.py     the clip manifest check (a `contaminated` clip taints every record that names it)
    verdicts.py   the verdict registry loader
    scripts/      the instruments that emit records: 71 files that import the harness
```

Run it from the repository root:

```bash
python3 -m harness.cli scope
python3 -m harness.cli check docs/wsp/verified/<file>.txt --records docs/wsp/verified/records/<file>.jsonl
python3 -m harness.cli db build && python3 -m harness.cli db write-block docs/RESEARCH-LEDGER.md
python3 harness/scripts/gate_sweep.py --help
```

`scripts/` still holds the 114 probes that do not import the harness: one-off instruments, benches and the
rendering utilities. A file that imports the harness belongs here; a file that does not is a probe.

Each moved instrument carries a self-contained path bootstrap (`_HARNESS_ROOT` from `__file__`), so it can be run
by path from anywhere - that replaced five different conventions (`sys.path.insert(0, "scripts")`,
`os.path.join(REPO, "scripts")`, two hardcoded absolute paths and the script-dir form), which is why the move was
mechanical rather than free.
