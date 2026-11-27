# The harness

Evidence machinery for this repo: every number that leaves a measurement is a **record** (value, unit, population
item-set, instrument, config, session, statistic), prose cites a record's id instead of restating the value, and a
verdict citing an `assumed` record fails mechanically.

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
    scripts/      187 instruments: probes, sweeps, benches and the record emitters
    tests/        the suite: 157 files, 439 tests under the app's interpreter, 1830 under the system one
```

## The tests are ours too

At the same base commit `backend/tests/` held one file - `unit_tests/test_frame.py`, the only test in the
repository - and no workflow ran pytest. The 156 files that exist today were written between 2026-11-15 and
2026-11-20 for the same measurements the harness records, so they live here and `backend/tests/unit_tests/` keeps
the upstream one. `harness/tests/conftest.py` puts the repository root and `backend/` on `sys.path`, which used to
be every caller's private `PYTHONPATH`.

Two interpreters, because the environment splits the dependencies: the app's python has torch and no PyYAML, the
system one has PyYAML and no torch. Both runs are green on what they can collect:

```bash
pytest harness/tests -q                       # system: 1783 passed, 51 skipped, 1 xfailed (torch files)
<app>/python/python/bin/python3 -m pytest harness/tests -q   # 439 passed, 1 skipped (the YAML db test)
```

## Why `scripts/` is nearly empty

`scripts/` is upstream's. At the commit this fork's evidence work started from (`8541c46c`, 2026-09-01) it held
exactly two files - `printchangelog.py` and `test_version_speed.py` - and those two are still there, untouched.
Everything else in it was written here between 2026-11-15 and 2026-11-20: 187 instruments, one per measurement the
queue asked for. The only other survivor is `autopush.sh`, and it stays because `packaging/systemd/rve-autopush.*`
points at `scripts/autopush.sh` by absolute path.

So the rule is not "a file that imports the harness belongs in it" - it is **ours or upstream's**. Our measurement
code lives in `harness/scripts/`; `scripts/` keeps the two upstream utilities and the systemd hook.

## Running

```bash
python3 -m harness.cli scope
python3 -m harness.cli check harness/docs/wsp/verified/<file>.txt --records harness/docs/wsp/verified/records/<file>.jsonl
python3 -m harness.cli db build && python3 -m harness.cli db write-block harness/docs/RESEARCH-LEDGER.md
python3 harness/scripts/gate_sweep.py --help
```

Each instrument carries a self-contained bootstrap derived from `__file__` that puts the repo root **and** the
instrument's own directory on `sys.path`, so it runs by path from anywhere and may import its siblings
(`price_predictor_probe` imports `cut_channels_naive`, and 88 such edges exist). That replaced five different
conventions the files had accumulated - `sys.path.insert(0, "scripts")`, `os.path.join(REPO, "scripts")`, two
hardcoded absolute paths, the script-dir form, and a repo-root constant computed with two `dirname`s that became
wrong the moment the file moved down a level.

Instruments that need torch run with the app's interpreter, not the system one:

```bash
/home/jorj/Загрузки/REAL-2/REAL-Video-Enhancer-RVE-2.4.2/python/python/bin/python3 harness/scripts/<probe>.py
```
