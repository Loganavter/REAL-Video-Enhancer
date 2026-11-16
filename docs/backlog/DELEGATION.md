# Delegating the RVE research queue

`queue.yaml` is what to do. This file is who does it and what they are allowed to
run, because most of these tasks are blocked by two facts that a generic worker
prompt would get wrong.

## Fact 1: the GPU is one lane, and it is the supervisor's lane

Every measurement in this queue goes through one physical device. The rule is
already in the ledger's provenance section because it was broken once: **exactly
one GPU job at a time**. A worker that runs `nvidia-smi dmon` while the
supervisor renders does not "measure variance", it produces the drift that has
already retracted three headlines.

Mechanism: `flock -n /var/tmp/opencode/gpu.lock <command>`. A blocked job exits
instead of queueing silently, and the failure is visible at the call site.

Consequence for planning: items marked `gpu: true` are **serial**, and their
order is a dependency chain, not a wish list. In particular E26 runs before E28:
if Winograd is the reason TensorRT wins on conv-dominated stacks, then Welder
(which has no Winograd) cannot win on ours, and running it first would spend a
day to produce a number we could have predicted.

## Fact 2: a clone cell cannot measure here

`agent-clone.sh` copies committed state. In this repo that means no bundled
interpreter, no `venv` (gitignored, absent) and no model weights (they live
outside the cell). **The path this section used to name —
`../REAL-2/REAL-Video-Enhancer-RVE-2.4.2/python/...` — does not exist on this host
(checked 2026-11-16); a worker that follows it will fail on the first command.**
**Never calibrate a quantisation model on the CPU at frame resolution (measured 2026-11-17).**
onnxruntime's `quantize_static` runs the model once per calibration frame on the CPU: 8 frames x
2.5 TFLOP of 3x3 convolutions at 1920x1080 on 16 threads, plus its arena. It took the desktop down to
the point where the window manager stuttered ("almost killed mutter") and the run had to be killed.
The same ranges come from forward hooks in torch on the GPU in about half a second per frame - that
is what `scripts/make_qdq.py` does, and it is also the only way to keep the one-GPU-job rule honest.

**TensorRT: how it is obtained here (measured 2026-11-17, after a 6-hour hang).** The app itself
pins the source of truth: `src/DownloadDeps.py` installs `tensorrt==10.12.0.36` with
`tensorrt_cu12`, `tensorrt-cu12_libs` and `tensorrt_cu12_bindings`, and the metapackage's own
`pyproject.toml` names `index_url = "https://pypi.nvidia.com/"`. Consequences, all observed:
- `pip install tensorrt` from **PyPI** resolves to a source tarball under cp313+ and hangs in
  "Preparing metadata" (it wants the NVIDIA index for its build deps). Two attempts sat for hours.
- The real wheels live on **pypi.nvidia.com**: `tensorrt-cu12-libs` is a `py2.py3-none-manylinux`
  wheel (any Python), but `tensorrt-cu12-bindings` is **cp310 only** - so the interpreter has to be
  3.10, whatever the venv for torch is.
- No `tensorrt` package exists in the pacman repos (only `tensorrt-debug`, symbols), and no
  `libnvinfer`/`trtexec` is installed system-wide.

The interpreter to use is the repo's own `.venv` (2026-11-16: torch 2.15.0+cu132,
triton 3.9.0, numpy; verified on the GPU - RTX 4060 Ti sm89, a Triton kernel runs).
The system `python3` is 3.14 with **no torch**, so a cell cannot even import the model;
measurement probes are run by the supervisor as `.venv/bin/python scripts/...` under
`flock -n /var/tmp/opencode/gpu.lock`. A cell still cannot: the venv is gitignored
and absent from the clone, and TensorRT's python bindings are not installed at all. A
worker that tries to reproduce a timing will burn its budget on
`ModuleNotFoundError` or on a missing `futsuu.pth`, and its report will blame the
code.

So the split is fixed:

| work | who | why |
|---|---|---|
| literature verification, budget arithmetic, ONNX inspection | research cell | CPU, read-only, no jail escape needed |
| writing probes, plugins, tests that run on CPU or `--dry` | code cell | the code is reviewed before it touches the GPU |
| every timing, every render, every acceptance statistic | supervisor, on the shared checkout | needs the GPU, the weights and the bundled interpreter |
| ledger rows, queue state, merge | supervisor | one writer for one file |

This inverts the usual delegation: workers prepare, the supervisor measures.
That is the honest shape for a project whose whole difficulty is measurement
drift.

## Creating a cell for this repo

```bash
agent-clone.sh "/home/jorj/Загрузки/projects/REAL-Video-Enhancer" rve-<task-id>
# cell:   ~/.cache/opencode/clones/rve-<task-id>/REAL-Video-Enhancer
# branch: agent/rve-<task-id>/work
```

`agent-clone.sh` accepts a path (anything that is not one of the three aliases is
passed through), and it writes its own pre-commit hook into the clone, so the
branch discipline holds inside the cell. Two differences from the three managed
repos that the prompt has to state explicitly, because nothing enforces them
here: this repo has **no `pre-push` hook and no repo-level guard config**, so
"never push `main`" is a prompt rule, and the supervisor checks the branch tip
rather than trusting the hook.

Delivery from a cell is always `git push origin agent/<task-id>/work` — inside a
clone, `origin` is the local source checkout, not GitHub. The fork is the
source checkout's `fork` remote; the supervisor moves the branch.

`DONE.md` at the clone root, untracked, sections per the machine protocol, and
`INTERFACE:` names every new symbol with its signature, because the plugin and
the probes are the only shared seams here.

## Uncommitted work has a lifecycle here too

In the sibling project two branch switches erased edits that existed only in a
worktree: the branch moved, the tree did not come with it, and nothing had ever
recorded what the tree looked like. The same hole is open here, and it is the
shared checkout that is exposed — a cell throws its tree away by design, the
supervisor's checkout is where hours of unmeasured work accumulate between
commits.

Two ways to close it, and they answer different questions:

- **Timer, automatic**: `scripts/autopush.sh` snapshots the working tree
  (staged + unstaged + untracked, `.gitignore`'d paths excluded) onto
  `refs/autopush/<branch>` without touching `HEAD`, the index or the tree. It
  skips a clean tree, a tree identical to the last snapshot, an in-progress
  merge (`MERGE_HEAD` or unmerged paths) and a detached `HEAD`. Ref name is the
  branch with `/` flattened to `--`, e.g.
  `refs/autopush/agent--rve-autopush--work`. Retention is bounded by
  `AUTOPUSH_KEEP` (default 48 snapshots, 12 h at the timer's interval): the ref
  always carries the newest snapshot, older ones fall off the end of the chain.
  The unit is `packaging/systemd/rve-autopush.{service,timer}`, 15 minutes,
  **not enabled by default** — a timer writing into a repository another lane is
  measuring in is the owner's decision, and the enable commands are in the
  service header.
- **Manual, before a switch**: `wip-save.sh [repo] [why]` from the machine
  toolkit records a one-off snapshot into `refs/wip/<branch>/<stamp>` (permanent,
  never pruned) and `git-safe-switch.sh <branch|commit>` saves before it
  switches. Reach for these when the thing you are about to do is exactly the
  thing that lost work last time; the timer only guarantees the last 15 minutes.

Recovery is the same for both, and neither puts a commit on your branch:

```bash
git for-each-ref refs/autopush                      # what exists, and when
git log refs/autopush/<name> --oneline -5           # snapshot timeline
git show refs/autopush/<name>:<path>                # content of one file
git checkout refs/autopush/<name> -- <path>         # restore one path
```

What `autopush.sh` deliberately does not do: no `git stash` (per-repository
global state, and a timer must not take it), no `git reset --hard`, no
`checkout`, no commit on any branch, and no push — this checkout's only remote is
the local source repo, so a snapshot is local by design. A pre-PR squash
therefore cannot drop it, which was the failure mode in the sibling project.

## Lanes

At most **three** cells at once, and never two with overlapping `owns_files`:

- **lane-lit** (research, `agent-clone.sh … --research`): verify the external
  numbers the queue cites, and answer three narrow questions: does the bundled
  torch's cuDNN expose the Winograd algorithm constants and does
  `cudnnGetConvolutionForwardAlgorithm_v7` return Winograd for
  `[N,H,W,64] x [64,3,3,64]`; can TRT's plan file be inspected for kernel names
  at all (the tactic dump we have shows `implicit_gemm`, the question is whether
  it enumerates the alternative); does Welder build on this host at all, and
  which of our ops (`PReLU`, `PixelShuffle`, residual add) it can express.
  Owns `REPORT.md` only. Every claim carries a command or a `file:line`.
- **lane-probes** (code): `scripts/probe_winograd.py` and
  `scripts/pyramid_budget.py`. Both are written and reviewed before the
  supervisor runs them; the probe must print the algorithm name it got, not just
  a time, or it reproduces the mistake E26 exists to correct.
- **lane-plugin** (code): `backend/src/pytorch/int8_plugin/` plus a CPU test of
  shapes, serialisation and the per-channel-scale path. No GPU run.
- **supervisor**: the GPU lane, the ledger, the queue, the merge, and the answer
  to the owner about weights.

`queue.yaml` is edited by the supervisor only, the same way `catalog.yaml` is in
the docs repo: the board is not a place where several agents write.

## Order

1. **Now**, CPU only: `rve-ledger-reopen-e16` (done in this commit),
   `rve-doc-literature-closures` (folded into it).
2. `lane-lit` in parallel with `lane-probes` and `lane-plugin` — three disjoint
   file sets, no GPU, so they can run together.
3. Supervisor: **E26** (1 h), the cheapest lever and the one that decides
   whether E28 is worth a day.
4. If E26 is positive, E28 and E27 follow; if negative, E28 moves to "expected
   loss" and E27 stays the next GPU item.
5. E29 phase 1 is CPU arithmetic and can run in `lane-probes` at any time; its
   phase 2 waits for the fused-prototype decision, i.e. weeks.
6. E30 is blocked on the owner, and blocked means asked, not deferred.
7. E31 phase 1 is CPU; phase 2 is a render, and it changes the metric, so it
   needs the owner in the loop before it costs GPU time.

## Acceptance, per task type

- **experiment**: the number, the reference it is compared against, the
  interleaving (3 rounds, alternating), and the count of runs. A single pair is
  not a measurement in this repo.
- **code**: the CPU test, plus the exact command the supervisor will run on the
  GPU. "It should work" is not a delivery.
- **research/decision**: every claim traceable to a command output, a `file:line`
  or a cited paper section. The literature is evidence about the literature, not
  about our engine; a 1.53x on a V100 against TensorRT is a reason to measure,
  never a prediction.
- **docs**: the ledger row states what was measured here, and marks external
  numbers as external.

## Timeboxes

S 60 min, M 120 min, L per phase. Heartbeats: pre-flight read ≤15 min, first
scoped edit ≤45 min, first push ≤60 min. At every wait slice: `agent-spin.sh`
plus `scripts/worker_progress.sh`, act on the worse verdict. Two consecutive
misses means kill, salvage, split. A cell that cannot run its task because it
needs the GPU posts `blocked` with the exact command it wanted to run, and the
supervisor runs it.
