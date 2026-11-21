# Protein Madness — motif-anchored design pipeline

Closed-loop protein design that improves motif-anchored designs against a fixed
scoring function, on two 12 GB GPUs.

One forward pass is: seed PDB with fixed motif residues → RFdiffusion partial
diffusion → LigandMPNN → Boltz-2 → hydrogen placement → scoring. A search loop
then expands the best designs through two branches and re-scores them until the
score stops improving.

## ▶ Start here

1. **[PIPELINE.md](PIPELINE.md)** — the specification: the stages, the scoring
   function, the search, the hardware limits, and the build phases. Read it
   first.
2. This README — how to run it, and **which parts are verified and which are
   not.**

```bash
python -m src.cli smoke --target config/targets/5fqj.yaml    # validate config, no GPU
python -m pytest -q                                           # 220 tests, no GPU
```

> The previous pipeline — multi-backend scoring tiers, Euler/Modal, the
> four-published-metrics scorer — now lives in **[outdated/](outdated/)** in
> full. It is kept for the record and is not maintained. This pipeline is
> deliberately much smaller: it is the base to build on, not a reduction in
> ambition.

## State: what is verified and what is not

The failure mode this table exists to prevent is a working search loop sitting
on top of an untested scorer, where every result looks plausible and none of it
means anything.

| Phase | What it covers | State |
| --- | --- | --- |
| **0** | Environment, per-stage VRAM and wall time, the ten-seed noise control | **not started** — needs the cluster |
| **1** | Config, tracking, one linear pass end to end | **partly done** — config, tracking and the `measure` table pass against the real seed; the linear pass has never been run with weights |
| **2** | Scoring, with unit tests against hand-computed geometry | **done** — 27 scoring tests + 22 confidence tests, all green |
| **3** | The search loop, sqlite store, sequence-hash cache | **code complete, never run** — logic is unit-tested; `resume` is deliberately not implemented |
| **4** | Two-worker GPU pool, process isolation | **code complete, never run** |

In detail:

| Area | State |
| --- | --- |
| `src/config.py` — schema, cross-validation, strict keys | real, 31 tests |
| `src/tracking.py` — PDB/CIF readers, `.trb` map, resolver | real, 36 tests, checked against the real 5FQJ seed and a real Boltz-2 CIF |
| `src/scoring/` — confidence, H-bond geometry, combination | real, 49 tests, values hand-derived in `tests/fixtures/build_fixtures.py` |
| `src/search/` — frontier, patience, both branches | real, 30 tests |
| `src/store.py` — lineage, leaderboard, cache | real, 35 tests |
| `src/stages/` — command construction, token guard, output parsing | real, 39 tests; **no model has ever been invoked** |
| `src/runner.py` — GPU pool and the per-scaffold loop | written, **not covered by tests and never run** |
| `src/cli.py` — `smoke`, `measure`, `report` | run by hand, work; `run` needs the cluster |
| `protonation.engine: reduce` | **refuses** — see `src/stages/protonate.py` for why |
| `src/cli.py resume` | **refuses** — the patience counter is not rebuilt from the database |

Nothing in this repo returns a plausible-looking number it did not compute. A
metric whose inputs are missing raises; it is never 0.0 and never NaN-with-a-shrug.

## Setup

The virtual environment is not in the repo and never will be: `.venv/` is
gitignored because it is machine-specific and hundreds of megabytes.

```powershell
python -m venv .venv
```

Activate it — this is where people trip:

| Shell | Command |
| --- | --- |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows cmd.exe | `.venv\Scripts\activate.bat` |
| Windows Git Bash | `source .venv/Scripts/activate` |
| macOS / Linux / WSL | `source .venv/bin/activate` |

```powershell
pip install -r env/requirements.txt
```

Then check it works. Both commands are CPU-only and need no model weights:

```powershell
python -m pytest -q
```

```powershell
python -m src.cli smoke --target config/targets/5fqj.yaml
```

`smoke` prints the resolved config — which bonds are measured where, against
what ideal, with which gate — and ends with `SMOKE TEST PASSED`.

GPU stacks are **not** in `requirements.txt`. RFdiffusion, LigandMPNN and
Boltz-2 pin incompatible torch/CUDA builds, so each is installed separately on
the cluster and reached by the paths in `config/pipeline.yaml`.

## Running it

**Check a target spec against a real structure** — the phase-1 acceptance
table. No GPU:

```powershell
python -m src.cli measure --target config/targets/5fqj.yaml --pdb data/5A_5FQJ.pdb
```

This resolves every named hydrogen bond through the full provenance chain and
prints its distance, angle and normalised terms. An atom it cannot resolve
raises rather than defaulting — which is the whole point, because a missing
atom scored as zero distance looks like a perfect bond.

**Run the search.** Needs both GPUs and all three model stacks:

```powershell
python -m src.cli run --target config/targets/5fqj.yaml --name run1
```

**Report on a finished or interrupted run:**

```powershell
python -m src.cli report --run runs/run1 --write
```

Writes `runs/run1/report.md`: a leaderboard per scaffold, the best design's
full lineage from the recursive ancestry query, and its per-bond geometry.

## Before the first real run

Phase 0, in order, from PIPELINE.md:

1. Fill in the three tool paths in `config/pipeline.yaml`
   (`rfdiffusion.run_inference`, `ligandmpnn.run_script`, `boltz.binary`).
2. Settle whether bf16 works on sm_75 and whether `--no_kernels` is required.
   Both have config entries already set to the expected answer; **confirm them,
   do not assume.** Write the findings to `runs/hardware_notes.md`.
3. Confirm the `contigs` string in `config/targets/5fqj.yaml` against the actual
   RFdiffusion build. The earlier runs used RFD3, whose syntax differs — see the
   PIPELINE.md appendix.
4. Run the ten-seed noise control. This is the one that decides whether the
   search is hill-climbing on signal or on noise, and it sets
   `run.eval_replicates` and `run.patience`.

Step 4 is worth the hour it costs. Patience of 2 on a noisy objective
terminates on noise rather than on convergence, and sidechain placement in a
designed pocket is exactly where a structure predictor is least reliable.

## How it fits together

```
config/pipeline.yaml ──┐
                       ├──> src/config.py ──> everything else
config/targets/*.yaml ─┘     (nothing downstream reads YAML)

seed PDB ──> rfdiffusion ──> ligandmpnn ──> boltz ──> protonate ──> scoring
               │ .trb           │ identity    │ 1..N      │            │
               └────────────────┴─────────────┴───────────┘            │
                     four explicit index maps, in Provenance           │
                                                                       v
                                      designs.db  <──  score + lineage + cache
```

Every number used in a computation comes from `config/pipeline.yaml`. A literal
float inside a scoring function is a bug, and
`tests/test_config.py::test_no_numeric_literals_in_scoring` is the grep for one,
run as a test over the AST so docstrings and `"utf-8"` do not trip it.

Each seed scaffold is an independent search with its own frontier, stopping
condition and leaderboard. There is no competition between scaffolds, so
results stay comparable across seeds and adding a seed to
`config/targets/5fqj.yaml` adds a search with no other change.

## The target

`config/targets/5fqj.yaml` is the 5FQJ / HG3 Kemp eliminase pocket with ligand
6NT. Two scored hydrogen bonds, both measured from `data/5A_5FQJ.pdb` in this
repo:

| | base | donor |
| --- | --- | --- |
| bond | ASP229 `OD2` ← 6NT `N3`-H | HIS291 `NE2`-H → 6NT `N1` |
| distance in the seed | 2.734 Å | 3.084 Å |
| acceptor-side angle | 106.6° | 126.6° |

A test asserts these against the PDB, so the target file cannot drift from the
structure it describes. Two points in it are still open — the donor residue
identity and the base angle's reference value — and both are flagged in the
file and in the PIPELINE.md appendix.

## Documents

**Live:**

- **[PIPELINE.md](PIPELINE.md)** — the specification, plus an appendix on how
  this repo differs from it and why.
- **[RUNLOG.md](RUNLOG.md)** — what was actually run and what it measured. The
  11.18 Å ligand-drift measurement in §1 is why this pipeline scores Boltz-2
  predictions rather than design-time geometry.
- **[DECISIONS.md](DECISIONS.md)** — append-only: what was decided and why. Add
  to it rather than editing old entries.

**Superseded — [outdated/](outdated/):** the whole previous pipeline, its
`src/`, `tests/`, `scripts/`, `configs/` and its documents
(`BRIEF.md`, `EVOLVE.md`, `DESIGN_RATIONALE.md`, and the pre-announcement set).
[outdated/README.md](outdated/README.md) maps each piece to what replaced it.
Nothing in there is maintained.
