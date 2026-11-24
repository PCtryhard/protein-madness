# Protein Madness — motif-anchored design pipeline

[![tests](https://github.com/PCtryhard/protein-madness/actions/workflows/tests.yml/badge.svg)](https://github.com/PCtryhard/protein-madness/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Closed-loop protein design that improves motif-anchored designs against a fixed
scoring function, on two 12 GB GPUs.

One forward pass is: seed PDB with fixed motif residues → RFdiffusion partial
diffusion → LigandMPNN → Boltz-2 → hydrogen placement → scoring. A search loop
then expands the best designs through two branches and re-scores them until the
score stops improving.

## Status: the challenge is over

**This repo is an archive.** The competition has closed and the **final 20
designs were chosen by hand** from the scored designs, not by a ranking script.
They are committed as sequences:

| file | |
| --- | --- |
| `data/designs.fasta` | the final 20 |
| `data/2kELO_10designs_submission.fa` | the 10-design submission |
| `data/2kELO_5designs_submission.fa` | the 5-design submission |

Each header carries the design's name and its `base`/`oxy` residue positions,
and the name is the one used throughout `data/rescored/` — so all 20 resolve
through `manifest.csv` back to a run, a lineage and a structure.

That they were picked by hand is visible in the picks: the 20 take the geometry
re-rank's **top three** but then spread to **ranks 1 through 92 across seven
variants** (01, 02, 03, 06, 08, 09, 12), rather than taking the top 20 by
score. Diversity was chosen over rank.

What is still worth having:

| | |
| --- | --- |
| **What was run and what it measured** | **[RUNLOG.md](RUNLOG.md)** — every batch, in order, with the jobs and the numbers. §15 (all 13 variants), §16 (three search cycles) and `data/final_run_60/` (the 12 top candidates, four more cycles) are the production runs |
| **Every scored design** | `data/01` … `data/13`, `data/final_run_60/`, `data/rescored/` — structures, leaderboards and provenance, all committed |
| **Re-running any of it** | **[EULER.md](EULER.md)** — the Euler procedure, start to finish. This is the part that is still live |
| **Why anything is the way it is** | **[DECISIONS.md](DECISIONS.md)**, append-only |

Nothing has been deleted in the cleanup: what the challenge no longer needs was
moved into **[outdated/](outdated/)**, which is where the previous pipeline
already lived. No cluster account name is committed anywhere — see
[DECISIONS.md](DECISIONS.md) for the placeholders used in the run artifacts.

## ▶ Start here

1. **[PIPELINE.md](PIPELINE.md)** — the specification: the stages, the scoring
   function, the search, the hardware limits, and the build phases. Read it
   first.
2. This README — how to run it, and **which parts are verified and which are
   not.**

```bash
python -m src.cli smoke --target config/targets/01_his_asp.yaml   # validate, no GPU
python -m pytest -q                                               # 283 tests, all green
python cluster/preflight.py                                       # can this box run it?
python scripts/noise_control.py --from-csv                        # is the score signal?
```

To run it on a cluster, start at **[EULER.md](EULER.md)**;
**[cluster/README.md](cluster/README.md)** has the layout and the hardware
limits behind it.

**On Euler**, a teammate reproduces the whole setup under their own account
with three scripts — the tools pin incompatible stacks, so each gets its own
environment, which is what upstream tells you to do:

```bash
bash cluster/euler/setup_rfd3.sh         # torch 2.7.1 + numpy 2, RFD3
bash cluster/euler/setup_ligandmpnn.sh   # torch 2.2.1 + numpy 1.23, python 3.11
bash cluster/euler/setup_boltz.sh        # boltz 2.2.1
python cluster/preflight.py              # checks every path before you queue
```

`config/pipeline.yaml` refers to them as `${HOME}/pm-*-env` and
`${SCRATCH}/pdh/...`, so there is nothing personal in it — a test asserts that.
The full procedure, including the deploy and the job submission, is
**[EULER.md](EULER.md)**.

## The score is noisier than the search assumes

`scripts/noise_control.py --from-csv` runs phase 0's noise control on the 32
designs × 5 seeds already in `data/No_mutation_run/`. No GPU, no new compute.
It says:

| term | noise floor | signal | ratio | |
| --- | --- | --- | --- | --- |
| base distance | 3.54 Å | 9.21 Å | 0.38 | marginal |
| donor distance | 4.16 Å | 5.30 Å | 0.79 | **mostly noise** |
| base angle | 25.5° | 27.5° | 0.93 | **mostly noise** |
| donor angle | 28.0° | 23.8° | 1.18 | **mostly noise** |
| iPDE | 0.62 | 1.52 | 0.41 | marginal |

"Noise floor" is the seed-to-seed spread for the *same* sequence; "signal" is
the spread between *different* designs. A ratio near or above 1 means the term
cannot tell designs apart at all — which is where both angle terms sit.

So before a real search: `run.eval_replicates: 3` and `run.patience: 4`. And
treat `scoring.geometry_weights.angle: 0.40` as suspect — the brief calls it
fixed, so it has not been changed, but this data does not support it.

Caveat, stated in the tool's own output: this is the *previous* pipeline's run,
with the severe ligand drift that `RUNLOG.md` §1 records. That drift inflates
the between-design spread, so these ratios are if anything optimistic. The
absolute within-design spreads — ±4 Å, ±28° — are the numbers that matter, and
they are large.

> The previous pipeline — multi-backend scoring tiers, Euler/Modal, the
> four-published-metrics scorer — now lives in **[outdated/](outdated/)** in
> full. It is kept for the record and is not maintained. This pipeline is
> deliberately much smaller: it is the base to build on, not a reduction in
> ambition.

## State: what is verified and what is not

The failure mode this table exists to prevent is a working search loop sitting
on top of an untested scorer, where every result looks plausible and none of it
means anything.

All five phases ran on Euler in the end. The table below is the state as of the
last run; **[RUNLOG.md](RUNLOG.md)** is what those runs measured, and it is the
honest account — several of its findings are retractions of earlier ones.

| Phase | What it covers | State |
| --- | --- | --- |
| **0** | Environment, per-stage VRAM and wall time, the ten-seed noise control | **done for the environment half** — `cluster/preflight.py` ran on Euler; the ten-seed noise control was never run as such, and `scripts/noise_control.py --from-csv` on the old run is the only noise estimate there is (see below) |
| **1** | Config, tracking, one linear pass end to end | **done** — ran end to end with weights on Euler, RUNLOG §4 |
| **2** | Scoring, with unit tests against hand-computed geometry | **done** — 27 scoring tests + 22 confidence tests, all green |
| **3** | The search loop, sqlite store, sequence-hash cache | **done** — ran to three cycles in §16, and to four per candidate in `data/final_run_60/`; `resume` is still deliberately not implemented |
| **4** | Two-worker GPU pool, process isolation | **done** — every production batch used it |

In detail:

| Area | State |
| --- | --- |
| `src/config.py` — schema, cross-validation, strict keys | real, 36 tests |
| `src/tracking.py` — PDB/CIF readers, `.trb` map, resolver | real, 36 tests, checked against the variant-01 seed and a real Boltz-2 CIF |
| `src/scoring/` — confidence, H-bond geometry, combination | real, 56 tests, values hand-derived in `tests/fixtures/build_fixtures.py` |
| the variants + the generator's angle provenance | real, 34 tests in `tests/test_variants.py` |
| `src/search/` — frontier, patience, both branches, seed-only | real, 33 tests, and run for real in §15, §16 and the final batch |
| `src/store.py` — lineage, leaderboard, cache | real, 35 tests |
| `src/stages/` — command construction, token guard, output parsing | real, 39 tests, and every stage has since driven its real model on Euler |
| `src/runner.py` — GPU pool and the per-scaffold loop | still not covered by tests, but it ran every production batch |
| `src/cli.py` — `smoke`, `measure`, `report`, `run` | all four run for real |
| `src/live.py` — live CSV, FASTA, progress | real, 19 tests incl. two-thread concurrency |
| `src/cli.py watch` | run by hand against a simulated run |
| `scripts/pull_run.sh` | `bash -n` clean; **never run against a real cluster** — runs were pulled by other means |
| `cluster/preflight.py` | run on this laptop and on Euler |
| `cluster/euler/*` | submitted, repeatedly; this is the path that was actually used |
| `scripts/build_targets.py` | the committed targets are not stale, but the generator **no longer runs unattended** — see below |
| `scripts/compare_no_mutation.py` | kept from the old pipeline, still runs against `data/No_mutation_run/` |
| `scripts/noise_control.py --from-csv` | real, run; the verdict above is its output |
| `scripts/noise_control.py` GPU mode | **refuses** — needs provenance maps read back from `designs.db`, not written |
| `config/pipeline.yaml` | loads, cross-validates against the targets, key-parity tested |
| seed-only runs (`max_cycles: 0`) | logic tested, and run with models |
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
python -m src.cli smoke --target config/targets/01_his_asp.yaml
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
python -m src.cli measure --target config/targets/01_his_asp.yaml --pdb data/01/inputs/01_his_asp_.pdb.pre-boltz
```

This resolves every named hydrogen bond through the full provenance chain and
prints its distance, angle and normalised terms. An atom it cannot resolve
raises rather than defaulting — which is the whole point, because a missing
atom scored as zero distance looks like a perfect bond.

**First real run: seed-only, 10 backbones × 8 sequences.** Generates and
scores cycle 0, writes a leaderboard, expands nothing. About an hour per
variant:

```powershell
python -m src.cli run --target config/targets/01_his_asp.yaml --name small-01
```

**A full search.** Needs both GPUs and all three model stacks, and is hours per
variant rather than one:

```powershell
python -m src.cli run --target config/targets/01_his_asp.yaml --name run1
```

**Run them all.** On Euler, as a job array — this is the path the production
batches used:

```powershell
sbatch cluster/euler/submit.sbatch
```

The single-machine AWS runner is in
[outdated/cluster/aws/](outdated/cluster/aws/): it was written, dry-run and
never submitted, because Euler was the cluster that got used.

**Watch it while it runs.** Every design is appended the moment it finishes —
nothing waits for the end of the run:

```powershell
python -m src.cli watch --run runs/small-01 --follow
```

```
runs/small-01/
  designs.csv       one row per design, appended and fsynced as it is scored
  sequences.fasta   the same designs as sequences
  progress.txt      counts, best so far, designs/hour
  structures/<design_id>/   the RFdiffusion backbone and the Boltz-2 prediction
  stages.jsonl      per-stage wall time and peak VRAM
  designs.db        sqlite, in WAL mode, readable while a worker writes it
```

`designs.csv` carries the score and every term, a column per hydrogen bond
(`<bond_id>__distance`, `<bond_id>__angle`), the logged Boltz metrics, the
sequence, and the paths to that design's backbone and predicted structure.
Gated designs name the bond that failed; errored designs get a row with the
error on it, which is most of the value of watching a run.

Structures are written by the tools straight to their final paths, so a
backbone or a `.cif` is on disk and complete while the next design is still
folding. `tail -f runs/small-01/designs.csv` works, and so does
`python -m src.cli report --run runs/small-01` mid-run.

**Copy a run to your own machine, mid-run or after:**

```powershell
bash scripts/pull_run.sh me@euler.ethz.ch:/cluster/home/me/protein-madness small-01
```

rsync, so it is safe to repeat while the run continues. By default it pulls the
CSV, the FASTA, the reports and the final structures but not RFdiffusion's
intermediate trajectories; `--all` takes everything.

**Report on a finished or interrupted run:**

```powershell
python -m src.cli report --run runs/run1 --write
```

Writes `runs/run1/report.md`: a leaderboard per scaffold, the best design's
full lineage from the recursive ancestry query, and its per-bond geometry.

## Before the first real run

Steps 1–3 of phase 0 are automated:

```powershell
python cluster/preflight.py --require-gpu --write runs/hardware_notes.md
```

It reports the GPUs, VRAM and compute capability, **actually runs a bf16
matmul** rather than assuming sm_75 cannot, checks whether the optimised Boltz
kernels are usable, verifies all three tool paths resolve, and writes phase 0's
findings where PIPELINE.md asks for them. Fix every `FAIL` before submitting
anything.

One thing it cannot do for you:

1. **The ten-seed noise control.** One fixed sequence, ten Boltz-2 seeds, all
   three score terms on each output. This is the one that decides whether the
   search is hill-climbing on signal or on noise, and it sets
   `run.eval_replicates` and `run.patience`.

It is worth the hour it costs, and it never got that hour — it is the one
phase-0 step that was skipped, so `run.eval_replicates` and `run.patience`
were set from the old run's spread rather than from a measurement of this
pipeline's. Patience of 2 on a noisy objective
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

Each variant is an independent search with its own frontier, stopping
condition and leaderboard. There is no competition between them, so results
stay comparable and they can run in any order, on any machine, in parallel or
one at a time.

## The 13 variants

`data/01` … `data/13` hold one motif-only PDB each: the fixed catalytic
residues plus the 6NT ligand, in 5FQJ numbering. Three positions vary, and the
filenames read `<291>_<229>_<261>`. Variants 01–07 came first; 08–13 complete
the grid by pairing an ASN at the nitro position with every base and donor:

| | A229 — base | A291 — donor | A261 — nitro |
| --- | --- | --- | --- |
| **01** `his_asp_` | ASP | HIS | — |
| **02** `his_asp_his` | ASP | HIS | HIS |
| **03** `asn_asp_his` | ASP | ASN | HIS |
| **04** `gln_asp_his` | ASP | GLN | HIS |
| **05** `his_glu_his` | GLU | HIS | HIS |
| **06** `asn_glu_his` | GLU | ASN | HIS |
| **07** `gln_glu_his` | GLU | GLN | HIS |
| **08** `his_asp_asn` | ASP | HIS | ASN |
| **09** `asn_asp_asn` | ASP | ASN | ASN |
| **10** `gln_asp_asn` | ASP | GLN | ASN |
| **11** `his_glu_asn` | GLU | HIS | ASN |
| **12** `asn_glu_asn` | GLU | ASN | ASN |
| **13** `gln_glu_asn` | GLU | GLN | ASN |

A229 accepts from the benzotriazole `N3`-H, A291 donates to the ligand `N1`,
and A261 (when present) donates to a nitro oxygen `O11`. One target file per
variant, generated from the PDBs:

```powershell
python scripts/build_targets.py
```

> **This now refuses rather than regenerating, and that is the designed
> behaviour.** Over the campaign each `data/0N/inputs/` accumulated both extra
> seed PDBs and run-output subdirectories, so the generator finds several
> candidate seeds per variant and will not guess between them — it exits
> non-zero naming them and asking for `--inputs`. Passing `--inputs .` does not
> help either, because three PDBs now sit directly in `data/01/inputs/`.
>
> The committed targets are **not** stale: `--check` never reports "out of
> date", which is the condition
> `tests/test_variants.py::test_generated_targets_are_not_stale` asserts, and
> that test is deliberately tolerant of a refusal. To regenerate, point
> `--inputs` at one subdirectory that every variant folder contains, and keep
> exactly one motif PDB per variant at the top of `inputs/`.

> **The seeds that production actually used were the FastRelax ones.** Each
> variant directory holds several candidate seeds; every batch from §15 onward
> ran from `data/0N/inputs/*_fastrelax_motif_af3.pdb`, which is the motif
> relaxed into a plausible pocket rather than modelled in and left clashing.
>
> An earlier round of refolded pockets for 01–03 was unusable — it put the
> motif residues 25–39 Å from the ligand, so nothing was interacting, and
> `scripts/build_targets.py` refused them rather than inventing stand-in
> ideals. `SEED_UNUSABLE_DISTANCE` in that script and the entry in
> [DECISIONS.md](DECISIONS.md) are what came out of it. The paragraphs below
> describe the original un-refolded seeds, and are kept because the clash
> table is what the scorer's short-contact gate was built against.

**Of the un-refolded seeds, only the ASP229 contact carries crystal geometry.** Its two contacts are the real
5FQJ values — ASP229 `OD2`···`N3` at 2.734 Å / 106.6°, HIS291 `NE2`···`N1` at
3.084 Å / 126.6°. Every mutated or added residue in 02–07 was modelled in
without repacking and its atoms land on top of the ligand:

| contact | in the seed | |
| --- | --- | --- |
| `GLU229 OE2···N3` | **1.677 Å** | impossible — modelled, clashes |
| `HIS261 NE2···O11` | **1.820 Å** | impossible — modelled, clashes |
| `ASN291 ND2···N1` | **4.823 Å** | too far — points away |
| `GLN291 NE2···N1` | **3.860 Å** | too far |

That is harmless for an RFdiffusion motif input, because the scaffold and
sidechains get rebuilt — but it means **the seed's own geometry is not a score
target.** So the ideal angles are not read out of these files: where the
acceptor exists unmutated in the crystal the measured value is used, and
otherwise the hybridisation value for that chemistry, marked as such in the
generated file. `tests/test_variants.py` asserts all of this, clashes included,
so if someone repacks the motif files the test fails and says to re-derive the
ideals.

Finding those clashes also exposed a gap in the scorer, now fixed: the distance
term is symmetric about the ideal, so a 1.7 Å contact scored zero exactly like
a 4.0 Å one — but only the long one gated, meaning a design with an atom driven
through the ligand could ride its confidence term into the frontier.
`scoring.distance.clash` (default 2.2 Å) gates the short end too.

## Documents

**Live:**

- **[PIPELINE.md](PIPELINE.md)** — the specification, plus an appendix on how
  this repo differs from it and why.
- **[EULER.md](EULER.md)** — the Euler procedure end to end, and the one
  constraint behind it: all three tools must be on the same machine, because
  the loop cannot be split.
- **[cluster/README.md](cluster/README.md)** — the cluster layout and the
  per-stage hardware limits.
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

The post-challenge cleanup moved four more things in there, unchanged:

| | why |
| --- | --- |
| `outdated/HANDOFF.md` | the procedure for when the refolded motif files arrive. They arrived; the FastRelax seeds are what ran |
| `outdated/cluster/aws/` | the single-machine runner, never submitted |
| `outdated/results/smoke/` | the previous pipeline's smoke output, against the old `configs/` layout |
| `outdated/logs/setup-perlin.log` | one env-build log, referenced by nothing |

## Continuous integration

[`.github/workflows/tests.yml`](.github/workflows/tests.yml) runs on every push
to `main`, every pull request, and on demand from the Actions tab. Everything
it does is CPU-only and needs no model weights, so a clean checkout is enough
to reproduce it:

| step | what it proves |
| --- | --- |
| `pytest -q` | the 283 unit tests |
| `smoke`, `measure` | the config, the provenance chain and the geometry terms resolve |
| `report` over `data/final_run_60/01-01-95cdcf` | a committed sqlite run store is readable, lineage query included |
| `noise_control --from-csv` | the committed leaderboards still parse |
| the pocket inspector build | all 314 rescored structures and the UTF-8 template load |
| `bash -n` over every `.sh` / `.sbatch` | no cluster script is syntactically broken |

It runs on **ubuntu-latest and windows-latest**. That is deliberate rather than
thorough-for-its-own-sake: development here is Windows-first while the runs
happened on Linux, and the two disagree about the default text encoding — which
is how `build_inspector.py` came to crash on Windows while working fine on
Euler. The inspector step is the guard against that recurring.

Not in CI, because both fail by design and would make a red badge meaningless:
`cluster/preflight.py` (no GPU or model stacks on a runner) and
`scripts/build_targets.py --check` (it refuses to guess between the seed
candidates in `data/0N/inputs/`, as described above).

## License

[MIT](LICENSE). The structures, leaderboards and run artifacts under `data/`
are covered by the same terms; `data/5FQJ.cif` and `data/6SS1.cif` are
PDB entries and carry the PDB's own terms, not ours.
