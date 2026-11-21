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
python -m src.cli smoke --target config/targets/01_his_asp.yaml   # validate, no GPU
python -m pytest -q                                               # 269 tests, no GPU
python cluster/preflight.py                                       # can this box run it?
python scripts/noise_control.py --from-csv                        # is the score signal?
```

To run it on a cluster, start at **[cluster/README.md](cluster/README.md)**. When
the refolded motif files arrive, **[HANDOFF.md](HANDOFF.md)** is the whole
procedure.

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

| Phase | What it covers | State |
| --- | --- | --- |
| **0** | Environment, per-stage VRAM and wall time, the ten-seed noise control | **tooling ready, not run** — `cluster/preflight.py` does the environment half; the noise control needs the cluster |
| **1** | Config, tracking, one linear pass end to end | **partly done** — config, tracking and the `measure` table pass against all seven seeds; the linear pass has never been run with weights |
| **2** | Scoring, with unit tests against hand-computed geometry | **done** — 27 scoring tests + 22 confidence tests, all green |
| **3** | The search loop, sqlite store, sequence-hash cache | **code complete, never run** — logic is unit-tested; `resume` is deliberately not implemented |
| **4** | Two-worker GPU pool, process isolation | **code complete, never run** |

In detail:

| Area | State |
| --- | --- |
| `src/config.py` — schema, cross-validation, strict keys | real, 36 tests |
| `src/tracking.py` — PDB/CIF readers, `.trb` map, resolver | real, 36 tests, checked against the variant-01 seed and a real Boltz-2 CIF |
| `src/scoring/` — confidence, H-bond geometry, combination | real, 56 tests, values hand-derived in `tests/fixtures/build_fixtures.py` |
| the seven variants + the generator's angle provenance | real, 34 tests in `tests/test_variants.py` |
| `src/search/` — frontier, patience, both branches, seed-only | real, 33 tests |
| `src/store.py` — lineage, leaderboard, cache | real, 35 tests |
| `src/stages/` — command construction, token guard, output parsing | real, 39 tests; **no model has ever been invoked** |
| `src/runner.py` — GPU pool and the per-scaffold loop | written, **not covered by tests and never run** |
| `src/cli.py` — `smoke`, `measure`, `report` | run by hand, work; `run` needs the cluster |
| `cluster/preflight.py` | run on this laptop; correctly reports what is missing. Never run on either cluster |
| `cluster/euler/*`, `cluster/aws/*` | `bash -n` clean, `run_all.sh --dry-run` verified; **never submitted** |
| `scripts/build_targets.py` | generates all 7 targets; a test asserts they are current |
| `scripts/compare_no_mutation.py` | kept from the old pipeline, still runs against `data/No_mutation_run/` |
| `scripts/noise_control.py --from-csv` | real, run; the verdict above is its output |
| `scripts/noise_control.py` GPU mode | **refuses** — needs provenance maps read back from `designs.db`, not written |
| `config/pipeline.small.yaml` | loads, cross-validates against all 7 targets, key-parity tested |
| seed-only runs (`max_cycles: 0`) | logic tested; never run with models |
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
python -m src.cli measure --target config/targets/01_his_asp.yaml --pdb data/01/inputs/5fqj-6cf4761/01_his_asp_.pdb
```

This resolves every named hydrogen bond through the full provenance chain and
prints its distance, angle and normalised terms. An atom it cannot resolve
raises rather than defaulting — which is the whole point, because a missing
atom scored as zero distance looks like a perfect bond.

**First real run: seed-only, 10 backbones × 8 sequences.** Generates and
scores cycle 0, writes a leaderboard, expands nothing. About an hour per
variant:

```powershell
python -m src.cli run --pipeline config/pipeline.small.yaml --target config/targets/01_his_asp.yaml --name small-01
```

**A full search.** Needs both GPUs and all three model stacks, and is hours per
variant rather than one:

```powershell
python -m src.cli run --target config/targets/01_his_asp.yaml --name run1
```

**Run all seven.** On the AWS box, one at a time, each using both cards:

```powershell
bash cluster/aws/run_all.sh
```

On Euler, as a 7-task array:

```powershell
sbatch cluster/euler/submit.sbatch
```

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

Two things it cannot do for you:

1. **Confirm the `contigs` strings.** They are generated by
   `scripts/build_targets.py` and are its construction, not something taken
   from a run that worked — the earlier runs used RFD3, whose contig syntax
   differs. Verify one against the actual RFdiffusion build before trusting a
   production run.
2. **The ten-seed noise control.** One fixed sequence, ten Boltz-2 seeds, all
   three score terms on each output. This is the one that decides whether the
   search is hill-climbing on signal or on noise, and it sets
   `run.eval_replicates` and `run.patience`.

Step 2 is worth the hour it costs. Patience of 2 on a noisy objective
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

## The seven variants

`data/01` … `data/07` hold one motif-only PDB each: the fixed catalytic
residues plus the 6NT ligand, in 5FQJ numbering. Three positions vary, and the
filenames read `<291>_<229>_<261>`:

| | A229 — base | A291 — donor | A261 — nitro |
| --- | --- | --- | --- |
| **01** `his_asp_` | ASP | HIS | — |
| **02** `his_asp_his` | ASP | HIS | HIS |
| **03** `asn_asp_his` | ASP | ASN | HIS |
| **04** `gln_asp_his` | ASP | GLN | HIS |
| **05** `his_glu_his` | GLU | HIS | HIS |
| **06** `asn_glu_his` | GLU | ASN | HIS |
| **07** `gln_glu_his` | GLU | GLN | HIS |

A229 accepts from the benzotriazole `N3`-H, A291 donates to the ligand `N1`,
and A261 (when present) donates to a nitro oxygen `O11`. One target file per
variant, generated from the PDBs:

```powershell
python scripts/build_targets.py
```

**Only variant 01 carries crystal geometry.** Its two contacts are the real
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
- **[cluster/README.md](cluster/README.md)** — how to run it on Euler or the
  AWS box, and the one constraint that decides which: all three tools must be
  on the same machine, because the loop cannot be split.
- **[HANDOFF.md](HANDOFF.md)** — what to do when the refolded motif files
  arrive. Three commands, and what the generator will tell you.
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
