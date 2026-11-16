# Onboarding

Zero to a passing smoke test in under 20 minutes. If any step takes longer than
it says here, stop and say so in the group chat — it means the repo is wrong,
not you.

## 1. Clone and install (5 min)

```bash
git clone <REMOTE> "Protein Madness" && cd "Protein Madness"
```

```bash
python -m venv .venv && source .venv/Scripts/activate && pip install -r env/requirements.txt
```

On macOS/Linux the activate line is `source .venv/bin/activate`.
With conda/mamba instead: `mamba env create -f env/environment.yml && mamba activate enzyme-hack`.

Python 3.12 is what everything is pinned against. The CPU path needs no GPU, no
model weights and no network.

> **Do not put the venv on OneDrive/Dropbox if you can avoid it.** Sync will
> fight you over thousands of small files. `python -m venv ~/.venvs/protein-madness`
> and activate from there works fine.

## 2. Smoke test (2 min)

```bash
bash scripts/smoke_test.sh
```

Expect `SMOKE TEST PASSED` in well under a minute. It runs the unit tests,
validates the toy target config, runs the whole pipeline on a committed toy
enzyme (hen egg-white lysozyme, PDB 1AKI), and checks that the leaderboard has
every Tier-1 metric with no NaNs and that all three figures exist.

If it fails, the output names the failing check. The most common cause is a
half-installed environment: `pip install -r env/requirements.txt` again and
re-read the first few lines of output.

## 3. Look at what it produced (3 min)

```bash
ls results/smoke/figures/ && head -3 results/smoke/leaderboard.md
```

That is the whole product in miniature: designs in, one table out, three
figures. On Sunday the only difference is which target config went in.

## 4. Your module (10 min)

Take your branch and read your module's source. Every module is a CLI with the
same contract, so learning one teaches you all of them.

```bash
git checkout -b module/<yours>
```

| Owner | Module | What it is | Start by reading |
| --- | --- | --- | --- |
| 1 | **integration / report** | Joins every stage CSV into `results/leaderboard.csv`, applies filters and ranking from `configs/pipeline.yaml`, draws the three figures, regenerates every number in `PITCH.md`. Owns `main` staying green. | `src/report/leaderboard.py` |
| 2 | **scoring** | The centre of the repo. Tier-1 (sequence: charge, GRAVY, complexity, catalytic integrity) and Tier-2 (structure: pLDDT, PAE between catalytic residues, self-consistency RMSD/TM, **catalytic geometry**, SASA). Every metric has a unit test with a hardcoded value on the toy enzyme. | `src/score/sequence.py`, then `src/score/structure.py` |
| 3 | **novelty** | How far our designs are from anything natural: MMseqs2 `easy-search` for sequence novelty, Foldseek for structural novelty. Both need a binary and a reference DB that are not in this repo. Your job this week is making them exist and recording them in `MANIFEST.md`. | `src/score/novelty.py` |
| 4 | **generation: MPNN** | LigandMPNN (ligand-aware — for enzymes this matters more than ProteinMPNN, it conditions on the substrate in the pocket) and ProteinMPNN. Wrappers exist; they have never been run with real weights. | `src/generate/ligandmpnn.py` |
| 5 | **generation: RFdiffusion** | Backbone generation and motif scaffolding. "Scaffold this catalytic motif" is the most likely shape of an enzyme-design challenge, so the motif-holding path is the one that matters. | `src/generate/rfdiffusion.py` |
| 6 | **compute** | Euler (sbatch, array jobs, 11 GB ceiling), Modal (the escape hatch), fold backends. Your deliverable this week is the three-way smoke-test agreement. | `src/backends/`, `env/euler_setup.sh` |

Everyone, whatever your module: read **`src/cli.py`** first. It is the contract
all the others obey, and it is 200 lines.

## 5. Add a metric, end to end

The shape of almost every change in this repo:

1. Write the function in `src/score/sequence.py` or `src/score/structure.py`.
   One metric, one function, returns a scalar or a small dict. The **first line
   of its docstring is the sentence you would say to a judge** — if you cannot
   write that sentence, do not write the metric.
2. Add a unit test in `tests/` with a hardcoded expected value on the toy
   enzyme. Tests are how we catch the numbering bug at hour 3 instead of hour 28.
3. Add it to the `all_*_metrics` aggregate so it reaches the CSV.
4. If it should filter or rank, add it to `configs/pipeline.yaml` — never
   hardcode a threshold.
5. `bash scripts/smoke_test.sh` must still pass before you push.

## Rules we actually hold each other to

1. **No fake implementations.** If you cannot implement something correctly
   without a weight, a DB or a decision we have not made: write the signature,
   `raise NotImplementedError` with one line saying exactly what is missing, and
   add it to `TODO.md`. A plausible-looking stub that silently returns wrong
   numbers is worse than nothing. This is the most important rule here.
2. **`main` always passes the smoke test.** Work on your branch. If you break
   `main`, revert first and debug second.
3. **Never "fix" a failing test by loosening the assertion.** Understand why it
   failed. Three of our own tests failed during the build and every one was a
   wrong expectation we then wrote down correctly — see the commit log.
4. **Commit often, small commits, real messages.** Assume six people are pushing
   at 3am.
5. **No model weights in git.** They go in a gitignored directory and their
   location goes in `MANIFEST.md`.
6. **Every threshold lives in `configs/pipeline.yaml`** and its reason in
   `DECISIONS.md`. At hour 25 nobody remembers why a number is 0.7.
7. **When unsure between two designs, pick the one easier to throw away.**

## Sharing results between machines

`results/` is gitignored — it holds structures, matrices and figures that would
bloat the repo and conflict constantly.

**We use a shared drive folder.** One folder, mirroring `results/`, one
subdirectory per run (`run1/`, `run2/`, ...). Copy your run directory in when it
finishes; never edit someone else's. `results/leaderboard.csv` is rebuilt from
stage CSVs by whoever is integrating, so you only ever need to share the stage
CSVs and the structures.

Link: _TBD — owner 1 creates it at kickoff and pastes it here._

The alternative we rejected: a `results` branch force-pushed by one person.
It works, but a force-push at 3am by the wrong person loses a night of compute.

## When something goes wrong

| Symptom | First thing to check |
| --- | --- |
| A metric looks too good | A numbering bug. Run `python -m src.io.validate_target configs/target.yaml`. Check the wild-type row scores sensibly. |
| A stage says "cached" and you wanted it to re-run | `--force`, or change the config (the cache key is the config hash). |
| A stage crashed at hour 20 | Just re-run the pipeline with the same `--out`. Completed stages are cached; only the broken one repeats. |
| `NaN` in a column | Intended. Check `tier2_notes` or the stage log for which input was missing. We never fill a missing metric with 0. |
| Everything on Euler is queued | Switch to Modal. Decide early — see PLAN.md. |
| ESMFold OOMs on a 2080Ti | Lower `fold.esmfold.chunk_size`, confirm `compute.precision: fp16`. fp32 does not fit in 11 GB. |
