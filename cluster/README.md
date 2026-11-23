# Running on a cluster

Seven variants, seven independent searches. Each is a separate scaffold with
its own frontier, stopping condition and leaderboard, and there is no
cross-scaffold competition — so they can run in any order, on any machine, in
parallel or one at a time, and the results stay comparable.

## Read this first: the loop cannot be split across two clusters

`RUNLOG.md` records the previous arrangement: **RFD3 and LigandMPNN on the AWS
box, Boltz-2 on Euler.** That worked when generation and folding were separate
batch steps a human moved files between.

It does not work here. The search is a closed loop — a cycle cannot choose its
next frontier until the designs it just produced have been scored, and scoring
needs the Boltz-2 output. A split would mean a round trip between two clusters
*per cycle*, twelve times per variant, with a human in the middle.

So **all three tools must be on the same machine**, and the first job is to
find out which machine that is:

```bash
python cluster/preflight.py
```

It checks the GPUs, the VRAM, whether bf16 actually works, and whether
RFdiffusion, LigandMPNN and Boltz-2 are all reachable from the paths in
`config/pipeline.yaml`. Run it on both clusters and compare. Three outcomes:

| preflight says | what to do |
| --- | --- |
| all three tools found on one cluster | run everything there |
| AWS has RFdiffusion + LigandMPNN but no Boltz-2 | `pip install boltz` there — it is a normal Python package and the 2×12 GB cards are what the pipeline was designed for |
| Euler has Boltz-2 but no RFdiffusion/LigandMPNN | install them there, or run on AWS |

If neither cluster can host all three, the pipeline needs a change it does not
currently have: a resumable, two-phase mode that stops after generation and
restarts after folding. That is real work, not a config flag. Say so rather
than hand-shuttling files for twelve cycles.

## ETH Euler

```bash
ssh <ethz-user>@euler.ethz.ch
cd /cluster/home/$USER/protein-madness
bash cluster/euler/setup.sh                      # ~10 min, login node
source ~/protein-madness-env/activate.sh
```

Then, **in this order**:

```bash
python cluster/preflight.py
```

Fix every `FAIL` before going further — most will be the three tool paths in
`config/pipeline.yaml`.

**Warm the Boltz cache on the login node.** Compute nodes have no outbound
internet, and Boltz downloads its weights on first use. Without this the first
array task hangs until the walltime kills it. One small prediction is enough;
any single-sequence input will do.

Then confirm the GPU story from inside a job and record it:

```bash
srun --gpus=1 --gres=gpumem:10g --time=00:20:00 python cluster/preflight.py --require-gpu --write runs/hardware_notes.md
```

That writes phase 0's findings to `runs/hardware_notes.md`, which is where
`PIPELINE.md` asks for them.

```bash
mkdir -p runs/slurm && sbatch cluster/euler/submit.sbatch
```

An array of 7, one task per variant, two GPUs each, **seed-only by default**.
For a full search:

```bash
sbatch --export=ALL,PM_PIPELINE=config/pipeline.yaml cluster/euler/submit.sbatch
```

A subset:

```bash
sbatch --array=1,5 cluster/euler/submit.sbatch
```

The reservation used for the hackathon was `ws_49734_gpu`; the `--account` and
`--reservation` lines in the script are commented out, so uncomment them if you
still have it.

## The provided AWS cluster — archived, never submitted

No scheduler, two GPUs, and the provided tooling already installed under
`/mnt/ref`. Euler is what the production batches ran on, so these two scripts
now live in [outdated/cluster/aws/](../outdated/cluster/aws/) and the paths
below are relative to there:

```bash
bash outdated/cluster/aws/setup.sh                # builds the venv, finds the tools
source ~/protein-madness-env/activate.sh
python cluster/preflight.py --require-gpu --write runs/hardware_notes.md
bash outdated/cluster/aws/run_all.sh --dry-run
bash outdated/cluster/aws/run_all.sh
```

`run_all.sh` runs them **one at a time**, each using both cards. That is
deliberate: the pipeline's own worker pool already puts one design on each GPU,
so two concurrent runs would put two models on one card. 12 GB is not 24 GB,
and Boltz-2 on a ~270-token system is near the limit of one card by itself.
`PM_JOBS` can raise it, but only if you have also set `run.gpus` to a single
device per run.

## Gotchas that have actually bitten this project

**CRLF line endings.** `RUNLOG.md` §4 records `sbatch` rejecting a submission
script because it had Windows line endings, with an error that does not mention
them. `.gitattributes` forces LF on `*.sh` and `*.sbatch`, and
`cluster/preflight.py` checks the live script directories. If you hit it
anyway: `sed -i 's/\r$//' <file>`.

**Compute nodes have no outbound internet on Euler.** Anything that downloads
must happen on the login node. We submit Boltz with `msa: empty`
(single-sequence) so nothing is fetched at fold time — keep it that way. The
MSA web API works on the login node and fails inside a job, sometimes by
hanging rather than erroring.

**`$SCRATCH` is purged after ~2 weeks and is not backed up.** Weights and the
Boltz cache live there. `runs/` lives in the repo, so the sqlite database and
the leaderboards survive; the structures are large and gitignored, so copy
anything you want to keep.

**The token budget is a hard refusal, not a clamp.** The seven variants expand
to at most 143 residues + 12 ligand atoms = 155 tokens, comfortably inside
`boltz.max_tokens: 400`. If you widen the contigs, check that number — the
wrapper refuses to submit rather than OOM, so an over-long design is logged as
errored and skipped, not silently truncated.

## Watching a run, and copying it to your own machine

Everything is written as designs finish. Nothing waits for the end.

On the cluster:

```bash
python -m src.cli watch --run runs/aws-01_his_asp --follow
tail -f runs/aws-01_his_asp/designs.csv
cat runs/aws-01_his_asp/progress.txt
```

`report` also works mid-run — `designs.db` is in WAL mode, so a reader in
another terminal does not block the workers and sees every design committed so
far.

On your own machine, repeatedly while the run continues:

```bash
bash scripts/pull_run.sh me@euler.ethz.ch:/cluster/home/me/protein-madness aws-01_his_asp
python -m src.cli watch --run runs/aws-01_his_asp
```

`pull_run.sh` is rsync, so repeating it only moves what changed. By default it
takes the CSV, the FASTA, the reports, the sqlite file and the final structures
(`*.cif`, `*.pdb`, the Boltz confidence JSONs, the input YAMLs) but not
RFdiffusion's intermediate trajectories. `--all` takes everything, which can be
gigabytes.

One caveat it also prints: copying a sqlite file that is actively being written
can give a torn snapshot. Mid-run, trust `designs.csv`. If you want a
consistent copy of a live database, ask sqlite on the cluster:

```bash
sqlite3 runs/<name>/designs.db ".backup runs/<name>/designs.snapshot.db"
```

## What a finished run leaves behind

```
runs/<name>/
  designs.csv         one row per design, appended as it is scored
  sequences.fasta     the same designs as sequences
  progress.txt        counts, best so far, designs/hour
  designs.db          sqlite: every design, its score, its lineage, the cache
  report.md           leaderboard + the best design's full ancestry
  structures/         one subdirectory per design (gitignored, large)
  stages.jsonl        per-stage wall time and peak VRAM
  hardware_notes.md   phase 0 findings, from preflight.py
```

`designs.csv` and `sequences.fasta` are not gitignored — they are small and
they are the result. Everything else under `runs/` is.

Re-read a finished or interrupted run without touching a GPU:

```bash
python -m src.cli report --run runs/<name>
```

`resume` is **not implemented** — it refuses rather than restarting the
patience counter from a half-known state. An interrupted run keeps everything
it scored; `report` will show it.
