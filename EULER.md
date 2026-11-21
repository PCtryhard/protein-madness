# Running this pipeline on ETH Euler

End to end, in six steps. Verified against Euler on 2026-11-21 (jobs 21874137,
21878946, 21879805). For what the pipeline *does*, see [PIPELINE.md](PIPELINE.md).

```
seed PDB  →  RFD3  →  LigandMPNN  →  Boltz-2  →  H-bond scorer  →  leaderboard
          (scaffold)  (sequence)    (refold)      (our scoring)
                           ↑                             │
                           └──── wiggle / mutate ←───────┘
```

---

## 1. Deploy the code

The clone at `~/protein-madness` is HTTPS with no stored credentials, so it
**cannot `git pull`**, and its working tree is newer than its index. Never run
`git checkout -- .`, `git reset --hard` or `git stash` there — you would
destroy deployed code. Deploy to a fresh directory instead:

```powershell
git archive --format=tar --prefix=pm-deploy/ HEAD | ssh malexandrov@euler.ethz.ch "tar x -C ~"
```

Tracked files only, ~36 MB, seed PDB included.

## 2. Build the three environments

Three, because the tools pin incompatible stacks: RFD3 needs numpy 2 and
torch 2.7.1+cu126; LigandMPNN pins numpy 1.23.5 and torch 2.2.1 and its
vendored openfold calls `np.int`, which numpy 2 removed. Collapsing them does
not work.

**Login node only** — compute nodes have no outbound internet.

```powershell
ssh malexandrov@euler.ethz.ch "cd ~/pm-deploy && bash cluster/euler/setup_rfd3.sh && bash cluster/euler/setup_ligandmpnn.sh && bash cluster/euler/setup_boltz.sh"
```

| env | Python | holds |
| --- | --- | --- |
| `~/pm-rfd3-env` | 3.12.8 | `rfd3` + our own deps; **the runner runs here** |
| `~/pm-mpnn-env` | 3.11.6 | LigandMPNN's stack, `setuptools<81` |
| `~/pm-boltz-env` | 3.12.8 | `boltz` 2.2.1 |

The runner shells out to each tool, so it needs none of them importable.
`ligandmpnn.python` names `pm-mpnn-env` explicitly — a bare `python` on Euler
resolves to one with no torch, which cost a job.

## 3. Stage the weights and tools

```powershell
ssh malexandrov@euler.ethz.ch "cd ~/pm-deploy && bash cluster/euler/stage_deps.sh"
```

Fetches LigandMPNN (MIT) and its weights, downloads the Boltz-2 weights **on
the login node** so no job ever tries to, and verifies RFD3 against
`cluster/euler/MANIFEST.sha256`. Idempotent.

**RFD3 is the one thing it cannot fetch**, and this repo does not ship it: the
`rfd3`/`rfd3na` packages carry no LICENSE, and `rfd3_latest.ckpt` is 2.6 GB —
past GitHub's 100 MB file limit. Copy both from a machine that has them (no new
credentials needed):

```powershell
ssh AWS "tar c -C <site-packages> rfd3 rfd3na" | ssh euler "tar x -C ~/pm-rfd3-env/lib/python3.12/site-packages"
```

Then re-run `stage_deps.sh` to check the checksums. `$SCRATCH` is purged after
~2 weeks, so if a job later dies on a missing checkpoint, start here.

## 4. Check it before spending an allocation

Every job failure so far was a setup error one of these catches, for free:

```powershell
ssh malexandrov@euler.ethz.ch "cd ~/pm-deploy && PYTHONPATH=. ~/pm-rfd3-env/bin/python cluster/preflight.py --target config/targets/01_his_asp.yaml"
```

Resolves every tool path, checks the RFD3 checkpoint and Boltz assets, and
**probes `ligandmpnn.python` for the modules it must import** — not just that
the file exists. Expect `PREFLIGHT PASSED` with one `WARN cuda` (no GPU on a
login node; fine).

```powershell
ssh malexandrov@euler.ethz.ch "cd ~/pm-deploy && PYTHONPATH=. ~/pm-rfd3-env/bin/python -m src.cli -v smoke --target config/targets/01_his_asp.yaml"
```

`-v` goes **before** the subcommand. Smoke validates the configs and the motif;
it does *not* check tool paths, so it is not a substitute for preflight.

Tests need a `python` on `PATH`, which Euler's modules do not provide:

```powershell
ssh malexandrov@euler.ethz.ch "mkdir -p ~/.pmbin && ln -sf \$(which python3) ~/.pmbin/python && cd ~/pm-deploy && PATH=\$HOME/.pmbin:\$PATH PYTHONPATH=. ~/pm-rfd3-env/bin/python -m pytest -q"
```

## 5. Submit

One script, parameterised by environment variable. `sbatch` is the SLURM
command; `run.sbatch` is the job script it takes.

```powershell
ssh malexandrov@euler.ethz.ch "cd ~/pm-deploy && PM_REPO=\$HOME/pm-deploy sbatch cluster/euler/run.sbatch"
```

| variable | default | meaning |
| --- | --- | --- |
| `PM_REPO` | `$HOME/pm-deploy` | where the code is |
| `PM_CONFIG` | `config/pipeline.yaml` | pipeline config |
| `PM_TARGET` | `config/targets/01_his_asp.yaml` | target |
| `PM_NAME` | `euler-01` | run name under `runs/` |
| `PM_WIGGLE_CHECK` | `0` | `1` also probes the wiggle branch first |

It runs preflight, smoke, the search, then writes the leaderboard. Override
SLURM settings on the command line:
`sbatch --time=01:00:00 --job-name=pm-fast cluster/euler/run.sbatch`.

A smaller config goes **outside `config/`** — `tests/test_config.py` enforces
exactly one pipeline config in that directory, deliberately:

```powershell
ssh malexandrov@euler.ethz.ch "cd ~/pm-deploy && PM_CONFIG=\$HOME/pm-runconf/fast.yaml PM_NAME=fast-01 sbatch cluster/euler/run.sbatch"
```

**Do not "optimise" `boltz.no_kernels` or `boltz.precision`.** Euler's GPU pool
is mixed — jobs have landed on a TITAN RTX (sm_75, pre-Ampere) and on RTX
3090s. The config must stay correct for the oldest card it might get.

## 6. Watch it, then read the leaderboard

Every append is flushed and `fsync`ed, so these are readable mid-run. The SLURM
log is in `$HOME` and so readable from any login node — unlike `/tmp`, which is
node-local.

```powershell
ssh malexandrov@euler.ethz.ch "tail -f ~/pm-deploy/runs/euler-01/progress.txt"
```

`runs/<name>/` holds `designs.csv` (a row per scored design, distance and angle
per bond), `sequences.fasta`, `progress.txt`, `stages.jsonl` (every command,
return code, wall time), `designs.db` and `report.md`.

```powershell
ssh malexandrov@euler.ethz.ch "cd ~/pm-deploy && PYTHONPATH=. ~/pm-rfd3-env/bin/python -m src.cli report --run runs/euler-01 --write"
```

### Reading the numbers honestly

A design is **gated** (score 0, still recorded) if a required bond is beyond
`distance.fail` 4.5 Å *or* below `distance.clash` 2.2 Å — too close counts too.
`gated_bonds` does not yet say which of the two.

Reference, job 21874137 (23m41s, RTX 3090): 24 designs, best **0.7037** with
bonds at 3.593 Å / 82.6° and 2.790 Å / 140.4°.

**About one real design in fifteen clears the gate (~7 %).** Size runs
accordingly — ten designs has an expected yield below one, and a run where
nothing clears stops with `every design in the pool was gated` and never
reaches the wiggle or mutate branches. Timing: RFD3 costs a fixed ~200 s for
5–8 backbones up front, then ~6 s LigandMPNN and ~35–50 s Boltz per sequence.

---

## Troubleshooting

| symptom | fix |
| --- | --- |
| dies in ~12 s on argparse | `-v` goes *before* the subcommand |
| `FileNotFoundError: config/pipeline.euler.yaml` | stale deploy — redeploy (step 1) |
| LigandMPNN has no torch | `ligandmpnn.python` must be `~/pm-mpnn-env/bin/python` |
| `np.int` / `pkg_resources` / `ml_collections` | rebuild `pm-mpnn-env` on Python 3.11 from LigandMPNN's own `requirements.txt` + `setuptools<81` |
| `MMCIFParseError: contained no atoms` | tools dispatch on file extension; handled by `_normalised_input()`. Do not rename files in `data/` |
| Boltz downloads on a compute node | run `stage_deps.sh` on a login node first |
| missing checkpoint after a quiet week | `$SCRATCH` purge — re-stage (step 3) |
| `every design in the pool was gated` | not a crash; nothing cleared 4.5 Å. Run more backbones |

## Known gaps

- `src/cli.py resume` is **not implemented** and raises on purpose. A dead run
  must be re-run, not continued.
- `src/cli.py measure` reads seeds, not Boltz outputs — Boltz renumbers the
  chain `1..N`, so it fails with `motif residues [('A', 291)] are absent`. The
  runner scores predictions correctly via `tracking`; the standalone CLI takes
  no mapping.
- `boltz.precision` is **dead config**: Boltz 2.2.1 has no `--precision` flag,
  so the value is validated and stored as provenance but never applied.
- `src/runner.py` has no unit tests, including `_relabel_spec_for_backbone`.
  It is covered only by running the pipeline.
- `cluster/euler/setup.sh` and `cluster/euler/submit.sbatch` are **stale** —
  older, using a different weights layout (`$SCRATCH/weights`). Use the three
  `setup_*.sh` scripts and `run.sbatch`.
- RFD3's own backbone sequence scored 0.5968 in job 21874137, competitive with
  LigandMPNN's designs. Interesting as a baseline; not a design we can claim.

## Three bugs fixed after the first complete run

**LigandMPNN's input record was ranked as a design.** Both records in its FASTA
carry `T=0.1`, so filtering on `"T="` kept the input sequence — RFD3's own
backbone identities — which Boltz then folded at ~50 s each. 5 of the 24
entries in job 21874137 were input records, including the **second-placed**
one. Now filtered on `id=`.

**The wiggle branch could never run.** Two faults: `rfd3 partial_T=10` fails
Hydra composition and `+partial_T=10` fails with `unexpected keyword argument`
— RFD3 takes partial diffusion as `partial_t` on the per-example design spec,
in **Ångströms of noise** (`t <= 15`), not as a timestep count. And the spec
was built in the *seed's* numbering (A229/A291) then pointed at a *backbone*,
where the motif sits elsewhere (A142/A179) and A291 does not exist at all. Now
`rfd3.partial_t: 2.0`, and the spec is relabelled through that parent's own
`seed_to_hal` map.

**Every backbone reported index 0.** The index was read by scanning the
filename backwards for the first digit, which finds `model_0`'s zero rather
than the batch number. That index feeds `mpnn_{index}`, so all backbones wrote
into one shared `mpnn_0/`. Nothing was corrupted — the FASTA is found by stem —
but the logs lied. Now parsed as the digit before `_model_`.
