# Protein Madness — enzyme design track

Repo for the **enzyme design track**, D-BSSE Protein Design Hackathon, Basel,
21–22 November 2026. Work starts Saturday 08:30; pitches Sunday 17:30.

## ▶ Start here

**The challenge is announced: design a de novo Kemp eliminase.** Two documents,
in this order:

1. **[BRIEF.md](BRIEF.md)** — what we are scored on, what the compute actually
   is, and the strategy. Read this first; it is the plan the team agreed from.
2. **[EVOLVE.md](EVOLVE.md)** — how to *run* the design loop, for both the
   de novo and the partial-diffusion arms. This is the practical guide.

The one thing to know before anything else: **we submit sequences, not
structures.** The organisers re-fold our FASTA with Boltz-2 and score the
result, and the organisers' own reference design puts the catalytic base
**11.18 Å** from the ligand after re-prediction — when the design had it at
2.53 Å. Closing that gap is the entire job.

```bash
pytest -q                                    # 166 tests, no GPU needed
python -m src.evolve.driver --help           # the design loop
```

> Historical note: this repo was built *before* the challenge was announced, to
> be target-agnostic. `PLAN.md`, `MANIFEST.md` and much of `TODO.md` still
> describe that world — Euler/Modal backends, ESMFold, an unknown target. They
> are kept for the record; **`BRIEF.md` supersedes them.**

**The differentiator here is evaluation, not generation.** Every team will
produce sequences; few will be able to say anything defensible about whether
their designs are good. `src/score/` is the centre of this repo and it was built
first.

## Quickstart

> **The virtual environment is not in the repo and never will be.** `.venv/` is
> gitignored on purpose: it is machine-specific and hundreds of megabytes, and a
> Windows venv does not work on macOS anyway. You create it yourself in step 2 —
> nothing is missing from your clone.

**1. Clone**

```bash
git clone https://github.com/PCtryhard/protein-madness.git && cd protein-madness
```

**2. Create the virtual environment**

```bash
python -m venv .venv
```

**3. Activate it** — the command depends on your shell, and this is where people trip:

| Shell | Command |
| --- | --- |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows cmd.exe | `.venv\Scripts\activate.bat` |
| Windows Git Bash | `source .venv/Scripts/activate` |
| macOS / Linux / WSL | `source .venv/bin/activate` |

Your prompt should now start with `(.venv)`. If PowerShell refuses with an
execution-policy error, either use Git Bash instead or run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first.

**4. Install the pinned dependencies and check it works**

```bash
pip install -r env/requirements.txt
```

```bash
bash scripts/smoke_test.sh
```

Takes about a minute, CPU only, no GPU or model weights needed. It must print
`SMOKE TEST PASSED`. On Windows run it from Git Bash; an activated venv is
picked up automatically, or point it at one explicitly with
`PYTHON=.venv/Scripts/python bash scripts/smoke_test.sh`.

Saturday morning, after the challenge is announced:

```bash
cp configs/target.example.yaml configs/target.yaml       # edit: structure, catalytic residues, geometry
python -m src.io.validate_target configs/target.yaml     # RUN THIS FIRST. It catches the off-by-one.
python scripts/run_pipeline.py --target configs/target.yaml --out results/run1
```

Full setup, module ownership and how we share results: **[ONBOARDING.md](ONBOARDING.md)**.

## Definition of done

The repo is **not ready for Saturday** until `scripts/smoke_test.sh` passes **on
the dev laptop, on Euler and on Modal, with the same toy enzyme producing the
same numbers on all three**. That three-way agreement is the actual goal of this
week's work — not feature count. A metric that gives a different answer on
Euler than on a laptop is a metric we cannot put in front of a judge.

Current state: passing on the dev laptop (~15 s). Euler and Modal are not yet
verified — see [TODO.md](TODO.md).

## How it fits together

```
configs/target.yaml ─┐
                     ├─> generate/ ─> designs.fasta ─> fold/ ─> structures ─┐
configs/pipeline.yaml┘                      │                               │
                                            └──> score/tier1 ──┐            │
                                                 score/tier2 <─────────────┘
                                                               │
                                     results/leaderboard.csv <─┘ ──> report/ ──> PITCH.md
```

Every stage is a CLI with one contract (`--in`, `--out`, `--config`), writes
`<out>/<tool>.csv` keyed by `design_id` plus a `.meta.json` recording git SHA,
config hash, backend, wall time and the exact command, and is a cached no-op if
its outputs exist and the config hash matches. So four people can work in
parallel, a fold backend can be swapped when Euler is contended, and a crash at
hour 20 costs one stage rather than the run.

**Everything writes to one table.** Every design that exists anywhere has a row
in `results/leaderboard.csv`, keyed by `design_id`. The pitch reads from it;
nothing in the pitch is typed by hand.

## What is real and what is not

Honesty about this is the point of the repo — see [TODO.md](TODO.md) for the
full list.

| Area | State |
| --- | --- |
| `src/io/` normalisation, numbering, target validator | real, tested against hostile renumbering |
| Tier-1 scoring (sequence) | real, tested, cross-checked against Biopython |
| Tier-2 scoring (structure, catalytic geometry, SASA, self-consistency) | real, tested on the toy enzyme |
| `generate/mutate.py` CPU baseline | real, tested |
| `report/` leaderboard + 3 figures + pitch assets | real, built against fake data |
| `fold/esmfold.py`, `fold/boltz.py` | real wrappers, **not yet run with weights** |
| `generate/ligandmpnn,proteinmpnn,rfdiffusion` | real wrappers, **not yet run with weights** |
| Sequence novelty (MMseqs2), structural novelty (Foldseek) | real, need a binary + DB; raise if absent |
| `fold/colabfold.py`, `fold/chai.py`, `backends/tamarind.py` | honest stubs, raise with what's missing |
| Tier-3 (OpenMM, Rosetta) | not started, deliberately |

Nothing in this repo returns a plausible-looking number it did not compute. A
metric whose inputs are missing is `NaN` with a note, or raises — never 0.0.

## Documents

- **[BRIEF.md](BRIEF.md) — the strategy. Start here.**
- **[EVOLVE.md](EVOLVE.md) — how to run the design loop (both arms).**
- [ONBOARDING.md](ONBOARDING.md) — zero to passing smoke test in under 20 minutes
- [PLAN.md](PLAN.md) — the 30-hour weekend timeline, owners, hard freeze
- [PITCH.md](PITCH.md) — Sunday's pitch skeleton; numbers fill in automatically
- [DECISIONS.md](DECISIONS.md) — append-only: what we decided and why
- [TODO.md](TODO.md) — everything stubbed, with exactly what is missing
- [MANIFEST.md](MANIFEST.md) — where every model weight and database lives, per backend
