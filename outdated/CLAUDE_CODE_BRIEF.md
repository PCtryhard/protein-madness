# Claude Code brief: enzyme-design hackathon repo

Paste this whole file into Claude Code as the opening message, or save it in an
empty directory and start with: `read CLAUDE_CODE_BRIEF.md and execute it`.

---

## 0. Fill these in before you start

- `TEAM_SIZE`: <FILL IN>
- `TEAM_MEMBERS` (names + what each wants to own): <FILL IN>
- `GITHUB_REMOTE`: <FILL IN or "none, local only for now">
- `MY_MACHINE`: <FILL IN: macOS/Linux, GPU or not, RAM>
- `ETH_USER`: clusteruser

Everything else below is fixed.

---

## 1. What you are building and why

A hackathon repo for the **enzyme design track** of the D-BSSE Protein Design
Hackathon, Basel, 21–22 November 2026. Work starts Saturday 08:30, pitches
Sunday 17:30. Team of `TEAM_SIZE`.

**The single hardest constraint: the challenge is not announced until Saturday
morning.** We do not know the target enzyme, the reaction, the substrate, or the
success metric. So the repo must be built so that on Saturday morning *only a
config file changes* and the whole pipeline runs end to end on the new target.

Design for that. Any code that hardcodes a target, a sequence, a reaction, or a
metric is wrong and will have to be rewritten under time pressure.

**Second constraint: the differentiator at this hackathon is evaluation, not
generation.** Every team will produce sequences. Few will be able to say
anything defensible about whether their designs are good, by Sunday afternoon.
So the scoring and reporting modules are the centre of this repo, not an
afterthought. Build them first and build them well.

---

## 2. Rules of engagement (non-negotiable)

1. **No fake implementations.** If you cannot implement something correctly
   without a model weight, a database, or a decision we have not made, write the
   function signature, raise `NotImplementedError` with a one-line message
   saying exactly what is missing, and add it to `TODO.md`. A plausible-looking
   stub that silently returns wrong numbers is worse than nothing. This is the
   single most important rule in this brief.
2. **Pin every version.** Exact versions in `environment.yml` and
   `requirements.txt`. No bare `pip install torch`.
3. **No model weights in git.** Weights are downloaded by a script into a
   gitignored directory and their location is recorded in `MANIFEST.md`.
4. **Every tool is a CLI with a fixed in/out contract** (section 4). No
   notebook-only code paths. Notebooks may only call into `src/`.
5. **Everything writes to one table.** Every design that exists anywhere in this
   project has a row in `results/leaderboard.csv`, keyed by `design_id`. One
   source of truth. The pitch reads from it.
6. **Commit often, small commits, real messages.** Assume four people will be
   pushing to this at 3am.
7. When you are unsure between two designs, pick the one that is easier to throw
   away.

---

## 3. Repo layout

Create exactly this:

```
.
├── README.md                  # what this is, 10-line quickstart
├── ONBOARDING.md              # a teammate goes from zero to passing smoke test
├── MANIFEST.md                # where every model weight / DB lives, per backend
├── PLAN.md                    # the 30-hour timeline (section 10)
├── DECISIONS.md               # append-only log: date, decision, why
├── TODO.md                    # anything you stubbed, with what's missing
├── PITCH.md                   # skeleton of Sunday's pitch (section 11)
├── .gitignore
├── env/
│   ├── environment.yml        # conda/mamba, pinned
│   ├── requirements.txt       # pip, pinned
│   ├── modal_image.py         # Modal image definition + one GPU function
│   └── euler_setup.sh         # module loads + venv on Euler, idempotent
├── configs/
│   ├── target.example.yaml    # THE file that changes Saturday morning
│   └── pipeline.yaml          # backend choices, batch sizes, thresholds
├── src/
│   ├── __init__.py
│   ├── ids.py                 # design_id generation, provenance
│   ├── io/                    # fasta/pdb/cif readers+writers, normalisation
│   ├── generate/              # sequence & backbone design wrappers
│   ├── fold/                  # structure prediction wrappers
│   ├── score/                 # <- the centre of gravity
│   ├── report/                # figures, tables, pitch assets
│   └── backends/              # local / euler / modal / tamarind dispatch
├── scripts/
│   ├── fetch_weights.sh
│   ├── smoke_test.sh          # definition of done, section 9
│   └── run_pipeline.py        # one entry point, takes a target config
├── data/
│   └── toy/                   # one small enzyme, committed, for smoke tests
├── notebooks/
│   └── figures.ipynb          # calls src/report only
└── results/
    └── .gitkeep               # contents gitignored
```

---

## 4. The CLI contract

Every wrapper in `generate/`, `fold/` and `score/` must be runnable as:

```
python -m src.<module>.<tool> --in <path> --out <path> --config configs/pipeline.yaml [--device ...]
```

and must:

- accept FASTA (sequences) or a directory of PDB/mmCIF (structures) as `--in`
- write its outputs into `--out` plus a single-row-per-design CSV at
  `<out>/<tool>.csv` whose first column is `design_id`
- write a `<out>/<tool>.meta.json` recording: tool name, version, git SHA,
  config hash, backend, wall time, and the exact command line
- be a no-op (exit 0, log "cached") if its outputs already exist and the config
  hash matches, unless `--force`

Why: this lets four people work on different modules simultaneously without
merge conflicts, lets us swap a fold backend when Euler is contended, and makes
the leaderboard join trivial. It also means a crash at hour 20 costs one stage,
not the run.

`src/ids.py` defines `design_id` as a short deterministic hash of
(sequence, generating method, config hash) so the same design produced twice
collides rather than duplicating.

---

## 5. Module specs

### `src/io/`
Readers and writers for FASTA, PDB, mmCIF. One normalisation function that
every entry point calls: strip insertion codes, renumber from 1, keep a single
chain unless told otherwise, record the original numbering in a sidecar so we
can map catalytic residue indices back. Getting residue numbering wrong is the
classic way to silently poison every downstream metric; add unit tests for it.

### `src/generate/`
Thin wrappers, each independently skippable:
- `proteinmpnn.py` — fixed-backbone sequence design
- `ligandmpnn.py` — same but ligand-aware. **For enzyme design this matters more
  than ProteinMPNN**: it conditions on the substrate in the pocket.
- `rfdiffusion.py` — backbone generation / motif scaffolding. Expose
  `--contigs` and a motif-holding mode, because "scaffold this catalytic motif"
  is the most likely shape of an enzyme-design challenge.
- `mutate.py` — cheap CPU baseline: enumerate point and double mutants around
  specified positions. Do not skip this. A systematic mutational scan around the
  active site is a legitimate submission, runs without a GPU, and is our
  fallback if everything else jams.

### `src/fold/`
`--backend {esmfold,colabfold,boltz,chai,tamarind}`, same contract for each.
Outputs per design: predicted structure, pLDDT (mean and per-residue), PAE
matrix, and for co-folding backends the ligand pose.

Read section 7 on memory before choosing defaults.

**MSA gotcha:** ColabFold-style backends want MSAs. Euler compute nodes have
heavily restricted outbound internet, so the MMseqs2 web API will fail there
even though it works on the login node. Either precompute MSAs before Friday
night and cache them in `results/msa/`, or stay on single-sequence models
(ESMFold, Boltz) on Euler. Put a loud note about this in `MANIFEST.md`.

### `src/score/` — build this first and most carefully

One function per metric, each taking a design and returning a scalar or small
dict, plus `score/run_all.py` that joins everything into the leaderboard.
Implement in this order:

**Tier 1 — cheap, CPU, no weights. Must work by Friday.**
- sequence identity to the parent/wild-type
- net charge at pH 7, theoretical pI, GRAVY, fraction of charged residues
- low-complexity and homopolymer runs; unpaired cysteine count
- catalytic residue integrity: given catalytic positions from the target config,
  are the expected residues still present at those positions
- novelty by sequence: max identity to the nearest hit in a reference DB via
  MMseqs2 `easy-search`. Reuse the distance-metric approach from my generative
  enzyme-design novelty audit if you find that code on this machine.

**Tier 2 — needs a folded structure.**
- mean pLDDT, and pLDDT restricted to the active-site residues (the global
  number hides exactly the thing we care about)
- PAE between catalytic residue pairs — a confident fold with a floppy active
  site is a failed design and the global metrics will not show it
- self-consistency: design -> fold -> align to the target backbone -> RMSD and
  TM-score. The standard field metric. Report it.
- **catalytic geometry**: distances and angles between the catalytic atoms,
  compared against the geometry in the reference structure. This is the
  enzyme-specific metric the binder teams have no reason to build, and it is the
  one that makes a pitch credible. Make the atom selections and the ideal
  geometry read from the target config, not hardcoded.
- ligand pose, when a co-folding backend is used: does the substrate stay in the
  pocket, ligand RMSD to the reference pose, contacts with catalytic residues
- hydrophobic patch area and a crude aggregation flag (needs SASA; use
  FreeSASA or Biotite)
- structural novelty: TM-score to the nearest PDB structure via Foldseek

**Tier 3 — only if time allows.** Short OpenMM minimisation and relaxed-state
RMSD; Rosetta-style per-residue energies if PyRosetta is available and licensed.
Stub these and move on.

Each metric gets a unit test on the toy enzyme with a hardcoded expected value.
Tests are how we catch the numbering bug at hour 3 instead of hour 28.

### `src/report/`
- `leaderboard.py` — join all stage CSVs, apply filters from `pipeline.yaml`,
  rank, write `results/leaderboard.csv` and a readable markdown table
- `figures.py` — exactly three figures, written as functions returning
  matplotlib figures, saved as both PNG (300 dpi) and PDF:
  1. score distribution of designs vs. the wild-type baseline, one panel per
     key metric
  2. novelty on one axis against predicted quality on the other, so we can point
     at the top-left quadrant and say "these are novel *and* confident"
  3. the best design: structure render with the active site highlighted, plus
     its per-residue pLDDT trace
- `pitch_assets.py` — regenerates every number and figure in `PITCH.md` from the
  current leaderboard in one command

Build these against *fake* leaderboard data (a generated CSV with the right
columns) so the figures exist and look right before any real design is made.
On Sunday you change the input path, not the plotting code.

### `src/backends/`
Dispatch layer. `local`, `euler` (writes and submits an sbatch script), `modal`,
`tamarind` (stub unless there's a documented API; the event code only arrives
during the hackathon, so assume the web UI and make the stub honest). Each
backend exposes `submit(cmd, resources) -> job_id` and `wait(job_id)`.

For Euler, generate real sbatch templates with `--gpus=1`,
`--gres=gpumem:10g`, a time limit, and `--array` support for batches. Keep the
array-job path working: 500 designs as 500 serial calls will not finish.

---

## 6. Target config

`configs/target.example.yaml` is the contract with Saturday morning. It must be
enough, on its own, to run the pipeline on a new enzyme. At minimum:

```yaml
name: toy_lysozyme
reference_structure: data/toy/ref.pdb
reference_sequence: data/toy/ref.fasta
chain: A
catalytic_residues:        # 1-indexed, in reference numbering
  - {resi: 35, resn: GLU, role: general_acid}
  - {resi: 52, resn: ASP, role: nucleophile}
catalytic_geometry:        # pairs/triples to measure
  - {atoms: ["35:OE1", "52:OD1"], ideal_distance: 9.1, tol: 1.5}
substrate:
  smiles: null
  reference_ligand: null
design_region:             # positions free to change
  mode: shell
  shell_radius: 8.0
  exclude: [35, 52]
objective: >
  one-paragraph plain-English statement of what the challenge asks for
success_metrics: []        # fill from the challenge brief Saturday
```

Write a validator (`python -m src.io.validate_target configs/target.yaml`) that
checks the structure exists, the catalytic residues really are those residue
types at those positions in that structure, and the geometry atom selections
resolve. Run it first thing Saturday. This catches the off-by-one that would
otherwise silently invalidate every number we present.

---

## 7. Compute: three backends, one hard ceiling

- **My machine** (`MY_MACHINE`) — development, Tier-1 scoring, figures.
- **ETH Euler** — ~32 reserved RTX 2080Ti, shared across ~70 participants.
  **sm_75, 11 GB VRAM, no bf16.** Everything must run in fp16 or fp32 under
  11 GB. ESMFold in fp32 does not fit; it needs fp16 plus chunked attention and
  is still tight. Boltz-2 and Chai-1 generally want more. Treat 11 GB as a hard
  wall and make it a config value, not an assumption buried in code. Also: no
  root, so conda/mamba in `$HOME` and weights in `$SCRATCH` (`$HOME` quota is
  small and model weights will blow through it).
- **Modal** — $30/month free credits, A100/H100 on demand, no event code needed.
  This is the escape hatch when Euler is contended or something needs more than
  11 GB. `env/modal_image.py` must contain a working image definition and one
  trivial GPU function that prints `torch.cuda.get_device_name()`. Building that
  image takes tens of minutes the first time, which is exactly the cost we do
  not want to pay on Saturday.
- **Tamarind.bio** — 200 free jobs per participant, event code arrives during
  the hackathon. Web UI. Treat as a manual fallback; `backends/tamarind.py` is a
  documented stub.

`scripts/fetch_weights.sh` must be runnable on each backend and must record what
it put where in `MANIFEST.md`.

---

## 8. Team setup

With `TEAM_SIZE` people, the modules above split cleanly. Generate
`ONBOARDING.md` such that a teammate who has never seen the repo can, in under
20 minutes: clone, create the environment, run the smoke test, see it pass, and
read a one-paragraph description of which module is theirs.

Also set up:
- a `main` branch that must always pass the smoke test, and one branch per module
- `DECISIONS.md` as an append-only log, because at hour 25 nobody will remember
  why a threshold is 0.7
- `results/` gitignored, with a note in `ONBOARDING.md` on how we share results
  between machines (pick one: a shared drive folder, or a `results` branch that
  only ever gets force-pushed by one person)

---

## 9. Definition of done: the smoke test

`scripts/smoke_test.sh` runs the entire pipeline on the committed toy enzyme at
deliberately tiny settings and must:

- complete in under 10 minutes, CPU-only, on a laptop
- produce at least 10 designs
- produce a `results/leaderboard.csv` with every Tier-1 metric populated and no
  NaNs in columns that are supposed to be filled
- produce all three figures as files
- exit non-zero if any of the above fails

A GPU-path variant (`scripts/smoke_test.sh --gpu`) does the same with one fold
backend enabled.

**The repo is not ready for Saturday until this passes on my machine, on Euler,
and on Modal, with the same toy enzyme producing the same numbers on all three.**
That three-way agreement is the actual goal of this week's work. Say so in
`README.md`.

---

## 10. `PLAN.md`

Write the weekend as a timeline with named owners and hard checkpoints. Shape it
around these fixed points: start 08:30 Saturday, challenge announced just before
that, overnight side-challenges for hints and extra compute, pitches 17:30
Sunday. Include a **hard freeze on new generation runs** at a stated Sunday hour,
after which the only remaining work is scoring, figures and the pitch. Teams lose
by generating until the last minute and presenting nothing legible.

---

## 11. `PITCH.md`

A skeleton with the headings and a one-line note on what evidence fills each,
written now so Sunday is filling blanks rather than inventing structure:

1. The problem, in the challenge's own terms
2. Our approach in one diagram
3. What we generated: counts, methods, compute used
4. **How we know it's any good**: the scoring stack, and the baseline we beat
5. Our best designs: the three figures
6. Novelty: how far these are from anything natural
7. What we'd do next and what we'd test experimentally
8. Honest limitations

Every number in it comes from `results/leaderboard.csv` via
`src/report/pitch_assets.py`. Nothing is typed in by hand.

---

## 12. Do not

- Download model weights into the repo
- Invent the challenge, the target, or the success metric
- Write code that only runs in a notebook
- Implement Tier-3 scoring before Tier-1 is tested
- Write a metric you cannot explain to a judge in one sentence
- "Fix" a failing test by loosening the assertion

---

## 13. Order of work

1. Skeleton, `.gitignore`, pinned env files, `README.md`
2. `src/io/` with numbering tests, and the toy enzyme in `data/toy/`
3. `src/ids.py`, the CLI contract helpers, `scripts/run_pipeline.py`
4. Tier-1 scoring, fully tested
5. `src/generate/mutate.py` (the CPU baseline) — now the smoke test can pass
6. `scripts/smoke_test.sh`, green on CPU
7. `src/report/` against fake leaderboard data, all three figures existing
8. `env/modal_image.py` and `env/euler_setup.sh`; get the trivial GPU function
   running on both
9. `src/fold/` with one backend, then Tier-2 scoring
10. `src/generate/` real wrappers, LigandMPNN before ProteinMPNN
11. `ONBOARDING.md`, `PLAN.md`, `PITCH.md`, `DECISIONS.md`, `MANIFEST.md`
12. Report: what is stubbed, what is untested, what you would not trust

Stop and ask me before: choosing a fold backend as default, adding any
dependency that needs a licence, or writing anything that assumes what the
challenge will be.
