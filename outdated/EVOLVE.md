# EVOLVE — how to run the design loop

Everything here works for **both arms**: fresh de novo backbones, and partial
diffusion from an existing design. Pick your arm in one function call; the rest
of the pipeline is identical.

If you only read one thing: **we are scored on four numbers, and the one we are
failing is geometry.** The organisers' own reference design puts the catalytic
base **11.18 Å** from the ligand after re-prediction, when the design had it at
2.53 Å. Closing that gap is the job. See [BRIEF.md](BRIEF.md) §1.

---

## 0. Setup, once

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows;  source .venv/bin/activate elsewhere
pip install -r env/requirements.txt
pytest -q                          # 166 tests, no GPU needed
```

The whole loop except the folding runs on CPU, so you can develop and test it
with the clusters busy or unreachable.

---

## 1. The loop in one picture

```
 backbone  ──>  sequence  ──>  fold with 6NT  ──>  score  ──>  mutate  ──┐
  (RFD3)      (LigandMPNN)      (Boltz-2 x N seeds)  (our copy of        │
     ▲                                                the judges')       │
     └───────────────────────────────────────────────────────────────────┘
```

Three commands, each does one step and exits. **Nothing is a daemon** — a
long-running process dies with your SSH session and takes the run with it.

```bash
python -m src.evolve.driver init    results/evo/myrun --founder <seq|fasta> --base 177 --donor 99
python -m src.evolve.driver ingest  results/evo/myrun          # after folding
python -m src.evolve.driver propose results/evo/myrun          # breed the next generation
```

Re-running any of them is safe. A generation that already has `scores.csv` is a
no-op, and no sequence is ever folded twice.

---

## 2. Choosing your arm

### De novo — a brand-new backbone every time

```python
from src.generate.rfd3 import denovo, write_input_json, rfd3_command

specs = [denovo(f"denovo_{i}") for i in range(8)]
write_input_json("inputs/Kemp.json", specs)
print(rfd3_command("inputs/Kemp.json", [s.name for s in specs], "results/rfd3", n_batches=4))
```

`contig="260-260"` holds the length at the competition maximum on purpose:
burial counts CA atoms near the catalytic base, and more protein means more
atoms available to be close.

### Partial diffusion — re-noise an existing design

```python
from src.generate.rfd3 import partial

specs = [partial(f"pd_{t}", "results/rfd3/best_design.pdb", partial_t=t)
         for t in (2.0, 4.0, 6.0, 8.0)]
```

`partial_t` is **angstroms of noise**, and RFD3 recommends ≤ 15. Treat it as a dial:

| `partial_t` | what you get |
|---|---|
| 2–4 Å | gentle refinement, recognisably the parent |
| 5–8 Å | real backbone change, same overall fold |
| 10–15 Å | substantial rearrangement, approaching de novo |

**Partial diffusion and the sequence loop compose.** Point mutations search
sequence space around a fixed backbone; partial diffusion searches backbone
space around a fixed design. A good pattern: run the sequence loop until it
stagnates, then partial-diffuse the best design and restart the sequence loop
on the new backbones.

**In both arms the catalytic motif is pinned** — base `A127` (OD1,CG,OD2), donor
`A50` (OE1,CD,NE2), and all of `6NT`. `src/generate/rfd3.py` refuses to build a
spec for a motif residue it has no fixed-atom list for, rather than letting RFD3
choose.

Optional third motif residue: `with_stack=True` adds the Trp44 π-stack. Harder
to scaffold, but a π-stacked ligand is a more confidently placed ligand — and
ligand drift is exactly our measured failure. Worth running as a parallel arm.

---

## 3. Folding

```python
from src.evolve.fold_batch import SiteProfile, write_fold_script

write_fold_script("gen_000/fold.sbatch", SiteProfile.euler(),
                  designs_dir="$SCRATCH/pdh/evo/myrun/gen_000/designs",
                  out_root="$SCRATCH/pdh/evo/myrun/gen_000/boltz",
                  seeds=[1, 2, 3, 4, 5])
```

- `SiteProfile.euler()` — 32× RTX 2080Ti, `--account=es_workshop`,
  `--reservation=ws_49734_gpu`, **`--mem-per-cpu` not `--mem`** (Euler's filter
  rejects `--mem` outright).
- `SiteProfile()` — AWS, `--constraint=g6.xlarge`. Never `g6.2xlarge` (takes the
  whole allocation for one GPU) and never `l4big` (queues forever).

Measured: **21 s per fold on an L4, 34 s on a 2080Ti**, using 3.3 GB of VRAM. One
Boltz process handles a whole generation, so the model loads once.

Two conventions that are not ours to change: the **ligand is always present**,
and the MSA is **`empty`** (measured: a de novo design gets an alignment of
depth 1 with zero non-query rows, so the alignment stage is pure cost).

---

## 4. Scoring — the four metrics

| Metric | Field | Direction | Reference design |
|---|---|---|---|
| iPDE | `ipde` | lower | 2.55 |
| Geometry — base→ligand | `base_to_ligand_a` | lower | **11.18 Å** |
| Geometry — donor→ligand | `donor_to_ligand_a` | lower | 7.48 Å |
| Burial | `base_burial_ca_count` | higher | 30 |
| Consistency | every `*_std` | lower | — |

`src/score/official.py` reproduces the organisers' own numbers **exactly** —
verified against their verification run, which is committed as a test fixture.
If `pytest tests/test_score_official.py` fails, our ranking is fiction; fix that
before anything else.

**Five seeds.** Boltz is run five separate times on the same sequence. A design
that scores well once and badly four times was lucky, and selection uses
`mean − k·spread` so it loses to a consistent one. Do not rank on one seed.

---

## 5. Eligibility — the rules that zero a design

```python
from src.score.eligibility import assess, max_sequence_identity, max_tm_score
```

| Rule | Limit | Tool |
|---|---|---|
| Length | ≤ 260 | free |
| Sequence identity to published enzymes | ≤ 70 % | `mmseqs` vs UniRef90 |
| TM-score to published structures | ≤ 0.7 | `foldseek` vs PDB |

Installed on the AWS head node at `/mnt/work/tools/bin/` (`foldseek`, `TMalign`).

**The identity search must run on a GPU node** — the provided UniRef90 index is
built for GPU prefilter only and a CPU search fails outright.

**"Not checked" never counts as "passed."** If a tool or database is missing,
these functions raise, and an unevaluated rule is recorded in `unchecked`. A
novelty score that silently returned 0.0 would report a design as novel exactly
when we failed to look.

> **If you are on the homologue arm, read this.** Starting from an existing
> enzyme is the fastest route to *both* disqualifiers. Run the identity check
> early and often rather than at submission.

---

## 6. Tuning the loop

All in `src/evolve/`, all parameterised, none of it hardcoded:

| What | Where | Default |
|---|---|---|
| Mutations per child | `MutationPolicy.k_weights` | mostly 1, tail to 5 |
| Conservative / model / radical | `MutationPolicy.mode_weights` | 0.55 / 0.30 / 0.15 |
| Where mutations land | `MutationPolicy.class_weights` | 60 % second shell |
| Fitness weights | `FitnessWeights` | geometry dominates |
| Hard gates | `HardGates` | base contact ≤ 12 Å |
| Population, parents, seeds | `RunConfig` | 24 / 4 / 5 |

`class_weights` is a **budget per class**, not a weight per position — the
surface is ~90 % of a protein and would otherwise swallow every mutation.

Two things happen automatically and are logged, never decided by a human at
hour 25: the mutation rate widens after two generations without improvement,
and again if the population collapses (`src/evolve/diversity.py`).

---

## 7. Known rough edges

- **The hard gate rejects the founder** (12.35 Å vs a 12.0 Å limit). Correct in
  principle, but a generation 0 containing *only* the founder would abort.
  Seed generation 0 with mutants, as `init` does.
- **Cross-cluster agreement is unverified.** AWS measured the founder at
  11.18 Å on one seed; Euler got 12.35 Å over two. Consistent within seed noise,
  but not proof. Same seed on both clusters would settle it. **Until then, do
  not pool AWS and Euler numbers in one leaderboard.**
- **The PDB database for TM-scores is not in place yet** (the Foldseek
  downloader followed a redirect to an error page). `TMalign` works for pairwise
  checks in the meantime.
- **Partial diffusion is wired but not yet run.** `partial_t` is read straight
  from RFD3's own input specification; nobody has executed it here.

---

## 8. Why not RFD4?

RFD4-Proteina exists and is public (BSD-3, ~1.7 GB of weights). We are staying
on RFD3 for this weekend, deliberately:

- The docs do **not** state that a small-molecule ligand can be held fixed, or
  that specific **side-chain atoms** can be pinned. Those are the two things our
  task cannot do without. Its PPI benchmark is described as using *backbone*
  conditioning, which is a hint in the wrong direction.
- It is a new ~7 GB environment with a new input format (annotated CIF / YAML),
  so the whole bridge into LigandMPNN would need rebuilding.
- RFD3 is installed, validated by the organisers' own job, and demonstrably
  works end to end today.

Not negligible, and not clearly better *for this task*. If someone has spare
time, the question that settles it is whether `sel(...)` can select side-chain
atoms and a ligand — see `examples/enzyme_design/input.cif` in their repo.
