# BRIEF — Kemp eliminase, Perlin × D-BSSE, 21–22 Nov 2026, team 04

Status: **draft for team discussion.** On `main` so everyone can read it;
**not yet agreed.** Argue with it, then edit it — §14 is the agenda.

This supersedes the "unknown target" assumptions in `PLAN.md` and `README.md`.
The challenge is announced, the compute is not what we planned for, and the
scoring function is *published*. All three change what we should do today.

---

## 0. The thesis, in one paragraph

We are not being asked to design a good enzyme. We are being asked to produce
sequences that score well on **four named, mechanical metrics computed by
Boltz-2**. Those metrics are knowable and cheap to re-implement. So our plan is
not "generate a lot and hope": it is **directed evolution against our own copy
of the judge's scoring function**. Start from one backbone, design a sequence,
refold it with the transition-state analogue bound, score it, mutate it, refold,
score, keep what improved. Every generation is a real measurement, every design
we submit has a documented lineage, and the GPU is only ever spent on a mutation
we have a reason to try. Everyone else will spend the weekend generating. We
spend it selecting.

---

## 1. What we are actually scored on

From the Perlin deck (slide 6), the score is these four things and nothing else:

| # | Metric | What it literally is | Where it comes from |
|---|---|---|---|
| 1 | **iPDE** | Boltz-2's estimated error on protein↔ligand token distances. Is the ligand confidently placed? | `confidence_*.json` from Boltz-2 |
| 2 | **Geometry** | Distances / angles / dihedrals vs. the ideal approach geometry for the **catalytic base** and the **H-bond donor** | measured on the Boltz-2 complex |
| 3 | **Catalytic base burial** | Count of CA atoms within **12 Å** of the catalytic base. Proxy for keeping the base's pKa by keeping it out of solvent | measured on the predicted backbone |
| 4 | **Consistency** | How well 1–3 reproduce across **5 Boltz-2 seeds**. Proxy for preorganisation | 5 replicates per sequence |

Three observations that drive everything today:

**(a) We submit sequences, not structures.** The organisers re-fold our FASTA
themselves with Boltz-2 and the 6NT ligand. A design that looks beautiful in
RFD3 and falls apart in Boltz scores zero. **Boltz-2 agreement is the product**,
which is exactly why the evolution loop closes through a real refold every
generation rather than through a cheap surrogate.

**(b) Metrics 2 and 3 are computable for free on a backbone**, on CPU, in
milliseconds — no folding needed. That makes them a filter on *proposals*
before we pay to evaluate them.

**(c) Metric 4 is the one everyone will ignore.** Teams will run one seed, pick
their best number and submit. Selecting on *mean minus spread* across seeds is
cheap, obvious once stated, and is also the only honest way to run selection —
see §4.4, because selecting on a single noisy seed is the fastest way to evolve
towards noise.

### The exact geometry the organisers measure

The reference pipeline (`example_e2e_kemp/src/verify_boltz.py`) measures
precisely two contacts on the Boltz output:

```
ASP(A127) OD2  →  6NT N3     # catalytic base to the TS analogue
GLN(A50)  NE2  →  6NT N1     # H-bond donor to the developing oxyanion
```

That is almost certainly the backbone of the official geometry term. **We
re-implement exactly this, atom name for atom name**, and treat anything we add
on top (angles, dihedrals, π-stack to Trp) as a secondary signal, never a
substitute.

### The motif, and the ligand we never take out

`inputs/HG3_17_kemp_motif.pdb` + `inputs/Kemp.json`, derived from HG3.17:

- **ASP A127** — catalytic base, fixed atoms `OD1, CG, OD2`
- **GLN A50** — H-bond donor, fixed atoms `OE1, CD, NE2`
- **TRP A44** — π-stacking, fixed atoms `CD1, NE1, CE2` — *only in the
  150-residue smoke entry, **dropped** from the 260-residue entry*
- **6NT A1305** — 6-nitrobenzotriazole, the stable transition-state analogue
  (crystal pose from PDB 4BS0 at 1.09 Å), all atoms fixed

**Every fold in this pipeline is a complex with 6NT present.** We never predict
the apo protein and we never score a design that was folded without the ligand.
The whole scoring function is about the ligand; a structure without it is not a
cheaper version of the measurement, it is a different measurement.

---

## 2. The binding constraint: we have two GPUs

This is the thing that invalidates most of `PLAN.md`. There is no Euler
allocation for this, no Modal, no Tamarind, no $30 of A100s. From the cluster
guide:

> Your team can run **8 logical CPUs of GPU workers at a time**. Either one
> g6.2xlarge, or two g6.xlarge.

- Every g6 is **one NVIDIA L4, 24 GB VRAM**.
- **Two `g6.xlarge` = two GPUs. One `g6.2xlarge` = one GPU.** Take the pair.
- The *only* reason to take the single big node is MMseqs2 MSA building, which
  wants ~28 GB system RAM.
- `l4big` (g6.4xlarge) is above our allocation and **queues forever instead of
  failing**. Never submit there. The example sbatch asks for
  `--constraint=g6.2xlarge`; that one line costs us half our compute.
- GPU nodes are dynamic (`idle~`, powered down). The first job of the day pays a
  several-minute spin-up. Keep the queue warm; don't let both nodes idle down
  between generations.

**Therefore: ~2 GPUs × ~30 usable hours ≈ 60 GPU-hours, total, for everything.**
Boltz-2 is the expensive step and the only step that produces score. The entire
design of the evolution loop below is shaped by that number.

### The MSA question — worth 2× our throughput

The reference pipeline runs MMseqs2 against UniRef90 to build an MSA for Boltz.
**For a de novo design this returns the query and nothing else** — the
organisers' own script prints a message saying so. If that holds, then:

- we skip the MSA stage inside the evolution loop entirely,
- which removes the 28 GB RAM requirement,
- which lets us run **two `g6.xlarge` in parallel instead of one `g6.2xlarge`**,
- which doubles the number of generations we get.

**Verify, don't assume**, in the first two hours: fold one design with the MSA
and without it, compare iPDE and geometry. If they match, drop MSA from the loop
and keep MMseqs2 only for the eligibility check in §7. *Record the comparison —
it is a pitch slide either way.*

---

## 3. Shape of the pipeline, and why these three tools

```
          ┌─────────────── GENERATION 0 ────────────────┐
RFD3  ──> backbone (motif + 6NT held)
            │  Gate A: free CPU filters on the backbone
            ▼
LigandMPNN ──> N founder sequences
            │  Gate B: free CPU filters on the sequence
            ▼
Boltz-2 (+6NT) ──> structures ──> OUR SCORER ──> fitness
          └─────────────────────────────────────────────┘
                              │
                              │   elites
                              ▼
          ┌──────────── GENERATION k ───────────────────┐
          │  MUTATE  (§4.2)                             │
          │     │  Gate A + Gate B again, free          │
          │     ▼                                       │
          │  Boltz-2 (+6NT) ──> OUR SCORER ──> fitness  │
          │     │                                       │
          │  SELECT (§4.4) + DIVERSITY GUARD (§4.5)     │
          └─────────────────────────────────────────────┘
                              │
                     every design, every generation
                              ▼
                   results/leaderboard.csv  ──> pitch
```

**Gate A — free, CPU, on a backbone:** motif geometry intact (exact atom names);
burial, i.e. CA count within 12 Å of the base CA (*this is official metric 3*);
buried fraction of 6NT SASA; radius of gyration. Rejects proposals before they
cost anything.

**Gate B — free, CPU, on a sequence:** LigandMPNN score; homopolymer runs,
unpaired Cys, net charge, low-complexity; length ≤ 260; catalytic residues
present and unchanged.

Only what survives both gates is folded. That is the whole economy of the
weekend.

### 3.1 Folding: Boltz-2 — and this one is not a choice

**We fold with Boltz-2 because the judge folds with Boltz-2.** Every metric in
§1 is defined as the output of a Boltz-2 run: iPDE is a Boltz confidence field,
geometry is measured on a Boltz complex, consistency is defined across *Boltz
seeds*. Any other folding model would be a surrogate for the thing we are
actually scored on, and a surrogate introduces exactly the error our selection
loop would then amplify.

This is worth stating plainly because it inverts the usual instinct. The
question is not "which folding model is most accurate?" — it is "which model is
the scoring function?" If the organisers scored with AF3, we would use AF3 and
the accuracy argument would be irrelevant.

Secondary reasons it is also the right tool on the merits: it **co-folds the
ligand** (6NT is a standard CCD code, so the TS analogue goes in by name), it is
single-sequence-capable so it does not need an MSA for a de novo design, it fits
24 GB at 260 residues, and it emits a per-ligand confidence breakdown that gives
us iPDE directly rather than something we have to approximate. The provided
container (`/mnt/ref/containers/boltz2.sif`, weights in
`/mnt/ref/data_weights/boltz2/`) is already validated by the organisers' own
verification job.

Consequence: **nothing in this pipeline is scored by ESMFold, Chai-1 or
ColabFold.** Those wrappers exist in our repo from the target-agnostic phase and
are parked (§11). Using one as a "cheap pre-filter" sounds attractive and is a
trap — it would rank designs on a different model's opinion.

### 3.2 Scaffolding: RFD3, not RFD4

The Perlin deck lists "RFdiffusion 3/4" among scaffolding options and newer is
usually better — but **RFD4 is not on the cluster.** What is provided is RFD3:
`/mnt/ref/software/envs/env-rfd3` with
`/mnt/ref/data_weights/rfd3/rfd3_latest.ckpt`. There is no RFD4 environment and
no RFD4 checkpoint anywhere under `/mnt/ref` (checked; the only checkpoint is
`rfd3_latest.ckpt`, 2.7 GB, dated 13 Nov). The folding side is **Boltz 2.2.1**
in `env-boltz2`.

We could in principle fetch RFD4 weights ourselves — the guide does permit other
tools (*"You may use other tools within the law and explore different
pipelines"*) — but the case against it is strong for this weekend:

- The organisers' `Kemp.json` input format, the motif PDB, the fixed-atom
  selections and the whole bridge into LigandMPNN are **written for RFD3's
  interface**. Porting that is hours we do not have.
- RFD3 here is doing something quite specific and it does it well: **all-atom
  motif scaffolding with a ligand present**, holding `ASP OD1/CG/OD2`,
  `GLN OE1/CD/NE2` and all of 6NT. That is the hard requirement, and RFD3 meets
  it.
- Backbone generation is **not our bottleneck** — Boltz folding is (§2). A
  better scaffolder buys us better founders; it does not buy us more
  generations. The marginal return is small and the integration risk is large.
- Everything in `/mnt/ref` is pre-validated by the organisers' verification job.
  An unvalidated stack that fails at hour 20 costs more than it could win.

**Recommendation: RFD3, and spend the saved time on the evolution loop.** If
someone wants to argue for RFD4, the honest framing is "a second founder source,
tried in parallel, after Gate 4" — not a replacement for the path that already
works. Worth asking an organiser whether RFD4 is deliberately absent; that is a
one-minute question with a useful answer.

### 3.3 Sequence design: LigandMPNN, specifically

Yes, MPNN — but the **ligand-aware** variant, and the distinction matters more
here than almost anywhere else. ProteinMPNN designs a sequence for a backbone
and is blind to what is sitting in the pocket. For an enzyme active site that is
precisely the wrong blindness: the residues we most need to get right are the
ones contacting 6NT, and ProteinMPNN cannot see 6NT. LigandMPNN conditions on
ligand atom context, which is why
`--ligand_mpnn_use_atom_context 1 --ligand_mpnn_use_side_chain_context 1` stay
on throughout.

It also earns a second role, which is easy to miss: **LigandMPNN's per-position
posterior is the proposal distribution for our model-guided mutations** (§4.2).
It is not only a sequence generator, it is the smart-library prior for the
evolution loop.

Checkpoint choice is a real decision, not a default. Provided under
`/mnt/ref/data_weights/ligandmpnn/`:

| Checkpoint | Training noise | Use |
|---|---|---|
| `ligandmpnn_v_32_005_25` | 0.05 Å | tightest to the input backbone |
| `ligandmpnn_v_32_010_25` | 0.10 Å | **the organisers' default; our baseline** |
| `ligandmpnn_v_32_020_25` | 0.20 Å | more tolerant of imperfect backbones |
| `ligandmpnn_v_32_030_25` | 0.30 Å | most tolerant, least faithful |
| `solublempnn_v_48_*` | — | solubility-biased; useful if designs look aggregation-prone |
| `ligandmpnn_sc_v_32_002_16` | — | side-chain packing, if we ever want packed side chains for Gate A |

Lower training noise means the design stays closer to the backbone it was given
— which is what we want, because RFD3 put the catalytic motif exactly where we
need it and we do not want the sequence model quietly arguing with that
geometry. **Start at `010_25` (matches the verified example), sweep `005` and
`020` once the loop turns**, and let measured iPDE decide. Temperature likewise:
0.1 is the example's conservative setting; 0.2–0.3 widens the founder pool when
we need diversity rather than fidelity.

---

## 4. The evolutionary design module

This is the centre of the repo now, the way `src/score/` was before. It is a
generational loop with an explicit state directory, driven by Slurm array jobs,
fully resumable.

### 4.1 Start with one backbone, and measure

We do not know what Boltz-2 costs at 260 residues on an L4. Nobody does yet. So
generation 0 is deliberately one backbone and a small founder population, and
its first job is **to produce the timing number**.

From that measurement the rest follows arithmetically:

```
generations ≈ (GPU-hours × 60) / (children_per_gen × seeds_per_child × minutes_per_fold)
```

Illustrative only, pending the real number: at 4 min/fold, 16 children/gen at 1
triage seed, plus 5-seed confirmation of the top 4, is ~2.2 GPU-h per
generation → roughly 18–20 generations across the weekend on two GPUs. If the
measurement comes back at 10 min/fold, that becomes 7 generations and we
immediately cut the population, not the generation count — **depth of lineage is
what this method buys; width is what RFD3 already gave us.**

Decide population size *after* Gate 1, from the measurement. Everything in this
section is parameterised, nothing is hardcoded.

### 4.2 The mutation operator

A child is its parent's sequence with **k point mutations**. Two things are
sampled: how many, and what.

**How many — an adaptive schedule.** Mostly single mutations, with a tail, and
the tail widens when the population stops improving:

| State | k = 1 | k = 2 | k = 3–5 |
|---|---|---|---|
| Improving | 0.70 | 0.22 | 0.08 |
| Stagnant (no elite improvement for 2 generations) | 0.40 | 0.35 | 0.25 |

This is a mutator phenotype, and it is the standard answer to a search that has
settled into a local optimum. The switch is automatic and logged, never a
judgement call at 03:00.

**Where — weighted by what actually moves the score.** Positions are not equal:

- **Never mutable:** the catalytic base and the H-bond donor, and any residue
  whose atoms were held fixed by RFD3. These are hard-coded exclusions, asserted
  on every child, and a child that violates one is discarded, not repaired.
- **Heavily weighted (~60 %):** second shell — residues with any heavy atom
  within ~10 Å of 6NT. These set the pocket shape and therefore iPDE and
  geometry.
- **Moderate (~30 %):** core positions outside the pocket. These set whether
  Boltz folds the thing confidently and reproducibly, i.e. metric 4.
- **Light (~10 %):** surface. Mostly solubility and a source of neutral drift,
  which is not useless — neutral drift is how real directed evolution escapes
  local optima.

**What — three proposal distributions, mixed.** This is the "similar, sometimes
not so similar" the loop needs:

| Mode | p | What it does |
|---|---|---|
| **Conservative** | 0.55 | Sample the substitution from the parent residue's BLOSUM62 row. Chemically similar, usually tolerated, small effect. |
| **Model-guided** | 0.30 | Sample from **LigandMPNN's own per-position posterior** for that backbone. This is the enzyme-design equivalent of a smart library: the model already knows what belongs at that position given the ligand. Strictly better than a generic matrix when it is available. |
| **Radical** | 0.15 | Uniform over the other 19. Occasionally this is the one that works, and without it the search cannot cross a chemical gap. |

Mode probabilities shift toward radical under stagnation, on the same trigger as
the k-schedule.

**Optional once two lineages exist: recombination.** Crossover between two
elites from different lineages — DNA shuffling, essentially. Cheap to implement,
and it is the classic way to combine independently discovered improvements.
Worth adding only after the basic loop is turning.

### 4.3 Refolding — the loop closes through Boltz

Each surviving child is folded as a complex with 6NT. Triage at one seed and
reduced sampling steps; elites re-folded at the organisers' settings
(200 sampling steps, 3 recycling steps) across 5 genuine seeds.

**Batching is not optional.** Model load on an L4 is minutes. One Boltz process
must take many YAMLs — one generation, one process, one model load. Paying the
load per design would cost more than the designs. This is the single most
valuable piece of engineering on the list in §11.

`--diffusion_samples N` gives N poses per pass far more cheaply than N seeds.
Use it for **triage** only: the organisers measure seed-to-seed, so confirmation
must use real seeds.

### 4.4 Selection, and the noise problem

Fitness is our replica of the official score. The danger is specific and worth
stating plainly: **a single-seed score is noisy, and selecting on a noisy score
evolves the population towards lucky seeds rather than good designs.** Two
generations of that and the run is worthless.

So selection is two-stage, every generation:

1. **Triage** every child at 1 seed. Rank.
2. **Confirm** the top few at 5 seeds. Only confirmed scores may become parents.

Rank on **mean − k · spread** across seeds. Pick `k` once, today, write it in
`DECISIONS.md`, and never tune it per-design — tuning a selection constant
mid-run is how you fool yourself. The consistency metric therefore does double
duty: it is official metric 4 *and* it is our noise control.

**Hard gates survive selection.** A child whose geometry has drifted outside
tolerance, or whose burial has fallen below the generation-0 floor, is dead
regardless of its iPDE. Otherwise the loop will happily discover that it can
maximise one metric by abandoning the active site.

### 4.5 The guard against convergence

Checked **every generation**, acted on **every second or third**:

**(a) Never fold the same sequence twice.** A content-addressed cache keyed by
the exact sequence. Free, and it quietly saves a surprising amount once
mutations start reverting.

**(b) Population collapse.** Track mean pairwise Hamming distance across the
population, per-position entropy, and the number of surviving lineages. When
diversity falls below threshold:
 - widen the mutation schedule (§4.2), and
 - **inject fresh blood**: new LigandMPNN sequences on the same backbone, or a
   second RFD3 backbone promoted into the population.

**(c) Structural convergence — the one that matters most.** If children keep
folding to the same structure as their parent and only the score jitters, we are
sampling noise, not exploring. Cheap proxy: CA-RMSD of each child's Boltz model
to its parent's. If RMSD stays near zero across a generation while scores move,
that movement is seed noise and the generation taught us nothing — widen the
mutation rate or change backbone.

**(d) Eligibility drift.** Evolution moves us away from HG3.17, which helps the
identity filter — but it could in principle move us *towards* some natural
protein. Re-run the MMseqs2 identity check on the elites every few generations,
not only at submission time.

### 4.6 State on disk, and why it is Slurm-shaped

```
results/evo/<run_id>/
  config.yaml            frozen parameters for this run
  gen_000/
    population.jsonl     one row per individual: id, parent_id, mutations, sequence
    designs/             the YAMLs handed to Boltz
    boltz/               raw model output
    scores.csv           our four metrics + spread, per individual
    elites.json          who becomes a parent, and why
  gen_001/ ...
  lineage.csv            the whole family tree, flat
```

One generation is one Slurm array job. The driver is a short script that reads
the last complete generation and submits the next — **not a long-running
process**, because a long-running process dies with your SSH session and takes
the run with it. Re-running the driver is always safe: a generation that already
has `scores.csv` is a no-op.

Every individual, including the dead ones, gets a row in
`results/leaderboard.csv`. The lineage of a submitted design — "this started as
backbone 7, picked up D→E at 93 in generation 4 and W at 118 in generation 9" —
is a pitch slide that no team doing one-shot generation can produce.

---

## 5. Design levers, metric by metric

**Burial (metric 3) → use all 260 residues.** The metric counts CA atoms within
12 Å of the base; more protein means more CAs available to be close. The cap is
260 and the example's `Kemp_HG3_17_base_donor` contig is already `260-260`.
**Keep it there.** Compact, globular folds bury more than elongated bundles —
select for low radius of gyration at Gate A. Counter-pressure: the ligand still
needs a channel. We want burial of the *base*, not occlusion of the *site*.

**iPDE (metric 1) → pocket enclosure.** iPDE rewards a ligand the model is
certain about, and certainty comes from many-sided contact: a deep, well-packed
pocket, not a surface groove. Keep `ligand_mpnn_use_atom_context 1` and
`use_side_chain_context 1`. Rank by buried fraction of 6NT SASA at Gate A.

**Geometry (metric 2) → RFD3 preserves it, Boltz loses it.** RFD3 holds the
fixed atoms by construction, so RFD3 geometry carries almost no information.
What matters is whether **Boltz reproduces it**, which is why geometry is scored
after the refold, never on the design.

**Consistency (metric 4) → see §4.4.** It is both a metric and the thing that
keeps selection honest.

### The one A/B worth running: does Trp44 come back?

The organisers' 260-residue entry scaffolds **base + donor only** and drops the
Trp44 π-stack the 150-residue smoke entry keeps. A three-residue motif is harder
to scaffold, but a π-stacked ligand is a *more confidently placed* ligand —
which is metric 1, and probably helps metric 4 too.

Run both as parallel lineages once the loop is turning and let measured iPDE
decide. Good pitch slide either way.

---

## 6. Calibration — the thing that wins the pitch

Rebuilding the scoring function is necessary. **Showing it is calibrated is what
makes a judge believe the ranking.**

Spend a small, fixed GPU budget (~6–8 Boltz runs, roughly 1 % of our compute) on
controls put through our *own* harness, exactly as a design would be:

| Control | Expected |
|---|---|
| **HG3.17** (evolved, highly active) | at or near the top |
| **HG3** (the parent, ~1000× less active) | clearly below HG3.17 |
| **HG3.17 D127N** (base knocked out) | collapses on geometry |
| A scrambled protein of similar length | fails everything |

If that ordering comes out, we can say on Sunday: *"our ranking reproduces the
known activity ordering of the Hilvert series; here is the plot."* If it does
**not**, that is more valuable still — it means the metric is noisier than it
looks, we learn it at hour six rather than hour twenty-eight, and our selection
constant `k` gets set from evidence.

These are controls for *our* ranking, never submissions — they would fail
eligibility instantly.

---

## 7. Eligibility — the silent killers

A disqualified design scores zero no matter how good it is. These are automated
checks, not opinions, and they run before submission.

| Rule | How we check it | Risk |
|---|---|---|
| **≤ 260 amino acids** | assert in the submission validator | low |
| **< 70 % identity to published enzymes** | `mmseqs search` against the provided UniRef90 DB (`/mnt/ref/data_weights/mmseqs2_db/targetDB_gpu`, 183 GB, uniref90_2022_05) | low for de novo + LigandMPNN — **but check every submitted sequence, and the elites every few generations** |
| **TM-score ≤ 0.7 to published structures** | **no Foldseek, no TMalign, no US-align on the cluster** | **our biggest blind spot** |
| Catalytic residues declared correctly in each FASTA header | per `SUBMISSION.md` + the shipped validator | low, if we read the spec early |

Two actions follow:

1. **Get `SUBMISSION.md`.** The guide says it sits "alongside this guide" in
   `/mnt/ref/hackathon/docs/`. **It is not there** — that directory holds only
   `GUIDE.md` and `GUIDE.html`. Ask an organiser in the first thirty minutes. It
   defines the header format and ships an offline validator. Submitting without
   having read it is the stupidest available way to lose.
2. **Solve the TM-score blind spot.** In order of preference: ask the organisers
   how they compute it and whether they will pre-check; or `pip install` a
   TM-score implementation into a user-writable env on the head node (it has
   outbound internet) and screen against a curated set of Kemp-eliminase and
   common α/β scaffolds.

**Corollary, and it is important: do not take the easy path of mutating
HG3.17 directly.** A redesign of a published enzyme is the fastest route to
>70 % identity and TM > 0.7 — a guaranteed zero. Our evolution starts from a
*de novo RFD3 backbone*, and the only thing inherited from HG3.17 is the
three-atom catalytic motif. That distinction is worth saying out loud in the
pitch, because it is the difference between evolution and plagiarism.

### Free lab validation — a second, separate objective

> *Every team will have the chance to submit at least 5 designs for lab
> validation, independent of score.*

Those five are not ranked and should **not** be our top five. Pick for
**mechanistic diversity**: different lineages, best-of-each-arm (2-residue vs
3-residue motif), one high-burial/low-iPDE and one high-iPDE/low-burial design
to probe which metric actually predicts activity. Saying exactly that shows we
understand the metrics are proxies.

---

## 8. Portability — Slurm-shaped, and friendly to Euler

Everything above runs as plain Slurm array jobs against a conda environment.
Nothing should hardcode `g6.xlarge`, `gpu:l4:1`, or `/mnt/ref`. Instead:

```
configs/site/pdh-aws.yaml     partition, constraint, gres, env root, weights root, max_vram_gb
configs/site/euler.yaml       partition, account, gres, module loads, $SCRATCH weights root
```

Code reads the site file; `PDH_SITE` (or `--site`) picks it. Adding a third
cluster should be a YAML file, not a patch.

Three honest notes:

1. **The organisers ask that competition compute stays on the provided
   cluster** — *"We ask you to use only the provided computing resources for
   fairness reasons."* So Euler support is for code portability, CPU-side work
   and life after the hackathon. **Scored design runs stay on the AWS cluster.**
   If anyone wants to use Euler for competition compute, that is a question for
   an organiser first, not a decision we make quietly.
2. **Euler's GPUs are RTX 2080Ti, 11 GB, sm_75, no bf16.** Boltz-2 on a
   260-residue complex is very unlikely to fit. The parts of this pipeline that
   *are* genuinely Euler-friendly are the CPU-side ones: Gate A and Gate B, the
   scorer, the mutation operator, MMseqs2 searches, leaderboard and figures. Say
   that rather than discovering it in a queue.
3. Our existing `src/backends/euler.py` already knows about sbatch, arrays and
   throttling. Keep it, strip the model-specific assumptions, and let it sit
   behind the same site interface as the AWS backend.

---

## 9. How we work: three phases, split on a trigger not a clock

Until real scores exist, splitting into six workstreams means five people
blocked on the sixth. So we converge first and split later — on a trigger,
not a time.

### Phase 0 — Everyone does the same thing (now → the example verifies)

**One** person runs `setup.sh`; everyone else reads while it queues. We have two
GPUs and six people, so six parallel example runs is self-inflicted contention.

Everyone, together:
- read `inputs/Kemp.json`, `HG3_17_kemp_motif.pdb` and `src/kemp-e2e.sbatch`
  out loud;
- watch the same verification log;
- then, on CPU with no GPU contention at all, **each person independently
  computes the four official metrics by hand from the example's output** and we
  compare answers.

That last step is the real purpose of Phase 0. Six people who have each
personally computed iPDE, the two contact distances and the burial count will
not later disagree about what the scorer is doing, and one of those six
implementations becomes `src/score/official.py`.

**Phase 0 ends when:** the verification passes and everyone can read
`summary.json` and say what each number means.

### Phase 1 — Three strands, one shared target (→ first scored generation)

Still effectively one task: get generation 0 of one backbone scored. Three
strands, because they are genuinely independent and each is small:

| Strand | What | Done when |
|---|---|---|
| **Scorer** | `src/score/official.py` | reproduces the example run's own `summary.json` numbers exactly |
| **Loop** | mutation operator, selection, state dirs, diversity guard — all CPU, all testable without a GPU | `pytest` green on a toy population; no GPU used |
| **Cluster** | batched Boltz driver, array submission, the timing number, MSA-vs-no-MSA | one process folds 8 YAMLs; minutes-per-fold known |

Pair up, two per strand. The loop strand needs no GPU at all, which is the point
— it can be fully built and tested while the cluster strand is still fighting
Slurm.

**Phase 1 ends when:** one backbone has a scored generation 0 in
`results/evo/<run>/gen_000/scores.csv`.

### Phase 2 — Split (after real designs and scores exist)

Now the work genuinely diverges and parallel effort stops colliding:

| # | Owns |
|---|---|
| 1 | Integration, leaderboard, pitch, **and the submission file** — one person owns the FASTA that gets uploaded, nobody else touches it |
| 2 | The scorer, calibration controls (§6), the `k` in mean − k·spread |
| 3 | Eligibility: MMseqs2 identity, the TM-score blind spot, `SUBMISSION.md` |
| 4 | RFD3 backbones — widening the founder pool, the Trp44 arm |
| 5 | The evolution loop: mutation schedules, diversity guard, lineage analysis |
| 6 | Cluster throughput — **keeping both GPUs busy is a full-time job** |

Person 6's real deliverable is a number on a whiteboard everyone can see: GPU
utilisation. An idle L4 at hour 18 is the most expensive thing that can happen
to us.

---

## 10. Timeline and hard gates

Gates are the point; clock times will slip. A gate that fails means we change
the plan, not that we debug past it.

**Immediately (first 30 minutes)**
- [ ] `bash /mnt/ref/hackathon/setup.sh` — **nobody has run it yet**
      (`/mnt/work/kemp_example` does not exist, `/mnt/work/projects` is empty).
      GPU nodes are powered down and must spin up. Nothing else can start until
      it passes. Run it, then read while it queues.
- [ ] Ask an organiser for **`SUBMISSION.md`**.
- [ ] Phase 0 reading, together, out loud.

**GATE 1 — the example runs end to end, and we know what one Boltz fold costs.**
Until we have minutes-per-fold at 260 residues on an L4, every population and
generation number in §4 is a guess and no plan is real.

**GATE 2 — `src/score/official.py` reproduces the example's own summary.**
If our metric does not match theirs on their own run, our ranking is fiction and
the evolution loop is optimising the wrong thing. Blocks everything downstream.

**GATE 3 — MSA decision made and written down.** Doubles or halves throughput.
Decide by early afternoon, not by evening.

**GATE 4 — generation 0 scored on one backbone.** The loop is real. Only now do
we set population size and generation count from the Gate 1 measurement.

**GATE 5 — generation 1 beats generation 0 on confirmed scores.** The moment we
know directed evolution is actually working here. If it is not, we fall back to
width: more RFD3 backbones, one-shot LigandMPNN, selection without evolution.
**That fallback decision belongs to Saturday evening, not Sunday morning.**

**GATE 6 — controls plotted (§6).** Saturday night, while there is time to react.

**GATE 7 — overnight queue is full and survives us sleeping.** Both GPUs, array
jobs, the driver resubmitting generations without a human. Sleep in shifts, two
awake at a time.

**GATE 8 — HARD FREEZE ON NEW GENERATIONS, Sunday ~12:00.** From `PLAN.md`, and
it holds harder now: five seeds per elite means the confirmation backlog is
longer than a one-shot pipeline's. After the freeze: drain the queue, finalise
the leaderboard, build the submission, rehearse. A design born at 14:00 cannot
be folded five times, scored, checked and defended by 17:30.

**GATE 9 — submission built and passed through the organisers' validator, with
time to spare.** Upload early; a better file can replace it.

---

## 11. What we need to build, in order

Dependencies run top to bottom. Items 1–3 unblock everything else.

| # | Thing | Acceptance test | Strand |
|---|---|---|---|
| 1 | `src/score/official.py` — the four metrics from one Boltz output dir, exact atom names | reproduces the example's `summary.json` numbers | Scorer |
| 2 | Batched Boltz driver — many YAMLs, one process, one model load | folds 8 designs in one job; minutes-per-fold recorded | Cluster |
| 3 | `src/backends/pdh_slurm.py` + `configs/site/*.yaml` — array submission, `--constraint=g6.xlarge`, `gpu:l4:1`, hard refusal to target `l4big` | an array of 8 runs and completes; site file swappable | Cluster |
| 4 | Gate A — burial, motif geometry, ligand SASA off a backbone, CPU | < 1 s per backbone; burial agrees with the scorer's metric 3 | Loop |
| 5 | `src/evolve/mutate.py` — the three-mode operator, k-schedule, position weights, hard exclusions | unit-tested on a toy sequence; never mutates base/donor | Loop |
| 6 | `src/evolve/select.py` — triage/confirm, mean − k·spread, hard gates | unit-tested on synthetic scores with injected noise | Loop |
| 7 | `src/evolve/driver.py` — generation state dirs, resumability, sequence cache | killing it mid-generation and re-running loses nothing | Loop |
| 8 | `src/evolve/diversity.py` — Hamming, entropy, lineage count, parent-child RMSD, injection trigger | flags a deliberately collapsed population in a test | Loop |
| 9 | Eligibility checker — MMseqs2 identity, length, catalytic residues present | runs on a FASTA, exits non-zero on a violation | Eligibility |
| 10 | Submission builder — FASTA + declared catalytic residues, refuses to emit a file that would be rejected | passes the organisers' validator | Integration |
| 11 | Calibration controls (§6) as a fixed, budgeted job | four controls scored through our own harness | Scorer |
| 12 | Lineage + leaderboard reporting, figures | every number in the pitch comes from `leaderboard.csv` | Integration |

**What we stop maintaining today**, and should not debug or mention in the
pitch: `src/backends/modal.py`, `src/backends/tamarind.py`, `src/fold/esmfold.py`,
`src/fold/colabfold.py`, `src/fold/chai.py`, `src/generate/rfdiffusion.py` (v1,
superseded by RFD3), `src/score/tier3.py`, most of `MANIFEST.md`. Half of
`TODO.md` is now moot — good news; it was honest bookkeeping about work we no
longer need. `src/io/` stays and matters more than ever: **three numbering
systems are now in play** (HG3.17 reference, RFD3 output, Boltz output) with
fixed-atom mappings that must survive all three, and `verify_boltz.py` maps
`A127 → position` through a bridge file where one off-by-one poisons every
geometry number and therefore every selection decision we make.

---

## 12. The pitch, written now and filled in later

1. *"The score is four metrics. We re-implemented all four on day one and
   verified our implementation against the organisers' own reference run."*
2. *"Then we ran directed evolution against it."* — the loop diagram, the
   generation count, the fitness trajectory.
3. *"We calibrated the ranking against the Hilvert series: HG3.17, HG3, and a
   base knockout. Here is the ordering our score produces."*
4. *"We selected on consistency across seeds, not on the best seed."* — the
   spread histogram, and a design that would have topped one seed and does not
   top five.
5. *"Here is the lineage of our best design"* — backbone, generation, every
   mutation, and what each one bought. No one-shot team can show this.
6. *"Here are the five we would put in the lab, and why those and not our top
   five."*
7. *"Here is what we did not get to, and what we would not claim."*

Point 7 is not modesty, it is what makes 1–6 credible. Every number comes out of
`results/leaderboard.csv` via `src.report.pitch_assets`. Nothing typed by hand.

---

## 13. Risks, and what we do instead

| If | Then |
|---|---|
| Boltz-2 costs 10+ min per fold on an L4 | cut population, not generations — depth of lineage is what this method buys. Decide at Gate 1. |
| Evolution does not beat generation 0 by Saturday evening | fall back to width: more RFD3 backbones, one-shot LigandMPNN, selection without evolution. Gate 5 exists to force this decision early. |
| The population collapses onto one lineage | §4.5 is automatic: widen mutations, inject fresh sequences, promote a second backbone. |
| We chase seed noise | §4.4: nothing becomes a parent on a single-seed score. Ever. |
| Both GPUs idle because a human is asleep or a job failed quietly | the driver resubmits generations; person 6 owns the utilisation number. |
| `SUBMISSION.md` never materialises | ask again, loudly, early. Build to the guide's format and get an organiser to eyeball it before the deadline. |
| We cannot check TM-score at all | diversify folds deliberately, submit more than the minimum, state the limitation in the pitch rather than pretending we checked. |
| The MSA turns out to matter | we lose the second GPU and halve throughput. Survivable: fewer, more carefully chosen designs. Find out early. |
| Our calibration controls come out in the wrong order | say so. A team that discovered its metric is noisy and reported it honestly beats a team that ranked on a number it never questioned. |
| Someone argues for an exception to the Sunday freeze | the answer is in this file and it is no. |
| A metric looks suspiciously good | assume a residue-numbering bug first. Three numbering systems are in play. |

---

## 14. Open questions for the team, in priority order

1. **Population size and generation count** — cannot be answered before Gate 1.
   Recommendation: one backbone, small founder population, measure, then decide.
2. **Do we trust the no-MSA shortcut?** Recommendation: verify empirically in
   the first two hours; it is worth 2× throughput.
3. **What is `k` in mean − k·spread?** Pick one number now, write it in
   `DECISIONS.md`, never tune it again. Ideally set it from the §6 controls.
4. **Trp44 in or out?** Recommendation: second lineage once the loop turns, let
   measured iPDE decide.
5. **How much GPU on calibration controls?** Recommendation: ~1 %, fixed,
   non-negotiable, spent early.
6. **How do we spend the five free lab-validation slots?** Recommendation:
   lineage diversity and a deliberate metric-probing spread, not the top five.
7. **Who owns the submission file?** One person. Recommendation: person 1.
8. **Do we stay on the provided stack (RFD3 / LigandMPNN / Boltz-2)?**
   Recommendation: yes — §3. Folding is not negotiable at all. Scaffolding is
   negotiable in principle but the integration cost is real and backbones are
   not our bottleneck. Ask an organiser whether RFD4's absence is deliberate.
9. **Which LigandMPNN checkpoint and temperature?** Recommendation: baseline at
   `ligandmpnn_v_32_010_25` / T=0.1 to match the verified example, sweep `005`
   and `020` and T=0.2–0.3 once the loop turns.
