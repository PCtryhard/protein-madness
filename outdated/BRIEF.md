# BRIEF — Kemp eliminase, Perlin × D-BSSE, 21–22 Nov 2026, team 04

Status: **draft for team discussion.** On `main` so everyone can read it;
**not yet agreed.** Argue with it, then edit it — §14 is the agenda.

This supersedes the "unknown target" assumptions in `outdated/PLAN.md` and `README.md`.
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

### MEASURED: what the organisers' own reference design actually scores

Verification job 1 completed on AWS (`ALL PARTICIPANT VERIFICATION CHECKS
PASSED`). Its `summary.json` is the most useful document we have, because it is
a worked example of the whole scoring chain on a real 260-residue design. Three
numbers out of it reshape the plan.

**1. The geometry collapses between design and re-prediction.**

| Contact | RFD3 design | Boltz-2 re-prediction |
|---|---|---|
| `A127 OD2 → 6NT N3` (catalytic base) | **2.53 Å** | **11.18 Å** |
| `A50 NE2 → 6NT N1` (H-bond donor) | **3.09 Å** | **7.48 Å** |

RFD3 built a textbook active site — 2.5 Å base-to-ligand, motif held to within
0.013 Å. Boltz-2, re-folding from **sequence alone**, put the ligand **11 Å away
from the catalytic base**. By the very model that scores us, the organisers'
reference design is catalytically dead.

**This is the whole problem, stated in one table.** It is not a flaw in the
example; it is what the task actually is. RFD3 and LigandMPNN can place a
perfect active site, and the thing that decides our score is whether that site
*survives being re-predicted from sequence*. Closing the 2.5 Å → 11.18 Å gap is
the optimisation problem, and it is exactly what the evolution loop in §4 is for.

It also means **the bar is low and the signal is huge.** A design whose Boltz
base-to-ligand distance is 4 Å instead of 11 Å is dramatically better than the
reference, and there is an enormous dynamic range to climb.

**2. Boltz-2 costs 21 seconds, not minutes.** From the log:
`Predicting DataLoader 0: 100%|...| 1/1 [00:21<00:00]` — one 260-residue
complex, 200 sampling steps, 3 recycling steps, on one L4. **Compute is not our
bottleneck.** See §4.1, which is rewritten around this.

**3. The MSA is confirmed useless for de novo designs.**
`kemp_design: {depth: 1, nonquery_rows: 0}` against the natural control's
`depth: 301`. The §2 shortcut is not a hope, it is measured: **drop MMseqs2 from
the design loop.**

Baseline to beat, from the same run: `complex_ipde` **2.55**, `iptm` 0.749,
`complex_plddt` 0.820, 2 ligand–protein pairs below 2.2 Å (i.e. mild clashes),
backbone CA-RMSD design-to-prediction 2.16 Å. LigandMPNN reported
`overall_confidence` 0.543, `ligand_confidence` 0.510.

Also useful: in the 260-residue design the mapped catalytic positions were
**base `A127` → position 177, donor `A50` → position 99**. Those are the
positions the mutation operator must never touch (§4.2).

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

This is the thing that invalidates most of `outdated/PLAN.md`. There is no Euler
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
  several-minute spin-up — the job sits in `CONFIGURING` while the instance
  physically boots. Keep the queue warm; don't let both nodes idle down between
  generations.

**The actual node names, confirmed on the cluster** (`sinfo -N`):

| Node | Feature | What it is | Use |
|---|---|---|---|
| `gpu-dy-l4small-1` | `g6.xlarge` | 4 CPU, 16 GB, 1×L4 | **our pair — run two of these** |
| `gpu-dy-l4small-2` | `g6.xlarge` | 4 CPU, 16 GB, 1×L4 | **our pair** |
| `gpu-dy-l4-1` | `g6.2xlarge` | 8 CPU, 32 GB, 1×L4 | MSA building only; consumes the whole allocation |
| `gpu-dy-l4big-1` | `g6.4xlarge` | 16 CPU, 64 GB | **never** — above allocation, queues forever |

So the evolution loop's sbatch header is:

```bash
#SBATCH --constraint=g6.xlarge     # NOT g6.2xlarge, which halves our GPUs
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --gres=gpu:l4:1
```

The organisers' example uses `--constraint=g6.2xlarge --cpus-per-task=8`, which
takes the entire allocation for one GPU. That is correct for their one-shot
verification job and wrong for ours.

**Therefore: ~2 GPUs × ~30 usable hours ≈ 60 GPU-hours from this cluster.**
Boltz-2 is the expensive step and the only step that produces score. The entire
design of the evolution loop below is shaped by that number.

**Plus Euler** (§8), where a fold takes 34 s and uses 3.3 GB of an 11 GB card.

**Be careful how you read the Euler capacity.** The 32-GPU reservation is **not
ours** — it was created by ETH for the whole event
(`Groups=urbanb-group,bsse-wicky-euler-hackathon`) and is shared with every
other participant, which matches the "32 GPUs shared across ~70 participants"
note in `DECISIONS.md`. We are one team among many on it. Treat our fair share
as a handful of GPUs at a time, not 32, and throttle array jobs accordingly.
Taking the whole reservation because Slurm would let us is a way to make
enemies and get the reservation policed.

**Compute has therefore stopped being the constraint on anything.** The binding
constraint for the rest of the weekend is how fast we can build the pipeline
that feeds it, and how well we choose what to generate. Every hour both clusters
sit idle is the real cost now.

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

### 4.1 MEASURED: compute is not the bottleneck

Gate 1 is answered. **One 260-residue Boltz-2 fold with ligand takes 21 seconds
on an L4** at the organisers' own settings (200 sampling steps, 3 recycling
steps, 1 diffusion sample). That is roughly an order of magnitude cheaper than
this brief originally assumed.

Budget, being deliberately pessimistic — call it 45 s per fold once model load
is amortised across a batch, and 2 GPUs for 20 usable hours:

```
2 × 20 × 3600 / 45  ≈  3200 folds
```

At 5 seeds per confirmed design that is **~640 fully confirmed designs**, or far
more if most folds are single-seed triage. For comparison, the original plan
feared 18 generations of 16 children.

**So the scarce resource is no longer GPU time. It is our own judgement about
what to generate and how to select.** Three consequences:

1. **Widen the funnel.** Population per generation should be in the dozens, not
   a handful. More RFD3 backbones as founders, more LigandMPNN sequences each.
2. **Be generous with seeds.** The consistency metric (§4.4) is cheap now.
   Confirm on 5 seeds routinely rather than only for elites, and stop worrying
   about the cost of honest selection.
3. **Spend the surplus on calibration and controls** (§6), not on more designs
   we cannot defend. The bottleneck has moved to *knowing which design is good*,
   which is precisely what §6 addresses.

Do still start with one backbone for generation 0 — not to measure cost, but to
get the loop end-to-end before scaling it. Everything stays parameterised.

**Caveat worth keeping:** 21 s is the predict step alone. Model load, I/O and
Slurm scheduling are extra, which is exactly why the batched driver (§4.3) still
matters — it is now the difference between 21 s and 21 s + a minute of overhead
per design, not a question of whether we can afford to fold at all.

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

**Geometry (metric 2) → RFD3 preserves it, Boltz loses it. This is now measured,
not predicted: 2.53 Å in the design, 11.18 Å in the re-prediction** (§1). RFD3
holds the fixed atoms by construction, so RFD3 geometry carries almost no
information — the reference design's motif was held to 0.013 Å and still
collapsed. What matters is only whether **Boltz reproduces it**, which is why
geometry is scored after the refold, never on the design.

**This is the primary optimisation target and should dominate the fitness
function.** Everything else — iPDE, burial — is a supporting signal. A design
that holds the base within 4 Å in the Boltz prediction is already far beyond the
reference, and the gap from 11 Å gives us a huge dynamic range to select on.

Hypotheses worth testing early, since the loop's whole job is closing this gap:
the pocket may be too open (favour enclosure, §5 iPDE lever); the 2 sub-2.2 Å
clashes in the reference suggest strain the model relieves by ejecting the
ligand; a three-residue motif with the Trp44 π-stack may anchor the ligand far
better than two polar contacts alone — which turns the §5 A/B below from a nice
experiment into the first thing to try.

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

### The two arms, and why this one is de novo

The team is running two approaches in parallel: **starting from an existing
similar enzyme**, and **de novo**. This brief and `src/evolve/` are the **de
novo arm**. Our founder is the RFD3-scaffolded, LigandMPNN-designed 260-mer from
the verification run — built around nothing but the three-atom catalytic motif,
a homologue of nothing. `scripts/founder_from_reference.py` extracts it and
records its provenance.

**The homologue arm carries an eligibility risk the de novo arm does not**, and
it is worth saying plainly to whoever owns it rather than discovering it at
submission:

- starting from a published enzyme is the fastest route to **>70 % sequence
  identity**, which disqualifies;
- keeping its fold is the fastest route to **TM > 0.7**, which also
  disqualifies;
- and we currently **cannot measure TM-score** (no Foldseek, no TMalign, no
  US-align anywhere on the cluster — §7), so that arm would be flying blind on
  the rule most likely to zero it.

That is not an argument against running it — a homologue will almost certainly
hold its catalytic geometry better under re-prediction, which is exactly the
thing the de novo arm is failing at (§1), so it is a genuinely informative
comparison. It is an argument for that arm checking sequence identity with
MMseqs2 against the provided UniRef90 DB **early and often**, and for getting an
answer from the organisers about how they compute TM-score.

The de novo arm's own eligibility position is comfortable: LigandMPNN sequences
on an RFD3 backbone are homologues of nothing, and the only thing inherited from
HG3.17 is the geometry of the base, the donor and the ligand. That distinction
is worth saying out loud in the pitch, because it is the difference between
design and redesign.

### Free lab validation — a second, separate objective

> *Every team will have the chance to submit at least 5 designs for lab
> validation, independent of score.*

Those five are not ranked and should **not** be our top five. Pick for
**mechanistic diversity**: different lineages, best-of-each-arm (2-residue vs
3-residue motif), one high-burial/low-iPDE and one high-iPDE/low-burial design
to probe which metric actually predicts activity. Saying exactly that shows we
understand the metrics are proxies.

---

## 8. Two clusters — Euler is permitted, and it is our second GPU pool

Everything above runs as plain Slurm array jobs against a conda environment.
Nothing should hardcode `g6.xlarge`, `gpu:l4:1`, or `/mnt/ref`. Instead:

```
configs/site/pdh-aws.yaml     partition, constraint, gres, env root, weights root, max_vram_gb
configs/site/euler.yaml       partition, account, gres, module loads, $SCRATCH weights root
```

Code reads the site file; `PDH_SITE` (or `--site`) picks it. Adding a third
cluster should be a YAML file, not a patch.

**Euler is cleared for use** — confirmed with the organisers in person at the
event (Max, Saturday morning). The written guide's "use only the provided
computing resources" line is superseded by that; if anyone outside the team asks,
the answer is that we checked and were told yes.

That makes Euler a second GPU pool, not a portability exercise, and the right
question becomes **which half of the pipeline goes where.**

### 8.1 The split: generate on AWS, fold on Euler

Boltz folding is our bottleneck (§2); RFD3 backbones are not. So:

| Stage | Cluster | Why |
|---|---|---|
| RFD3 scaffolding | **AWS** | the checkpoint, the env and the validated `Kemp.json` interface are all there, and backbones are not the bottleneck |
| LigandMPNN | **AWS** (Euler optional) | small, fast, weights are public and MIT-licensed if we want a second copy |
| **Boltz-2 folding** | **Euler** + AWS | **this is the whole point** — it is the expensive step, the weights are public, and more folding directly buys more generations |
| Gates A/B, scorer, mutation, selection | either; CPU | trivial either side |
| MMseqs2 eligibility | AWS | the 183 GB UniRef90 DB is already there |

Folding is also the *easiest* stage to move: Boltz-2 is a pip install with
publicly downloadable weights, and if the no-MSA decision (§2) holds, a fold job
needs no network at run time at all — which neatly sidesteps Euler's biggest
operational trap.

**Do not copy `rfd3_latest.ckpt` off `/mnt/ref`.** Its redistribution terms are
not something we have checked, and we do not need to: RFD3 stays on AWS. Boltz-2
(MIT) and LigandMPNN (MIT) are ours to install anywhere.

### 8.2 The gate that makes this safe: identical scores, both clusters

This is the part that can quietly ruin the run, and it is not optional.

**A design folded on Euler and a design folded on AWS must produce the same
numbers, or we cannot put them in one leaderboard** — and if we cannot pool
them, the evolution loop is selecting across two incomparable score scales and
every comparison between generations is corrupted.

Different GPU architectures (L4 is sm_89; Euler's cards vary), different kernel
paths, TF32 and fp16 defaults, and `--no_kernels` on or off can all shift
confidence numbers slightly. Slightly is enough: our selection signal between
generations may be small.

So, before a single Euler-folded design enters the leaderboard:

1. Fold **the same sequence, same seed, same flags, same Boltz version (2.2.1)**
   on both clusters.
2. Compare iPDE, both contact distances, burial and pLDDT.
3. If they agree to within a tolerance we write down today — pool freely.
4. If they do not — **do not mix them.** Use Euler for triage only and keep the
   5-seed confirmation runs (§4.4) on one cluster, so that every number that
   decides a parent came from the same machine.

This is the old `README.md` "three-way smoke test" promise, except now it has
teeth and a deadline. It is **Gate 3b**, and it blocks pooling, not folding —
Euler can start producing triage folds before it passes.

### 8.3 What Euler actually is — surveyed, not guessed

Logged in as `malexandrov@euler.ethz.ch` (key auth works). Hard facts:

**Share: `es_workshop`** — 80 CPU cores, 1024 GiB RAM, **32 GPUs**. Associations
exist for `gpu`, `gpuhe`, `gpupr`, `hpc` and `bigmem`.

**There is a dedicated hackathon reservation**, and it is sitting empty:

```
ReservationName=ws_49734_gpu     PartitionName=gpu.120h   Flags=FLEX,MAGNETIC
Nodes=eu-lo-g2-[023,026,034],eu-lo-g3-043        NodeCnt=4  CoreCnt=328
StartTime=2026-11-21T08:00  EndTime=2026-11-22T20:00       State=ACTIVE
Groups=urbanb-group,bsse-wicky-euler-hackathon
```

All four nodes were **`IDLE+RESERVED` with zero CPUs allocated** when surveyed —
nobody is using them. The reservation runs to Sunday 20:00, i.e. past our pitch.
`MAGNETIC` means eligible jobs land there without asking.

Every one of those nodes is:

```
Gres=gpu:nvidia_geforce_rtx_2080_ti:8(S:0-1),gpumem:no_consume:11G
```

4 nodes × 8 GPUs = the 32 GPUs in the share, **all of them 11 GB RTX 2080Ti,
sm_75** — exactly the hardware `DECISIONS.md` warned about. So we probed it
rather than argued about it.

**MEASURED (job 21802893): Boltz-2 fits, easily.**

| | AWS | Euler reservation |
|---|---|---|
| GPUs | 2 | **32** |
| VRAM each | 24 GB | 11 GB |
| Peak VRAM used, 260 res + 6NT | — | **3354 MiB of 11264** |
| Seconds per fold | 21 | **34** |
| Availability | ours | SHARED event reservation, to Sun 20:00 |

A 260-residue complex uses **under a third of a 2080Ti**. The feared bf16
problem did not materialise: `torch.cuda.is_bf16_supported()` returned **True**
on sm_75, and the run completed with exit code 0 at only 1.6× the L4's time.
Recorded plainly because this brief predicted the opposite.

So the 2080Ti's **can** fold, which was the open question.

**But the reservation is shared, not ours** — see §2. `Users=(null)`,
`Groups=urbanb-group,bsse-wicky-euler-hackathon`: ETH created it for the event,
and the nodes already show other people's jobs on them. Our fair share is a few
GPUs at a time. The useful conclusion is not "we have 32 GPUs", it is **"Euler
can fold, so we are not bottlenecked on the 2 AWS cards"**.

Two follow-ups worth a few minutes each: whether two or three Boltz processes
can share one card (3.3 GB fits three times into 11 GB), and the §8.2
cross-cluster agreement check before pooling Euler and AWS numbers in one
leaderboard.

Setup that now exists on Euler, ready to use:
`$SCRATCH/pdh/venv` (Boltz **2.2.1**, pinned to match AWS; torch 2.15.0+cu132),
`$SCRATCH/pdh/cache` (7.6 GB of weights, staged from the login node).

Other facts settled while surveying:

- **`outdated/TODO.md` B6 is resolved**: the account is `es_workshop`.
- **`outdated/TODO.md` B4 is resolved**: the GRES suffix is an upper/lower-case `G` —
  the nodes advertise `gpumem:no_consume:11G`, so `--gres=gpumem:11g` parses.
  To reach bigger cards *outside* the reservation, raise that number.
- **The login node reaches PyPI** (HTTP 200), so `pip install boltz` and weight
  staging work there. Compute nodes still have no outbound internet — stage
  weights on the login node, never inside a job, or a retrying client will look
  exactly like a hang until walltime kills it.
- `cuda/13.0.2` and `eth_proxy` modules are present.
- **`$SCRATCH` is purged after ~2 weeks and is not backed up** (482 T, 73 %
  full). Fine for a weekend; results come home to git or `/mnt/work`, never left
  on Euler scratch.

### 8.4 If 11 GB is not enough

Two fallbacks, in order:

1. **Request bigger cards outside the reservation.** `es_workshop` has `gpu`,
   `gpuhe` and `gpupr` associations, and Euler as a whole has RTX 4090 and 3090
   (24 GB), A100 40/80 GB, and RTX PRO 6000 (96 GB) nodes. Ask with
   `--gres=gpumem:24g`. Cost: we leave the reservation and join the general
   queue, so throughput becomes unpredictable — acceptable for overnight batches,
   not for the interactive loop.
2. **Euler does CPU-side work only** — Gates A and B, the scorer, the mutation
   operator, analysis — and all folding stays on AWS. §8.1 as originally written.

`src/backends/euler.py` already knows sbatch, arrays and throttling — keep it,
strip the model-specific assumptions, and run it behind the same site interface
as the AWS backend.

### 8.5 Site profiles

```
configs/site/pdh-aws.yaml     partition gpu, constraint g6.xlarge, gres gpu:l4:1, /mnt/ref roots
configs/site/euler.yaml       partition gpu.120h, account es_workshop, gres gpu:1 + gpumem:11g, $SCRATCH weights
```

Code reads the site file; `PDH_SITE` (or `--site`) picks it. Nothing hardcodes
`g6.xlarge`, `gpu:l4:1` or `/mnt/ref`. Adding a third cluster is a YAML file,
not a patch. **Every leaderboard row records which site folded it** — without
that column, §8.2 is unenforceable after the fact.

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

**GATE 1 — PASSED.** Verification job 1 completed, all checks passed, and one
260-residue Boltz-2 fold costs **21 s** on an L4. Compute is not the bottleneck;
§4.1 is rewritten around this. The run also handed us the baseline to beat and
the geometry-collapse finding in §1.

**GATE 2 — PASSED.** `src/score/official.py` reproduces the organisers'
verification run exactly — iPDE 2.5525107383728027, base contact
11.178812026977539 Å, donor contact 7.478554725646973 Å, min ligand distance
2.063716411590576 Å, 2 clashes: every value identical to their `summary.json`.
Their real 260-residue output is committed as the test fixture, and every
expected value is read from their file rather than typed in, so the test asserts
agreement with *them* and cannot drift into agreeing with ourselves. 91 tests
pass. **We now measure with the judges' ruler.**

Burial, which their summary does not expose, comes out at **30 CA atoms within
12 Å** for the reference design. That is the metric-3 baseline to beat.

**GATE 3 — PASSED.** The MSA question is settled empirically: the de novo design
got `depth: 1, nonquery_rows: 0` while the natural control got 301. **Drop
MMseqs2 from the design loop**, keep it for the eligibility check (§7).

**GATE 3b — PASSED, and not marginally.** Boltz-2 folded a 260-residue complex
with 6NT on an 11 GB RTX 2080Ti using a **peak of 3354 MiB — under a third of
the card** — in **34 s** (vs 21 s on the L4). `torch.cuda.is_bf16_supported()`
returned **True** on sm_75, so the bf16 worry in this brief's earlier draft was
wrong; recorded here because a wrong prediction we acted on is worth as much as
a right one.

Consequence: **all 32 reserved GPUs are usable.** Total capacity is now absurd
relative to need — roughly one fold per second in aggregate across the
reservation. Worth testing, since 3.3 GB fits three times into 11 GB: whether
two or three Boltz processes can share one card. Even without that, compute has
stopped being a constraint on anything.

**GATE 3c — do both clusters agree?** Same sequence, same seed, same flags, both
clusters (§8.2). Until this passes, Euler folds may be used for triage but must
not be pooled into the leaderboard.

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

**GATE 8 — HARD FREEZE ON NEW GENERATIONS, Sunday ~12:00.** From `outdated/PLAN.md`, and
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
| 3b | Euler: account + GPU survey, Boltz-2 installed, weights staged on the login node, `configs/site/euler.yaml` | one Boltz-2 fold completes on Euler; cross-cluster comparison run (§8.2) recorded | Cluster |
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
superseded by RFD3), `src/score/tier3.py`, most of `outdated/MANIFEST.md`. Half of
`outdated/TODO.md` is now moot — good news; it was honest bookkeeping about work we no
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
| Euler and AWS disagree on the same sequence | do not pool. Euler does triage only; every score that decides a parent comes from one cluster. §8.2. |
| ~~Euler turns out to be 11 GB cards only~~ | **Resolved: it folds in 3.3 GB.** All 32 GPUs usable. |
| Compute sits idle because nothing is built to feed it | **this is now the real risk.** 34 GPUs are available and the pipeline to use them does not exist yet. The scarce resource is build time, not GPU time. |
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
8. **Who owns the Euler bring-up, and what is its timebox?** Recommendation:
   one person, answers §8.3's first two questions before lunch, and if Boltz-2
   has not folded on Euler by Saturday evening we drop it. It is upside, not
   budget — nothing in §4 may depend on it.
9. **Do we stay on the provided stack (RFD3 / LigandMPNN / Boltz-2)?**
   Recommendation: yes — §3. Folding is not negotiable at all. Scaffolding is
   negotiable in principle but the integration cost is real and backbones are
   not our bottleneck. Ask an organiser whether RFD4's absence is deliberate.
10. **Which LigandMPNN checkpoint and temperature?** Recommendation: baseline at
   `ligandmpnn_v_32_010_25` / T=0.1 to match the verified example, sweep `005`
   and `020` and T=0.2–0.3 once the loop turns.
