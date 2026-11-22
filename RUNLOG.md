# RUNLOG — what we actually ran, and what it told us

Saturday 21 November 2026. Append-only, newest at the bottom. `BRIEF.md` is the
plan; this is the evidence. Every number here came off a cluster, not a guess.

**Cluster shorthand:** *AWS* = the provided 2-GPU cluster (`13.38.53.98`), where
RFD3 and LigandMPNN live. *Euler* = ETH, where Boltz-2 folding runs on the
shared hackathon reservation `ws_49734_gpu`.

---

## 1. The organisers' example, end to end — AWS job 1 ✅

`bash /mnt/ref/hackathon/setup.sh`. All participant checks passed.

**The single most important measurement of the day** came out of its
`summary.json`:

| Contact | In the RFD3 design | After Boltz re-prediction |
|---|---|---|
| base `ASP OD2 → 6NT N3` | **2.53 Å** | **11.18 Å** |
| donor `GLN NE2 → 6NT N1` | 3.09 Å | 7.48 Å |

RFD3 built a textbook active site and held the motif to **0.013 Å**. Boltz,
re-folding from sequence alone, put the ligand 11 Å from the catalytic base.
**We are scored on the right-hand column.** Closing that gap is the whole task.

Baseline to beat: `complex_ipde` **2.55**, burial **30**, 2 clashes under 2.2 Å.

Two more facts worth the run on their own:

* **A fold costs 21 s** on an L4, not the 4–10 min assumed. Compute stopped
  being the constraint.
* **The MSA is useless to us**: a de novo design got an alignment of depth 1
  with zero non-query rows, against 301 for a natural control. MMseqs2 left the
  design loop.

## 2. Our scorer vs theirs — GATE 2 ✅

`src/score/official.py` reproduces their verification numbers **exactly**:
iPDE `2.5525107383728027`, base contact `11.178812026977539`, donor
`7.478554725646973`, min distance `2.063716411590576`, 2 clashes. Their real
output is committed as a test fixture, and every expected value is read from
*their* file, so the test cannot drift into agreeing with us.

Burial, which their summary never exposes, is **30** for the reference design.

## 3. Can Euler fold? — Euler job 21802893 ✅

Predicted: no — Boltz uses bf16 AMP and the 2080Ti (sm_75) has none.
**Measured: yes, easily.** Peak **3354 MiB of 11264**, 34 s per fold, exit 0,
and `torch.cuda.is_bf16_supported()` returned `True` on sm_75.

The prediction in this repo was wrong and is recorded as wrong.

## 4. The loop, end to end — Euler job 21806470 ✅

4 designs × 2 seeds from the inherited founder. 8 folds, 12½ min, all scored,
parents selected, generation 1 bred.

| design | base→ligand | iPDE | burial | |
|---|---|---|---|---|
| founder | 12.35 Å ±1.14 | 2.17 | 29.5 | failed gate |
| V100A | **5.91 Å ±0.11** | 2.50 | 33.5 | best |
| V232I | 5.29 Å ±1.32 | 2.33 | 34.5 | |
| S172N+H237T | 7.16 Å ±0.85 | 2.77 | 31.5 | |

All three mutants more than halved the catalytic distance. The believable one is
**V100A**, whose two seeds agreed to ±0.11 Å; V232I scored better on the mean
with a ±1.32 Å spread, which is exactly what the five-seed rule exists to catch.

Three things only a real run could find: the hard gate rejected the founder
(12.35 vs a 12.0 Å limit); the diversity guard fired correctly and widened
mutations by itself; `sbatch` rejected our script for CRLF line endings.

## 5. Fresh de novo backbones — AWS job 3 ✅

8 backbones, ~45 s each, two arms: `bd` (base + donor) and `bdw` (+ Trp44
π-stack). Gate A scored them on CPU in seconds:

```
backbone            res  base-lig  donor-lig  burial   minD  rmsdB  rmsdD
Kemp_denovo_bd_0    260      2.53       3.09      40   1.97   0.52   3.60
Kemp_denovo_bd_1    260      2.53       3.09      46   2.50   0.33   1.18
Kemp_denovo_bd_2    260      2.53       3.09      59   2.30   3.39   1.47
Kemp_denovo_bd_3    260      2.53       3.09      45   2.53   0.20   0.65
Kemp_denovo_bdw_0   260      2.53       3.10      43   1.63   0.22   2.48
Kemp_denovo_bdw_1   260      2.53       3.10      48   2.11   0.20   0.51
Kemp_denovo_bdw_2   260      2.53       3.10      56   1.85   1.62   1.19
Kemp_denovo_bdw_3   260      2.53       3.10      53   2.53   0.54   0.95
```

**Burial 40–59 against the reference design's 30** — every new backbone beats it
on metric 3 before anything else is done. The geometry columns are identical
everywhere, because RFD3 pins those atoms: **backbone geometry carries no
information about design quality.** The discriminator is `rmsdB`/`rmsdD`, RFD3's
own motif-fidelity.

## 6. TM-score, finally measured ⚠️

113 structures pulled from RCSB by full-text search (*Kemp eliminase*, *Kemp
elimination*, *benzisoxazole*, *computational enzyme design*) — including `4BS0`
and `6SS1` (HG3.17 with 6NT) and `2RKX` (KE07). TM-aligned against all 8
backbones:

```
Kemp_denovo_bd_0   0.499  3E0L   ok      Kemp_denovo_bdw_0  0.528  9PM1  ok
Kemp_denovo_bd_1   0.587  3TP4   ok      Kemp_denovo_bdw_1  0.602  6SS1  ok
Kemp_denovo_bd_2   0.703  3I1C   FAIL    Kemp_denovo_bdw_2  0.705  3I1C  FAIL
Kemp_denovo_bd_3   0.379  9R7K   ok      Kemp_denovo_bdw_3  0.369  3TP4  ok
```

**Two of eight are ineligible**, against `3I1C`, an engineered Diels-Alderase.
RFD3 trained on the PDB; designed enzymes are in the PDB; it rediscovered one.

**And they are the two with the best burial (59 and 56).** Optimising burial
walks toward disqualification, so TM-score is a **hard gate** in `HardGates`,
not a ranking term to trade against burial.

> **Lower bound, not a clearance.** 113 Kemp-related entries is not the PDB.
> `bd_1` at 0.587 and `bdw_1` at 0.602 could cross 0.7 against something we did
> not search. The full check needs the Foldseek PDB database, which is
> rate-limited at its host (HTTP 429, Cloudflare 1027) — three attempts failed.

## 7. Sequence design — AWS job 4 ✅

LigandMPNN on all 8 backbones, 4 sequences each at T=0.1 and T=0.2, atom and
side-chain context on. 64 sequences, all 260 residues, catalytic residues
intact. The 2 ineligible backbones were dropped automatically, leaving
**48 designs from 6 backbones, 24 per arm**.

Sequence eligibility: the earlier proof-of-concept designs returned **0.000
identity** against UniRef90 — zero hits. De novo + LigandMPNN is nowhere near
the 70 % filter. *(That search must run on a GPU node: the provided UniRef90
index is GPU-prefilter-only and a CPU search fails outright.)*

## 8. First real fold batch — Euler jobs 21817015 / 21817016 ⏳

48 designs, 1 triage seed, split across 2 GPUs. **Running.**

## 9. Third arm: extra ligand contacts — AWS job 5 ⏳

The measured failure is ligand drift, and the supplied motif contacts the
ligand at only two points. 6NT's nitro oxygens (`O11`, `O21`) are strong H-bond
acceptors touched by nothing. So a third arm adds RFD3 conditioning —
`select_hbond_acceptor` on those oxygens, `select_buried` on the catalytic base
— while leaving the two scored contacts pinned exactly as before.

Deliberately additive: it cannot cost us anything on the geometry term. The
semantics of these conditioning fields are read off the installed
`DesignInputSpecification` and are **not verified by us**, which is why it runs
as a parallel arm and is judged on measured iPDE. **Running.**

---

## 10. Scoring audit against mentor guidance

A mentor's framing: co-fold the ligand, weight **distances** above **angles**
but use both, minimise the **amount of change**, and score **stability**.
Audited honestly, we had about half of it.

| Asked for | Before | Now |
|---|---|---|
| Co-fold the ligand | ✅ 6NT present in every fold, never apo | unchanged |
| Distances | ✅ both scored contacts | unchanged, still dominant |
| Angles | ❌ **not measured at all** | `base_angle_deg`, `donor_angle_deg` |
| Amount of change | ❌ not scored | `design_rmsd_a`: designed structure vs its re-prediction |
| Stability | ⚠️ `complex_plddt` carried but unused in ranking | a fitness term with a floor at 0.70 |
| Consistency | ✅ mean − k·spread | now also spans angles, stability and change |

The angle is `root-contact···ligand` — `CG-OD2···N3` for the base,
`CD-NE2···N1` for the donor. It answers a question distance cannot: a
carboxylate 3 Å from the ligand but *pointing away* is not a catalytic base.

Weighted as instructed: **distance 1.0 per Å, angle 0.15 per 10° off ideal**,
so a contact at the right distance and the wrong orientation loses, but angle
only ever breaks ties between designs that already sit sensibly. The 120° ideal
is an approximation, which is part of why the weight is low.

`design_rmsd_a` is the organisers' own `backbone_ca_rmsd_from_rfd3`
(2.16 Å on the reference). Low change means Boltz agrees with what we designed
— precisely the failure in §1.

## 11. Fold batch 1, first attempt — FAILED ❌

Euler jobs 21817015/16 reported `COMPLETED`, exit 0, and produced **nothing**:

```
RuntimeError: Found directory .../chunk_a/boltz_seed1 instead of .fasta or .yaml
```

Self-inflicted: the output directory was placed *inside* the input directory,
and Boltz scans its input for YAMLs only. Slurm still called it a success —
a reminder that job state is not evidence. `write_fold_script` now refuses this
arrangement, with a regression test. Cost ~5 min on two GPUs; no designs lost.

---

## 12. Calibration: does a real Kemp eliminase score well on our metrics? ✅

The check the whole ranking rests on. Scored **PDB 4BS0** — HG3.17 with 6NT
bound at 1.09 Å, about as good as a Kemp eliminase gets — with our own metrics,
directly on the crystal structure, no folding involved:

| | HG3.17 (4BS0) | 6SS1 (Q50A variant) | Organisers' de novo design |
|---|---|---|---|
| base `OD2→N3` | **2.53 Å** | 2.61 Å | **11.18 Å** |
| donor `NE2→N1` | **3.10 Å** | — (donor mutated away) | 7.48 Å |
| base angle `CG-OD2···N3` | **130.9°** | 129.2° | — |
| donor angle `CD-NE2···N1` | **133.6°** | — | — |
| burial | **47** | 48 | **30** |
| clashes < 2.2 Å | **0** | 0 | **2** |

Our metrics recognise the real enzyme on every axis. That is the sanity check
passing.

**It also corrected two numbers I had guessed:**

* `ideal_angle_deg` was **120°**, reasoned from sp2 geometry. The real enzyme
  sits at **130.9°/133.6°** — we would have penalised HG3.17 by 11° for being
  correct. Now **131°**, measured.
* `burial_target` was **35**, set as "a bit above the reference design's 30".
  The real enzyme buries its base with **47**. Aiming at a design we already
  know fails was the wrong anchor. Now **47**.

A third control arrives free: **6SS1 is the Q50A donor knockout**. Its donor
contact is unmeasurable because NE2 does not exist, which exercises the
NaN-not-zero rule on a real structure rather than a synthetic one.

Note HG3.17 is **299 residues**, over the competition's 260 limit — fine as a
control, never submittable.

Five tests now pin this: a real Kemp eliminase must beat the reference design,
must pay zero geometry penalty, and must pass our own hard gates. If a future
change breaks any of them, the change is wrong.

---

## 13. First real scores on our own de novo designs ✅ — Euler 21817976/81

48 designs folded (10.7 min, 2 GPUs), scored at 1 triage seed. **36 of 48 pass
the hard gates.** Top of the table:

```
design                      arm    base  donor    ang   iPDE  bur  pLDDT   cost
Kemp_denovo_bd_1_T0.1_1     bd     2.36   3.48  135.2   0.80   53  0.959   0.65
Kemp_denovo_bd_1_T0.2_1     bd     2.39   3.80  125.0   1.37   50  0.918   1.00
Kemp_denovo_bdw_1_T0.1_3    bdw    3.46   4.01  165.4   1.35   48  0.925   1.07
Kemp_denovo_bd_1_T0.2_3     bd     3.14   4.08  151.7   1.21   49  0.875   1.09
Kemp_denovo_bd_1_T0.1_2     bd     2.84   4.16  133.4   1.52   47  0.832   1.26
```

**The best design matches or beats the real evolved enzyme on every axis:**

| | our best | HG3.17 crystal | organisers' reference |
|---|---|---|---|
| base `OD2→N3` | **2.36 Å** | 2.53 Å | 11.18 Å |
| base angle | 135.2° | 130.9° | — |
| burial | **53** | 47 | 30 |
| iPDE | **0.80** | — | 2.55 |
| pLDDT | 0.959 | — | 0.820 |

That is a 45-second de novo backbone, re-predicted from sequence alone,
holding its catalytic contact tighter than a laboratory-evolved enzyme.

### The backbone dominates, not the sequence

`bd_1` produced **7 of the top 12** designs. `bdw_3` produced the **8 worst**,
every one between 23 and 27 Å. Sequence design and temperature barely move a
design compared with which backbone it came from.

### Gate A did NOT predict which backbones would work ⚠️

This matters and it is uncomfortable:

| backbone | Gate A burial | Gate A rmsdB | median base contact after folding |
|---|---|---|---|
| `bd_1` | 46 | 0.33 | **2.84 Å** — best |
| `bdw_3` | **53** (2nd best) | 0.54 | **26.4 Å** — worst |

`bdw_3` looked *good* on the free metrics and produced garbage. The free
backbone metrics measure the structure **we designed**; what decides the score
is the structure **Boltz rebuilds**, and §1 already showed those can differ by
9 Å. Gate A is still worth running — it is free, and it caught the TM failures
— but **it cannot be trusted to rank backbones.** Fold a couple of sequences
per backbone and rank on that instead.

### Trp44 arm: no evidence of benefit yet

`bd` median 4.73 Å, `bdw` median 10.89 Å. But that gap is driven almost
entirely by `bdw_3` being catastrophic, and `bdw_1` sits third overall. With
two usable backbones per arm, backbone quality swamps any arm effect. **Not
decided.**

---

## 14. Scoring bug: the contact penalty was one-sided ⚠️ fixed

Raised by Max: *is the distance metric an optimal middle, or closer-is-better?*

It was **closer-is-better, and that was wrong.** `contact_penalty` charged
nothing below 3.5 Å, so a **1.5 Å contact scored as perfect** — a distance at
which the atoms are simply overlapping.

Why a band is right:

* The scored contacts are **hydrogen bonds** — ASP `OD2` to the acidic `N3-H`
  of the benzotriazole, GLN `NE2` to `N1`. An O···N hydrogen bond stops
  improving around 2.5 Å; below that it is van der Waals repulsion.
* **HG3.17 sits at 2.53 Å**, not at 1.5 Å. Directed evolution had every
  opportunity to go shorter and did not.
* The organisers describe their own term as comparing *"against the ideal
  approach geometry"* — a deviation from a target, which is two-sided. Our
  one-sided version was probably wrong about their metric, not just the physics.

Now a window of **[2.4, 3.5] Å**: free inside, distance-to-edge outside, and
**too-close costs double** because a clash destabilises the complex while a long
contact is merely weak.

**Re-ranked batch 1: the order is unchanged**, top design 0.65 → 0.73 (its
2.36 Å now pays a small floor penalty). So this changes none of today's
conclusions — but it closes a hole that the evolution loop would have found and
exploited, because searching for extreme values is exactly what it does.

### On "are we just rebuilding HG3.17?"

Partly, and deliberately — but the part that counts is novel.

| | inherited from HG3.17 | ours |
|---|---|---|
| catalytic constellation | 3 atoms of ASP, 3 of GLN, the 6NT pose (4BS0) | — |
| sequence | nothing — **0.000 identity** to UniRef90 | all 260 residues |
| fold | nothing — best design's closest PDB match is 3TP4 at TM 0.587 | the whole scaffold |

The challenge supplies that motif; the task is to build a *novel protein* that
holds it. The identity and TM filters exist to enforce exactly that split.

**One circularity we introduced ourselves, and should keep visible:** §12 set
`ideal_angle_deg` to 131° *measured from HG3.17*. That makes our score reward
"geometrically like HG3.17" by construction, so it **cannot recognise a better
geometry that differs from HG3.17's**. It is still the best-grounded number
available — the only experimentally validated high-activity Kemp geometry we
have — but it is an assumption, not a discovery.

---

## 15. All 13 variants, 20 backbones each — Euler 21893949…21894359 ✅

The first batch wide enough to rank variants rather than guess. Every variant
we have, run from one deployed tree at `origin/main` 852df71, submitted through
`cluster/euler/run.sbatch`.

**Config** (`~/pm-runconf/overnight.yaml`): 20 backbones × 3 sequences at
cycle 0, `max_cycles: 1`, `recycling_steps: 5`. **950 designs, 63 scored
(6.6%).** All 13 jobs `COMPLETED`, exit 0.

| rank | variant | best | scored/total | best came from |
|---|---|---|---|---|
| 1 | **10** | **0.8988** | 3/76 | `mutate` |
| 2 | 01 | 0.8296 | 10/76 | `mutate` |
| 3 | 09 | 0.8065 | 8/76 | `seed` |
| 4 | 03 | 0.8059 | 5/76 | `wiggle` |
| 5 | 06 | 0.7286 | 9/76 | `seed` |
| 6 | 07 | 0.7249 | 4/76 | `seed` |
| 7 | 11 | 0.7026 | 8/76 | `seed` |
| 8 | 08 | 0.6655 | 4/76 | `seed` |
| 9 | 05 | 0.6061 | 5/76 | `seed` |
| 10 | 13 | 0.5333 | 6/76 | `wiggle` |
| 11 | 02 | 0.2254 | 1/70 | `seed` |
| — | 04 | — | 0/60 | nothing cleared the gate |
| — | 12 | — | 0/60 | nothing cleared the gate |

Variant 10's winner is strong on every term, not carried by one:
confidence 0.912, catalytic 0.918, non-catalytic 0.981, angle 0.768.

### The binding constraint is the conjunction, not either bond

Each required bond lands on its own 10–37% of the time. **Both at once, ~6%.**
Variant 12 is the clean illustration: 14/60 designs placed the catalytic bond
inside the gate, 6/60 placed the ASN291 bond, and **0 placed both** — against
1.4 expected if the two were independent.

So the per-bond numbers in earlier sections overstate how close we are. Adding
backbones buys hits linearly; the conjunction is what has to be attacked.

### `max_cycles: 1` starves the search exactly when it is needed ⚠️

`wiggle` and `mutate` only spawn from a parent that cleared the gate. A variant
that misses on all of cycle 0 therefore gets **no search at all** — 04 and 12
produced 60 designs where every other variant produced 76, and their search
branches never ran.

The pipeline silently degenerates to random sampling in precisely the case
where searching would help most. Four of the eleven scoring variants found
their best design in a `wiggle` or `mutate` child, so the branches are
earning their keep where they run. **Raise `max_cycles` next batch.**

### Recycles 3 → 5 made no measurable difference

Three variants have before/after runs at the same benchmark. Pooled cycle-0
double-hit rate:

| recycles | double hits | designs | rate |
|---|---|---|---|
| 3 | 6 | 108 | 5.6% |
| 5 | 10 | 178 | 5.6% |

Identical. Per-bond rates are flat to better (09's catalytic bond 19% → 37%).
The three variants moved in *different directions*, which is the shape of
noise, not of an effect. **Recycles are not the lever.**

### Variant 12's earlier crown was a small-sample artefact ⚠️

12 led at 0.8653 on a 52-design run and scored **zero** on 60 designs here.
That earlier lead rested on 5 lucky cycle-0 hits out of 36 (1.8 expected). Any
ranking built on ~36 designs per variant should be treated as provisional —
including the top-3 selection that drove the earlier parallel batch.

### The distance metric now contradicts §14's conclusion ⚠️ decided deliberately

§14 concluded that **too-close should cost double**, because a clash
destabilises the complex while a long contact is merely weak. The metric
shipped for this batch does the opposite: full marks at 2.9 Å, falling to zero
at 0.5 Å below and 4.5 Å above, so a short deviation costs **0.67×** what the
same long deviation costs, and only sub-0.5 Å overlap gates.

This was a deliberate team decision, not a regression, and it is recorded here
because it reverses a documented finding. The §14 reasoning (HG3.17 sits at
2.53 Å; directed evolution could have gone shorter and did not) has not been
rebutted — it was overridden. If short contacts start winning the leaderboard,
this is the first thing to re-examine.

### Hardware, so wall-clock is not misread

Split across RTX 3090 (`eu-g4-*`, 1:06–1:25) and RTX 4090 (`eu-g6-*`,
0:45–0:55), with two jobs per node on some 3090s. **Runtime differences
between variants here are the card and node contention, not the biology.**
Design counts and scores are comparable; elapsed time is not.

### What is committed

Leaderboards, reports, sequences and a config snapshot (`pipeline.yaml`,
`target.yaml`, `motif.pdb`) for all 13 runs, plus the full
rfd3/ligandmpnn/boltz artefacts for the **63 designs that cleared the gate**
— 888 files, 15 MB. The ~230 MB of gated designs stays on Euler under
`~/pm-live/runs/overnight-NN/`; it is regenerable and was deliberately kept
out of git history.

---

## 16. Three search cycles instead of one — Euler 21899142…21899152 ✅

The §15 batch found that `max_cycles: 1` starves the search: `wiggle` and
`mutate` only spawn from a parent that cleared the gate, so a variant whose
cycle 0 produced nothing got no search at all and degenerated into plain
random sampling. This batch tests the fix.

Two parameters moved from the §15 config, and only two:

| | §15 overnight | §16 depth |
|---|---|---|
| `run.max_cycles` | 1 | **3** |
| `boltz.recycling_steps` | 5 | **3** |

Recycles came down to pay for the extra cycles. §15 had already measured them
as neutral, and this batch re-measures that properly (see below). Everything
else — 20 backbones, 3 sequences each, `keep_per_cycle: 2`, `partial_t: 2.0` —
is unchanged. Variants 04 and 12 were dropped: neither scored a single design
in §15, so they had nothing to search from.

11 jobs, all `COMPLETED`, longest 1:48 against a 2:05 wall. 1,122 designs,
142 scored (12.7%), against 830 designs and 63 scored (7.6%) for the same 11
variants in §15.

### Cycles 2 and 3 do run

This was the untested part: at cycle 2 the `wiggle` branch uses a *wiggle
child* as its parent, a code path no previous run had exercised. It works.
Variant 06's cycle 2 wiggled from `c01.mutate.995a1f` — a search child used as
a parent — and produced scoring designs.

Three variants (03, 06, 07) stopped after cycle 2 rather than running all
three. That is the early-stop rule firing correctly: `no improvement above
min_delta=0.005 for 2 consecutive cycles`. Those three are also exactly the
three whose best design came from cycle 0, so the rule stopped runs that had
genuinely stalled. Nothing was truncated by the wall clock.

### The branch rates are the finding

Pooled over all 1,122 designs, by how the design was generated:

| branch | designs | cleared gate | rate |
|---|---|---|---|
| `seed` (fresh RFD3) | 660 | 41 | **6.2%** |
| `wiggle` (partial re-diffusion) | 342 | 50 | **14.6%** |
| `mutate` (new sequence, same backbone) | 120 | 51 | **42.5%** |

Search children clear the gate at 21.9% against 6.2% for random seeds —
Fisher p ≈ 1.5 × 10⁻¹⁴. `mutate` beats `wiggle` at p ≈ 1.3 × 10⁻⁹. The
ordering replicates: in §15 the same three branches ran 5.5% / 12.7% / 25.0%,
with `mutate` on only 44 designs. Pooling the two batches, `mutate`'s
advantage is not a small-sample artefact.

By cycle, the enrichment appears immediately and does not decay:

| cycle | designs | scored | rate |
|---|---|---|---|
| 0 | 660 | 41 | 6.2% |
| 1 | 158 | 35 | 22.2% |
| 2 | 176 | 33 | 18.8% |
| 3 | 128 | 33 | 25.8% |

Six of the 11 variants found their best design in cycle 2 or 3 — designs that
could not exist under `max_cycles: 1`.

### `mutate` is starved, and that is the next lever

Observed in every cycle of every variant: `mutate` produced **exactly 4
designs per cycle** whether the frontier held one parent or two, while
`wiggle` scaled 6 → 12 with the parent count. So `mutate` takes 4 of a full
cycle's 16 designs (25%) and returns 51 of the batch's 142 hits (36%) from 11%
of the designs. It is also the cheapest branch — no re-diffusion, just
LigandMPNN and Boltz on an existing backbone.

Raising `mutate`'s share is the highest-confidence change available to the
next run, and it shifts budget from the 6.2% branch to the 42.5% one.

### Recycles 5 → 3: now a controlled measurement

Cycle 0 is identical in shape between §15 and §16 — 20 backbones × 3
sequences, same 11 variants, 660 designs on each side — and `recycling_steps`
is the only parameter that differs. That makes this a clean A/B, far larger
than the §15 attempt:

| `recycling_steps` | cycle-0 designs | scored | rate |
|---|---|---|---|
| 5 (§15) | 660 | 36 | 5.5% |
| 3 (§16) | 660 | 41 | 6.2% |

Fisher p = 0.64. Recycles are neutral for gate-clearing, and dropping them
bought roughly 30% more wall clock for free. **Do not spend time tuning
`recycling_steps`.**

### Depth raised the floor, not the ceiling

| variant | §15 best | §16 best | Δ | scored §15 → §16 |
|---|---|---|---|---|
| 11 | 0.7026 | **0.8373** | +0.135 | 8/76 → **24/108** |
| 03 | 0.8059 | 0.8323 | +0.026 | 5/76 → 9/92 |
| 01 | 0.8296 | 0.8187 | −0.011 | 10/76 → 12/108 |
| 08 | 0.6655 | 0.8007 | +0.135 | 4/76 → 17/108 |
| 10 | **0.8988** | 0.7951 | −0.104 | 3/76 → 16/108 |
| 06 | 0.7286 | 0.7762 | +0.048 | 9/76 → 14/92 |
| 13 | 0.5333 | 0.7672 | +0.234 | 6/76 → 15/102 |
| 09 | 0.8065 | 0.6863 | −0.120 | 8/76 → 11/102 |
| 05 | 0.6061 | 0.6780 | +0.072 | 5/76 → 14/108 |
| 02 | 0.2254 | 0.6716 | +0.446 | 1/70 → 8/102 |
| 07 | 0.7249 | 0.5735 | −0.151 | 4/76 → 2/92 |

Best score improved in 7 of 11; the number of designs clearing the gate
improved in 10 of 11. But **the single best design did not improve**: §15's
variant 10 at 0.8988 still leads, and §16's best is 0.8373.

Best of this batch, `11.c03.wiggle.7b5b29`, descended from a cycle-2 wiggle
child:

| | |
|---|---|
| score | 0.8373 |
| confidence / catalytic / non-catalytic / angle | 0.844 / 0.820 / 0.952 / 0.767 |
| GLU229 OE2···N3 | 2.22 Å, 122.8° |
| HIS291 NE2···N1 | 2.72 Å, 89.7° |
| ASN261 ND2···O11 | 2.79 Å, 117.6° |
| iptm / complex pLDDT | 0.880 / 0.822 |

### Variant 10's crown was a lucky draw

§15 ranked variant 10 first on 0.8988 from **3 scored designs out of 76**.
Given 108 designs and 16 hits here — five times the sample — it topped out at
0.7951. This is the second time a §15-era leader has failed to reproduce; §15
itself retracted variant 12's 0.8653 for the same reason.

**The overnight top-3 was selection on small samples and should not be used to
pick variants.** Across both batches the variants that reproduce are **11, 03
and 01**. Variant 11 has both the top §16 score and the best hit rate (24/108,
22%), and is the defensible first pick if the next step is to narrow and go
deep.

### Caveat

Two parameters moved together, so depth-versus-breadth is strictly a joint
test. The cycle-0 slice isolates recycles cleanly and shows them neutral,
which is what licenses attributing the rest to the extra cycles — but that is
an inference from one controlled slice, not a second controlled experiment.

Committed here: the 11 leaderboards, config snapshots, and RFD3 / LigandMPNN /
Boltz-2 artefacts for the 142 scored designs — 1,652 files, 28 MB. The gated
designs stay on Euler under `~/pm-live/runs/depth-NN/`; they are regenerable
and were deliberately kept out of git history.

---

## Open and unresolved

| | State |
|---|---|
| `SUBMISSION.md` | **Still not obtained.** Defines the header format and ships a validator. Ask an organiser. |
| Foldseek PDB database | Blocked at source (HTTP 429). TM-screen is targeted-only until it lands. |
| Cross-cluster agreement | AWS measured the founder at 11.18 Å on one seed, Euler at 12.35 Å over two. Consistent within noise, **not verified**. Do not pool AWS and Euler numbers in one leaderboard until the same seed is run on both. |
| Evolution on de novo designs | Has run only on the inherited founder. The 48 fresh designs have never been folded. |
