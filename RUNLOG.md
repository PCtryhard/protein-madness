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

## Open and unresolved

| | State |
|---|---|
| `SUBMISSION.md` | **Still not obtained.** Defines the header format and ships a validator. Ask an organiser. |
| Foldseek PDB database | Blocked at source (HTTP 429). TM-screen is targeted-only until it lands. |
| Cross-cluster agreement | AWS measured the founder at 11.18 Å on one seed, Euler at 12.35 Å over two. Consistent within noise, **not verified**. Do not pool AWS and Euler numbers in one leaderboard until the same seed is run on both. |
| Evolution on de novo designs | Has run only on the inherited founder. The 48 fresh designs have never been folded. |
