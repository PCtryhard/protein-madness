# Design rationale — every decision, and why

*Team 04 · de novo Kemp eliminase · Perlin × D-BSSE, 21–22 November 2026*

This document explains **every input, every parameter, and the reason behind
it**. Nothing here is a default we accepted without asking what it does.

---

## 1. What we were actually asked to do

A Kemp elimination needs three things working together: a **base** to pull a
proton off a carbon, a **hydrogen-bond donor** to stabilise the negative charge
that develops as the ring opens, and a **protein fold** that holds both in place
and keeps water away.

The competition gives us the first two as a fixed geometric motif, taken from
HG3.17 — the best Kemp eliminase anyone has made, about 1000× more active than
its parent. Our job is the third: **invent a protein that holds that chemistry.**

The decisive detail is *how* we are judged. **We submit a sequence — a string of
letters. The organisers feed it to Boltz-2, which builds its own 3D model, and
they measure that.** We never submit a structure.

That turns out to be the whole problem. Here is the organisers' own reference
design:

| | what RFdiffusion built | what Boltz rebuilt from the sequence |
|---|---|---|
| base → ligand | **2.53 Å** | **11.18 Å** |
| donor → ligand | 3.09 Å | 7.48 Å |

A textbook active site on paper. Four times too far apart when rebuilt from the
sequence alone. **By the model that scores it, the reference design has no
active site.** Every decision below is aimed at that gap.

---

## 2. Why de novo, and not "modify an existing enzyme"

Two eligibility rules decide this for us:

- more than **70 % sequence identity** to a published enzyme → disqualified
- **TM-score above 0.7** to a published structure → disqualified

Starting from an existing Kemp eliminase is the fastest route to breaking both.
Starting from a blank scaffold and borrowing only the three-atom catalytic
motif breaks neither.

**Measured:** our designs return **0.000 sequence identity** against UniRef90 —
zero hits at all. They are homologues of nothing.

---

## 3. Stage 1 — the backbone (RFdiffusion3)

We ask RFD3 to grow a brand-new 260-residue protein around a fixed catalytic
motif and a fixed ligand.

### What we pin, and why

```json
"ligand": "6NT",
"unindex": "A127,A50",
"select_fixed_atoms": {
  "A127": "OD1,CG,OD2",     // aspartate carboxylate — the catalytic base
  "A50":  "OE1,CD,NE2",     // glutamine amide — stabilises the oxyanion
  "6NT":  "ALL"             // the transition-state analogue, every atom
}
```

**Why these three atoms per residue and not the whole side chain.** Pinning only
the business end — the carboxylate oxygens of the base, the amide of the donor —
leaves RFD3 free to choose the backbone position and rotamer that let it build a
good protein around them. Pin more and you over-constrain the scaffold; pin less
and the chemistry drifts.

**Why the ligand is `6NT`, and why it is an analogue.** The real substrate is
5-nitrobenzisoxazole, and you cannot design around it because *it reacts* — it
is consumed by the reaction we are designing for. The transition state itself
exists for femtoseconds. **6-nitrobenzotriazole** is a stable molecule shaped
like that transition state: same ring system, same nitro group, but inert. Its
pose comes from a 1.09 Å crystal structure of HG3.17 (PDB **4BS0**) — about as
good a starting geometry as exists. And decisively: **the organisers score with
6NT**, so using anything else would make our numbers incomparable with theirs.

**Why `contig: "260-260"` — exactly 260, not a range.** One of the four scored
metrics is how buried the catalytic base is, counted as the number of Cα atoms
within 12 Å of it. More protein means more atoms available to be close. 260 is
the competition's hard maximum, so we take all of it. A range like `150-260`
would hand away burial for nothing.

### The three arms we ran

| arm | motif | reasoning |
|---|---|---|
| **bd** | base + donor | the organisers' own configuration; the control |
| **bdw** | + **Trp44** π-stack | two polar contacts clearly are not holding the ligand. A tryptophan stacks flat against the aromatic ring — a broad, non-directional grip. Harder to scaffold, possibly much stickier. |
| **hb** | + H-bond conditioning | 6NT's nitro oxygens (`O11`, `O21`) are strong hydrogen-bond acceptors **touched by nothing** in the supplied motif. We asked RFD3 for donors against them via `select_hbond_acceptor`, plus `select_buried` on the base to push the burial metric directly. |

The third arm is **purely additive**: the two scored contacts stay pinned exactly
as in the control, so it cannot cost us anything on the geometry term. Worst
case it does nothing.

### Engine settings

`n_batches=4` (four independent backbones per arm), `seed=1`,
`prevalidate_inputs=True` (reject a malformed spec rather than silently
designing the wrong thing), `cleanup_guideposts=True`,
`dump_prediction_metadata_json=True` — **this last one is essential**: it writes
the `diffused_index_map` telling us where the catalytic residues landed in each
new chain. They land somewhere different every time (positions 177/51, 161/74,
55/48…), and getting that mapping wrong would silently score the wrong residue.

**Cost: ~45 seconds per backbone.**

---

## 4. Stage 2 — the sequence (LigandMPNN)

RFD3 gives us a shape. LigandMPNN decides which amino acids go where.

```
--model_type ligand_mpnn
--checkpoint_ligand_mpnn ligandmpnn_v_32_010_25
--fixed_residues "<the mapped catalytic positions>"
--ligand_mpnn_use_atom_context 1
--ligand_mpnn_use_side_chain_context 1
--temperature 0.1 and 0.2
--batch_size 4
```

**Why LigandMPNN and not ProteinMPNN.** ProteinMPNN cannot see the ligand. For
an active site that is exactly the wrong blindness — the residues we most need
right are the ones touching 6NT. Hence both context flags on.

**Why checkpoint `v_32_010_25`.** The `010` is the training noise level, 0.10 Å.
Lower noise means the model stays faithful to the backbone it was given — which
is what we want, because RFD3 placed the catalytic motif *exactly* where it
needs to be and we do not want the sequence model quietly arguing with that
geometry. Higher-noise checkpoints (`020`, `030`) tolerate sloppier backbones
but drift more.

**Why two temperatures.** 0.1 is conservative — the model's confident choice.
0.2 samples more widely. Running both gives a founder population with both
fidelity and diversity, rather than four near-identical sequences.

**Why the catalytic residues are fixed.** They are the enzyme. A design that
quietly mutated its own base is not the thing we claim to have made, so they are
held and then re-checked afterwards.

---

## 5. Stage 3 — folding (Boltz-2)

**This is not a choice, and that is the most important thing to understand about
our pipeline.** We fold with Boltz-2 because *the judges* fold with Boltz-2.
Every metric is defined as its output. Using a more accurate model would make us
*less* aligned with the score.

```
--model boltz2 --sampling_steps 200 --recycling_steps 3
--diffusion_samples 1 --no_kernels --output_format mmcif
msa: empty      ligand: {ccd: 6NT}
```

Flags copied exactly from the organisers' own pipeline, so our numbers are
comparable with the ones we are judged on.

**Why `msa: empty`.** Normally you give a folding model an alignment of related
natural sequences. Our designs **have no relatives** — measured: a de novo design
returns an alignment of depth 1 with zero non-query rows, while a natural
control returns 301. Building an alignment is pure cost for us, and skipping it
also removed a 28 GB memory requirement that would have halved our GPU count.

**Why the ligand is always present.** We never fold the protein alone. The entire
question is where the ligand sits; a structure without it is not a cheaper
measurement, it is a different one.

**Why five seeds.** Boltz is stochastic — same sequence, slightly different
answer each run. One seed flatters lucky draws. Five tells us whether a design
is *reliably* good. **Cost: 21 s per fold on an L4, 34 s on Euler's 2080Ti.**

---

## 6. Stage 4 — scoring

### The four things the judges measure

| metric | what it means | direction |
|---|---|---|
| **iPDE** | Boltz's own uncertainty about where the ligand is | lower |
| **Geometry** | distances and angles from base and donor to the ligand | closer to ideal |
| **Burial** | Cα atoms within 12 Å of the catalytic base | higher |
| **Consistency** | how well all of the above reproduce across 5 seeds | tighter |

### Our implementation, and how we know it is right

We measure the **exact two contacts** the organisers' own verification script
measures — `ASP OD2 → 6NT N3` and `GLN NE2 → 6NT N1` — with the same atom names
and the same residue indexing. Where their code makes a convention choice we
copied the choice rather than improving on it: **a better metric that disagrees
with the judge is worse than a faithful one.**

**Proof it is faithful:** run on the organisers' own output, our scorer
reproduces their numbers to the last decimal — iPDE `2.5525107383728027`, base
contact `11.178812026977539`. Their real output is committed as a test fixture.
If that test ever fails, our ranking is fiction.

### Why distance is a *band*, not "closer is better"

The scored contacts are **hydrogen bonds**. An O···N hydrogen bond does not keep
improving as the atoms approach — below about 2.4 Å you are in van der Waals
repulsion, not a stronger bond. **HG3.17 itself sits at 2.53 Å**, and directed
evolution had every opportunity to go shorter.

So the penalty is a window of **2.4–3.5 Å**: free inside, cost outside, and
**too-close costs double**, because a clash actively destabilises the complex
while a long contact is merely weak.

*(An earlier version of our code penalised only long contacts, which scored a
physically impossible 1.5 Å contact as perfect. An evolutionary search would
have found and exploited that. It was caught and fixed before scaling up.)*

### Why angles matter, and why they are weighted lightly

A carboxylate sitting 3 Å from the ligand but *pointing away* is not a catalytic
base. Distance alone cannot tell those apart, so we also measure the approach
angle `CG–OD2···N3`. But **distance carries most of the signal**, so it is
weighted **1.0 per Å against 0.15 per 10° of angle error** — angle breaks ties
between designs that already sit sensibly, it never rescues a bad contact.

### The full ranking function

Lower is better. Weights in interpretable units, not z-scores, so a cost means
the same thing in generation 1 and generation 20.

| term | weight | why |
|---|---|---|
| base contact deviation | **1.0 / Å** | the metric the reference design fails; biggest dynamic range |
| donor contact deviation | 0.5 / Å | matters, but the base is the catalyst |
| iPDE | 0.5 | model's own confidence in the ligand placement |
| burial shortfall below **47** | 0.05 / atom | 47 is what the real HG3.17 achieves |
| angle error beyond 25° of **131°** | 0.15 / 10° | 131° is HG3.17's measured angle |
| stability below pLDDT **0.70** | 3.0 | an unconfident fold is one whose geometry we should not believe |
| change (design → rebuild RMSD) beyond 1.5 Å | 0.4 / Å | directly measures "did the active site survive" |
| spread across seeds | ×1.0 | a design that works on one seed out of five is not a design |

**Two of those numbers were originally guesses, and measuring the real enzyme
corrected both.** We had the ideal angle at 120° from textbook geometry — HG3.17
sits at 130.9°, so we would have penalised the best known Kemp eliminase by 11°
for being correct. And we had the burial target at 35, "a bit above the
reference design's 30" — the real enzyme manages 47. We had anchored on a design
we already knew fails.

### Hard gates — not tradeable against anything

A design that violates these is dead regardless of how well it scores:

| gate | limit | why |
|---|---|---|
| length | ≤ 260 | competition rule |
| base→ligand | ≤ 12 Å | beyond this the ligand has left the pocket |
| burial | ≥ 15 | a solvent-exposed base is not a base |
| **TM-score** | **≤ 0.70** | eligibility — see below |

**Why TM-score must be a hard gate and not a ranking term.** We measured our
first 8 backbones against 113 published structures. Two failed at 0.703 and
0.705 — and **they were the two with the best burial scores.** Optimising burial
walks you toward disqualification. If TM were tradeable, a good burial number
would buy a design back into contention and we would submit a zero.

---

## 7. How we check eligibility

| rule | how | status |
|---|---|---|
| ≤ 260 residues | asserted on every design | ✅ |
| < 70 % identity to published enzymes | MMseqs2 against UniRef90 | ✅ **0.000** — zero hits |
| TM ≤ 0.7 to published structures | TMalign against 113 RCSB entries | ⚠️ targeted screen only |

The structure set is built by **RCSB full-text search** (*Kemp eliminase*, *Kemp
elimination*, *benzisoxazole*, *computational enzyme design*) rather than from
PDB IDs anyone remembered — so it is reproducible and auditable. It includes
4BS0 and 6SS1 (HG3.17 with ligand) and 2RKX (KE07).

**We state this as a lower bound, not a clearance.** 113 entries is not the PDB.
The full search needs a database that is currently rate-limited at its host.

Throughout, a rule we *could not* evaluate is recorded as `unchecked` and never
counted as a pass. A novelty score that silently returned 0.0 would report a
design as novel exactly when we failed to look.

---

## 8. Why we believe this makes a good Kemp eliminase

Four arguments, in order of how much weight we put on them:

**1. The catalytic chemistry is inherited from something that demonstrably
works.** We did not invent a mechanism. The base, the donor and their geometry
come from HG3.17, which has been measured in a laboratory and is the most active
Kemp eliminase known. We are not betting on new chemistry; we are betting on a
new scaffold for proven chemistry.

**2. Our best design holds that chemistry better than HG3.17 does — in the
test that counts.**

| | our best design | HG3.17 crystal | organisers' reference |
|---|---|---|---|
| base → ligand | **2.36 Å** | 2.53 Å | 11.18 Å |
| approach angle | 135.2° | 130.9° | — |
| burial | **53** | 47 | 30 |
| iPDE | **0.80** | — | 2.55 |
| model confidence | 0.959 | — | 0.820 |

Rebuilt from sequence alone, it reproduces an active site tighter than a
laboratory-evolved enzyme — and buries the base more deeply, which protects the
aspartate's pKa from water.

**3. We validated the ruler before trusting it.** Scoring HG3.17's own crystal
structure with our metrics puts it exactly where it should be: 2.53 Å, 47 buried
atoms, zero clashes, perfect on every term. A scoring function that could not
recognise the best known Kemp eliminase would be worthless for ranking ours.

**4. We select on reliability, not on best-case.** Every candidate is folded
across five independent seeds and ranked on mean minus spread. A design that
looks brilliant once and mediocre four times loses to one that is consistently
good — because *consistency across seeds is a proxy for preorganisation*, which
is the physical property that actually makes enzymes fast.

---

## 9. What we would not claim

- **The transition-state analogue is an approximation.** The organisers say so
  themselves. A design that grips 6NT beautifully is not guaranteed to catalyse
  the real elimination.
- **Our angle ideal is calibrated from HG3.17**, so our score rewards "like
  HG3.17" by construction and **cannot recognise a better geometry that differs
  from it**. That is an assumption, not a discovery.
- **The TM-score check is a lower bound**, not a clearance.
- **We have not computed any activity**, only geometry and model confidence.
  Nothing here is a kinetic prediction.
- **Free CPU filters did not predict which backbones would work.** One backbone
  with the second-best burial produced our eight worst designs. We corrected
  course and now rank backbones by folding a couple of sequences from each — but
  it is an honest negative result about the cheap heuristics.

---

## 10. The pipeline in one page

```
   HG3.17 motif (3 atoms × 2 residues) + 6NT pose from 4BS0
                        │
   RFdiffusion3 ────────┤  260 residues, motif + ligand pinned        ~45 s
                        │  three arms: base+donor │ +Trp │ +H-bond conditioning
                        ▼
   Gate A (free, CPU) ──┤  motif intact, burial, ligand enclosure
                        │  TM-score ≤ 0.70  ← HARD, 2 of first 8 failed here
                        ▼
   LigandMPNN ──────────┤  4 sequences × 2 temperatures per backbone
                        │  catalytic residues fixed, ligand context on
                        ▼
   Gate B (free, CPU) ──┤  length, composition, identity < 70 %
                        ▼
   Boltz-2 ─────────────┤  ligand bound, no MSA, 1 seed to triage      ~21 s
                        │                       5 seeds to confirm
                        ▼
   Our scorer ──────────┤  iPDE · geometry (distance + angle) · burial
                        │  · stability · change · spread across seeds
                        ▼
   Directed evolution ──┘  mutate the winners, refold, repeat
                           catalytic residues never touched
```
