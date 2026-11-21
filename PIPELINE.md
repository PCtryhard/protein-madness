# PIPELINE — motif-anchored protein design with score-guided search

Nov 21, 2026 · @Maximilian Zane Alexandrov

**This is the specification the repo implements.** It supersedes everything in
[outdated/](outdated/). `README.md` says which build phases have passed; this
file says what is being built. Where the two disagree, the README is the state
and this file is the intent.

Repo-specific notes that are not part of the specification — the real 5FQJ
numbers, RFD3 vs RFdiffusion, and the two open questions as they now stand —
are in the appendix at the end.

## Purpose

Build a closed-loop design pipeline that improves motif-anchored designs against a fixed scoring function, running on two 12 GB GPUs.

One forward pass is: seed PDB with fixed motif residues, then RFdiffusion partial diffusion, then LigandMPNN, then Boltz-2, then hydrogen placement, then scoring. A search loop then expands the best designs through two branches and re-scores them until the score stops improving.

Each seed scaffold is an independent search with its own frontier, stopping condition and leaderboard. There is no competition between scaffolds, so results stay comparable across seeds and the number of seeds can change without altering behaviour.

The scoring function is weighted towards Boltz-2 interface confidence, with hydrogen-bond distance and linearity as secondary terms. Every weight, threshold and model parameter lives in one file.

Two things in the original sketch do not work as stated and are corrected below: atom-level freezing does not exist in RFdiffusion, and a 180 degree hydrogen-bond angle cannot be measured on Boltz-2 output, which contains no hydrogens.

```
                         ONE FORWARD PASS
  seed PDB  ->  RFdiffusion  ->  LigandMPNN  ->  Boltz-2  ->  H placement  ->  score
  (motif        (partial         (ligand-        (complex     (only if the     (0..1,
   fixed)        diffusion)       aware seq)      from seq)    angle rule       1 best)
                                                              needs it)

                     THE SEARCH, PER SCAFFOLD, PER CYCLE

                        +-----------------+
                        |    frontier     |  keep_per_cycle best, never pruned
                        +--------+--------+
                                 |
                 +---------------+---------------+
                 |                               |
         wiggle branch                     mutate branch
   re-noise the RFdiffusion          single mutants within
   backbone for partial_T steps      shell_radius of the ligand
                 |                               |
         RFdiffusion                             |  (no new backbone needed:
                 |                               |   Boltz-2 repacks implicitly)
         LigandMPNN                              |
                 |                               |
                 +-------------+-----------------+
                               |
                            Boltz-2  <-- sequence-hash cache sits here
                               |
                          H placement
                               |
                            scoring
                               |
                 next frontier = best of (both branches + previous frontier)
                               |
                 stop when no gain > min_delta for `patience` cycles
```

The mutation branch returns straight to Boltz-2 because a point mutation needs no new backbone, and Boltz-2 repacks the neighbourhood implicitly when it predicts the complex from sequence.

## Hardware: two RTX 2080 Ti, 12 GB each

Two cards at 12 GB is not 24 GB. Every model instance must fit inside one card, so the parallelism available is two independent designs at a time, not one larger job split across both.

Boltz-2 is the memory bottleneck. NVIDIA's packaged Boltz-2 NIM states a 48 GB minimum, and the open-source `boltz` package is much lighter but still limited: reports put consumer 12 GB cards at roughly 300 to 500 tokens before exhausting VRAM. On Linux there is no CPU fallback, so the run fails rather than slowing down.

Keep the whole system, protein plus ligand atoms, under 400 tokens, and preferably under 300. Build a hard refusal into the Boltz-2 wrapper rather than discovering the limit mid-search.

Turing, compute capability 7.5, has no native bf16. The optimised attention kernels Boltz ships target Ampere and later. Expect to need `--no_kernels` and a non-bf16 precision, and verify both on day one rather than assuming either way.

Memory knobs, in the order to cut them: `diffusion_samples`, then `recycling_steps`, then system size. Dropping `diffusion_samples` to 1 costs the least accuracy per GB recovered.

Throughput sets the mutation branch budget, so measure it in phase 0 instead of estimating. As a planning figure, a few minutes per Boltz-2 call on a 250 token system gives order 50 to 100 designs per hour across both cards. Full saturation of 12 positions is 228 mutants, which is several hours for one cycle of one scaffold.

Two consequences follow and are built into the design below. The mutation branch must be restricted by shell radius and a reduced alphabet, and every Boltz-2 result must be cached by sequence hash, because a mutation search regenerates the same sequences constantly.

## Repo layout and configuration

Every tunable lives in `config/pipeline.yaml`. A numeric literal inside a scoring function or a stage wrapper is a bug, and the review for each phase should grep for one.

```markdown
config/
  pipeline.yaml          # all tunables, single source of truth
  targets/<name>.yaml    # per-target: seeds, motif, H-bonds, ligand
src/
  config.py              # pydantic schema, loads and validates both files
  tracking.py            # structure readers, AtomRef resolution, provenance
  stages/                # base.py rfdiffusion.py ligandmpnn.py boltz.py protonate.py
  scoring/               # confidence.py hbond.py combine.py
  search/                # tree.py wiggle.py mutate.py
  store.py               # sqlite: lineage, leaderboard, sequence-hash cache
  runner.py              # two-worker GPU queue, the per-scaffold loop
  cli.py                 # run | resume | report | smoke | measure
tests/
  fixtures/              # hand-built PDBs with known geometry + their generator
runs/<run_name>/
  designs.db             # sqlite lineage and cache
  structures/            # one subdir per design
  stages.jsonl           # per-stage wall time and peak VRAM
  hardware_notes.md      # phase 0 findings
```

Two additions to the layout as originally sketched, both because two modules
needed the same thing: `src/store.py` holds the sqlite layer, which `runner.py`
and `cli.py` both use, and `src/stages/base.py` holds the subprocess runner and
the `StageResult` every wrapper returns. `cli.py` also grew a `measure`
subcommand, which is the phase-1 acceptance table and needs no GPU.

The config file, with a recommended range and a centre value on anything that is genuinely a choice:

```yaml
run:
  name: null                   # set per run
  seed: 0
  gpus: [0, 1]
  max_cycles: 12               # range 5-30 (12)
  patience: 2                  # cycles without improvement before a scaffold stops
  min_delta: 0.005             # smallest score gain that counts as improvement
  eval_replicates: 1           # Boltz-2 seeds averaged per design; range 1-5 (1)

rfdiffusion:
  run_inference: /path/to/RFdiffusion/scripts/run_inference.py
  ckpt: Base_ckpt.pt
  partial_T: 10                # re-noising timesteps; range 2-20 (10)
  diffuser_T: 50               # full trajectory length; range 25-100 (50)
  designs_per_parent: 4        # range 2-16 (4)
  extra_args: []

ligandmpnn:
  run_script: /path/to/LigandMPNN/run.py
  checkpoint: ligandmpnn_v_32_010_25.pt
  temperature: 0.1             # range 0.05-0.3 (0.1)
  sequences_per_backbone: 4    # range 1-8 (4)
  omit_AA: "CX"
  pack_side_chains: false

boltz:
  binary: boltz
  cache_dir: ~/.boltz
  recycling_steps: 3           # range 1-5 (3); cut second on OOM
  diffusion_samples: 1         # range 1-5 (1); cut first on OOM
  sampling_steps: 200          # range 50-200 (200)
  use_potentials: false
  no_kernels: true             # expected on sm_75; confirm in phase 0
  precision: "32"              # bf16 unsupported on Turing; confirm in phase 0
  max_tokens: 400              # refuse to submit beyond this

protonation:
  engine: pdbfixer             # pdbfixer | reduce | none
  ph: 7.0

scoring:
  weights:
    confidence: 0.70           # range 0.5-0.9 (0.7)
    geometry: 0.30
  geometry_weights:
    distance: 0.60             # fixed by spec
    angle: 0.40                # fixed by spec
  confidence:
    metric: complex_ipde       # complex_ipde | iptm | ligand_iptm | motif_pae
    ipde_best: 0.5             # angstrom, maps to 1.0
    ipde_worst: 5.0            # angstrom, maps to 0.0
  distance:
    ideal: 2.9                 # angstrom, heavy-atom donor to acceptor; range 2.7-3.0
    tolerated: 3.5             # angstrom, maps to 0.0; range 3.2-3.6 (3.5)
    fail: 4.5                  # angstrom, any required bond beyond this zeroes the design
  angle:
    tolerance: 60.0            # degrees of deviation mapped to 0.0; range 30-70 (60)
    rule: acceptor_side_always # split_by_donor_type | acceptor_side_always
  aggregate: mean              # mean | min across the named bonds

search:
  keep_per_cycle: 2            # frontier size per scaffold; range 1-5 (2)
  wiggle:
    enabled: true
    parents_per_cycle: 4       # range 1-8 (4)
    source: rfdiffusion_backbone   # rfdiffusion_backbone | boltz_prediction
  mutate:
    enabled: true
    parents_per_cycle: 2       # range 1-4 (2)
    shell_radius: 5.0          # angstrom from ligand; range 4-8 (5)
    alphabet: "ADEFGHIKLMNQRSTVWY"
    max_mutants_per_parent: 80 # range 20-250 (80)
```

## Target spec

One file per target declares the seeds, the ligand, the fixed motif and the named hydrogen bonds. Everything downstream refers to a hydrogen bond by its string `id`, never by residue number, because the numbers change and the id does not.

On freezing: RFdiffusion holds whole residues, not individual atoms. Motif scaffolding fixes residues named in a contig string, and partial diffusion re-noises and re-denoises the whole structure. There is no switch that pins three atoms and lets the rest of a sidechain move. The spec below therefore fixes whole residues, which is what you confirmed.

```yaml
name: example_target

seeds:
  - {id: seed_a, pdb: inputs/seed_a.pdb, contigs: "20-40/A45-49/10-30/A102-102/20-40"}
  - {id: seed_b, pdb: inputs/seed_b.pdb, contigs: "20-40/A45-49/10-30/A102-102/20-40"}
  - {id: seed_c, pdb: inputs/seed_c.pdb, contigs: "20-40/A45-49/10-30/A102-102/20-40"}

ligand:
  id: LIG
  ccd: null                            # prefer a CCD code whenever one exists
  smiles: "CC(=O)Nc1ccccc1"            # used only when ccd is null
  reference_sdf: inputs/lig_named.sdf  # authoritative atom names for the SMILES path

motif:
  fixed:                               # whole residues, never mutated, never redesigned
    - {chain: A, resnum: 45}
    - {chain: A, resnum: 46}
    - {chain: A, resnum: 49}
    - {chain: A, resnum: 102}

hbonds:
  - id: lig_n3_to_asp45_od1
    donor:     {site: ligand,  atom: N3}
    acceptor:  {site: protein, chain: A, resnum: 45, atom: OD1}
    acceptor_antecedent: {site: protein, chain: A, resnum: 45, atom: CG}
    rotatable_donor: false             # amide N-H, H position is determined
    required: true

  - id: ser49_og_to_lig_o1
    donor:     {site: protein, chain: A, resnum: 49, atom: OG}
    donor_antecedent: {site: protein, chain: A, resnum: 49, atom: CB}
    acceptor:  {site: ligand,  atom: O1}
    rotatable_donor: true              # hydroxyl, H position is chosen by the protonator
    required: true
    acceptor_ideal_angle: 120.0        # lone-pair directionality, see Hydrogens
```

The seeds list is open-ended. Add a fourth or tenth entry and the runner starts another independent search with no other change. Contigs are per seed because the motif may sit at different positions in each.

`required: true` means the bond is subject to the distance gate. A bond marked `required: false` contributes to the score but cannot zero a design.

## Identity tracking through the pipeline

Yes, the hydrogen bonds can be tracked, and the mechanism is a chain of four explicit index maps carried with each design. Nothing is inferred by proximity at any point.

A hydrogen bond is declared once against seed numbering and keeps its string id forever. What changes is where its atoms live, and each stage contributes one map.

**Seed to RFdiffusion output.** RFdiffusion writes a `.trb` pickle beside every output PDB. It contains `con_ref_idx0` and `con_hal_idx0`, the zero-indexed reference and generated positions of every motif residue, plus `con_ref_pdb_idx` and `con_hal_pdb_idx` as chain and residue-number pairs, and `complex_con_hal_idx0` for multi-chain runs. Load it and build a dict from seed `(chain, resnum)` to output `(chain, resnum)`. Assert that every residue in `motif.fixed` appears. A missing one means the contig string and the motif list disagree, which must stop the run rather than produce a design whose motif silently moved.

**RFdiffusion to LigandMPNN.** LigandMPNN preserves the numbering of its input PDB, so this map is the identity. Pass fixed positions in output numbering, and after the call assert the residue type at each motif position is unchanged. LigandMPNN redesigning a fixed residue is a configuration error that is otherwise invisible.

**LigandMPNN to Boltz-2.** Boltz-2 takes sequences, not a PDB, and numbers its output 1 to N per chain in submitted order. Do not assume an offset of zero. Compute the index of each motif residue in the exact string you submitted, then assert the residue type at that index in the output structure. If the assertion fails, the sequence was reordered or trimmed somewhere.

**Ligand atom names.** This is the one place a name can be lost. With a CCD code, Boltz-2 uses CCD atom names and `N3` resolves directly. With SMILES, Boltz assigns its own atom naming and your reference names do not survive. Build an RDKit molecule from `reference_sdf`, which carries your names, build one from the output block, and map atoms with `GetSubstructMatch` for a one-to-one correspondence. Cache the map per ligand, since it is identical for every design. Prefer a CCD code and treat the graph-matching path as the fallback.

The resolver is one object per design:

```python
@dataclass(frozen=True)
class AtomRef:
    site: Literal["protein", "ligand"]
    atom: str
    chain: str | None = None
    resnum: int | None = None

class Provenance:
    design_id: str
    parent_id: str | None
    scaffold_id: str
    seed_to_hal: dict[tuple[str, int], tuple[str, int]]
    hal_to_boltz: dict[tuple[str, int], tuple[str, int]]
    ligand_atom_map: dict[str, int]

    def resolve(self, ref: AtomRef, structure) -> np.ndarray:
        """Return xyz. Raise AtomResolutionError on any miss."""
```

The whole `Provenance` object is serialised into the design's database row. A design whose maps are lost is not re-derivable, and the lineage is worthless without them.

One rule above all others here: `resolve` raises on a miss and never returns a default. A missing atom scored as zero distance looks like a perfect bond, and a missing atom scored as infinite distance looks like a failed design. Both corrupt the search silently, which is worse than a crash.

## Hydrogens and the 180 degree criterion

Boltz-2 writes no hydrogens, so the angle you specified cannot be measured on its raw output. This needs a decision before the scorer is written, because the obvious workaround gives the wrong target value.

The angle that idealises near 180 degrees is donor, hydrogen, acceptor:

```latex
\theta_{\mathrm{D-H}\cdots\mathrm{A}} \to 180^\circ
```

The heavy-atom angle at the donor is a different quantity. For an sp2 or sp3 donor its ideal value is roughly 109 to 120 degrees, set by hybridisation:

```latex
\theta_{\mathrm{D_{ant}-D}\cdots\mathrm{A}} \approx 109^\circ \text{ to } 120^\circ
```

So applying a 180 degree target to heavy atoms, which is what happens if hydrogens are simply skipped, penalises good geometry and rewards a nearly linear antecedent arrangement that is physically wrong. This is the single easiest way to get a plausible-looking pipeline that optimises in the wrong direction.

Add a protonation stage between Boltz-2 and scoring. PDBFixer, through `openmm.app.Modeller.addHydrogens` at pH 7, is the least friction; `reduce` from the Richardson lab is the alternative and is better at orienting rotatable groups.

The remaining subtlety is that for rotatable donors the protonation tool chooses where the hydrogen goes, and most such tools place it to optimise hydrogen bonding. Measuring D-H to A on a serine hydroxyl then partly measures what PDBFixer decided, not what Boltz-2 predicted, and the term will look better than the prediction deserves.

The recommended split, which is why `rotatable_donor` is in the target spec:

| Donor type | Examples | Angle measured | Ideal |
| --- | --- | --- | --- |
| Fixed hydrogen | backbone N-H, Asn, Gln, Arg, Trp, amide N-H | donor, H, acceptor | 180 degrees |
| Rotatable hydrogen | Ser, Thr, Tyr hydroxyl, Lys ammonium | acceptor antecedent, acceptor, donor | from `acceptor_ideal_angle`, typically 120 degrees |

For rotatable donors the scorer measures directionality at the acceptor instead, which needs no hydrogens and cannot be gamed by the protonator. The distance term is always heavy-atom donor to acceptor and is unaffected by any of this.

If you would rather keep one rule for every bond, say so and the scorer collapses to heavy-atom distance plus acceptor-side directionality throughout, dropping the protonation stage entirely. That is the more conservative option and loses the least to tooling artefacts.

## Scoring function

Every term is normalised to the interval from 0 to 1 with 1 best, because the weights are otherwise meaningless. Mixing a raw iPDE in angstroms with an angular deviation in degrees would let the units decide the weighting instead of you.

Your 0.6 and 0.4 split the geometry sub-score. Confidence sits above that split with its own weight, defaulting to 0.7 against 0.3, which is the interpretation of "heavily weighted" and is the first number to revisit.

Confidence, from `complex_ipde`, where lower is better:

```latex
c = \mathrm{clip}\!\left(\frac{\mathrm{ipde}_{\mathrm{worst}} - \mathrm{ipde}}{\mathrm{ipde}_{\mathrm{worst}} - \mathrm{ipde}_{\mathrm{best}}},\; 0,\; 1\right)
```

Distance, per bond, symmetric about the ideal so that a 2.4 angstrom clash is penalised as well as a 3.4 angstrom stretch:

```latex
d_i = \mathrm{clip}\!\left(1 - \frac{\lvert r_i - r_{\mathrm{ideal}} \rvert}{r_{\mathrm{tol}} - r_{\mathrm{ideal}}},\; 0,\; 1\right)
```

Angle, per bond, against that bond's own ideal from the table in the previous section:

```latex
a_i = \mathrm{clip}\!\left(1 - \frac{\lvert \theta_i - \theta_{\mathrm{ideal},i} \rvert}{\theta_{\mathrm{tol}}},\; 0,\; 1\right)
```

Combined, with the aggregation over bonds set by `scoring.aggregate`:

```latex
S = w_c \, c + w_g \left( 0.6 \cdot \underset{i}{\mathrm{agg}}\, d_i + 0.4 \cdot \underset{i}{\mathrm{agg}}\, a_i \right)
```

Use `mean` while exploring and `min` when you want every named bond to form, since `min` makes one bad bond dominate. The gate is separate: any bond with `required: true` beyond `distance.fail` sets the total to zero, and the design is recorded as gated rather than deleted, so the search can see what it rejected.

Log these for every design without scoring on them, so the weighting can be revisited offline without recomputing anything: `iptm`, `ptm`, `ligand_iptm`, `complex_plddt`, `complex_ipde`, the mean PAE restricted to motif-residue and ligand-atom pairs, the LigandMPNN sequence score, and every raw distance and angle per named bond.

One caveat on the metric choice. `complex_ipde` is a global interface figure and averages over the whole interface, including regions you do not care about. The mean PAE over just the motif to ligand pairs is a more sensitive signal for this objective. Log both from the start and keep `scoring.confidence.metric` switchable, so swapping is a config change and not a rewrite.

## Search, lineage and stopping

Each scaffold runs its own elitist hill-climb. A cycle expands the current frontier through both branches, scores everything, and forms the next frontier from the best designs across the branches and the previous frontier together. Including the previous frontier is what guarantees the best score never regresses.

The wiggle branch re-noises a parent for `partial_T` steps, re-denoises, passes the backbone to LigandMPNN, then to Boltz-2. The mutation branch enumerates single mutants of a parent sequence and goes straight to Boltz-2, skipping diffusion and LigandMPNN entirely. That skip is sound, because Boltz-2 predicts the whole complex from sequence and so repacks the neighbourhood implicitly.

One correction on the wiggle branch. Re-noising the Boltz-2 prediction feeds RFdiffusion a backbone from a different distribution than the crystal-like structures it was trained on. Re-noise the RFdiffusion backbone that produced the parent, carrying the parent's sequence forward. `search.wiggle.source` defaults to `rfdiffusion_backbone` for this reason; the other setting is there so the assumption can be tested rather than trusted.

Mutable positions are those within `shell_radius` of any ligand atom, excluding every residue in `motif.fixed`. If the count of candidate mutants exceeds `max_mutants_per_parent`, sample without replacement rather than truncating in sequence order, which would bias towards whichever end of the chain comes first.

Stopping, per scaffold: halt when the best score has not improved by more than `min_delta` for two consecutive cycles. `min_delta` has to exist, otherwise floating-point noise in the seventh decimal registers as improvement and the search never stops. Every cycle's best design and its full ancestry are kept. The tree is never pruned, only the frontier is.

One sqlite table carries the whole search:

```sql
CREATE TABLE designs (
  design_id        TEXT PRIMARY KEY,
  parent_id        TEXT,
  scaffold_id      TEXT NOT NULL,
  cycle            INTEGER NOT NULL,
  branch           TEXT NOT NULL,      -- seed | wiggle | mutate
  mutation         TEXT,               -- e.g. A72V, null outside the mutate branch
  sequence         TEXT NOT NULL,
  seq_hash         TEXT NOT NULL,
  score            REAL,
  term_confidence  REAL,
  term_distance    REAL,
  term_angle       REAL,
  gated            INTEGER DEFAULT 0,
  status           TEXT NOT NULL,      -- scored | gated | errored
  error            TEXT,
  hbond_raw        TEXT,               -- json: per-bond distance and angle
  extra_metrics    TEXT,               -- json: iptm, ptm, plddt, motif_pae, ...
  provenance       TEXT NOT NULL,      -- json: the three index maps
  structure_path   TEXT,
  created_at       TEXT NOT NULL
);
CREATE INDEX idx_seq_hash ON designs(seq_hash);
CREATE INDEX idx_scaffold_cycle ON designs(scaffold_id, cycle);
```

The `seq_hash` index is the Boltz-2 cache. Before submitting, look up the hash of sequence plus ligand plus Boltz settings; on a hit, reuse the stored result. A mutation search regenerates identical sequences constantly, and on two cards this cache is worth more than any other optimisation in the build.

Ancestry is a recursive query on `parent_id`, which is why lineage needs no separate structure:

```sql
WITH RECURSIVE line(id) AS (
  SELECT ?  UNION ALL
  SELECT d.parent_id FROM designs d JOIN line ON d.design_id = line.id
  WHERE d.parent_id IS NOT NULL
)
SELECT * FROM designs WHERE design_id IN (SELECT id FROM line);
```

## Build phases

Build in this order and do not start a phase until the previous one passes its acceptance test. The failure mode worth guarding against is a working search loop sitting on top of an untested scorer, where every result looks plausible and none of it means anything.

**Phase 0: environment and the noise control.** Smoke-test each tool separately on one small system, recording peak VRAM and wall time per stage. Settle whether bf16 works on sm\_75 and whether `--no_kernels` is required, and write the findings to `runs/hardware_notes.md`.

Then the control that matters most. Take one fixed sequence and ligand, run Boltz-2 ten times with ten different seeds, and compute all three score terms on each output.

Acceptance: a table of mean and standard deviation per term across the ten seeds. The decision rule is that if the standard deviation of the geometry sub-score is more than about half the spread you expect between genuinely different designs, the geometry terms are mostly prediction noise. In that case raise `eval_replicates` to 3 and `patience` to 4 before building the search.

This is worth the hour it costs. The entire search is hill-climbing on this signal, and patience of 2 on a noisy objective terminates on noise rather than on convergence. Sidechain placement in a designed pocket is exactly where a structure predictor is least reliable, and the angle term asks for precision at roughly the scale of the model's own error.

**Phase 1: config, tracking, one linear pass.** The two config files with a pydantic schema, then one seed through all five stages exactly once, no search.

Acceptance: for a single design, print a table of each named hydrogen bond with its resolved atom indices at every stage, plus the final distance and angle. Every motif residue traced from seed numbering to Boltz-2 numbering. Any unresolvable atom raises rather than defaulting.

**Phase 2: scoring with unit tests.** Hand-build two small PDBs in `tests/fixtures` with geometry you have computed yourself: one with a 2.9 angstrom, 180 degree bond, one with 3.4 angstroms at 130 degrees.

Acceptance: the scorer returns the term values you calculated by hand, the gate fires on a fixture with a 5 angstrom bond, and both aggregation modes are covered.

**Phase 3: the search loop.** Both branches, elitism, patience, the sqlite store and the sequence-hash cache.

Acceptance: a three-cycle run on one scaffold completes; the recursive ancestry query returns the best design's full lineage; re-submitting a sequence already seen hits the cache instead of Boltz-2, verified by a counter in the run log.

**Phase 4: scale-out.** The two-worker GPU pool, one stage per card at a time, process isolation so no two models ever share a card.

Acceptance: a full multi-scaffold run with both GPUs busy, no OOM, and peak VRAM logged per stage per card.

## Rules and non-goals

Fail loudly. Every one of these corrupts a search quietly if it is handled with a fallback instead of an exception:

- Never score a design whose hydrogen-bond atoms could not be resolved. Raise, record `status = errored`, move on.
- Never infer a residue mapping from proximity or sequence alignment. If the `.trb` lacks a motif residue, stop the run.
- Never substitute a distance-based hydrogen-bond detector for the bonds named in the target spec. The spec names them; find those exact atoms, or fail.
- Never silently clamp `max_tokens`. Refuse the submission and log it.
- Assert residue identity at every motif position after every stage, not only at the end.
- Never catch an exception around a whole design and continue with partial scores.

Every number used in a computation comes from config. A literal float inside a scoring function is a bug, and each phase review should grep for one.

Do not build, in this version:

- No Rosetta or FoldX integration.
- No molecular dynamics and no minimisation beyond hydrogen placement.
- No cross-scaffold competition and no shared frontier.
- No sharding of a single model across both GPUs.
- No web interface. A CLI, the sqlite file, and a `report` subcommand that writes a markdown leaderboard per scaffold.

Two open questions to settle before phase 3, both of which change the search rather than the plumbing. Whether the angle rule is split by donor type as recommended, or collapsed to acceptor-side directionality for every bond. And whether `eval_replicates` stays at 1, which phase 0 answers with data rather than judgement.

---

## Appendix: how this repo differs from the specification above

Added when the spec was implemented, 21 November 2026. Everything above this
line is the brief as written; everything below is what was found on contact
with the repo. None of it changes the design, and all of it is checked by a
test so it cannot drift.

### The angle rule is a config switch, and the conservative option is the default

The brief offers two rules and recommends the split by donor type. Both are
implemented, selected by `scoring.angle.rule`, and the shipped default is
`acceptor_side_always` — the conservative one. Three reasons:

1. **The repo already measured acceptor-side angles.** `RUNLOG.md` §3 records
   the scored contacts as `CG-OD2···N3` and `CD-NE2···N1`, i.e. antecedent at
   the acceptor, root-contact to ligand. That is the same quantity
   `acceptor_side_always` measures, so the new pipeline stays comparable with
   the numbers already collected.
2. **Both 5FQJ bonds have fixed hydrogens.** The donors are a benzotriazole
   N3-H and an imidazole NE2-H, so under the split rule *both* would be
   measured `D-H···A` and *both* would need the protonation stage. The split
   rule's advantage — that it avoids measuring a protonator's choices on
   rotatable donors — buys nothing on this target, while the stage it requires
   adds a dependency (OpenMM + PDBFixer) and a failure mode.
3. **It is the option that loses least to tooling artefacts**, which is the
   brief's own assessment.

Switching is one line in `config/pipeline.yaml`. `config.py` refuses the
combination `rule: split_by_donor_type` with `protonation.engine: none`,
because that pairing silently applies a 180° target to heavy-atom geometry —
the exact failure the Hydrogens section warns about.

### RFdiffusion does not do atom-level freezing; RFD3 does

The brief's correction is right for RFdiffusion, and the target spec fixes
whole residues accordingly. But the runs already in `data/No_mutation_run/`
used **RFD3**, whose input JSON carries a `select_fixed_atoms` key — atom-level
freezing does exist there.

This matters in one concrete way: the two tools' contig syntax is not the same.
RFD3 took `{"contig": "260-260", "unindex": "A48,A225,..."}`; RFdiffusion takes
`contigmap.contigs=[...]` plus `diffuser.partial_T`. `src/stages/rfdiffusion.py`
targets RFdiffusion as specified, and the `contigs` string in
the generated target files is flagged as unverified against the actual
RFdiffusion build. **Confirm it in phase 1 before trusting a run.** If the
project moves to RFD3 instead, the wrapper and the contig field change together.

### The real target's numbers are measured, not assumed

The target files are built from the motif PDBs in this repo, not
from the brief's example. What the seed actually contains:

| | base | donor |
|---|---|---|
| bond | ASP229 `OD2` ← 6NT `N3`-H | HIS291 `NE2`-H → 6NT `N1` |
| distance | 2.734 Å | 3.084 Å |
| acceptor-side angle | `CG-OD2···N3` = 106.6° | `N2-N1···NE2` = 126.6° |

`tests/test_tracking.py::test_real_target_bonds_resolve_and_measure_as_documented`
asserts these, so the target file cannot drift from the structure it describes.

Two things in it are still open, and both are flagged in the file itself:

* **The donor residue.** This seed has HIS291. The organisers' reference motif
  (`RUNLOG.md` §1) uses a GLN at the equivalent position, and 4BS0 has the
  donor mutated away. If the intended motif is GLN, the angle must be
  re-measured rather than carried over.
* **The base angle.** 106.6° here; `RUNLOG.md` §3 reports 130.9° for the same
  contact in 4BS0. Different structures, so both can be right, but the ideal
  should be revisited once the reference structure is settled.

### The example target above is missing two fields, and now has them

`config/targets/example_target.yaml` is the spec's example, and it did not
load. Two gaps, both found by `config.load()` refusing it:

* `lig_n3_to_asp45_od1` has no `acceptor_ideal_angle`. Fine under
  `split_by_donor_type`, where a fixed-H donor is measured `D-H···A` against
  180°, but invalid under the `acceptor_side_always` default.
* `ser49_og_to_lig_o1` has no `acceptor_antecedent`. This one was invalid under
  **both** rules: its donor is rotatable, so the split rule also measures it at
  the acceptor, and there was no atom named to measure against.

Both are filled in, with comments saying which rule needs which field, and
`tests/test_config.py` now loads every shipped target under both rules so a
documented example cannot go stale again.

### The ligand token count is a lookup table, not an estimate

`boltz.max_tokens` is only a guard if its input is right, so
`KNOWN_LIGAND_ATOM_COUNTS` in `src/stages/boltz.py` maps CCD code to heavy-atom
count and **raises for an unknown code** rather than guessing. 6NT is 12 heavy
atoms, counted from the seed PDB and cross-checked against a real Boltz-2
prediction by a test. 5FQJ is therefore 260 + 12 = **272 tokens**, inside the
300 the brief prefers.

### The target is seven variants, and their seeds carry deliberate clashes

The spec assumes one target with several seeds. The data is seven *targets*:
`data/01..07`, each a motif-only PDB with a different catalytic constellation
(ASP or GLU at 229, HIS/ASN/GLN at 291, an optional HIS at 261). The motif
residues and the hydrogen bonds differ per variant, and the `Target` schema
holds one motif and one bond list per file, so each gets its own target file.
`scripts/build_targets.py` generates all seven from the PDBs, and a test runs
it with `--check` so a hand-edit cannot drift.

That also changes cycle 0: these are motif-only inputs, so cycle 0 is motif
scaffolding rather than partial diffusion of a full-length seed. The runner
already did the right thing here (`partial=False` for the seed cycle,
`partial=True` for the wiggle branch), but the contig strings are now
scaffolding contigs — `20-40/A229-229/10-30/A291-291/20-40` and similar — and
they are **this repo's construction, not a verified string.** Still phase 1's
job to confirm.

Only variant 01's geometry is real. The rest were modelled without repacking:
`GLU229 OE2···N3` at 1.677 Å, `HIS261 NE2···O11` at 1.820 Å, `ASN291 ND2···N1`
at 4.823 Å. Fine for an RFdiffusion input, useless as a score target, so the
ideal angles come from the crystal where the acceptor is unmutated and from
hybridisation otherwise.

### The distance gate was asymmetric; it now has a clash end

Finding those sub-2 Å contacts exposed a gap the spec left. The distance term
is symmetric about the ideal, so 1.7 Å and 4.1 Å both score zero — but
`is_gated` only fired above `distance.fail`. A design with an atom driven
through the ligand would score zero on geometry and still be ranked on its
confidence term, which at weight 0.7 is most of the total.

`scoring.distance.clash` (default 2.2 Å) gates the short end too. Setting it
very small recovers the spec's one-sided behaviour exactly.

### Running it: the loop cannot be split across two clusters

`RUNLOG.md` has RFD3 and LigandMPNN on the AWS box and Boltz-2 on Euler. That
split cannot drive this pipeline: a cycle cannot pick its next frontier until
its designs are scored, so a split means a cluster round trip per cycle, twelve
times per variant.

`cluster/preflight.py` is the tool that settles where to run — it checks the
GPUs, runs a real bf16 matmul instead of assuming sm_75 cannot, and verifies
all three tool paths resolve. `cluster/README.md` has the decision table and
the gotchas that have actually bitten this project, CRLF line endings in
`sbatch` scripts among them.

### The two open questions, as they now stand

* **The angle rule** — answered, with a default and a reason, above. Flip it in
  config if you disagree; nothing else changes.
* **`eval_replicates`** — still open, and still phase 0's to answer with data.
  `run.eval_replicates` works and `average_replicates()` is tested, including
  the rule that a design gated in *any* replicate is gated. What has not been
  run is the ten-seed noise control itself.
