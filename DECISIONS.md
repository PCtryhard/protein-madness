# Decisions

Append-only. At hour 25 nobody will remember why a threshold is 0.7.

Format: date — decision — why. Newest at the bottom. Never edit an old entry;
add a new one that supersedes it and say so.

---

**2026-11-16 — Residue numbering is normalised to 1..N everywhere, with a sidecar map.**
Every structure passes through `src/io/structure.normalise`: one chain, insertion
codes stripped, renumbered from 1, original numbering preserved in a
`NumberingMap`. Config indices are written in the *reference* numbering and
translated once, in `src/io/target.load_reference`.
Why: getting residue numbering wrong is the classic way to silently poison every
downstream metric, and PDB files in the wild have offsets, insertion codes and
decoy chains. `tests/test_io.py` renumbers the toy enzyme hostilely (chain E,
+100 offset, insertion codes, a decoy chain placed first) and asserts the
resolved catalytic positions and geometry come out identical.

**2026-11-16 — Net charge and pI delegate to Biopython rather than our own Henderson-Hasselbalch.**
Our own implementation agreed with Biopython on GRAVY exactly but differed by
~0.1 pH unit on pI (different terminal-pKa convention).
Why: a judge can reproduce a Biopython number in one line. Two implementations
of the same quantity is one too many, and ours was not better.

**2026-11-16 — A metric whose inputs are missing is NaN with a note, never 0.0.**
`catalytic_integrity` with no catalytic residues configured is NaN, not 1.0.
Missing PAE is NaN with `tier2_notes="pae:not produced by this fold backend"`.
Database-backed novelty raises `NoveltyUnavailable` rather than substituting the
parent-identity metric.
Why: a plausible-looking stub that silently returns wrong numbers is worse than
nothing, and 0.0 or 1.0 are both numbers a filter will act on.

**2026-11-16 — `novelty_vs_parent` and `seq_novelty` are different columns and never merged.**
`novelty_vs_parent` = 1 − identity to the wild-type (always available).
`seq_novelty` = 1 − max identity to the nearest hit in a reference DB (needs MMseqs2).
Why: they are different claims. "Far from the wild-type" is not "far from
anything natural", and a judge will ask which one we measured.

**2026-11-16 — Default fold backend is ESMFold; Boltz-2 on Modal is the ligand-aware escape hatch.**
Decided with Max.
Why: ESMFold needs no MSA, so it works on Euler compute nodes, which have
restricted outbound internet and would fail any MMseqs2-web-API backend. It fits
Euler's 11 GB RTX 2080Ti in fp16 with chunked attention (fp32 does not fit).
Boltz-2 generally wants more than 11 GB, so it runs on Modal's A100s — it earns
its place by co-folding the substrate, which is the enzyme-relevant signal
ESMFold cannot give us. ColabFold was rejected as default because its MSAs must
all be precomputed before Friday night and we do not know the target yet.

**2026-11-16 — The 11 GB VRAM ceiling is a config value (`compute.max_vram_gb`), not an assumption in code.**
`compute.max_residues` makes oversized jobs refuse up front instead of OOMing on
a shared GPU, and `compute.allow_bf16: false` is a hardware fact about Euler's
sm_75 cards, not a preference.
Why: 32 GPUs shared across ~70 participants. An OOM at hour 20 costs us a slot
in the queue as well as the run.

**2026-11-16 — LigandMPNN and ProteinMPNN revert positions outside the design region.**
After sampling, any position not in `design_region` is set back to the parent
residue.
Why: the target config states which positions we are allowed to change. A design
that quietly changed others is not the thing we claim to have made, and the
claim is what the pitch rests on.

**2026-11-16 — RFdiffusion has no default `contigs` and refuses to run without one.**
`--suggest-contigs` prints a starting point built from the catalytic residues,
clearly labelled as a starting point.
Why: a wrong contig string scaffolds the wrong motif and nothing downstream
would notice.

**2026-11-16 — Leaderboard thresholds live in `configs/pipeline.yaml`, never in code.**
Current hard filter: `catalytic_integrity == 1.0`. Soft: `mean_plddt >= 70`,
`unpaired_cys_delta <= 0`, `max_homopolymer_run <= 4`.
Why: "why 70?" must have exactly one answer and one place to change. 70 is the
conventional pLDDT confidence cut; it is soft, not hard, because on an unknown
target we do not yet know the distribution. Catalytic integrity is hard because
a design that lost a catalytic residue is dead regardless of every other number.

**2026-11-16 — Ranking is a weighted sum of z-scores computed over hard-filter survivors only.**
Why: a flood of dead designs would otherwise distort the scale and make the
survivors look uniformly excellent. A design missing one metric scores 0 for
that term (the population mean) and records the fact in
`score_metrics_missing`, so a design is never rewarded for not being measured.

**2026-11-16 — The CPU-only leaderboard is deliberately not meaningfully ranked.**
With no fold stage, every ranking term is absent and `src/report/leaderboard.py`
logs "no ranking terms available: 'score' is 0 and 'rank' is arbitrary".
Why: you cannot rank enzyme designs without folding them. Inventing a
sequence-only ranking would produce a leaderboard that looks authoritative and
means nothing. The warning is the correct output.

**2026-11-16 — Figures are built against generated fake data, before any real design exists.**
`src/report/fake_data.py` writes a leaderboard-shaped table; every row carries
`is_fake=True`, every figure drawn from it is stamped FAKE DATA, and
`pitch_assets.py` refuses such a table without `--allow-fake`.
Why: on Sunday we change an input path, not plotting code. The stamp exists so
a dry-run figure can never walk into the pitch deck.

---

**2026-11-21 — The pipeline is replaced by the motif-anchored one in `PIPELINE.md`; the whole previous pipeline moves to `outdated/`.**
`src/`, `tests/`, `scripts/`, `configs/`, `notebooks/`, `BRIEF.md`, `EVOLVE.md`,
`DESIGN_RATIONALE.md` and the Euler/Modal env files are now under `outdated/`.
The new pipeline is one path — RFdiffusion partial diffusion, LigandMPNN,
Boltz-2, hydrogen placement, scoring — with a per-scaffold elitist hill-climb
over two branches.
Why: the old pipeline ran and produced the baseline in `data/No_mutation_run/`,
but it carried four fold backends, four compute backends and three scoring
tiers, most of which were never exercised. It was too complex to build on. The
new one is deliberately smaller and is the base to extend, not a reduction in
ambition. `data/`, `runs/`, `RUNLOG.md` and this file stay live.

**2026-11-21 — Residue numbering is tracked by four explicit maps, not normalised to 1..N.**
Supersedes the 2026-11-16 entry on normalising to 1..N with a sidecar map. A
hydrogen bond is declared once against seed numbering, keeps its string id
forever, and its atoms are resolved through `Provenance`: seed → RFdiffusion
output (from the `.trb` pickle), → LigandMPNN (identity, asserted), → Boltz-2
(1..N in submitted order, computed from the exact submitted string), plus a
ligand atom-name map.
Why: normalising means every stage has to agree on one numbering, and the stages
do not — RFdiffusion renumbers, Boltz-2 renumbers again from 1. Carrying the
maps with the design makes each stage's renumbering explicit and assertable.
The old rule's purpose is kept: nothing is inferred by proximity or alignment,
and a residue that cannot be traced stops the run.

**2026-11-21 — The angle term defaults to acceptor-side directionality for every bond, not the split by donor type.**
`scoring.angle.rule: acceptor_side_always`. Both rules are implemented and the
switch is one config line; `config.py` refuses `split_by_donor_type` combined
with `protonation.engine: none`.
Why three reasons, in order. (1) `RUNLOG.md` §3 shows the old scorer already
measured `CG-OD2···N3` — acceptor-side — so the new default keeps the numbers
comparable. (2) Both 5FQJ bonds have chemically fixed hydrogens (benzotriazole
N3-H, imidazole NE2-H), so the split rule would measure `D-H···A` on both and
require the protonation stage for both; its advantage on rotatable donors buys
nothing here while adding an OpenMM/PDBFixer dependency. (3) `PIPELINE.md` calls
it the option that loses least to tooling artefacts. This is one of the two
questions that document left open; the other, `eval_replicates`, is still open
and is phase 0's to answer with data.

**2026-11-21 — The ligand heavy-atom count is a lookup table that raises on an unknown code, not an estimate.**
`KNOWN_LIGAND_ATOM_COUNTS` in `src/stages/boltz.py`. 6NT is 12, counted from
`data/5A_5FQJ.pdb` and cross-checked against a real Boltz-2 prediction by a test.
Why: `boltz.max_tokens` is the only guard against an OOM that kills a worker
mid-search, and on Linux there is no CPU fallback. A guessed input makes the
guard worthless, so an unrecognised ligand refuses rather than estimating.

**2026-11-21 — `resume` and `protonation.engine: reduce` refuse rather than half-work.**
Both raise with an explanation of what is missing.
Why: `resume` would have to rebuild a scaffold's patience counter from stored
rows, and a resume that silently restarts the patience clock changes where the
search stops. `reduce` writes its whole PDB to stdout while `run_command` keeps
only the tail, so wiring it up naively would write a truncated structure and
every angle measured on it would be wrong. A loud refusal costs an hour; either
of those costs a run.

---

**2026-11-21 — The target is seven generated files, one per `data/0N` variant, not one file with seven seeds.**
`scripts/build_targets.py` writes `config/targets/01_his_asp.yaml` through
`07_gln_glu_his.yaml`; `tests/test_variants.py` runs it with `--check` so the
generated files cannot drift from the generator.
Why: the seven differ in which residues are fixed and which hydrogen bonds are
scored, and the `Target` schema holds one motif and one bond list per file. They
could not be seeds of a single target even if we wanted that. Generating them
keeps the chemistry table (ASP→OD2/CG, GLU→OE2/CD, HIS→NE2/CD2, ASN→ND2/CG,
GLN→NE2/CD) in one place instead of copied seven times, which is exactly the
kind of thing that is wrong in one file out of seven.

**2026-11-21 — Ideal angles are taken from the crystal only where the acceptor is unmutated; elsewhere from hybridisation.**
Variant 01's two contacts are real 5FQJ geometry (2.734 Å/106.6°,
3.084 Å/126.6°). Everything else in 02–07 was modelled in without repacking:
`GLU229 OE2···N3` = 1.677 Å, `HIS261 NE2···O11` = 1.820 Å, `ASN291 ND2···N1` =
4.823 Å, `GLN291 NE2···N1` = 3.860 Å. Those get 120° (sp2 carboxylate, sp2
nitro oxygen) with a comment saying so.
Why: reading an ideal out of a clash would make the scorer chase an impossible
geometry. `tests/test_variants.py` asserts the clash distances as they are, so
if the motif files are ever repacked the test fails and tells you to re-derive
the ideals from the new coordinates. The ligand-side ideal (126.6° at `N1`) is
the same in every variant by construction, because the acceptor is a ligand
atom and its lone-pair direction does not depend on which residue donates.

**2026-11-21 — The distance gate has a clash end, `scoring.distance.clash`, default 2.2 Å.**
Supersedes nothing; it adds to the spec, which named only `distance.fail`.
Why: the distance term is symmetric about the ideal, so 1.7 Å and 4.1 Å score
the same zero, but only the long one gated. A design with an atom driven through
the ligand would therefore score zero on geometry and still be ranked on its
confidence term, which carries weight 0.7. Three of the seven shipped motif
files contain sub-2 Å contacts, so this was not hypothetical. Setting `clash`
very small recovers the spec's one-sided behaviour.

**2026-11-21 — All three model stacks must live on one machine; the loop is not splittable.**
`cluster/preflight.py` is the tool that decides which machine. `cluster/README.md`
states the constraint and what to do in each outcome.
Why: `RUNLOG.md` records RFD3 and LigandMPNN on AWS and Boltz-2 on Euler, which
worked when generation and folding were separate batch steps with a human
moving files. The search is a closed loop — a cycle cannot choose its next
frontier until its designs are scored — so that split would mean a cluster round
trip per cycle, twelve times per variant. If neither cluster can host all three,
the pipeline needs a resumable two-phase mode it does not have, and that is real
work rather than a config flag.

**2026-11-21 — `.gitattributes` forces LF on `*.sh` and `*.sbatch`, and preflight checks it.**
Why: `RUNLOG.md` §4 records `sbatch` rejecting a submission script for CRLF line
endings, with an error that does not mention line endings. The repo is edited on
Windows, so without this every new script is born broken. The check is scoped to
`cluster/`, `scripts/` and `env/` on purpose — `data/` holds an archived sbatch
file that does have CRLF, and flagging it would train people to ignore the check.

---

**2026-11-21 — The noise control uses 5 seeds across several sequences, not 10 on one.**
`scripts/noise_control.py --seeds 5 --sequences 3`.
Why: the precision of an SD estimate goes as sigma/sqrt(2(n-1)) — about 35% at
n=5 against 24% at n=10 — which argues for 10 only if you measure one sequence.
The quantity that matters is the MEAN within-design SD over many designs, and
averaging over m designs tightens it by sqrt(m). Five seeds on three sequences
(15 calls) estimates the noise floor better than ten on one (10 calls), and it
also shows whether the noise is sequence-dependent, which ten seeds on one
sequence cannot. Five is also this project's existing convention: `RUNLOG.md` §4
calls it "the five-seed rule" and `data/No_mutation_run/` holds exactly five
seeds per design, so staying at five keeps the numbers comparable.

**2026-11-21 — Measured from the existing 5-seed data: both angle terms are prediction noise.**
`python scripts/noise_control.py --from-csv` over 32 designs x 5 seeds gives
within/between ratios of 0.93 (base angle), 1.18 (donor angle), 0.79 (donor
distance), 0.38 (base distance), 0.41 (iPDE).
Why it matters: a ratio at or above 1 means Boltz-2's seed-to-seed variation on
the same sequence is as large as the variation between genuinely different
designs, so the term carries no information about design quality. The
recommendation is `run.eval_replicates: 3` and `run.patience: 4` before any real
search. `scoring.geometry_weights.angle: 0.40` is marked "fixed by spec" and has
NOT been changed, but this data does not support it and that should be revisited
with the user rather than silently. Caveat recorded in the tool's own output:
this is the previous pipeline's run, whose ligand drift inflates the
between-design spread, so the ratios are if anything optimistic.

**2026-11-21 — `run.max_cycles: 0` means seed-only, and is the shape of the first real run.**
`config/pipeline.small.yaml`: 10 backbones x 8 sequences, no expansion. Checked
in `ScaffoldSearch.seed_frontier` rather than in the runner's loop, because the
loop would otherwise run one whole cycle before the stopping condition was
consulted.
Why: 80 Boltz-2 calls per variant is about an hour and exercises the entire
forward pass at scale, producing a leaderboard. A full search with the default
profile is 224 calls per cycle for up to 12 cycles. Finding out that the forward
pass works should not cost a day.

**2026-11-21 — `build_targets.py` prefers ideals measured in the seed, and says per bond where each came from.**
Three sources, printed as a table: `measured` (the seed's contact is a plausible
H-bond, 2.2-3.5 A, so its own angle is the target), `crystal` (not plausible,
but the acceptor exists unmutated in 5FQJ), `chemistry` (neither — hybridisation
value, with a warning).
Why: the refolded motif files are coming, and the whole update should be "drop
them in, re-run one command". Preferring the seed is right because the seed's
geometry is what RFdiffusion will be asked to reproduce. The plausibility window
is bounded by `tolerated` rather than `fail` because a 3.9 A contact sits inside
the fail gate and still scores zero, so its angle is not a target either. The
script also flags a plausible distance with an implausible direction (an
acceptor-side angle outside 90-180 deg), which is what a sidechain translated
but not repacked looks like, and would otherwise install a nonsense target
silently.

**2026-11-21 — The newest `inputs/` subdirectory wins, by modification time.**
`data/0N/inputs/<subdir>/`, chosen by mtime with the name as tiebreak, always
printed; `--inputs NAME` overrides.
Why: lexicographic order would prefer the shipped `5fqj-6cf4761` over a new
`01_refold`, silently generating targets from the old clashing files. Mtime
matches what anyone dropping a folder in means. Nothing is deleted, so the
originals stay for comparison.

---

**2026-11-21 — `DesignStore` opens sqlite with `check_same_thread=False` and serialises every statement behind an RLock.**
Found by writing a test that simulates the real threading model: two workers
inserting at once.
Why this was urgent: `sqlite3.connect()` defaults to `check_same_thread=True`,
and the GPU worker pool calls `store.insert()` from its worker threads, not from
the thread that opened the store. The pipeline would have raised
`ProgrammingError` on the very first design — on the cluster, after the first
RFdiffusion call — and nothing in the suite exercised two threads writing, so
nothing caught it. `check_same_thread=False` alone is not enough, because a
sqlite connection is then not safe for concurrent use, hence the lock.
`tests/test_live.py::test_the_store_is_usable_from_another_thread` is the
regression.

**2026-11-21 — The run directory is readable while the run is going: designs.csv, sequences.fasta, progress.txt, appended per design.**
`src/live.py`. Every append is taken under a lock, then flushed AND fsynced.
Why: a multi-hour search whose output only becomes readable at the end tells you
nothing about whether it is working, and anything that has to reach another
machine must be copyable mid-run. The fsync is what makes a row safe for a
reader on another host or an rsync to see — without it a reader can catch a
truncated line. A design takes minutes, so an fsync per row costs nothing.
The CSV carries a column per hydrogen bond, names the bond that gated a design
rather than only flagging it, and gives errored designs a row with the error on
it. Structures needed no change: the stage wrappers already write to their final
paths rather than to temporary directories, so a backbone or a prediction is
complete on disk while the next design is still folding.

**2026-11-21 — `designs.csv` and `sequences.fasta` are not gitignored; everything else under `runs/` is.**
Why: they are one row per design, they are the result, and the previous pipeline
committed `leaderboard.csv` the same way. `progress.txt` is transient and is
ignored, as are the structures, the sqlite files and the slurm logs.
