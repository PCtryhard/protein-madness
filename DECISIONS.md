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
