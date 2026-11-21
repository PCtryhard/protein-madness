# PLAN

Team of 6. Hackathon: Basel, Saturday 21 – Sunday 22 November 2026.
Fixed points: **start 08:30 Saturday** (challenge announced just before),
overnight side-challenges for hints and extra compute, **pitches 17:30 Sunday**.

Owner names are placeholders — fill them in at the kickoff and commit the
change. Module descriptions are in [ONBOARDING.md](ONBOARDING.md).

| # | Owner | Module | Branch |
| --- | --- | --- | --- |
| 1 | Max | integration, leaderboard, pitch | `main` + `module/report` |
| 2 | _TBD_ | Tier-1 + Tier-2 scoring (`src/score/`) | `module/score` |
| 3 | _TBD_ | novelty: MMseqs2 + Foldseek, reference DBs | `module/novelty` |
| 4 | _TBD_ | generation: LigandMPNN / ProteinMPNN | `module/generate-mpnn` |
| 5 | _TBD_ | generation: RFdiffusion / motif scaffolding | `module/generate-rfd` |
| 6 | _TBD_ | compute: Euler + Modal, fold backends | `module/compute` |

---

## Before Saturday (this week)

The weekend is won or lost here. **Every hour spent now is worth three on
Saturday**, because on Saturday the bottleneck is wall-clock, not ideas.

| When | What | Owner | Done when |
| --- | --- | --- | --- |
| Mon–Tue | Everyone: clone, run the smoke test, read your module | all | each person has seen `SMOKE TEST PASSED` locally |
| Tue | Euler accounts work; `env/euler_setup.sh` runs clean | 6 | trivial GPU job prints the device name on a 2080Ti |
| Tue | Modal account + `env/modal_image.py` image **pre-built** | 6 | `modal run env/modal_image.py` prints the GPU name |
| Wed | Weights fetched on Euler (`$SCRATCH`) and Modal | 6 | `MANIFEST.md` lists real paths, not plans |
| Wed | ESMFold runs on Euler within 11 GB, fp16 + chunking | 6 | toy enzyme folded on a 2080Ti, mean pLDDT recorded |
| Wed–Thu | MMseqs2 + a reference DB; Foldseek + PDB DB | 3 | `seq_novelty` and `struct_novelty` populate for the toy enzyme |
| Thu | LigandMPNN + ProteinMPNN produce real designs on the toy enzyme | 4 | `designs.fasta` with >50 designs, catalytic residues intact |
| Thu | RFdiffusion motif-scaffolding path runs end to end | 5 | 8 backbones held on the toy enzyme's catalytic motif |
| Thu | Tier-2 verified on real predicted structures, not just the reference | 2 | pLDDT, PAE, geometry all populated, no `tier2_notes` surprises |
| **Fri** | **Three-way smoke test agreement** | all | laptop == Euler == Modal, same toy enzyme, same numbers |
| Fri | Dry run: pretend a challenge, write a target.yaml, run everything, build the pitch | all | `PITCH.md` fully populated from a fake challenge in < 2 h |
| Fri night | Precompute MSAs for anything plausible, if we go near ColabFold | 3 | cached in `results/msa/` or explicitly dropped |

**The Friday dry run is not optional.** It is the only thing that tells us how
long Saturday actually takes.

---

## Saturday

| Time | What | Owner | Checkpoint |
| --- | --- | --- | --- |
| 08:30 | Challenge announced. **Nobody writes code for 30 minutes.** Read it together, out loud. | all | one paragraph in `configs/target.yaml:objective`, agreed by all six |
| 09:00 | Write `configs/target.yaml`: structure, chain, catalytic residues, geometry, design region, success metrics | 1+2 | — |
| 09:30 | **`python -m src.io.validate_target configs/target.yaml` exits 0** | 2 | **HARD GATE: nothing else starts until this passes** |
| 09:30 | In parallel: fetch/prepare the target structure, check it is what the brief means | 3 | — |
| 10:00 | CPU baseline running: `generate.mutate` + Tier-1 on the real target | 2 | first real `leaderboard.csv` rows exist |
| 10:00 | GPU paths warming: ESMFold on Euler, Boltz on Modal, with the real target | 6 | one real design folded by 11:00 |
| 11:00 | **Checkpoint 1: we have designs and scores for the real target.** Even bad ones. | all | leaderboard has >10 real rows |
| 11:00–14:00 | Generation at scale: LigandMPNN on the real backbone; RFdiffusion if the challenge is a scaffolding problem | 4+5 | — |
| 11:00–14:00 | Novelty DBs pointed at the real target's family | 3 | — |
| 14:00 | **Checkpoint 2: the full pipeline has run end to end on the real target.** | all | all three figures exist with real data |
| 14:00–18:00 | Scale up what is working. Kill what is not. | all | — |
| 18:00 | **Checkpoint 3: pick the approach.** Whatever is not working by now is abandoned, not debugged. | all | one sentence in `DECISIONS.md` |
| 18:00–20:00 | Dinner + side-challenges (hints, extra compute). Send one person, not three. | 3 | — |
| 20:00–24:00 | Big generation + folding runs queued. Array jobs on Euler, Modal for anything over 11 GB. | 4+5+6 | queue is full overnight |
| 24:00 | **Checkpoint 4: overnight runs are submitted and will survive us sleeping.** | 6 | `squeue` shows the array; Modal jobs detached |

**Sleep in shifts.** Two people awake at a time, rotating. A team that is
awake for 30 hours pitches badly.

---

## Sunday

| Time | What | Owner | Checkpoint |
| --- | --- | --- | --- |
| 08:00 | Collect overnight results, rebuild the leaderboard | 1 | leaderboard reflects everything generated |
| 08:00–12:00 | Last generation round, informed by overnight scores | 4+5 | — |
| 12:00 | **HARD FREEZE ON NEW GENERATION RUNS.** | all | **Nothing new is generated after 12:00. No exceptions, no "just one more".** |
| 12:00–14:00 | Fold and score everything already generated. Queue drains; nothing new enters it. | 2+6 | every design has a leaderboard row |
| 14:00 | **Checkpoint 5: the leaderboard is final.** | all | `results/leaderboard.csv` frozen, committed |
| 14:00–15:30 | `python -m src.report.pitch_assets` → every number and figure in `PITCH.md` | 1 | no `[NOT COMPUTED]` left that we could have computed |
| 15:30–16:30 | Write and rehearse the pitch. Out loud. Twice. With a timer. | all | runs to time |
| 16:30–17:00 | Rehearse the questions: "how do you know it's any good?", "what's your baseline?", "what would you test?" | all | every answer points at a number in the leaderboard |
| 17:00 | Laptop, adapter, PDF export, backup on a USB stick and in the cloud | 1 | figures open on someone else's machine |
| **17:30** | **Pitch** | all | — |

### Why the freeze at 12:00

Teams lose by generating until the last minute and presenting nothing legible.
Five and a half hours sounds generous and is not: folding a backlog, rebuilding
the leaderboard, regenerating figures and rehearsing twice fills it exactly.
A design generated at 14:00 cannot be folded, scored, checked and defended by
17:30 — so it cannot go in the pitch, so generating it was wasted compute that
also delayed the designs that *will* go in.

If someone argues for an exception at 12:00, the answer is in this file and it
is no.

---

## Failure modes and what we do instead

| If | Then |
| --- | --- |
| Euler is contended / queue is hours deep | Modal ($30 credit, A100s). `--backend modal`. Decide by 11:00, not 16:00. |
| Everything GPU jams | `generate.mutate` + Tier-1 is a legitimate submission on CPU. A systematic active-site scan with honest scoring beats a pile of unscored designs. |
| The challenge is a scaffolding problem, not a redesign | RFdiffusion motif-holding (owner 5) becomes the main path; `--suggest-contigs` gives a starting point in seconds. |
| The challenge gives no usable substrate | Drop Boltz, stay on ESMFold, say so in the pitch's limitations slide. |
| A metric looks too good | Assume a numbering bug first. Re-run `validate_target`, check `tier2_notes`, check the wild-type row scores sensibly. |
| We have no novelty DB by Saturday | Report `novelty_vs_parent` only, and say in the pitch that it measures distance from the wild-type, not from nature. Do not relabel it. |
| Somebody breaks `main` | `main` must always pass the smoke test. Revert first, debug on a branch second. |
