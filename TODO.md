# TODO — everything that is stubbed, and exactly what is missing

Rule of engagement #1: no fake implementations. Everything listed here raises
`NotImplementedError` (or records a TODO instead of downloading) rather than
returning a plausible wrong value. Each entry says what is missing, not just
that something is.

Owner column is empty where nobody has claimed it.

---

## Compute backends (`src/backends/`)

| # | Where | Missing | Owner |
|---|---|---|---|
| B1 | `src.backends.modal.ModalBackend.submit` | A generic "run this repo command on Modal" path. Needs three things: (a) the repo inside the image (`Image.add_local_dir(..., copy=True)`), (b) a remote command-runner function in `env/modal_image.py` to exec it, (c) the weights-storage decision in B2. Workaround that *is* implemented: `ModalBackend.spawn("<fn>")` / `.call("<fn>")` against a deployed function. | |
| B2 | `src.backends.modal.ModalBackend.upload_weights` | **A decision, not code.** Weights on Modal: a `modal.Volume` mounted at `/weights` and populated once, or baked into the image? Volume = slower first read, small images, one upload. Baked = fast cold start, tens-of-minutes rebuild whenever a checkpoint changes. Until this is decided, `scripts/fetch_weights.sh --backend modal` writes to `${ENZYME_WEIGHTS:-/weights}`, which is container-local and evaporates. | |
| B3 | `src.backends.tamarind` (every automated call) | A documented Tamarind.bio API, and the hackathon event code (arrives during the event). Deliberately a manual web-UI fallback: `instructions()` prints the runbook and `missing_outputs()` checks which `<design_id>.pdb` files came back. Do not automate this against a guessed API. | |
| B4 | `src.backends.euler.EulerBackend` — `--gres=gpumem` units | The ETH docs give the option as `--gres=gpumem:<size>[units]` but no example value, so the accepted suffix is unverified. Default is `g` (`gpumem:10g`), overridable via `EulerBackend(gpumem_units=...)`. **Verify with one real submission on Euler**; a GRES parse error is the symptom. | |
| B5 | `src.backends.euler.EulerBackend` — array size limit | Slurm's `MaxArraySize` on Euler is not documented and not known. If a 500-task array is rejected, split it. Use `submit_array(..., throttle=N)` regardless: 32 GPUs are shared with ~70 people. | |
| B6 | `src.backends.euler.EulerBackend` — `--account` | Euler GPU nodes are shareholder-only (`-A/--account=<share_name>`). We do not know which share the hackathon uses. Pass `get_backend("euler", account="...")` or set `SBATCH_ACCOUNT`. Submission fails with a clear Slurm error if it is wrong. | |
| B7 | `src.backends.local.LocalBackend` — job ids | Job state is in memory, so ids do not survive the submitting process. Fine for the smoke test; if we ever need to re-attach to a local run, it needs a state file. Low priority. | |

## Environment / compute setup

| # | Where | Missing | Owner |
|---|---|---|---|
| E1 | `env/modal_image.py` | **Not missing — but not yet built.** The first image build takes tens of minutes. `modal deploy env/modal_image.py` must be run before Saturday, not on Saturday. Nobody has confirmed it ran. | |
| E2 | `env/modal_image.py` | Images for the actual model stacks (esmfold / boltz / mpnn). The committed image is torch + numpy and the trivial `gpu_check` function only. Deliberate: one image per stack so a change to one does not rebuild the others. | |
| E3 | `env/euler_setup.sh` | Unverified on the real cluster: `stack/2024-06` + `python/3.12.8` + `cuda/12.1.1` are documented to exist, but the combination has not been loaded by us. Also unverified: that `torch==2.5.1+cu121` actually runs on sm_75 on these nodes. **Run `env/euler_setup.sh` and then the `srun` one-liner it prints.** | |
| E4 | three-way smoke test | The definition of done (README) is the toy enzyme producing the **same numbers** on laptop, Euler and Modal. Not done, not attempted. | |

## Weights and databases (`scripts/fetch_weights.sh`, `MANIFEST.md`)

| # | Component | Missing | Owner |
|---|---|---|---|
| W1 | Boltz-2 | No verified download-only command. `boltz` fetches weights into its `--cache` directory on the first prediction; the HuggingFace repo id is **not** recorded here because we have not verified it. Warm the cache with a dummy prediction on a machine with internet, then re-run the fetch script to record the paths. Boltz-2 wants >11 GB → Modal, not Euler. | |
| W2 | LigandMPNN | Relies on the upstream `get_model_params.sh`. If upstream moves it, the script records a TODO instead of guessing a `files.ipd.uw.edu` URL. Also unverified: that `ligandmpnn_v_32_010_25` (the checkpoint named in `configs/pipeline.yaml`) is among what the script downloads. | |
| W3 | RFdiffusion | Relies on the upstream `scripts/download_models.sh`. The per-checkpoint URLs are hash paths we are not confident about and will not hardcode. | |
| W4 | MMseqs2 / Foldseek reference DBs | Opt-in (`--only mmseqs_db`, `--only foldseek_db`) and not fetched by default: tens of GB each. Until they exist, `score.tier1.novelty.enabled` and `score.tier2.foldseek.enabled` stay `false` — Tier-1 raises rather than writing a fabricated novelty column. Which reference DB we pitch novelty against is also a **decision** (UniRef50? a PDB-seq DB? the challenge's own set?) and belongs in `DECISIONS.md`. | |
| W5 | ESMFold | Fetched from HF `facebook/esmfold_v1`, which we are confident is the self-contained repo `transformers.EsmForProteinFolding` loads. Unverified by us on this machine. Needs `huggingface_hub`, which is not in `env/requirements.txt`. | |

## Fold backends (`src/fold/`)

| # | Where | Missing | Owner |
|---|---|---|---|
| F1 | `src.fold.colabfold` | **A decision about where MSAs come from.** ColabFold needs them and the MMseqs2 web API is unreachable from Euler compute nodes without `module load eth_proxy` (and is bandwidth-limited even then). Either precompute MSAs on the login node into `results/msa/`, or run ColabFold only on Modal. Raises until then: a structure built from a failed or truncated MSA looks fine and makes every downstream confidence number wrong. | |
| F2 | `src.fold.chai` | A verified Chai-1 API and output layout. Nobody has run it. Boltz-2 already covers ligand co-folding with a verified CLI, so this is only worth doing if Boltz is unavailable or clearly worse on the real target. | |
| F3 | `src.fold.boltz` — npz array keys | The Boltz docs name the output files (`plddt_<stem>_model_0.npz`, `pae_<stem>_model_0.npz`) but **not the array keys inside them**. `_read_npz_matrix` takes the single array when there is one and otherwise matches by name, raising rather than guessing between several. Verify against one real run. | |
| F4 | `src.fold.boltz` — is PAE written by default? | The docs list `pae_*.npz` in the output tree but also document `--write_full_pae` as default-False. We pass `--write_full_pae` explicitly. Confirm a PAE file actually appears. | |
| F5 | `src.fold.esmfold` | Never run with real weights. The fp16 + `chunk_size` path is the thing that must fit 11 GB; that it does is from the ESMFold docs, not from our own measurement. | |

## Scoring

| # | Where | Missing | Owner |
|---|---|---|---|
| S1 | `src.score.tier3.minimise_and_rmsd` | An OpenMM force-field/solvent decision, and `openmm` is not a pinned dependency. Deliberately last: Tier-2 self-consistency RMSD asks a similar question far more cheaply. | |
| S2 | `src.score.tier3.per_residue_energies` | **PyRosetta needs a licence** that nobody has set up. Adding a licensed dependency is one of the three things the brief says to stop and ask about — ask Max before starting. | |
| S3 | `src.score.structure.ligand_pose_metrics` — `ligand_rmsd` | Only computed when the design's ligand atom names match the reference's exactly; otherwise left NaN with a warning. A real atom-name mapping (or a graph match on the molecule) is needed for co-folded ligands whose naming differs. Contacts, burial and minimum catalytic distance work regardless. | |
| S4 | Tier-2 on real predicted structures | Every Tier-2 test runs against the reference structure or a perturbed copy of it. None has run against an actual ESMFold/Boltz output, where the B-factor column, residue numbering and any ligand come from the model rather than from us. **This is the most likely place for a silent numbering bug to survive.** | |
| S5 | `catalytic_geometry` ideals | `configs/target.example.yaml` sets `ideal_distance: 7.05` from the toy structure itself, so the toy's deviation is only config rounding. For the real target, decide deliberately whether each ideal comes from the reference structure (leave `ideal_*` out) or from the challenge brief / mechanism (set it). A wrong ideal makes every design look equally bad. | |

## Pipeline and reporting

| # | Where | Missing | Owner |
|---|---|---|---|
| P1 | CPU-only leaderboard ranking | With no fold stage there are no ranking terms, so `score` is 0 and `rank` is arbitrary — logged loudly. This is correct, not a bug: you cannot rank enzyme designs without folding them. Do not "fix" it by inventing a sequence-only ranking. | |
| P2 | `scripts/smoke_test.sh --gpu` | Never run. Needs a CUDA box with ESMFold weights. | |
| P3 | `src.generate.ligandmpnn` / `proteinmpnn` / `rfdiffusion` | Real wrappers, never executed — no weights on this machine. The CLI flags are from each project's documented interface, not from a run. Expect at least one flag mismatch on first contact. | |
| P4 | Reference ligand in the target config | `substrate.reference_ligand` is threaded into Tier-2 and Boltz but no toy target exercises it (1AKI has no ligand). The ligand path is tested only with a synthetic 2-atom ligand in `tests/test_score_tier2.py`. | |
