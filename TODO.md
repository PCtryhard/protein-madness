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
