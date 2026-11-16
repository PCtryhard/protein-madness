# MANIFEST — where every model weight and database lives

One table per backend. Nothing in this list is in git: `.gitignore` excludes
`weights/` and every checkpoint extension. If a path below is empty on your
machine, run `scripts/fetch_weights.sh` — it fills it and **appends what it
actually did** to the download log at the bottom of this file.

```
bash scripts/fetch_weights.sh --list
bash scripts/fetch_weights.sh                   # default model set, auto backend
bash scripts/fetch_weights.sh --only esmfold
bash scripts/fetch_weights.sh --backend euler   # -> $SCRATCH/weights
```

---

## ⚠ THE INTERNET GOTCHA — read this before you debug a hung job

**ETH Euler compute nodes have no outbound internet.** They can reach the ETH
network and nothing else, unless the job does `module load eth_proxy`, and that
proxy is bandwidth- and connection-limited.
(https://docs.hpc.ethz.ch/connections/firewall/)

Consequences, all of which have cost people a night:

1. **The MMseqs2 / ColabFold MSA web API works on the login node and fails
   inside a job.** Some clients do not fail loudly — they retry until the
   walltime kills the task. So either
   * precompute MSAs on the **login node** into `results/msa/` and set
     `fold.colabfold.msa_mode: precomputed`, or
   * stay on **single-sequence** models on Euler (ESMFold is the default fold
     backend precisely because of this; Boltz is also single-sequence).
2. **Download weights on the login node only.** `scripts/fetch_weights.sh`
   warns if `$SLURM_JOB_ID` is set.
3. **Nothing may hit HuggingFace at run time.** Point the model loaders at the
   local paths below. Setting `HF_HUB_OFFLINE=1` in a job turns a silent hang
   into an immediate error, which is what we want.
4. Modal containers *do* have internet, and Tamarind is a browser. The gotcha is
   Euler-specific — which is exactly why the one three-way-identical smoke test
   (laptop / Euler / Modal) is the definition of done.

**`$SCRATCH` is purged after ~2 weeks and is not backed up.** Weights there will
vanish. Re-run the fetch script; do not move them to `$HOME` (50 GB and a
500,000-file quota that a model stack will eat).

---

## Laptop / development machine

Weights root: **`<repo>/weights/`** (gitignored).

| Component | Path under the root | Size (approx) | How it gets there |
|---|---|---|---|
| ESMFold | `esmfold_v1/` | 2.6 GB | `fetch_weights.sh --only esmfold` (HF `facebook/esmfold_v1`) |
| ProteinMPNN | `ProteinMPNN/` | 100 MB | git clone; checkpoints are committed upstream |
| LigandMPNN | `LigandMPNN/model_params/` | 500 MB | git clone + upstream `get_model_params.sh` |
| RFdiffusion | `RFdiffusion/models/` | 3 GB | git clone + upstream `scripts/download_models.sh` |
| Boltz-2 | `boltz/` | TODO | `boltz` CLI fills its `--cache` on first prediction |
| MMseqs2 ref DB | `mmseqs/uniref50` | tens of GB | `fetch_weights.sh --only mmseqs_db` (opt-in) |
| Foldseek PDB DB | `foldseek/pdb` | tens of GB | `fetch_weights.sh --only foldseek_db` (opt-in) |

A laptop realistically holds ESMFold + the MPNNs. The reference databases are
an Euler job; the Tier-1 novelty metric stays disabled locally
(`score.tier1.novelty.enabled: false`) rather than reporting a fabricated column.

## ETH Euler

Weights root: **`$SCRATCH/weights`** = `/cluster/scratch/$USER/weights`.
Environment: **`$HOME/enzyme-hack-env`**, created by `env/euler_setup.sh`.
Every sbatch script written by `src/backends/euler.py` sources
`$HOME/enzyme-hack-env/activate.sh`, which exports
`ENZYME_WEIGHTS=$SCRATCH/weights`.

Same table as above, rooted at `$SCRATCH/weights`. Euler-specific notes:

* **Never** in `$HOME`: 50 GB and 500,000 files, and Conda/weights blow through
  the file count. (https://docs.hpc.ethz.ch/hardware/storage/)
* GPUs are RTX 2080Ti: **sm_75, 11 GB, no bf16**. That is
  `compute.max_vram_gb: 11.0` in `configs/pipeline.yaml` and
  `src.backends.base.check_fits` refuses anything larger before submission.
* Boltz-2 and Chai-1 generally want more than 11 GB → they are Modal jobs. Do
  not download their weights to `$SCRATCH` expecting to run them here.
* `module load eth_proxy` is in the generated `activate.sh`, so a stray download
  inside a job fails fast instead of hanging.

## Modal

Weights root: **not decided yet — see TODO.md.** A `modal.Volume` mounted at
`/weights` and populated once is the obvious answer; baking weights into the
image is the alternative (faster cold start, slower and larger builds). Nobody
has made the call, so `scripts/fetch_weights.sh --backend modal` writes to
`${ENZYME_WEIGHTS:-/weights}` and `ModalBackend.upload_weights()` raises
`NotImplementedError` naming this decision.

What *is* set up: `env/modal_image.py` defines the pinned image and one GPU
function (`gpu_check`). **Pre-build it before Saturday** — the first build is
tens of minutes:

```
pip install modal && modal setup
modal deploy env/modal_image.py            # builds the image AND deploys the app
modal run env/modal_image.py::gpu_check    # prints torch.cuda.get_device_name()
```

## Tamarind.bio

No weights and no local paths: it is a web UI with 200 free jobs per
participant and an event code that arrives during the hackathon. There is no
public API we have seen, so `src/backends/tamarind.py` raises
`NotImplementedError` on every automated call and instead prints an operator
runbook. The only thing it needs back is one file per design:

```
<fold output dir>/<design_id>.pdb
```

`src.backends.tamarind.missing_outputs(design_ids, fold_out_dir)` says which
designs have not come back yet.

---

## Download log

Appended by `scripts/fetch_weights.sh`. Newest last.
