#!/bin/bash
# Download model weights and reference databases into a GITIGNORED directory,
# and append what landed where to MANIFEST.md.
#
#   bash scripts/fetch_weights.sh --list
#   bash scripts/fetch_weights.sh                       # default set, auto backend
#   bash scripts/fetch_weights.sh --only esmfold
#   bash scripts/fetch_weights.sh --backend euler       # -> $SCRATCH/weights
#   bash scripts/fetch_weights.sh --only foldseek_db --dry-run
#
# Rules this script enforces so nobody has to remember them at 3am:
#   * weights NEVER go in the repo. Locally they go to ./weights, which is in
#     .gitignore, and the script refuses to run if that line is missing.
#   * on Euler they go to $SCRATCH/weights, because $HOME is 50 GB / 500k files
#     and will not survive a model stack. $SCRATCH is purged after ~2 weeks.
#   * everything is idempotent: an entry that is already there is skipped.
#   * every success APPENDS a row to MANIFEST.md. The manifest is the answer to
#     "where are the weights on this machine", per machine.
#
# ############################################################################
# #  RUN THIS ON A EULER *LOGIN* NODE, NEVER INSIDE A JOB.                   #
# #  Euler compute nodes have no outbound internet unless `module load       #
# #  eth_proxy`, and that proxy is bandwidth-limited. A 3 GB download from    #
# #  inside an array job will hang until the walltime kills it.              #
# ############################################################################
#
# Honesty policy: where we are NOT confident of the exact public URL, this
# script uses the upstream project's OWN downloader (its repo is the source of
# truth for its URLs), or records a TODO in MANIFEST.md and TODO.md instead of
# guessing. A wrong URL that 404s is annoying; a wrong URL that silently fetches
# the wrong checkpoint invalidates every number we present.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$REPO_DIR/MANIFEST.md"

# The interpreter used for python-side downloads. Should be the ACTIVE
# environment's python (the venv from env/euler_setup.sh, or .venv locally),
# because that is where huggingface_hub lives. Override with PYTHON_BIN=...
PYTHON_BIN="${PYTHON_BIN:-${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

BACKEND="auto"
WEIGHTS_DIR=""
DRY_RUN=0
ONLY=""

# Default set: the model weights. The reference databases (mmseqs_db,
# foldseek_db) are hundreds of GB and are opt-in via --only.
DEFAULT_COMPONENTS="esmfold proteinmpnn ligandmpnn rfdiffusion boltz2"
ALL_COMPONENTS="$DEFAULT_COMPONENTS mmseqs_db foldseek_db"

log()  { printf '[fetch_weights] %s\n' "$*"; }
warn() { printf '[fetch_weights] WARNING: %s\n' "$*" >&2; }
die()  { printf '[fetch_weights] ERROR: %s\n' "$*" >&2; exit 1; }
run()  { if [ "$DRY_RUN" = 1 ]; then printf '[dry-run] %s\n' "$*"; else eval "$@"; fi; }

usage() {
    cat <<EOF
usage: bash scripts/fetch_weights.sh [options]

  --backend {auto,local,euler,modal}  where we are (default: auto-detect)
  --dir PATH                          override the weights directory
  --only NAME[,NAME...]               fetch just these components
  --list                              list components and exit
  --dry-run                           print what would happen, download nothing
  -h, --help                          this

components: $ALL_COMPONENTS
default set: $DEFAULT_COMPONENTS
EOF
}

# ---------------------------------------------------------------- arguments

while [ $# -gt 0 ]; do
    case "$1" in
        --backend) BACKEND="${2:-}"; shift 2 ;;
        --dir)     WEIGHTS_DIR="${2:-}"; shift 2 ;;
        --only)    ONLY="$(printf '%s' "${2:-}" | tr ',' ' ')"; shift 2 ;;
        --list)    printf '%s\n' $ALL_COMPONENTS; exit 0 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *)         usage >&2; die "unknown argument '$1'" ;;
    esac
done

# ---------------------------------------------------------------- where

if [ "$BACKEND" = "auto" ]; then
    if [ -n "${SLURM_CLUSTER_NAME:-}" ] || [ -n "${SCRATCH:-}" ]; then
        BACKEND="euler"
    elif [ -n "${MODAL_TASK_ID:-}" ] || [ -n "${MODAL_ENVIRONMENT:-}" ]; then
        BACKEND="modal"
    else
        BACKEND="local"
    fi
    log "auto-detected backend: $BACKEND"
fi

if [ -z "$WEIGHTS_DIR" ]; then
    case "$BACKEND" in
        euler)
            [ -n "${SCRATCH:-}" ] || die "--backend euler but \$SCRATCH is not set"
            WEIGHTS_DIR="$SCRATCH/weights" ;;
        modal)
            # On Modal this must be a mounted modal.Volume, not container-local
            # storage, or it evaporates with the container. See TODO.md.
            WEIGHTS_DIR="${ENZYME_WEIGHTS:-/weights}" ;;
        local)
            WEIGHTS_DIR="$REPO_DIR/weights" ;;
        *) die "unknown backend '$BACKEND' (auto|local|euler|modal)" ;;
    esac
fi

# Never, ever into the repo. The single sanctioned in-repo location is
# <repo>/weights, which .gitignore excludes; anything else inside the repo is
# refused outright rather than trusted to be ignored.
case "$WEIGHTS_DIR" in
    "$REPO_DIR/weights"|"$REPO_DIR/weights/"*)
        grep -qE '^[[:space:]]*weights/' "$REPO_DIR/.gitignore" 2>/dev/null \
            || die "$WEIGHTS_DIR is inside the repo and '.gitignore' has no 'weights/'
       line. Refusing: model weights must never be committed." ;;
    "$REPO_DIR"|"$REPO_DIR"/*)
        die "refusing to download weights into the repo at
         $WEIGHTS_DIR
       The only in-repo location allowed is '$REPO_DIR/weights' (gitignored).
       On Euler use --backend euler (\$SCRATCH/weights); elsewhere pass --dir
       with a path outside the repo." ;;
esac

if [ "$BACKEND" = "euler" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    warn "SLURM_JOB_ID is set: this looks like a COMPUTE NODE. Compute nodes have
         no outbound internet without 'module load eth_proxy', and the proxy is
         rate-limited. Run this on a login node instead."
fi

mkdir -p "$WEIGHTS_DIR"
WEIGHTS_DIR="$(cd "$WEIGHTS_DIR" && pwd)"
log "backend=$BACKEND  weights dir: $WEIGHTS_DIR"

# ---------------------------------------------------------------- manifest

HOSTNAME_S="$(hostname 2>/dev/null || echo unknown)"

manifest_init() {
    [ "$DRY_RUN" = 1 ] && return 0
    if [ ! -f "$MANIFEST" ]; then
        die "$MANIFEST does not exist. It is committed; restore it from git."
    fi
    if ! grep -q '^## Download log' "$MANIFEST"; then
        printf '\n## Download log\n\nAppended by `scripts/fetch_weights.sh`. Newest last.\n' >> "$MANIFEST"
    fi
}

# record <component> <path> <source> <note>
record() {
    local comp="$1" path="$2" source="$3" note="${4:-}"
    local size="unknown"
    if [ -e "$path" ]; then
        size="$(du -sh "$path" 2>/dev/null | cut -f1 || echo unknown)"
    fi
    log "RECORDED $comp -> $path ($size)"
    [ "$DRY_RUN" = 1 ] && return 0
    {
        printf -- '- **%s** | backend `%s` | host `%s` | %s\n' \
            "$comp" "$BACKEND" "$HOSTNAME_S" "$(date -Is)"
        printf -- '  - path: `%s` (%s)\n' "$path" "$size"
        printf -- '  - source: %s\n' "$source"
        if [ -n "$note" ]; then printf -- '  - note: %s\n' "$note"; fi
    } >> "$MANIFEST"
    return 0
}

# record_todo <component> <what is missing>
record_todo() {
    local comp="$1" what="$2"
    warn "$comp NOT fetched: $what"
    [ "$DRY_RUN" = 1 ] && return 0
    {
        printf -- '- **%s** | backend `%s` | host `%s` | %s\n' \
            "$comp" "$BACKEND" "$HOSTNAME_S" "$(date -Is)"
        printf -- '  - **TODO, NOT DOWNLOADED**: %s\n' "$what"
    } >> "$MANIFEST"
    return 0
}

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || { record_todo "$2" "'$1' is not on PATH ($3)"; return 1; }
}

have_python_pkg() {
    "$PYTHON_BIN" -c "import $1" >/dev/null 2>&1
}

# ---------------------------------------------------------------- components

# ESMFold. The HuggingFace repo `facebook/esmfold_v1` is self-contained: it is
# what `transformers.EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")`
# pulls, language-model weights included. ~2.6 GB.
fetch_esmfold() {
    local dest="$WEIGHTS_DIR/esmfold_v1"
    if [ -d "$dest" ] && [ -n "$(ls -A "$dest" 2>/dev/null)" ]; then
        log "esmfold: already present, skipping"; return 0
    fi
    if ! have_python_pkg huggingface_hub; then
        record_todo "esmfold" "python package 'huggingface_hub' is not installed in
    $PYTHON_BIN. Activate the project environment first, then:
      python -m pip install huggingface_hub
    and re-run. (Set PYTHON_BIN=... to point at a different interpreter.)"
        return 0
    fi
    log "esmfold: downloading facebook/esmfold_v1 -> $dest"
    run "'$PYTHON_BIN' -c \"from huggingface_hub import snapshot_download; \
snapshot_download('facebook/esmfold_v1', local_dir='$dest')\""
    record "esmfold" "$dest" \
        "HuggingFace repo \`facebook/esmfold_v1\` via huggingface_hub.snapshot_download" \
        "single-sequence, no MSA, so it is the Euler-safe fold backend. Must run fp16 + chunked attention to fit 11 GB."
}

# ProteinMPNN ships its checkpoints INSIDE its git repo
# (vanilla_model_weights/, soluble_model_weights/, ca_model_weights/), so the
# clone IS the download. ~100 MB.
fetch_proteinmpnn() {
    local dest="$WEIGHTS_DIR/ProteinMPNN"
    need_cmd git proteinmpnn "install git / module load git" || return 0
    if [ -d "$dest/.git" ]; then
        log "proteinmpnn: repo present, fetching updates"
        run "git -C '$dest' pull --ff-only"
    else
        run "git clone --depth 1 https://github.com/dauparas/ProteinMPNN '$dest'"
    fi
    record "proteinmpnn" "$dest" \
        "git clone https://github.com/dauparas/ProteinMPNN (checkpoints are committed in the repo)" \
        "checkpoints under vanilla_model_weights/, soluble_model_weights/, ca_model_weights/"
}

# LigandMPNN keeps its checkpoints OUT of the repo and ships its own downloader
# (get_model_params.sh). We run the upstream script rather than hardcoding URLs
# we have not verified: the repo is the source of truth for them.
fetch_ligandmpnn() {
    local dest="$WEIGHTS_DIR/LigandMPNN"
    need_cmd git ligandmpnn "install git / module load git" || return 0
    if [ -d "$dest/.git" ]; then
        log "ligandmpnn: repo present, fetching updates"
        run "git -C '$dest' pull --ff-only"
    else
        run "git clone --depth 1 https://github.com/dauparas/LigandMPNN '$dest'"
    fi
    if [ "$DRY_RUN" = 1 ]; then
        printf '[dry-run] run %s/get_model_params.sh %s/model_params\n' "$dest" "$dest"
        return 0
    fi
    if [ -f "$dest/get_model_params.sh" ]; then
        run "bash '$dest/get_model_params.sh' '$dest/model_params'"
        record "ligandmpnn" "$dest/model_params" \
            "upstream \`get_model_params.sh\` from https://github.com/dauparas/LigandMPNN" \
            "configs/pipeline.yaml expects checkpoint \`ligandmpnn_v_32_010_25\`; confirm that file is in model_params/"
    else
        record_todo "ligandmpnn" "cloned $dest but it has no get_model_params.sh.
    Upstream changed its download path. Read $dest/README.md and fetch the
    checkpoints by hand; do NOT guess a files.ipd.uw.edu URL."
    fi
}

# RFdiffusion checkpoints are large (~3 GB for the full set) and live behind
# URLs we have not verified. The repo ships scripts/download_models.sh; use it.
fetch_rfdiffusion() {
    local dest="$WEIGHTS_DIR/RFdiffusion"
    need_cmd git rfdiffusion "install git / module load git" || return 0
    if [ -d "$dest/.git" ]; then
        log "rfdiffusion: repo present, fetching updates"
        run "git -C '$dest' pull --ff-only"
    else
        run "git clone --depth 1 https://github.com/RosettaCommons/RFdiffusion '$dest'"
    fi
    if [ "$DRY_RUN" = 1 ]; then
        printf '[dry-run] run %s/scripts/download_models.sh %s/models\n' "$dest" "$dest"
        return 0
    fi
    if [ -f "$dest/scripts/download_models.sh" ]; then
        mkdir -p "$dest/models"
        run "bash '$dest/scripts/download_models.sh' '$dest/models'"
        record "rfdiffusion" "$dest/models" \
            "upstream \`scripts/download_models.sh\` from https://github.com/RosettaCommons/RFdiffusion" \
            "~3 GB for the full set. Base_ckpt.pt is the one the motif-scaffolding path needs."
    else
        record_todo "rfdiffusion" "cloned $dest but scripts/download_models.sh is absent.
    Read $dest/README.md for the current checkpoint URLs. Do NOT guess the
    files.ipd.uw.edu hash paths."
    fi
}

# Boltz-2: the `boltz` package downloads its own weights on first use into the
# directory given by --cache. There is no download-only command we have
# verified, so this creates the cache dir and records how to fill it rather
# than inventing a HuggingFace repo id.
fetch_boltz2() {
    local dest="$WEIGHTS_DIR/boltz"
    mkdir -p "$dest"
    if command -v boltz >/dev/null 2>&1; then
        record_todo "boltz2" "the 'boltz' CLI is installed but we have NOT verified a
    download-only command. Boltz fetches its weights on the first prediction
    into the --cache directory. Warm the cache ON A MACHINE WITH INTERNET with a
    one-residue dummy prediction:
      boltz predict <tiny.fasta> --cache '$dest' --out_dir /tmp/boltz_warmup
    then confirm the files landed in $dest and re-run this script to record them.
    Reminder: Boltz-2 generally wants MORE than 11 GB of VRAM, so it is a Modal
    job, not a Euler job."
    else
        record_todo "boltz2" "the 'boltz' package is not installed here (it is not in
    env/requirements.txt: the core env is CPU-only). Boltz-2 is the ligand-aware
    escape hatch and runs on Modal, not on the 11 GB Euler GPUs. Install it in
    the Modal image, then warm its --cache there."
    fi
    log "boltz2: cache directory prepared at $dest (still empty)"
    return 0
}

# MMseqs2 reference DB for the sequence-novelty metric. `mmseqs databases`
# is the documented, self-updating downloader, so no URLs are hardcoded.
# UniRef50 is ~10 GB compressed and much more unpacked; this is opt-in.
fetch_mmseqs_db() {
    local dest="$WEIGHTS_DIR/mmseqs/uniref50"
    local db_name="${ENZYME_MMSEQS_DB:-UniRef50}"
    need_cmd mmseqs mmseqs_db "conda install -c bioconda mmseqs2, or module avail mmseqs2 on Euler" || return 0
    if [ -f "$dest.dbtype" ]; then
        log "mmseqs_db: $dest already built, skipping"; return 0
    fi
    mkdir -p "$(dirname "$dest")"
    log "mmseqs_db: building $db_name -> $dest (tens of GB, slow)"
    run "mmseqs databases '$db_name' '$dest' '$WEIGHTS_DIR/mmseqs/tmp'"
    record "mmseqs_db" "$dest" "\`mmseqs databases $db_name\` (MMseqs2's own downloader)" \
        "set score.tier1.novelty.reference_db to this path and novelty.enabled: true"
}

# Foldseek reference DB for the structural-novelty metric (TM-score to the
# nearest PDB entry). `foldseek databases PDB` is the documented downloader.
fetch_foldseek_db() {
    local dest="$WEIGHTS_DIR/foldseek/pdb"
    need_cmd foldseek foldseek_db "conda install -c bioconda foldseek" || return 0
    if [ -f "$dest.dbtype" ]; then
        log "foldseek_db: $dest already built, skipping"; return 0
    fi
    mkdir -p "$(dirname "$dest")"
    log "foldseek_db: building PDB -> $dest"
    run "foldseek databases PDB '$dest' '$WEIGHTS_DIR/foldseek/tmp'"
    record "foldseek_db" "$dest" "\`foldseek databases PDB\` (Foldseek's own downloader)" \
        "set score.tier2.foldseek.reference_db to this path and foldseek.enabled: true"
}

# ---------------------------------------------------------------- run

manifest_init

COMPONENTS="${ONLY:-$DEFAULT_COMPONENTS}"
for comp in $COMPONENTS; do
    case " $ALL_COMPONENTS " in
        *" $comp "*) ;;
        *) die "unknown component '$comp'; known: $ALL_COMPONENTS" ;;
    esac
done

log "components: $COMPONENTS"
for comp in $COMPONENTS; do
    log "--- $comp"
    "fetch_$comp"
done

log "done. weights dir: $WEIGHTS_DIR"
[ "$DRY_RUN" = 1 ] || log "appended to $MANIFEST -- commit it so every machine's layout is recorded"
