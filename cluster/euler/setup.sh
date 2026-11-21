#!/bin/bash
# Set up this pipeline on ETH Euler. Idempotent: safe to re-run, and re-running
# is the intended way to pick up a changed env/requirements.txt.
#
#   ssh <ethz-user>@euler.ethz.ch
#   cd /cluster/home/$USER/protein-madness
#   bash cluster/euler/setup.sh                   # ~10 min the first time
#   source "$HOME/protein-madness-env/activate.sh"
#   python cluster/preflight.py                   # on the login node
#   srun --gpus=1 --gres=gpumem:10g --time=00:20:00 \
#        python cluster/preflight.py --require-gpu --write runs/hardware_notes.md
#
# Adapted from the previous pipeline's env/euler_setup.sh (now in
# outdated/env/), which was written against a working Euler account. The module
# stack and the reasoning below are carried over unchanged; what differs is the
# dependency set and that this pipeline needs Boltz-2 rather than ESMFold.
#
# ############################################################################
# #  LOUD WARNING: COMPUTE NODES HAVE NO OUTBOUND INTERNET.                  #
# #                                                                          #
# #  Consequences for THIS pipeline specifically:                            #
# #    - Boltz-2 downloads its weights on first use. That download MUST      #
# #      happen on the login node. Run one tiny prediction there to warm     #
# #      ~/.boltz before submitting anything, or the first array task will   #
# #      hang until the walltime kills it.                                   #
# #    - We submit with `msa: empty` (single-sequence), so no MSA server is  #
# #      contacted. Keep it that way: the ColabFold MSA API works on the     #
# #      login node and fails inside a job, sometimes by hanging.            #
# #    - If a job really must reach out: module load eth_proxy. It is        #
# #      bandwidth- and connection-limited; do not point 7 array tasks at    #
# #      it at once.                                                         #
# #  https://docs.hpc.ethz.ch/connections/firewall/                          #
# ############################################################################
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${PM_VENV:-$HOME/protein-madness-env}"
ACTIVATE="$VENV_DIR/activate.sh"

# Pinned module stack. Python 3.12 to match env/requirements.txt.
# Bump deliberately and record it in DECISIONS.md.
STACK="${PM_STACK:-stack/2024-06}"
GCC_MOD="${PM_GCC:-gcc/12.2.0}"
PYTHON_MOD="${PM_PYTHON:-python/3.12.8}"
CUDA_MOD="${PM_CUDA:-cuda/12.1.1}"

# torch for the reserved GPUs: RTX 2080Ti, sm_75, 11 GB, NO native bf16.
# cu121 wheels cover sm_75. Deliberately NOT in env/requirements.txt: that file
# is CPU-only and each model stack pins its own GPU build.
TORCH_VERSION="${PM_TORCH:-2.5.1}"
TORCH_INDEX="https://download.pytorch.org/whl/cu121"

log() { printf '[euler setup] %s\n' "$*"; }
die() { printf '[euler setup] ERROR: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- sanity

[ -f "$REPO_DIR/env/requirements.txt" ] \
    || die "cannot find $REPO_DIR/env/requirements.txt"

if [ -z "${SCRATCH:-}" ]; then
    die "\$SCRATCH is not set. This script is for ETH Euler, where \$SCRATCH is
       /cluster/scratch/\$USER. On a laptop you want a plain venv:
         python -m venv .venv && pip install -r env/requirements.txt"
fi

# Lmod's `module` is a shell function and is not exported into every
# non-interactive shell. Bring it in rather than failing obscurely.
if ! type module >/dev/null 2>&1; then
    for init in /cluster/apps/lmod/lmod/init/bash /usr/share/lmod/lmod/init/bash \
                /etc/profile.d/modules.sh; do
        [ -r "$init" ] && { set +u; . "$init"; set -u; break; }
    done
fi
type module >/dev/null 2>&1 || die "'module' is not available in this shell. Run this on a
       Euler LOGIN node (ssh \$USER@euler.ethz.ch). Nothing was changed."

# ---------------------------------------------------------------- modules

load_modules() {
    set +u    # Lmod scripts trip over `set -u`
    module purge >/dev/null 2>&1 || true
    module load "$STACK"      || die "module load $STACK failed (see 'module avail stack')"
    module load "$GCC_MOD"    || die "module load $GCC_MOD failed"
    module load "$PYTHON_MOD" || die "module load $PYTHON_MOD failed (see 'module avail python')"
    module load "$CUDA_MOD"   || log "WARNING: $CUDA_MOD not loaded; GPU jobs may fail"
    module load eth_proxy     || log "WARNING: eth_proxy not loaded; downloads may fail"
    set -u
}
log "loading modules: $STACK $GCC_MOD $PYTHON_MOD $CUDA_MOD eth_proxy"
load_modules
log "python: $(command -v python3) ($(python3 --version 2>&1))"

# ---------------------------------------------------------------- scratch dirs

# $SCRATCH is 2.5 TB, no backup, purged after ~2 weeks. Weights and the Boltz
# cache go there; NEVER in $HOME (file-count quota) and NEVER in the repo.
WEIGHTS_DIR="$SCRATCH/weights"
BOLTZ_CACHE="$SCRATCH/boltz-cache"
mkdir -p "$WEIGHTS_DIR" "$BOLTZ_CACHE"
log "weights dir : $WEIGHTS_DIR"
log "boltz cache : $BOLTZ_CACHE"
log "NOTE: \$SCRATCH is purged after ~2 weeks and is NOT backed up."

# ---------------------------------------------------------------- venv

if [ -x "$VENV_DIR/bin/python" ]; then
    log "venv exists: $VENV_DIR (reusing)"
else
    log "creating venv: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi
set +u; . "$VENV_DIR/bin/activate"; set -u

python -m pip install --upgrade pip setuptools wheel
log "installing env/requirements.txt (pinned)"
python -m pip install -r "$REPO_DIR/env/requirements.txt"

if python -c "import torch" >/dev/null 2>&1; then
    log "torch already installed: $(python -c 'import torch; print(torch.__version__)')"
else
    log "installing torch==$TORCH_VERSION from $TORCH_INDEX (sm_75 / cu121, ~2.5 GB)"
    python -m pip install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"
fi

if python -c "import boltz" >/dev/null 2>&1; then
    log "boltz already installed"
else
    log "installing boltz (login node: this needs the network)"
    python -m pip install boltz || log "WARNING: boltz install failed; see cluster/README.md"
fi

# ---------------------------------------------------------------- activate.sh

# Rewritten every run on purpose: it is generated, not edited. Every sbatch
# script sources exactly this path.
log "writing $ACTIVATE"
cat > "$ACTIVATE" <<EOF
#!/bin/bash
# GENERATED by cluster/euler/setup.sh on $(date -Is). Do not edit; re-run setup.
set +u
module purge >/dev/null 2>&1 || true
module load $STACK
module load $GCC_MOD
module load $PYTHON_MOD
module load $CUDA_MOD || true
# eth_proxy: compute nodes have no outbound internet without it, and it is
# rate-limited. Loaded so a stray download fails fast instead of hanging.
module load eth_proxy || true
. "$VENV_DIR/bin/activate"
set -u
export PM_WEIGHTS="\${PM_WEIGHTS:-\$SCRATCH/weights}"
export BOLTZ_CACHE="\${BOLTZ_CACHE:-\$SCRATCH/boltz-cache}"
export PYTHONPATH="$REPO_DIR\${PYTHONPATH:+:\$PYTHONPATH}"
export OMP_NUM_THREADS="\${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="\${SLURM_CPUS_PER_TASK:-1}"
EOF
chmod +x "$ACTIVATE"

# ---------------------------------------------------------------- report

log "done."
cat <<EOF

  Environment : $VENV_DIR
  Activate    : source "$ACTIVATE"
  Weights     : $WEIGHTS_DIR
  Boltz cache : $BOLTZ_CACHE
  Repo        : $REPO_DIR

  Next, in order:

    source "$ACTIVATE"

    # 1. Point config/pipeline.yaml at this cluster's tools, then check:
    python cluster/preflight.py

    # 2. WARM THE BOLTZ CACHE ON THE LOGIN NODE. Compute nodes cannot download.
    #    One tiny prediction is enough. See cluster/README.md.

    # 3. Confirm the GPU story from inside a job, and record it:
    srun --gpus=1 --gres=gpumem:10g --time=00:20:00 \\
         python cluster/preflight.py --require-gpu --write runs/hardware_notes.md

    # 4. Only then submit:
    sbatch cluster/euler/submit.sbatch

  Reminder: compute nodes have no outbound internet. We submit Boltz with
  'msa: empty' so nothing is fetched at fold time -- keep it that way.
EOF
