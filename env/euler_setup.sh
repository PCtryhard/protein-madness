#!/bin/bash
# Set up this repo's environment on ETH Euler. Idempotent: safe to re-run, and
# re-running is the intended way to pick up a changed env/requirements.txt.
#
#   ssh <ethz-user>@euler.ethz.ch
#   cd /cluster/home/$USER/Protein-Madness        # wherever you cloned it
#   bash env/euler_setup.sh                       # ~10 min the first time
#   source "$HOME/enzyme-hack-env/activate.sh"    # every new shell
#
# What it does, and why it is shaped this way:
#   * module loads  -- there is no root on Euler, so nothing is apt-installed.
#   * a venv in $HOME -- $HOME is 50 GB / 500k files, which a venv fits and a
#     conda environment plus model weights does not. The ETH docs explicitly
#     warn that conda on the Lustre /cluster/work filesystem behaves badly.
#   * weights in $SCRATCH/weights -- 2.5 TB, no backup, purged after ~2 weeks.
#     NEVER in $HOME (file-count quota) and NEVER in the repo (see .gitignore).
#   * it writes $HOME/enzyme-hack-env/activate.sh, which is the single file that
#     every sbatch script sources (see src/backends/euler.py DEFAULT_SETUP).
#
# ############################################################################
# #  LOUD WARNING: COMPUTE NODES HAVE NO OUTBOUND INTERNET.                  #
# #                                                                          #
# #  Euler compute nodes cannot reach anything outside the ETH network by    #
# #  default. So:                                                            #
# #    - the MMseqs2 / ColabFold MSA WEB API WORKS ON THE LOGIN NODE AND     #
# #      FAILS INSIDE A JOB. It does not fail loudly in every client; some   #
# #      hang until the walltime kills them. Precompute MSAs into            #
# #      results/msa/ from the login node, or stay on single-sequence        #
# #      models (ESMFold, Boltz) on Euler.                                   #
# #    - weight downloads must happen on the LOGIN node                      #
# #      (scripts/fetch_weights.sh), never from inside a job.                #
# #    - if a job really must reach out: module load eth_proxy. It is        #
# #      bandwidth- and connection-limited, so do not point 500 array tasks  #
# #      at it.                                                              #
# #  This is documented at https://docs.hpc.ethz.ch/connections/firewall/    #
# ############################################################################
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ENZYME_VENV:-$HOME/enzyme-hack-env}"
ACTIVATE="$VENV_DIR/activate.sh"

# Pinned module stack. Python 3.12 to match env/requirements.txt (resolved and
# tested on 3.12.10). Bump deliberately and record it in DECISIONS.md.
STACK="${ENZYME_STACK:-stack/2024-06}"
GCC_MOD="${ENZYME_GCC:-gcc/12.2.0}"
PYTHON_MOD="${ENZYME_PYTHON:-python/3.12.8}"
CUDA_MOD="${ENZYME_CUDA:-cuda/12.1.1}"

# torch for the reserved GPUs: RTX 2080Ti, sm_75, 11 GB, NO bf16.
# cu121 wheels cover sm_75. This is NOT in env/requirements.txt on purpose:
# the core environment is CPU-only and each backend pins its own GPU stack.
TORCH_VERSION="${ENZYME_TORCH:-2.5.1}"
TORCH_INDEX="https://download.pytorch.org/whl/cu121"

log()  { printf '[euler_setup] %s\n' "$*"; }
die()  { printf '[euler_setup] ERROR: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- sanity

[ -f "$REPO_DIR/env/requirements.txt" ] || die "cannot find $REPO_DIR/env/requirements.txt"

if [ -z "${SCRATCH:-}" ]; then
    die "\$SCRATCH is not set. This script is for ETH Euler; on Euler \$SCRATCH is
       /cluster/scratch/\$USER. If you are on a laptop you want a plain venv:
         python -m venv .venv && pip install -r env/requirements.txt"
fi

# Lmod's `module` is a shell function and is not exported into every
# non-interactive shell. Try to bring it in rather than failing obscurely.
if ! type module >/dev/null 2>&1; then
    for init in /cluster/apps/lmod/lmod/init/bash /usr/share/lmod/lmod/init/bash \
                /etc/profile.d/modules.sh; do
        # shellcheck disable=SC1090
        [ -r "$init" ] && { set +u; . "$init"; set -u; break; }
    done
fi
type module >/dev/null 2>&1 || die "'module' is not available in this shell. Run this on a
       Euler LOGIN node (ssh \$USER@euler.ethz.ch), or source your Lmod init
       script first. Nothing was changed."

# ---------------------------------------------------------------- modules

# Lmod scripts trip over `set -u`; relax it only around the module calls.
load_modules() {
    set +u
    module purge >/dev/null 2>&1 || true
    module load "$STACK"      || die "module load $STACK failed (see 'module avail stack')"
    module load "$GCC_MOD"    || die "module load $GCC_MOD failed"
    module load "$PYTHON_MOD" || die "module load $PYTHON_MOD failed (see 'module avail python')"
    # CUDA and the proxy are best-effort: a CPU-only login shell does not need
    # them, and eth_proxy is in the core layer so it loads without a stack.
    module load "$CUDA_MOD"   || log "WARNING: $CUDA_MOD not loaded; GPU jobs may fail"
    module load eth_proxy     || log "WARNING: eth_proxy not loaded; downloads may fail"
    set -u
}
log "loading modules: $STACK $GCC_MOD $PYTHON_MOD $CUDA_MOD eth_proxy"
load_modules
log "python: $(command -v python3) ($(python3 --version 2>&1))"

# ---------------------------------------------------------------- weights dir

WEIGHTS_DIR="$SCRATCH/weights"
mkdir -p "$WEIGHTS_DIR"
log "weights dir: $WEIGHTS_DIR  (fill it with scripts/fetch_weights.sh ON THE LOGIN NODE)"
log "NOTE: \$SCRATCH is purged after ~2 weeks and is NOT backed up."

# ---------------------------------------------------------------- venv

if [ -x "$VENV_DIR/bin/python" ]; then
    log "venv exists: $VENV_DIR (reusing)"
else
    log "creating venv: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
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

# ---------------------------------------------------------------- activate.sh

# Rewritten every run on purpose: it is generated, not edited. src/backends/euler.py
# sources exactly this path in every sbatch script it writes.
log "writing $ACTIVATE"
cat > "$ACTIVATE" <<EOF
#!/bin/bash
# GENERATED by env/euler_setup.sh on $(date -Is). Do not edit; re-run the setup.
# Sourced by every sbatch script written by src/backends/euler.py.
set +u
module purge >/dev/null 2>&1 || true
module load $STACK
module load $GCC_MOD
module load $PYTHON_MOD
module load $CUDA_MOD || true
# eth_proxy: compute nodes have no outbound internet without it, and it is
# rate-limited. Loaded here so a stray download fails fast instead of hanging.
module load eth_proxy || true
. "$VENV_DIR/bin/activate"
set -u
export ENZYME_WEIGHTS="\${ENZYME_WEIGHTS:-\$SCRATCH/weights}"
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
  Weights     : $WEIGHTS_DIR   (scripts/fetch_weights.sh --backend euler)
  Repo        : $REPO_DIR

  Next:
    source "$ACTIVATE"
    bash scripts/fetch_weights.sh --backend euler      # LOGIN NODE ONLY
    srun --gpus=1 --gres=gpumem:10g --time=00:10:00 \\
         python -c "import torch; print(torch.cuda.get_device_name())"

  Reminder: compute nodes have no outbound internet. MSA web APIs fail there.
EOF
