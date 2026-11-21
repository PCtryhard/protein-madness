#!/usr/bin/env bash
#SBATCH --job-name=pm-our-01
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4g
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm/pm-our-01-%j.log
#
# OUR pipeline, on Euler, variant 01.
#
# Three environments, because the three tools pin incompatible torch builds:
#   pm-rfd3-env    torch 2.7.1+cu126, rfd3
#   pm-boltz-env   torch 2.15.0, boltz
#   (LigandMPNN runs under pm-rfd3-env; it needs torch and numpy, nothing exotic)
#
# The runner shells out to each tool, so it does not need them importable
# itself -- it runs under pm-rfd3-env, which has the structure libraries our
# own code wants (numpy, pydantic, pyyaml).
set -e -o pipefail

REPO=$HOME/protein-madness
RFD3_ENV=$HOME/pm-rfd3-env
cd "$REPO"

set +u
module purge >/dev/null 2>&1 || true
module load stack/2024-06 gcc/12.2.0 python/3.12.8 >/dev/null 2>&1 || true
module load cuda/12.6.0 >/dev/null 2>&1 || module load cuda/12.1.1 >/dev/null 2>&1 || true
set -u

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
# Keep caches off $HOME, which has a 500k file quota.
export XDG_CACHE_HOME="$SCRATCH/pdh/cache-xdg"
export TORCH_HOME="$SCRATCH/pdh/cache-torch"
export TMPDIR="$SCRATCH/pdh/tmp"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$TMPDIR" "$REPO/runs/slurm"

echo "================================================================"
echo " OUR pipeline, variant 01"
echo " host    : $(hostname)"
echo " job     : ${SLURM_JOB_ID:-interactive}"
echo " started : $(date -Is)"
echo "================================================================"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader || true
echo

# Fail before the allocation is spent if the config or target is wrong.
"$RFD3_ENV/bin/python" -m src.cli smoke \
    --pipeline config/pipeline.euler.yaml \
    --target config/targets/01_his_asp.yaml

echo
echo "=== the search ==="
"$RFD3_ENV/bin/python" -m src.cli run \
    --pipeline config/pipeline.euler.yaml \
    --target config/targets/01_his_asp.yaml \
    --name euler-01 -v

echo
echo "=== leaderboard ==="
"$RFD3_ENV/bin/python" -m src.cli report --run runs/euler-01 --write
echo "finished: $(date -Is)"
