#!/usr/bin/env bash
#SBATCH --job-name=pm-final-60
#SBATCH --array=1-60%3
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --time=06:00:00
#SBATCH --output=data/final_run_60/logs/%A_%a.out

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/protein-madness-env/activate.sh"
source "${PM_MODEL_ENV_FILE:-$HOME/pm-perlin-envs/launcher.env}"
python -u -m scripts.final_run_60 --rank "$SLURM_ARRAY_TASK_ID"
