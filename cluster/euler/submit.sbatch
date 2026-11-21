#!/bin/bash
#SBATCH --job-name=pm-search
#SBATCH --array=1-7
#SBATCH --gpus=2
#SBATCH --gres=gpumem:10g
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4g
#SBATCH --time=12:00:00
#SBATCH --output=runs/slurm/%x-%A_%a.out
#SBATCH --error=runs/slurm/%x-%A_%a.out
#
# One array task per variant: seven independent searches, each a separate
# scaffold with its own frontier, stopping condition and leaderboard. There is
# no cross-scaffold competition, so running them as independent tasks changes
# nothing about the results and lets SLURM schedule them as capacity appears.
#
#   mkdir -p runs/slurm
#   sbatch cluster/euler/submit.sbatch
#   sbatch --array=1,5 cluster/euler/submit.sbatch     # just variants 01 and 05
#
# Each task asks for TWO GPUs because the pipeline's worker pool runs one
# design per card. If the reservation cannot give two, run with
# `--gpus=1` and set run.gpus: [0] -- it halves throughput and changes nothing
# else.
#
# BEFORE THE FIRST SUBMISSION, in this order:
#   1. bash cluster/euler/setup.sh
#   2. python cluster/preflight.py            (login node, fix every FAIL)
#   3. warm the Boltz cache on the login node -- compute nodes have no network
#   4. srun ... python cluster/preflight.py --require-gpu --write runs/hardware_notes.md
# cluster/README.md explains why each step is there.
#
# The reservation used for the hackathon was ws_49734_gpu. Uncomment and set it
# if you have one; without it this queues on the general GPU partition.
##SBATCH --account=ws_49734_gpu
##SBATCH --reservation=ws_49734_gpu

set -euo pipefail

REPO_DIR="${PM_REPO:-$SLURM_SUBMIT_DIR}"
ACTIVATE="${PM_ACTIVATE:-$HOME/protein-madness-env/activate.sh}"
RUN_PREFIX="${PM_RUN_PREFIX:-euler}"
# Seed-only by default, same reasoning as cluster/aws/run_all.sh. Submit with
#   sbatch --export=ALL,PM_PIPELINE=config/pipeline.yaml cluster/euler/submit.sbatch
# for a full search.
PIPELINE="${PM_PIPELINE:-config/pipeline.small.yaml}"

cd "$REPO_DIR"

[ -r "$ACTIVATE" ] || {
    echo "ERROR: $ACTIVATE not found. Run: bash cluster/euler/setup.sh" >&2
    exit 1
}
# shellcheck disable=SC1090
source "$ACTIVATE"

# Pick this task's target. Sorted so the array index maps to the variant
# number: task 1 -> 01_*, task 7 -> 07_*.
mapfile -t TARGETS < <(ls -1 config/targets/0*.yaml | sort)
INDEX="${SLURM_ARRAY_TASK_ID:-1}"
if [ "$INDEX" -lt 1 ] || [ "$INDEX" -gt "${#TARGETS[@]}" ]; then
    echo "ERROR: array index $INDEX is outside 1..${#TARGETS[@]}" >&2
    exit 1
fi
TARGET="${TARGETS[$((INDEX - 1))]}"
NAME="${RUN_PREFIX}-$(basename "$TARGET" .yaml)"

echo "================================================================"
echo " host        : $(hostname)"
echo " job         : ${SLURM_JOB_ID:-none} task ${INDEX}"
echo " target      : $TARGET"
echo " pipeline    : $PIPELINE"
echo " run name    : $NAME"
echo " gpus        : ${CUDA_VISIBLE_DEVICES:-unset}"
echo " started     : $(date -Is)"
echo "================================================================"

nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

# Fail before burning the allocation if the config or the target is wrong.
python -m src.cli smoke --pipeline "$PIPELINE" --target "$TARGET"

# The search. Scores, structures and lineage all land under runs/$NAME/.
python -m src.cli run --pipeline "$PIPELINE" --target "$TARGET" --name "$NAME"

# A leaderboard per scaffold plus the best design's full lineage.
python -m src.cli report --run "runs/$NAME" --write

echo "finished: $(date -Is)"
