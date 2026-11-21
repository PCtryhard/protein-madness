#!/usr/bin/env bash
#SBATCH --job-name=pm-run
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4g
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm/%x-%j.log
#
# THE Euler submission script. One, parameterised by environment variable, so
# there is no family of near-identical copies drifting apart.
#
#   sbatch cluster/euler/run.sbatch
#   PM_TARGET=config/targets/01_his_asp.yaml PM_NAME=euler-01 \
#       sbatch cluster/euler/run.sbatch
#   PM_CONFIG=$HOME/pm-runconf/fast.yaml PM_NAME=fast-01 \
#       sbatch cluster/euler/run.sbatch
#   PM_WIGGLE_CHECK=1 sbatch cluster/euler/run.sbatch   # also test the wiggle
#
# Knobs, all optional:
#   PM_REPO          where the code is            (default $HOME/pm-deploy)
#   PM_CONFIG        pipeline config              (default config/pipeline.yaml)
#   PM_TARGET        target config                (default 01_his_asp)
#   PM_NAME          run name under runs/         (default euler-01)
#   PM_WIGGLE_CHECK  1 = run the wiggle probe first
#
# A smaller config belongs OUTSIDE config/ -- tests/test_config.py enforces
# exactly one pipeline config in that directory, deliberately.
#
# Override the SLURM settings on the command line when you need to, e.g.
#   sbatch --time=01:00:00 --job-name=pm-fast cluster/euler/run.sbatch
set -e -o pipefail

REPO=${PM_REPO:-$HOME/pm-deploy}
CONFIG=${PM_CONFIG:-config/pipeline.yaml}
TARGET=${PM_TARGET:-config/targets/01_his_asp.yaml}
NAME=${PM_NAME:-euler-01}
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
echo " motif-anchored pipeline"
echo " host    : $(hostname)"
echo " job     : ${SLURM_JOB_ID:-interactive}"
echo " repo    : $REPO"
echo " config  : $CONFIG"
echo " target  : $TARGET"
echo " run     : $NAME"
echo " started : $(date -Is)"
echo "================================================================"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader || true
echo

# Fail before the allocation is spent, not twelve minutes in. This resolves
# every tool path and probes the LigandMPNN interpreter for its imports.
echo "=== preflight ==="
"$RFD3_ENV/bin/python" cluster/preflight.py \
    --pipeline "$CONFIG" --target "$TARGET" --require-gpu

echo
echo "=== smoke ==="
"$RFD3_ENV/bin/python" -m src.cli smoke --pipeline "$CONFIG" --target "$TARGET"

if [ "${PM_WIGGLE_CHECK:-0}" = "1" ]; then
    echo
    echo "=== wiggle probe ==="
    "$RFD3_ENV/bin/python" cluster/euler/check_wiggle.py
fi

echo
echo "=== the search ==="
"$RFD3_ENV/bin/python" -m src.cli -v run \
    --pipeline "$CONFIG" --target "$TARGET" --name "$NAME"

echo
echo "=== leaderboard ==="
"$RFD3_ENV/bin/python" -m src.cli report --run "runs/$NAME" --write || true

echo
echo "=== summary ==="
"$RFD3_ENV/bin/python" - "$NAME" <<'PYEOF'
import collections
import csv
import sys

name = sys.argv[1]
try:
    rows = list(csv.DictReader(open("runs/%s/designs.csv" % name)))
except FileNotFoundError:
    print("no designs.csv")
    raise SystemExit(0)


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


print("designs  : %d" % len(rows))
print("by branch: %s" % dict(collections.Counter(r["branch"] for r in rows)))
print("by status: %s" % dict(collections.Counter(r["status"] for r in rows)))
scored = [r for r in rows if r["status"] == "scored"]
if scored:
    best = max(scored, key=lambda r: num(r["score"]))
    print("best     : %.4f  %s" % (num(best["score"]), best["design_id"]))
else:
    print("best     : nothing cleared the gate")
PYEOF
echo "finished: $(date -Is)"
