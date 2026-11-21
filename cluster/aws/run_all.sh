#!/bin/bash
# Run the seven variant searches on a single machine with no scheduler.
#
#   bash cluster/aws/run_all.sh --dry-run        # print what it would do
#   bash cluster/aws/run_all.sh 01 05            # just these variants
#   bash cluster/aws/run_all.sh                  # all seven, one at a time
#   bash cluster/aws/run_all.sh --full 01        # a real search, not seed-only
#
# THE PROFILE DEFAULTS TO SEED-ONLY.
#
# config/pipeline.small.yaml is 10 backbones x 8 sequences with max_cycles 0:
# about an hour per variant, and it exercises the whole forward pass and writes
# a leaderboard without committing to a multi-hour search. That is what you
# want first, so it is the default here rather than something to remember.
#
# --full switches to config/pipeline.yaml (max_cycles 12, both branches, 224
# Boltz-2 calls per cycle). PM_PIPELINE=<path> overrides either.
#
# CONCURRENCY, AND WHY THE DEFAULT IS 1
#
# The pipeline already uses both GPUs inside a single run: its worker pool puts
# one design on each card. Starting two runs at once would put two models on
# one card, which is the thing PIPELINE.md forbids -- 12 GB is not 24 GB, and
# Boltz-2 on a ~270-token system is close to the limit of one card on its own.
#
# So the default is one run at a time, each using both cards. PM_JOBS exists
# for the case where you have deliberately set run.gpus to a single device per
# run; if you raise it without doing that, you will OOM.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

JOBS="${PM_JOBS:-1}"
RUN_PREFIX="${PM_RUN_PREFIX:-aws}"
DRY_RUN=0
FULL=0
WANTED=()

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --full) FULL=1 ;;
        -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
        -*) echo "unknown option: $1" >&2; exit 2 ;;
        *) WANTED+=("$1") ;;
    esac
    shift
done

if [ -n "${PM_PIPELINE:-}" ]; then
    PIPELINE="$PM_PIPELINE"
elif [ "$FULL" -eq 1 ]; then
    PIPELINE="config/pipeline.yaml"
else
    PIPELINE="config/pipeline.small.yaml"
fi
[ -r "$PIPELINE" ] || { echo "pipeline config not found: $PIPELINE" >&2; exit 1; }

# Say plainly which kind of run this is. Starting a 6-hour search when you
# meant a 1-hour smoke run is the mistake this line exists to prevent.
MAX_CYCLES="$(python - "$PIPELINE" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["run"].get("max_cycles", "?"))
PY
)"
if [ "$MAX_CYCLES" = "0" ]; then
    RUN_KIND="seed-only (max_cycles 0)"
else
    RUN_KIND="FULL SEARCH (max_cycles $MAX_CYCLES) -- hours per variant"
fi

mapfile -t ALL_TARGETS < <(ls -1 config/targets/[0-9][0-9]_*.yaml | sort)
[ "${#ALL_TARGETS[@]}" -gt 0 ] || { echo "no config/targets/0*.yaml found" >&2; exit 1; }

TARGETS=()
if [ "${#WANTED[@]}" -eq 0 ]; then
    TARGETS=("${ALL_TARGETS[@]}")
else
    for want in "${WANTED[@]}"; do
        match=""
        for target in "${ALL_TARGETS[@]}"; do
            case "$(basename "$target")" in
                "$want"_*) match="$target"; break ;;
            esac
        done
        [ -n "$match" ] || { echo "no target matches '$want'" >&2; exit 1; }
        TARGETS+=("$match")
    done
fi

LOG_DIR="runs/logs"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo " host     : $(hostname)"
echo " repo     : $REPO_DIR"
echo " variants : ${#TARGETS[@]}"
echo " pipeline : $PIPELINE"
echo " kind     : $RUN_KIND"
echo " parallel : $JOBS run(s) at a time"
echo " logs     : $LOG_DIR"
echo "================================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null || \
    echo "(nvidia-smi unavailable)"
echo

# Validate every target before starting anything. Cheap, CPU-only, and it
# catches a bad config before hours of GPU time rather than after.
for target in "${TARGETS[@]}"; do
    printf '  checking %-34s ' "$(basename "$target")"
    if python -m src.cli smoke --pipeline "$PIPELINE" --target "$target" >/dev/null 2>&1; then
        echo "ok"
    else
        echo "FAILED"
        echo
        echo "  smoke test failed for $target. Full output:" >&2
        python -m src.cli smoke --pipeline "$PIPELINE" --target "$target" || true
        exit 1
    fi
done
echo

run_one() {
    local target="$1"
    local name="${RUN_PREFIX}-$(basename "$target" .yaml)"
    local log="$LOG_DIR/${name}.log"

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  would run: python -m src.cli run --pipeline $PIPELINE --target $target --name $name"
        return 0
    fi

    echo "  [$(date +%H:%M:%S)] starting $name  (log: $log)"
    {
        echo "=== $name  started $(date -Is) on $(hostname) ==="
        python -m src.cli run --pipeline "$PIPELINE" --target "$target" --name "$name"
        echo "=== search done, writing report ==="
        python -m src.cli report --run "runs/$name" --write
        echo "=== $name  finished $(date -Is) ==="
    } >"$log" 2>&1
    echo "  [$(date +%H:%M:%S)] finished $name"
}

FAILED=()
if [ "$JOBS" -le 1 ]; then
    for target in "${TARGETS[@]}"; do
        run_one "$target" || FAILED+=("$target")
    done
else
    echo "WARNING: PM_JOBS=$JOBS. Only do this if run.gpus names ONE device," >&2
    echo "         or two models will land on one card and OOM." >&2
    echo
    running=0
    for target in "${TARGETS[@]}"; do
        run_one "$target" &
        running=$((running + 1))
        if [ "$running" -ge "$JOBS" ]; then
            wait -n || true
            running=$((running - 1))
        fi
    done
    wait
fi

echo
if [ "$DRY_RUN" -eq 1 ]; then
    echo "dry run: nothing was executed"
    exit 0
fi

if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "FAILED: ${#FAILED[@]} run(s):" >&2
    for target in "${FAILED[@]}"; do echo "  - $target" >&2; done
    echo "Check $LOG_DIR for the output." >&2
    exit 1
fi

echo "all ${#TARGETS[@]} run(s) finished. Leaderboards:"
for target in "${TARGETS[@]}"; do
    name="${RUN_PREFIX}-$(basename "$target" .yaml)"
    echo "  runs/$name/report.md"
done
