#!/bin/bash
# Copy a running (or finished) run from a cluster to this machine. Safe to run
# repeatedly while the run is still going: rsync only moves what changed.
#
#   bash scripts/pull_run.sh <user@host>:<repo> <run-name> [dest]
#
# Examples
#   bash scripts/pull_run.sh me@euler.ethz.ch:/cluster/home/me/protein-madness small-01
#   bash scripts/pull_run.sh me@13.38.53.98:~/protein-madness aws-01_his_asp ~/runs
#   watch -n 60 "bash scripts/pull_run.sh me@host:~/protein-madness small-01"
#
# WHAT IT PULLS, AND WHAT IT LEAVES BEHIND
#
# Everything small and readable comes every time: designs.csv, sequences.fasta,
# progress.txt, report.md, hardware_notes.md, stages.jsonl, and the sqlite
# database.
#
# The structures are the bulk -- one directory per design, each with an
# RFdiffusion backbone, a LigandMPNN run and a Boltz-2 prediction. By default
# only the FINAL files are pulled (*.cif, *.pdb, the Boltz confidence JSONs and
# the input YAMLs), not the intermediate trajectories RFdiffusion writes. Pass
# --all for literally everything, which can be many gigabytes.
#
# A note on the sqlite file: copying a database that is being written to can
# give you a torn snapshot. The CSV is the thing to trust mid-run; the database
# is pulled for completeness and is reliable once the run has finished. If you
# want a consistent copy of a live database, ask sqlite for one on the cluster:
#   sqlite3 runs/<name>/designs.db ".backup runs/<name>/designs.snapshot.db"
set -euo pipefail

ALL=0
ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --all) ALL=1 ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        -*) echo "unknown option: $1" >&2; exit 2 ;;
        *) ARGS+=("$1") ;;
    esac
    shift
done

if [ "${#ARGS[@]}" -lt 2 ]; then
    echo "usage: bash scripts/pull_run.sh <user@host>:<repo> <run-name> [dest]" >&2
    echo "       bash scripts/pull_run.sh --help" >&2
    exit 2
fi

REMOTE="${ARGS[0]}"
RUN="${ARGS[1]}"
DEST="${ARGS[2]:-runs}"

command -v rsync >/dev/null 2>&1 || {
    echo "rsync is not installed. On Windows use Git Bash with rsync, or WSL." >&2
    exit 1
}

case "$REMOTE" in
    *:*) ;;
    *) echo "first argument must look like user@host:/path/to/repo" >&2; exit 2 ;;
esac

SRC="${REMOTE%%:*}:${REMOTE#*:}/runs/$RUN/"
mkdir -p "$DEST/$RUN"

echo "from : $SRC"
echo "to   : $DEST/$RUN/"
echo

# --partial so an interrupted transfer of a large structure resumes rather than
# restarting. No --delete: a file removed on the cluster should not vanish from
# a copy you may be relying on.
COMMON=(-az --partial --human-readable --info=stats1,progress2)

if [ "$ALL" -eq 1 ]; then
    rsync "${COMMON[@]}" "$SRC" "$DEST/$RUN/"
else
    rsync "${COMMON[@]}" \
        --include='*/' \
        --include='designs.csv' \
        --include='designs.*.csv' \
        --include='sequences.fasta' \
        --include='progress.txt' \
        --include='report.md' \
        --include='hardware_notes.md' \
        --include='stages.jsonl' \
        --include='designs.db' \
        --include='*.cif' \
        --include='*.pdb' \
        --include='confidence_*.json' \
        --include='*.yaml' \
        --exclude='*' \
        "$SRC" "$DEST/$RUN/"
fi

echo
echo "pulled. Have a look with:"
echo "  python -m src.cli watch --run $DEST/$RUN"
echo "  python -m src.cli report --run $DEST/$RUN"
