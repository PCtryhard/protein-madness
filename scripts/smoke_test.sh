#!/usr/bin/env bash
# Definition of done. Runs the whole pipeline on the committed toy enzyme at
# deliberately tiny settings.
#
#   scripts/smoke_test.sh          CPU only, must finish in under 10 minutes
#   scripts/smoke_test.sh --gpu    same, with one fold backend enabled
#
# The repo is not ready for Saturday until this passes on the dev laptop, on
# Euler and on Modal, with the SAME toy enzyme producing the SAME numbers.
#
# Exits non-zero if any check fails. Every check prints PASS or FAIL.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GPU=0
FOLD_BACKEND="${FOLD_BACKEND:-esmfold}"
KEEP=0
for arg in "$@"; do
  case "$arg" in
    --gpu) GPU=1 ;;
    --keep) KEEP=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

PY="${PYTHON:-python}"
OUT="${SMOKE_OUT:-results/smoke}"
TARGET="configs/target.example.yaml"
MIN_DESIGNS=10
DEADLINE_S=600

FAILURES=0
START=$(date +%s)

pass() { echo "PASS  $*"; }
fail() { echo "FAIL  $*"; FAILURES=$((FAILURES + 1)); }
step() { echo; echo "--- $* ---"; }

step "environment"
echo "python: $($PY --version 2>&1), $($PY -c 'import sys;print(sys.executable)')"
echo "repo:   $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git repo')"
$PY -c "import numpy, pandas, biotite, Bio, matplotlib, yaml" 2>/dev/null \
  && pass "core imports" \
  || { fail "core imports (pip install -r env/requirements.txt)"; echo; echo "SMOKE TEST FAILED"; exit 1; }

step "clean output directory"
[ "$KEEP" -eq 1 ] || rm -rf "$OUT"
mkdir -p "$OUT"

step "unit tests"
if $PY -m pytest -q tests; then pass "unit tests"; else fail "unit tests"; fi

step "target config validates"
if $PY -m src.io.validate_target "$TARGET"; then pass "target validates"; else fail "target validates"; fi

STAGES="validate,mutate,tier1"
if [ "$GPU" -eq 1 ]; then
  step "GPU visible"
  $PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null \
    && pass "cuda available: $($PY -c 'import torch;print(torch.cuda.get_device_name(0))' 2>/dev/null)" \
    || fail "cuda not available (--gpu requested)"
  STAGES="$STAGES,fold,tier2"
fi
STAGES="$STAGES,leaderboard,figures"

step "pipeline: $STAGES"
PIPELINE_ARGS=(--target "$TARGET" --out "$OUT" --stages "$STAGES" --max-designs 60)
[ "$GPU" -eq 1 ] && PIPELINE_ARGS+=(--fold-backend "$FOLD_BACKEND" --device cuda)
if $PY scripts/run_pipeline.py "${PIPELINE_ARGS[@]}"; then pass "pipeline ran"; else fail "pipeline ran"; fi

step "checks"
LEADERBOARD="$OUT/leaderboard.csv"
if [ -s "$LEADERBOARD" ]; then pass "leaderboard exists"; else fail "leaderboard exists: $LEADERBOARD"; fi

# >= 10 designs, every Tier-1 metric populated, no NaNs where a value is required.
TIER1_REQUIRED="length,net_charge_ph7,isoelectric_point,gravy,fraction_charged,max_homopolymer_run,low_complexity_fraction,cysteine_count,catalytic_integrity,sequence_identity"
if [ -s "$LEADERBOARD" ]; then
  $PY - "$LEADERBOARD" "$MIN_DESIGNS" "$TIER1_REQUIRED" <<'EOF'
import sys
import pandas as pd
path, min_designs, required = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(",")
df = pd.read_csv(path)
ok = True
if len(df) < min_designs:
    print(f"FAIL  >= {min_designs} designs (got {len(df)})"); ok = False
else:
    print(f"PASS  >= {min_designs} designs (got {len(df)})")
missing = [c for c in required if c not in df.columns]
if missing:
    print(f"FAIL  tier-1 columns present (missing: {', '.join(missing)})"); ok = False
else:
    print("PASS  tier-1 columns present")
nan_cols = [c for c in required if c in df.columns and df[c].isna().any()]
if nan_cols:
    for c in nan_cols:
        print(f"FAIL  no NaNs in '{c}' ({int(df[c].isna().sum())} of {len(df)} rows)")
    ok = False
else:
    print("PASS  no NaNs in required tier-1 columns")
if "is_fake" in df.columns and df["is_fake"].fillna(False).astype(bool).any():
    print("FAIL  leaderboard contains fake rows"); ok = False
else:
    print("PASS  leaderboard contains no fake rows")
if "catalytic_integrity" in df.columns:
    wt = df[df.get("method", "").astype(str) == "wildtype"]
    if len(wt) and float(wt.iloc[0]["catalytic_integrity"]) != 1.0:
        print("FAIL  wild-type keeps its catalytic residues"); ok = False
    elif len(wt):
        print("PASS  wild-type keeps its catalytic residues")
sys.exit(0 if ok else 1)
EOF
  [ $? -eq 0 ] || FAILURES=$((FAILURES + 1))
fi

for f in fig1_distributions fig2_novelty_vs_quality fig3_best_design; do
  for ext in png pdf; do
    if [ -s "$OUT/figures/$f.$ext" ]; then pass "figure $f.$ext"; else fail "figure $f.$ext"; fi
  done
done

ELAPSED=$(( $(date +%s) - START ))
if [ "$ELAPSED" -le "$DEADLINE_S" ]; then
  pass "finished in ${ELAPSED}s (limit ${DEADLINE_S}s)"
else
  fail "finished in ${ELAPSED}s (limit ${DEADLINE_S}s)"
fi

echo
if [ "$FAILURES" -eq 0 ]; then
  echo "SMOKE TEST PASSED (${ELAPSED}s)"
  exit 0
fi
echo "SMOKE TEST FAILED: $FAILURES check(s) failed (${ELAPSED}s)"
exit 1
