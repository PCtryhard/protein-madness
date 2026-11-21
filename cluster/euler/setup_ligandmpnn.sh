#!/usr/bin/env bash
# LigandMPNN's own environment on Euler, from its own requirements.txt.
#
# Why a separate env rather than reusing pm-rfd3-env: LigandMPNN pins
# numpy 1.23.5, and its vendored openfold uses np.int, which numpy 2 removed.
# pm-rfd3-env has numpy 2.5.4 because RFD3 wants it. The two stacks cannot
# coexist, which is exactly why the AWS cluster ships three separate envs.
#
# Python 3.11 because numpy 1.23.5 has no wheel for 3.12 and cannot build
# there (pkgutil.ImpImporter was removed). 3.11.6 here matches AWS's 3.11.18.
#
# Nothing custom: the tool's own requirements file, the tool's own pins.
set -e -o pipefail

VENV="$HOME/pm-mpnn-env"
L="$SCRATCH/pdh/tools/LigandMPNN"

set +u
module purge >/dev/null 2>&1 || true
module load stack/2024-06 gcc/12.2.0 python/3.11.6 >/dev/null 2>&1 || true
module load cuda/12.1.1 >/dev/null 2>&1 || true
module load eth_proxy >/dev/null 2>&1 || true
set -u

echo "python: $(command -v python3) ($(python3 --version 2>&1))"
test -f "$L/requirements.txt"

if [ -x "$VENV/bin/python" ]; then
    echo "venv exists: $VENV"
else
    python3 -m venv "$VENV"
    echo "created $VENV"
fi
set +u; . "$VENV/bin/activate"; set -u
python -m pip install --upgrade -q pip setuptools wheel

echo
echo "=== installing LigandMPNN's own requirements.txt ==="
python -m pip install -r "$L/requirements.txt" 2>&1 | tail -12

# ProDy 2.4.1 calls pkg_resources, which setuptools 81 removed. Not in the
# upstream requirements because it used to come for free.
python -m pip install -q "setuptools<81"

echo
echo "=== versions ==="
python - <<'PY'
import importlib.metadata as md
for name in ("torch", "numpy", "ProDy", "ml-collections", "dm-tree", "scipy"):
    try:
        print(f"  {name:<16} {md.version(name)}")
    except md.PackageNotFoundError:
        print(f"  {name:<16} MISSING")
PY

echo
echo "=== the whole LigandMPNN import chain ==="
cd "$L"
python -c "
import data_utils, model_utils, sc_utils
print('ALL LIGANDMPNN MODULES IMPORT OK')
"

echo
echo "=== run.py responds ==="
python "$L/run.py" --help 2>&1 | head -6

echo
echo "DONE: $VENV"
