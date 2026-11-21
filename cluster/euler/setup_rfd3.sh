#!/usr/bin/env bash
# Build the RFD3 environment on Euler from PyPI.
#
# RFD3 itself is not installable -- it has no pip metadata and was hand-placed
# into site-packages on the AWS reference mount -- but 122 of its 140
# dependencies are ordinary PyPI pins. Those get installed here; the handful
# that are private get copied separately.
#
# torch comes from the PyTorch cu126 index rather than the freeze, because
# pinning the nvidia-* wheels alongside it fights the index resolver.
#
# Login node only: Euler compute nodes have no outbound internet.
set -e -o pipefail

VENV="$HOME/pm-rfd3-env"
REQ="$HOME/rfd3_requirements.txt"

set +u
module purge >/dev/null 2>&1 || true
module load stack/2024-06 gcc/12.2.0 python/3.12.8 >/dev/null 2>&1 || true
module load cuda/12.6.0 >/dev/null 2>&1 || module load cuda/12.1.1 >/dev/null 2>&1 || true
module load eth_proxy >/dev/null 2>&1 || true
set -u

echo "python: $(command -v python3) ($(python3 --version 2>&1))"
test -f "$REQ"

if [ -x "$VENV/bin/python" ]; then
    echo "venv exists: $VENV"
else
    python3 -m venv "$VENV"
    echo "created $VENV"
fi
set +u; . "$VENV/bin/activate"; set -u
python -m pip install --upgrade -q pip setuptools wheel

echo
echo "=== 1/2 torch 2.7.1+cu126 ==="
if python -c "import torch; assert torch.__version__.startswith('2.7.1')" 2>/dev/null; then
    echo "already present: $(python -c 'import torch; print(torch.__version__)')"
else
    python -m pip install "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu126
fi

echo
echo "=== 2/2 the other 122 pins ==="
# Not --quiet: if a pin cannot be satisfied on this Python, the resolver's
# message is the whole diagnosis.
python -m pip install -r "$REQ" 2>&1 | tail -25

echo
echo "=== result ==="
python - <<'PY'
import importlib.metadata as md
for name in ("torch", "atomworks", "hydride", "biotite", "lightning",
             "hydra-core", "einx", "jaxtyping", "numpy", "scipy"):
    try:
        print(f"  {name:<12} {md.version(name)}")
    except md.PackageNotFoundError:
        print(f"  {name:<12} MISSING")
PY
echo
echo "NOTE: rfd3, rfd3na, foundry, coreweave_evaluations and cwevals are still"
echo "      missing -- they have no PyPI release and are copied separately."
