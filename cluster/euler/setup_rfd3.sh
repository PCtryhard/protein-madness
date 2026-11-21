#!/usr/bin/env bash
# Build the RFD3 environment on Euler.
#
#   bash cluster/euler/setup_rfd3.sh
#
# RFD3 ships inside RosettaCommons' Foundry package, BSD 3-Clause, on PyPI as
# `rc-foundry`. The `[rfd3]` extra pulls RFD3 and its dependencies; the wheel
# provides the `rfd3` and `rfd3na` packages and an `rfd3` console script
# (rfd3.cli:app), which is exactly what config/pipeline.yaml's rfd3.binary
# points at.
#
# This replaces an older arrangement that installed 122 PyPI pins by hand and
# then copied rfd3/rfd3na out of a hackathon AWS mount. That is no longer
# necessary: rc-foundry 0.2.0's rfd3 tree was verified byte-for-byte identical
# to the hand-placed copy the recorded runs used. The old pin list is kept as
# cluster/euler/rfd3_requirements.txt for reference only.
#
# Python must be >=3.12,<3.13, which is what the module below provides.
#
# LOGIN NODE ONLY -- Euler compute nodes have no outbound internet.
set -e -o pipefail

VENV="${PM_RFD3_ENV:-$HOME/pm-rfd3-env}"

set +u
module purge >/dev/null 2>&1 || true
module load stack/2024-06 gcc/12.2.0 python/3.12.8 >/dev/null 2>&1 || true
module load cuda/12.6.0 >/dev/null 2>&1 || module load cuda/12.1.1 >/dev/null 2>&1 || true
module load eth_proxy >/dev/null 2>&1 || true
set -u

# Keep pip's download cache OFF $HOME. rc-foundry[rfd3] pulls torch's full GPU
# stack -- nvidia-* at ~1.9 GB and triton at ~1 GB -- and pip caches every
# wheel it downloads. Left at its default that is ~10 GB in ~/.cache/pip,
# against a 45 GB soft quota on a $HOME that also has a 500k file limit.
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${SCRATCH:-/tmp}/pdh/cache-pip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRATCH:-/tmp}/pdh/cache-xdg}"
mkdir -p "$PIP_CACHE_DIR" "$XDG_CACHE_HOME"

echo "python: $(command -v python3) ($(python3 --version 2>&1))"
echo "pip cache: $PIP_CACHE_DIR"
echo
echo "NOTE: this downloads roughly 4 GB (torch + CUDA libraries) and takes"
echo "      5-15 minutes on a cold cache. rfd3 itself is about 5 MB and is"
echo "      installed last, after its dependencies."


if [ -x "$VENV/bin/python" ]; then
    echo "venv exists: $VENV"
    # A working env is not worth clobbering: rc-foundry would re-resolve torch
    # and could replace a build that is known good on this cluster's GPUs.
    if "$VENV/bin/python" -c "import rfd3" >/dev/null 2>&1; then
        echo
        echo "rfd3 already imports in $VENV -- nothing to do."
        echo "To rebuild from scratch, remove the directory first, or point"
        echo "PM_RFD3_ENV at a new path."
        "$VENV/bin/python" - <<'PY'
import importlib.metadata as md
try:
    print("  rc-foundry   %s (pip)" % md.version("rc-foundry"))
except md.PackageNotFoundError:
    print("  rc-foundry   not a pip install (hand-placed rfd3 tree)")
import torch
print("  torch        %s" % torch.__version__)
PY
        exit 0
    fi
else
    python3 -m venv "$VENV"
    echo "created $VENV"
fi

set +u; . "$VENV/bin/activate"; set -u
python -m pip install --upgrade -q pip setuptools wheel

echo
echo "=== rc-foundry[rfd3] from PyPI ==="
# Not --quiet: if a dependency cannot be resolved on this Python, the
# resolver's message is the whole diagnosis.
python -m pip install "rc-foundry[rfd3]" 2>&1 | tail -20

echo
echo "=== our own structure libraries ==="
python -m pip install -q "numpy>=2" pyyaml pydantic pytest

echo
echo "=== result ==="
python - <<'PY'
import importlib.metadata as md

for name in ("rc-foundry", "torch", "numpy", "pydantic", "pyyaml"):
    try:
        print("  %-12s %s" % (name, md.version(name)))
    except md.PackageNotFoundError:
        print("  %-12s MISSING" % name)

import rfd3            # noqa: F401
import rfd3na          # noqa: F401
from rfd3.cli import app   # noqa: F401
print("  rfd3 / rfd3na import, rfd3.cli:app present")

import torch
print("  cuda available: %s" % torch.cuda.is_available())
PY

echo
echo "The rfd3 console script is at $VENV/bin/rfd3, which is what"
echo "config/pipeline.yaml's rfd3.binary expects."
echo
echo "Next: bash cluster/euler/stage_deps.sh   (fetches the checkpoints)"
