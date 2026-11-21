#!/usr/bin/env bash
# Install the boltz package on Euler, into its own venv.
#
# The weights are already there -- 14 GB under $SCRATCH/pdh/cache, in exactly
# the layout boltz expects (boltz2_conf.ckpt, boltz2_aff.ckpt, mols.tar,
# mols/) -- so this is only the Python package.
#
# A SEPARATE venv on purpose. ~/enzyme-hack-env belongs to the retired
# pipeline and holds a working torch 2.5.1+cu121; letting pip resolve boltz
# into it could upgrade torch underneath something that currently works. A new
# env costs disk and nothing else.
#
# Login node only: compute nodes on Euler have no outbound internet.
set -e -o pipefail

VENV="$HOME/pm-boltz-env"

set +u
module purge >/dev/null 2>&1 || true
module load stack/2024-06 gcc/12.2.0 python/3.12.8 >/dev/null 2>&1 || true
module load cuda/12.1.1 >/dev/null 2>&1 || true
module load eth_proxy >/dev/null 2>&1 || true
set -u

echo "python: $(command -v python3) ($(python3 --version 2>&1))"

if [ -x "$VENV/bin/python" ]; then
    echo "venv exists: $VENV"
else
    python3 -m venv "$VENV"
    echo "created $VENV"
fi

set +u; . "$VENV/bin/activate"; set -u
python -m pip install --upgrade -q pip setuptools wheel

if python -c "import boltz" >/dev/null 2>&1; then
    echo "boltz already installed"
else
    echo "installing boltz (this pulls torch and friends; several minutes)"
    python -m pip install boltz
fi

echo
echo "=== versions ==="
python - <<'PY'
import importlib.metadata as md
for name in ("boltz", "torch", "numpy"):
    try:
        print(f"  {name}: {md.version(name)}")
    except md.PackageNotFoundError:
        print(f"  {name}: NOT INSTALLED")
PY
echo
echo "=== boltz CLI ==="
command -v boltz && boltz --help 2>&1 | head -5 || echo "boltz CLI not on PATH"
echo
echo "=== weights already staged ==="
ls -la "$SCRATCH/pdh/cache/" 2>/dev/null | head -8
echo
echo "DONE. To fold on a GPU node:"
echo "  srun --gpus=1 --gres=gpumem:20g --time=01:00:00 --pty bash"
echo "  source $VENV/bin/activate"
echo "  boltz predict IN.yaml --cache \$SCRATCH/pdh/cache --out_dir OUT \\"
echo "    --accelerator gpu --devices 1 --output_format mmcif"
