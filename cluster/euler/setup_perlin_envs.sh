#!/usr/bin/env bash
# Recreate the Perlin model stacks on an Euler login node.
set -euo pipefail

: "${SCRATCH:?Run on Euler with SCRATCH set}"
command -v conda >/dev/null || { echo "Conda must be available in this shell." >&2; exit 1; }

envs="${PM_PERLIN_ENVS:-$HOME/pm-perlin-envs}"
assets="${PM_PERLIN_ASSETS:-$SCRATCH/pm-perlin}"
rfd3="$envs/rfd3"
mpnn="$envs/ligandmpnn"
boltz="$envs/boltz2"
source_dir="$assets/LigandMPNN"
mpnn_weights="$assets/weights/ligandmpnn_v_32_010_25.pt"
rfd3_weights="$assets/weights/rfd3"
boltz_cache="$assets/boltz-cache"
mkdir -p "$envs" "$assets/weights" "$rfd3_weights" "$boltz_cache"

make_env() {
    local prefix=$1 python_version=$2
    if [[ ! -x "$prefix/bin/python" ]]; then
        conda create -y -p "$prefix" "python=$python_version" pip
    fi
}

make_env "$rfd3" 3.12
make_env "$mpnn" 3.11.18
make_env "$boltz" 3.11.18

for prefix in "$rfd3" "$boltz"; do
    "$prefix/bin/python" -m pip install 'torch==2.7.1+cu126' \
        --index-url https://download.pytorch.org/whl/cu126
done
"$rfd3/bin/python" -m pip install 'rc-foundry[rfd3]==0.2.0' \
    'atomworks==2.2.1' 'torch==2.7.1+cu126'
"$boltz/bin/python" -m pip install 'boltz==2.2.1' 'numpy==1.26.4' \
    'torch==2.7.1+cu126'

revision=26ec57ac976ade5379920dbd43c7f97a91cf82de
if [[ ! -d "$source_dir/.git" ]]; then
    git clone https://github.com/dauparas/LigandMPNN.git "$source_dir"
    git -C "$source_dir" checkout --detach "$revision"
fi
if [[ "$(git -C "$source_dir" rev-parse HEAD)" != "$revision" ]]; then
    echo "Unexpected LigandMPNN revision in $source_dir" >&2
    exit 1
fi
"$mpnn/bin/python" -m pip install 'torch==2.2.1+cu121' \
    --index-url https://download.pytorch.org/whl/cu121
"$mpnn/bin/python" -m pip install -r "$source_dir/requirements.txt"

if [[ ! -s "$mpnn_weights" ]]; then
    curl -fL --retry 2 https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_010_25.pt \
        -o "$mpnn_weights.part"
    mv "$mpnn_weights.part" "$mpnn_weights"
fi
"$rfd3/bin/foundry" install rfd3 --checkpoint-dir "$rfd3_weights"
"$boltz/bin/python" - "$boltz_cache" <<'PY'
from pathlib import Path
import sys
from boltz.main import download_boltz2

download_boltz2(Path(sys.argv[1]))
PY

adapter="$envs/ligandmpnn_adapter.py"
"$mpnn/bin/python" - "$adapter" "$mpnn/bin/python" "$source_dir/run.py" <<'PY'
from pathlib import Path
import sys

adapter, python, script = sys.argv[1:]
Path(adapter).write_text(
    f"import os\nimport sys\nos.execv({python!r}, [{python!r}, {script!r}, *sys.argv[1:]])\n"
)
PY

for prefix in "$rfd3" "$mpnn" "$boltz"; do
    "$prefix/bin/python" -m pip check
done
"$rfd3/bin/rfd3" --help >/dev/null
"$mpnn/bin/python" "$adapter" --help >/dev/null
"$boltz/bin/boltz" predict --help >/dev/null

mapfile -t rfd3_checkpoints < <(find "$rfd3_weights" -type f -name '*.ckpt' | sort)
if [[ ${#rfd3_checkpoints[@]} -ne 1 ]]; then
    printf 'Expected one RFD3 checkpoint under %s; found %s\n' \
        "$rfd3_weights" "${#rfd3_checkpoints[@]}" >&2
    exit 1
fi
{
    printf 'export PM_RFD3_BIN=%q\n' "$rfd3/bin/rfd3"
    printf 'export PM_RFD3_CKPT=%q\n' "${rfd3_checkpoints[0]}"
    printf 'export PM_MPNN_SCRIPT=%q\n' "$adapter"
    printf 'export PM_MPNN_CKPT=%q\n' "$mpnn_weights"
    printf 'export PM_BOLTZ_BIN=%q\n' "$boltz/bin/boltz"
    printf 'export PM_BOLTZ_CACHE=%q\n' "$boltz_cache"
} > "$envs/launcher.env"

printf 'Model CLIs passed. Run: source %q\n' "$envs/launcher.env"
