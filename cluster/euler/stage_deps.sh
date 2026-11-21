#!/usr/bin/env bash
# Stage every external dependency this pipeline needs on Euler.
#
#   bash cluster/euler/stage_deps.sh
#
# LOGIN NODE ONLY -- compute nodes have no outbound internet, which is the
# whole reason this script exists as a separate step.
#
# Three of the four tools come from public sources and this script fetches
# them in full:
#
#   LigandMPNN   github.com/dauparas/LigandMPNN (MIT) + its own get_model_params.sh
#   Boltz-2      pip, then its weights downloaded here rather than on a node
#   our code     the repo you are standing in
#
# RFD3 does not. It has no pip metadata, no LICENSE and no public download; it
# was hand-placed into site-packages from the hackathon's AWS reference mount.
# This script therefore VERIFIES RFD3 against cluster/euler/MANIFEST.sha256
# and tells you what is missing -- it cannot fetch it, and the repo does not
# redistribute it. See the RFD3 section of EULER.md.
#
# Idempotent: anything already present and correct is left alone.
set -e -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PDH="${PM_PDH:-$SCRATCH/pdh}"

TOOLS="$PDH/tools"
WEIGHTS="$PDH/weights"
CACHE="$PDH/cache"

ok()   { printf '  \033[32mOK\033[0m    %s\n' "$1"; }
todo() { printf '  \033[33mTODO\033[0m  %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; }

set +u
module purge >/dev/null 2>&1 || true
module load stack/2024-06 gcc/12.2.0 python/3.12.8 >/dev/null 2>&1 || true
module load eth_proxy >/dev/null 2>&1 || true
set -u

echo "staging into $PDH"
mkdir -p "$TOOLS" "$WEIGHTS/rfd3" "$CACHE"
echo

# ---------------------------------------------------------------- LigandMPNN
echo "[1/4] LigandMPNN (public, MIT)"
if [ -f "$TOOLS/LigandMPNN/run.py" ]; then
    ok "$TOOLS/LigandMPNN"
else
    git clone --depth 1 https://github.com/dauparas/LigandMPNN.git \
        "$TOOLS/LigandMPNN"
    ok "cloned LigandMPNN"
fi

# Its weights land in model_params/ by default; we keep them in a lowercase
# sibling directory because the config names the checkpoint explicitly.
# NOTE: tools/LigandMPNN (the repo) and tools/ligandmpnn (the weights) are two
# different directories on a case-sensitive filesystem. Both are correct.
echo "[2/4] LigandMPNN weights (public)"
if [ -f "$TOOLS/ligandmpnn/ligandmpnn_v_32_010_25.pt" ]; then
    ok "$TOOLS/ligandmpnn/ligandmpnn_v_32_010_25.pt"
else
    mkdir -p "$TOOLS/ligandmpnn"
    (
        cd "$TOOLS/LigandMPNN"
        bash get_model_params.sh "$TOOLS/ligandmpnn"
    )
    if [ -f "$TOOLS/ligandmpnn/ligandmpnn_v_32_010_25.pt" ]; then
        ok "downloaded LigandMPNN weights"
    else
        bad "get_model_params.sh did not produce ligandmpnn_v_32_010_25.pt"
        exit 1
    fi
fi

# ---------------------------------------------------------------- Boltz-2
echo "[3/4] Boltz-2 weights (public, downloaded HERE not on a compute node)"
BOLTZ="$HOME/pm-boltz-env/bin/boltz"
WANT_CONF="$CACHE/boltz2_conf.ckpt"
WANT_MOLS="$CACHE/mols"
if [ -f "$WANT_CONF" ] && [ -e "$WANT_MOLS" ]; then
    ok "$CACHE already holds boltz2_conf.ckpt and mols/"
elif [ ! -x "$BOLTZ" ]; then
    todo "run cluster/euler/setup_boltz.sh first, then re-run this script"
else
    # Boltz populates its cache on first prediction. A one-residue job is the
    # cheapest way to trigger the download without reserving a GPU.
    TMPYAML="$(mktemp -d)/tiny.yaml"
    cat > "$TMPYAML" <<'YAML'
version: 1
sequences:
- protein:
    id: A
    sequence: GGGG
    msa: empty
YAML
    echo "  triggering the Boltz weight download (several GB, a few minutes)..."
    "$BOLTZ" predict "$TMPYAML" --cache "$CACHE" --accelerator cpu \
        --devices 1 --recycling_steps 1 --sampling_steps 10 \
        --diffusion_samples 1 --out_dir "$(dirname "$TMPYAML")/out" \
        >/dev/null 2>&1 || true
    if [ -f "$WANT_CONF" ]; then
        ok "Boltz weights staged in $CACHE"
    else
        bad "Boltz did not populate $CACHE -- check eth_proxy and retry"
        exit 1
    fi
fi

# ---------------------------------------------------------------- RFD3
echo "[4/4] RFD3 (NOT public, NOT redistributed by this repo)"
MANIFEST="$HERE/MANIFEST.sha256"
RFD3_CKPT="$WEIGHTS/rfd3/rfd3_latest.ckpt"
RFD3_REQ="$HOME/rfd3_requirements.txt"
missing=0

if [ -f "$RFD3_CKPT" ]; then
    if [ -f "$MANIFEST" ]; then
        want=$(awk '/rfd3_latest\.ckpt/ {print $1; exit}' "$MANIFEST")
        echo "  checksumming 2.6 GB, this takes a minute..."
        got=$(sha256sum "$RFD3_CKPT" | awk '{print $1}')
        if [ "$want" = "$got" ]; then
            ok "rfd3_latest.ckpt matches MANIFEST.sha256"
        else
            bad "rfd3_latest.ckpt does NOT match the manifest"
            echo "        expected $want"
            echo "        got      $got"
            missing=1
        fi
    else
        ok "rfd3_latest.ckpt present (no manifest to check it against)"
    fi
else
    todo "missing $RFD3_CKPT"
    missing=1
fi

if [ -f "$RFD3_REQ" ]; then
    ok "$RFD3_REQ"
else
    cp "$HERE/rfd3_requirements.txt" "$RFD3_REQ"
    ok "copied rfd3_requirements.txt to \$HOME (setup_rfd3.sh expects it there)"
fi

if "$HOME/pm-rfd3-env/bin/python" -c "import rfd3" >/dev/null 2>&1; then
    ok "the rfd3 package imports"
else
    todo "the rfd3 package is not importable in ~/pm-rfd3-env"
    missing=1
fi

echo
if [ "$missing" -ne 0 ]; then
    cat <<'EOF'
RFD3 is incomplete, and this script cannot fix that.

  * The rfd3 / rfd3na Python packages (~2.4 MB) carry no LICENSE, so this
    repo does not redistribute them.
  * rfd3_latest.ckpt is 2.6 GB, which exceeds GitHub's 100 MB file limit and
    would need a paid LFS data pack, so it is not in the repo either.

To get them, copy both from a machine that already has them -- no new
credentials needed:

  ssh AWS "tar c -C <site-packages> rfd3 rfd3na" \
    | ssh euler "tar x -C ~/pm-rfd3-env/lib/python3.12/site-packages"

  ssh AWS "cat <path>/rfd3_latest.ckpt" \
    > /tmp/rfd3_latest.ckpt   # then scp it to $SCRATCH/pdh/weights/rfd3/

Then re-run this script to verify the checksums, and ask the organisers what
the redistribution terms actually are before putting RFD3 anywhere else.
EOF
    exit 1
fi

echo "all dependencies staged. Next: cluster/preflight.py"
