#!/usr/bin/env bash
#SBATCH --job-name=pm-fastrelax-01
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --time=06:00:00
#SBATCH --output=runs/slurm/%x-%j.out

set -eo pipefail
cd "${PM_REPO:-$SLURM_SUBMIT_DIR}"
source "$HOME/protein-madness-env/activate.sh"
name="fastrelax-01-$SLURM_JOB_ID"
motif="${PM_MOTIF:-data/01/inputs/01_fastrelax_motif_af3.pdb}"
test -s "$motif" || { echo "Missing motif: $motif" >&2; exit 1; }

for stage in inputs rfd3 ligandmpnn boltz scores; do
    mkdir -p "data/01/$stage/$name"
done
run_inputs="$PWD/data/01/inputs/$name"
export PM_RFD3_BIN="${PM_RFD3_BIN:-$HOME/rfd3-env/bin/rfd3}"
export PM_BOLTZ_BIN="${PM_BOLTZ_BIN:-$HOME/protein-madness-env/bin/boltz}"
export PM_BOLTZ_CACHE="${PM_BOLTZ_CACHE:-$SCRATCH/boltz-cache}"
export PM_MOTIF="$motif"
python - "$run_inputs" <<'PY'
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import yaml

out = Path(sys.argv[1])
scratch = Path(os.environ["SCRATCH"])
home = Path.home()
roots = [scratch / directory for directory in ("weights", "pdh", "tools")]
roots += [home / directory for directory in ("LigandMPNN", "tools")]

def existing(var, filename, parent=None):
    if os.environ.get(var):
        matches = [Path(os.environ[var]).expanduser()]
    else:
        matches = sorted({p.resolve() for root in roots if root.is_dir()
                          for p in root.rglob(filename) if p.is_file()
                          and (parent is None or p.parent.name.lower() == parent)})
    if len(matches) != 1 or not matches[0].is_file():
        raise SystemExit(f"Set {var} to your existing file. Found: {matches}")
    return str(matches[0].resolve())

def executable(var):
    path = Path(os.environ[var]).expanduser()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise SystemExit(f"Missing executable: {var}={path}")
    return str(path.absolute())

cfg = yaml.safe_load(Path("config/pipeline.euler.yaml").read_text())
rfd3 = executable("PM_RFD3_BIN")
help_result = subprocess.run([rfd3, "--help"], capture_output=True, text=True, check=True)
if any(line.split()[:1] == ["design"] for line in help_result.stdout.splitlines()):
    wrapper = out / "rfd3-design.sh"
    wrapper.write_text(f"#!/usr/bin/env bash\nexec {shlex.quote(rfd3)} design \"$@\"\n")
    wrapper.chmod(0o755)
    rfd3 = str(wrapper)

cfg["rfd3"].update(binary=rfd3, checkpoint=existing("PM_RFD3_CKPT", "rfd3_latest.ckpt"))
mpnn = existing("PM_MPNN_SCRIPT", "run.py", parent="ligandmpnn")
cfg["ligandmpnn"].update(
    run_script=mpnn,
    checkpoint=existing("PM_MPNN_CKPT", "ligandmpnn_v_32_010_25.pt"),
)
cache = Path(os.environ["PM_BOLTZ_CACHE"]).expanduser().resolve()
required = ("boltz2_conf.ckpt", "boltz2_aff.ckpt", "mols.tar", "mols")
if missing := [asset for asset in required if not (cache / asset).exists()]:
    raise SystemExit(f"Boltz cache {cache} is missing: {', '.join(missing)}")
cfg["boltz"].update(binary=executable("PM_BOLTZ_BIN"), cache_dir=str(cache), assets_dir=None)

target = yaml.safe_load(Path("config/targets/01_his_asp.yaml").read_text())
motif = out / "motif.pdb"
shutil.copyfile(os.environ["PM_MOTIF"], motif)
target["seeds"][0]["pdb"] = str(motif)
for name, data in (("pipeline.yaml", cfg), ("target.yaml", target)):
    (out / name).write_text(yaml.safe_dump(data, sort_keys=False))
PY

pipeline="$run_inputs/pipeline.yaml"
target="$run_inputs/target.yaml"
python -m src.cli smoke --pipeline "$pipeline" --target "$target"
python -m src.cli measure --pipeline "$pipeline" --target "$target" --pdb "$run_inputs/motif.pdb"

# Keep native outputs and copy each stage to its matching data/01 folder.
export_pm_stages() {
    local status=$?
    trap - EXIT
    python - "$name" <<'PY'
from pathlib import Path
import shutil
import sys

name = sys.argv[1]
run = Path("runs") / name
structures = run / "structures"
folders = {"rfd3": "rfd3", "ligandmpnn": "mpnn_*", "boltz": "boltz_r*"}
for stage, pattern in folders.items():
    dest = Path("data/01") / stage / name
    for folder in sorted(structures.rglob(pattern)):
        if folder.is_dir():
            shutil.copytree(folder, dest / folder.relative_to(structures), dirs_exist_ok=True)
dest = Path("data/01/scores") / name
for filename in ("designs.csv", "sequences.fasta", "progress.txt", "report.md", "stages.jsonl"):
    file = run / filename
    if file.is_file():
        shutil.copy2(file, dest / filename)
PY
    exit "$status"
}
trap export_pm_stages EXIT
python -m src.cli run --pipeline "$pipeline" --target "$target" --name "$name"
python -m src.cli report --run "runs/$name" --write
