"""Boltz-2 input and output for a whole generation at once.

**Batching is the point.** Boltz spends minutes loading the model and 21-34 s
actually folding (BRIEF.md 4.1), so one process per design would cost more in
startup than in work. Boltz's ``predict`` accepts a *directory* of YAMLs, so a
generation is one directory, one process, one model load.

Two conventions that are not ours to choose and must not drift:

* **the ligand is always present.** Every fold is a complex with the
  transition-state analogue. A structure without it is not a cheaper
  measurement, it is a different one.
* **``msa: empty``.** Measured on the organisers' own run: a de novo design gets
  an MSA of depth 1 with zero non-query rows, while a natural control gets 301.
  The alignment stage is pure cost for us, and skipping it removes the 28 GB
  host-memory requirement that would otherwise halve our GPU count.

Seeds: Boltz takes one ``--seed`` per invocation, so N seeds means N runs of the
same directory into N output directories. That is metric 4, and it is cheap.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..score.official import CatalyticSite, OfficialScore, score_prediction

log = logging.getLogger("pipeline.evolve.fold_batch")

LIGAND_CCD = "6NT"


def design_yaml(sequence: str, ligand_ccd: str = LIGAND_CCD) -> dict:
    """The Boltz input spec for one design, ligand bound, no MSA."""
    if not sequence:
        raise ValueError("empty sequence")
    return {
        "version": 1,
        "sequences": [
            {"protein": {"id": "A", "sequence": sequence, "msa": "empty"}},
            {"ligand": {"id": "B", "ccd": ligand_ccd}},
        ],
    }


def write_generation(
    out_dir: Path, designs: dict[str, str], ligand_ccd: str = LIGAND_CCD
) -> Path:
    """Write one YAML per design into ``out_dir``. Returns that directory.

    The file stem is the design id, which is how results are matched back:
    Boltz names its output directories after the input stems.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not designs:
        raise ValueError("no designs to write")
    for design_id, sequence in designs.items():
        if "/" in design_id or "\\" in design_id:
            raise ValueError(f"design id {design_id!r} is not a safe filename")
        path = out_dir / f"{design_id}.yaml"
        # newline="\n" always: these files are written on a laptop and read on a
        # Linux cluster, and Python would otherwise translate to CRLF on Windows.
        path.write_text(
            yaml.safe_dump(design_yaml(sequence, ligand_ccd), sort_keys=False),
            newline="\n",
        )
    log.info("wrote %d design YAMLs to %s", len(designs), out_dir)
    return out_dir


# ------------------------------------------------------------------ output

def prediction_paths(boltz_out: Path, design_id: str) -> tuple[Path, Path] | None:
    """(structure, confidence) for one design, or None if it did not come back.

    A missing design is a fact to record, not an exception: in a generation of
    fifty, one failure should not lose the other forty-nine.
    """
    boltz_out = Path(boltz_out)
    hits = list(boltz_out.rglob(f"{design_id}_model_0.cif"))
    confs = list(boltz_out.rglob(f"confidence_{design_id}_model_0.json"))
    if len(hits) != 1 or len(confs) != 1:
        return None
    return hits[0], confs[0]


def collect_generation(
    boltz_out: Path,
    design_ids: list[str],
    site: CatalyticSite,
    seed: int | None = None,
) -> dict[str, OfficialScore]:
    """Score every design that came back from one Boltz run."""
    scores: dict[str, OfficialScore] = {}
    for design_id in design_ids:
        paths = prediction_paths(boltz_out, design_id)
        if paths is None:
            log.warning("design %s produced no prediction in %s", design_id, boltz_out)
            continue
        try:
            scores[design_id] = score_prediction(paths[0], paths[1], site, seed=seed)
        except (ValueError, OSError, IndexError) as exc:
            log.warning("design %s could not be scored: %s", design_id, exc)
    log.info("scored %d/%d designs from %s", len(scores), len(design_ids), boltz_out)
    return scores


def collect_seeds(
    seed_dirs: dict[int, Path],
    design_ids: list[str],
    site: CatalyticSite,
) -> dict[str, list[OfficialScore]]:
    """Gather every seed's score for every design.

    Feed the result to :func:`src.score.official.aggregate_seeds` per design to
    get metric 4.
    """
    out: dict[str, list[OfficialScore]] = {design_id: [] for design_id in design_ids}
    for seed, directory in sorted(seed_dirs.items()):
        for design_id, score in collect_generation(directory, design_ids, site, seed=seed).items():
            out[design_id].append(score)
    return {k: v for k, v in out.items() if v}


# -------------------------------------------------------------- submission

@dataclass(frozen=True)
class SiteProfile:
    """Cluster-specific submission settings. Never hardcode these in code.

    Defaults are the AWS cluster; ``euler()`` is the 32-GPU reservation. Adding
    a third cluster should be another classmethod or a YAML file, not a patch
    through the codebase.
    """

    name: str = "pdh-aws"
    partition: str = "gpu"
    constraint: str | None = "g6.xlarge"
    account: str | None = None
    reservation: str | None = None
    cpus: int = 4
    gpus: int = 1
    #: Euler rejects --mem and demands --mem-per-cpu; AWS accepts --mem.
    mem_per_cpu: str | None = None
    mem: str | None = "14G"
    gres: str = "gpu:l4:1"
    extra: tuple[str, ...] = ()
    python: str = "python"
    boltz: str = "boltz"
    cache: str = ""

    @classmethod
    def euler(cls) -> "SiteProfile":
        """32x RTX 2080Ti, measured at 3.3 GB and 34 s per 260-residue fold."""
        return cls(
            name="euler",
            partition="gpu.120h",
            constraint=None,
            account="es_workshop",
            reservation="ws_49734_gpu",
            cpus=4,
            mem_per_cpu="6G",
            mem=None,
            gres="gpumem:10g",
            extra=("--gpus=1",),
            python="$SCRATCH/pdh/venv/bin/python",
            boltz="$SCRATCH/pdh/venv/bin/boltz",
            cache="$SCRATCH/pdh/cache",
        )

    def sbatch_header(self, job_name: str, log_path: str, time_limit: str = "04:00:00") -> list[str]:
        lines = [
            "#!/bin/bash -l",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --partition={self.partition}",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            f"#SBATCH --cpus-per-task={self.cpus}",
            f"#SBATCH --time={time_limit}",
            f"#SBATCH --output={log_path}",
        ]
        if self.account:
            lines.append(f"#SBATCH --account={self.account}")
        if self.reservation:
            lines.append(f"#SBATCH --reservation={self.reservation}")
        if self.constraint:
            lines.append(f"#SBATCH --constraint={self.constraint}")
        if self.mem_per_cpu:
            lines.append(f"#SBATCH --mem-per-cpu={self.mem_per_cpu}")
        elif self.mem:
            lines.append(f"#SBATCH --mem={self.mem}")
        if self.gres:
            lines.append(f"#SBATCH --gres={self.gres}")
        lines.extend(f"#SBATCH {flag}" for flag in self.extra)
        return lines


def fold_command(
    profile: SiteProfile, designs_dir: str, out_dir: str, seed: int, *, sampling_steps: int = 200
) -> str:
    """The Boltz invocation. Flags match the organisers' kemp-e2e.sbatch exactly.

    Deviating from their flags would make our numbers incomparable with the
    scores we are actually judged on, so the only knob exposed here is
    ``sampling_steps`` - lowered for triage, left at 200 for confirmation.
    """
    return (
        f'"{profile.boltz}" predict "{designs_dir}" '
        f'--model boltz2 --cache "{profile.cache}" --out_dir "{out_dir}" '
        f"--accelerator gpu --devices 1 --no_kernels "
        f"--sampling_steps {sampling_steps} --recycling_steps 3 --diffusion_samples 1 "
        f"--num_workers 0 --preprocessing-threads 1 --output_format mmcif --seed {seed}"
    )


def write_fold_script(
    path: Path,
    profile: SiteProfile,
    designs_dir: str,
    out_root: str,
    seeds: list[int],
    *,
    job_name: str = "pdh-fold",
    sampling_steps: int = 200,
) -> Path:
    """An sbatch script that folds one generation at every requested seed."""
    # Boltz scans the input directory and refuses anything that is not a .yaml
    # or .fasta -- including a subdirectory. Writing output inside the input
    # directory therefore kills the run with
    #   "Found directory .../boltz_seed1 instead of .fasta or .yaml"
    # It cost us a 48-design batch, so it is checked here rather than trusted.
    designs_norm = str(designs_dir).replace("\\", "/").rstrip("/")
    out_norm = str(out_root).replace("\\", "/").rstrip("/")
    if out_norm == designs_norm or out_norm.startswith(designs_norm + "/"):
        raise ValueError(
            f"out_root {out_root!r} is inside designs_dir {designs_dir!r}; Boltz "
            f"scans the input directory and will refuse the subdirectory it finds"
        )

    path = Path(path)
    lines = profile.sbatch_header(job_name, f"{out_root}/fold-%j.log")
    lines += [
        "",
        "set -o pipefail",
        'export HF_HUB_OFFLINE=1   # a hang on a compute node is worse than an error',
        f'mkdir -p "{out_root}"',
        "",
    ]
    for seed in seeds:
        lines += [
            f'echo "=== seed {seed} at $(date +%T)"',
            fold_command(profile, designs_dir, f"{out_root}/seed_{seed}", seed,
                         sampling_steps=sampling_steps),
            "",
        ]
    lines.append('echo "=== fold batch complete"')
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" is not cosmetic: sbatch refuses a script containing CRLF
    # ("Batch script contains DOS line breaks"), and we write these on Windows.
    path.write_text("\n".join(lines) + "\n", newline="\n")
    log.info("wrote fold script %s (%d seeds, site=%s)", path, len(seeds), profile.name)
    return path
