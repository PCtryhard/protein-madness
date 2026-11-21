"""RFdiffusion partial diffusion: re-noise a backbone and re-denoise it.

On freezing: RFdiffusion holds whole residues, not individual atoms. Motif
scaffolding fixes residues named in a contig string, and partial diffusion
re-noises and re-denoises the whole structure. There is no switch that pins
three atoms and lets the rest of a sidechain move, so the target spec fixes
whole residues.

(RFD3, which the earlier runs in this repo used, does expose a
`select_fixed_atoms` option. That is a different tool with a different contig
syntax; see the note in config/targets/5fqj.yaml. This wrapper targets
RFdiffusion as PIPELINE.md specifies.)

Every output PDB gets a `.trb` pickle beside it, which is what `tracking.py`
reads to build the seed -> output residue map. A run whose `.trb` is missing or
lacks a motif residue stops: the alternative is a design whose motif silently
moved.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import MotifCfg, RFdiffusionCfg
from .base import StageError, StageResult, run_command


@dataclass(frozen=True)
class RFdiffusionOutput:
    """One generated backbone and its constraint map."""

    pdb: Path
    trb: Path
    index: int


def build_command(
    cfg: RFdiffusionCfg,
    input_pdb: Path,
    contigs: str,
    out_prefix: Path,
    num_designs: int,
    partial: bool,
) -> list[str]:
    """The Hydra-style command line RFdiffusion takes.

    `partial` selects the branch: a partial-diffusion step sets
    `diffuser.partial_T`, which re-noises only part of the trajectory and so
    keeps the parent's fold. Without it the run is a fresh motif-scaffolding
    job from the seed.
    """
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python", cfg.run_inference,
        f"inference.output_prefix={out_prefix}",
        f"inference.input_pdb={input_pdb}",
        f"inference.num_designs={num_designs}",
        f"contigmap.contigs=[{contigs}]",
        f"diffuser.T={cfg.diffuser_T}",
    ]
    if cfg.ckpt:
        cmd.append(f"inference.ckpt_override_path={cfg.ckpt}")
    if partial:
        cmd.append(f"diffuser.partial_T={cfg.partial_T}")
    cmd.extend(cfg.extra_args)
    return cmd


def run(
    cfg: RFdiffusionCfg,
    input_pdb: str | Path,
    contigs: str,
    out_prefix: str | Path,
    num_designs: int | None = None,
    partial: bool = True,
    gpu: int | None = None,
    timeout: float | None = None,
) -> tuple[list[RFdiffusionOutput], StageResult]:
    """Generate backbones, and return them paired with their `.trb` files."""
    input_pdb = Path(input_pdb)
    if not input_pdb.exists():
        raise StageError(f"rfdiffusion: input PDB not found: {input_pdb}")

    out_prefix = Path(out_prefix)
    count = cfg.designs_per_parent if num_designs is None else num_designs
    # The device is pinned with CUDA_VISIBLE_DEVICES below, not on the command
    # line, so the command itself does not depend on which card it lands on.
    cmd = build_command(cfg, input_pdb, contigs, out_prefix, count, partial=partial)

    env_overrides = {}
    if gpu is not None:
        env_overrides["CUDA_VISIBLE_DEVICES"] = str(gpu)

    result = run_command(
        cmd, stage="rfdiffusion", env_overrides=env_overrides, timeout=timeout
    )
    return collect_outputs(out_prefix, count), result


def collect_outputs(out_prefix: Path, expected: int) -> list[RFdiffusionOutput]:
    """Pair each output PDB with its `.trb`, or raise.

    A PDB without its `.trb` is unusable: the motif map cannot be built, and
    inferring it from the structure is exactly the proximity-based guess the
    rules forbid.
    """
    out_prefix = Path(out_prefix)
    outputs: list[RFdiffusionOutput] = []
    for index in range(expected):
        pdb = Path(f"{out_prefix}_{index}.pdb")
        trb = Path(f"{out_prefix}_{index}.trb")
        if not pdb.exists():
            continue
        if not trb.exists():
            raise StageError(
                f"rfdiffusion wrote {pdb} but no {trb}. Without the .trb the "
                "motif cannot be traced, and inferring the mapping from the "
                "structure is forbidden. Check the RFdiffusion version."
            )
        outputs.append(RFdiffusionOutput(pdb=pdb, trb=trb, index=index))

    if not outputs:
        parent = out_prefix.parent
        listing = sorted(p.name for p in parent.glob("*"))[:30] if parent.exists() else []
        raise StageError(
            f"rfdiffusion produced no PDBs at {out_prefix}_*.pdb. "
            f"Directory holds: {listing or 'nothing'}"
        )
    return outputs


def assert_motif_in_trb(trb_path: str | Path, motif: MotifCfg) -> None:
    """Fail fast if the contig string and `motif.fixed` disagree.

    Thin wrapper over tracking.seed_to_hal_map, kept here so a stage can check
    its own output before the design travels any further.
    """
    from ..tracking import seed_to_hal_map

    seed_to_hal_map(trb_path, motif)
