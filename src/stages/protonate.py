"""Hydrogen placement between Boltz-2 and scoring.

Boltz-2 writes no hydrogens, so a donor-hydrogen-acceptor angle cannot be
measured on its raw output. The angle that idealises near 180 degrees is
D-H...A; the heavy-atom angle at the donor is a different quantity whose ideal
is roughly 109 to 120 degrees, set by hybridisation. Applying a 180 degree
target to heavy atoms -- which is what happens if hydrogens are simply skipped
-- penalises good geometry and rewards a nearly linear antecedent arrangement
that is physically wrong.

Whether this stage runs at all depends on `scoring.angle.rule`:

* `acceptor_side_always` needs no hydrogens. The stage is skipped and nothing
  in the score depends on a protonator's choices.
* `split_by_donor_type` needs hydrogens for its fixed-H donors, so the stage
  must run and `protonation.engine` must not be `none` (config.py enforces
  that pairing at load time).

The remaining subtlety, and the reason `rotatable_donor` exists in the target
spec: for rotatable donors most protonation tools place the hydrogen to
optimise hydrogen bonding. Measuring D-H...A on a serine hydroxyl then partly
measures what PDBFixer decided, not what Boltz-2 predicted. Those bonds are
measured at the acceptor under both rules.
"""

from __future__ import annotations

from pathlib import Path

from ..config import ProtonationCfg
from .base import StageError, StageResult, run_command


def needed(rule: str) -> bool:
    """Whether the configured angle rule requires hydrogens at all."""
    return rule == "split_by_donor_type"


def run(
    cfg: ProtonationCfg,
    input_structure: str | Path,
    output_pdb: str | Path,
    timeout: float | None = None,
) -> tuple[Path, StageResult | None]:
    """Add hydrogens. Returns the path to use for scoring.

    With `engine: none` the input is returned unchanged -- which is correct
    only under `acceptor_side_always`, and config.py refuses the other pairing.
    """
    input_structure = Path(input_structure)
    output_pdb = Path(output_pdb)
    if not input_structure.exists():
        raise StageError(f"protonation: input not found: {input_structure}")

    if cfg.engine == "none":
        return input_structure, None
    if cfg.engine == "pdbfixer":
        return _run_pdbfixer(cfg, input_structure, output_pdb), None
    if cfg.engine == "reduce":
        return _run_reduce(cfg, input_structure, output_pdb, timeout)
    raise StageError(f"unknown protonation engine {cfg.engine!r}")


def _run_pdbfixer(
    cfg: ProtonationCfg, input_structure: Path, output_pdb: Path
) -> Path:
    """PDBFixer via openmm.app.Modeller.addHydrogens at the configured pH.

    The least-friction option, and in-process rather than a subprocess.
    Imported lazily so the rest of the pipeline, and the whole test suite, runs
    without OpenMM installed.
    """
    try:
        from openmm.app import PDBFile
        from pdbfixer import PDBFixer
    except ImportError as exc:
        raise StageError(
            "protonation.engine='pdbfixer' needs pdbfixer and openmm, which are "
            "not installed. Install them, use engine='reduce', or set "
            "scoring.angle.rule='acceptor_side_always', which needs no hydrogens."
        ) from exc

    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    fixer = PDBFixer(filename=str(input_structure))
    fixer.findMissingResidues()
    # Deliberately NOT filling in missing residues: a predicted structure has
    # none, and adding any would change the sequence the design is scored as.
    fixer.missingResidues = {}
    fixer.findNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(cfg.ph)
    with output_pdb.open("w", encoding="utf-8") as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh, keepIds=True)

    if not output_pdb.exists() or output_pdb.stat().st_size == 0:
        raise StageError(f"protonation: pdbfixer wrote nothing to {output_pdb}")
    return output_pdb


def _run_reduce(
    cfg: ProtonationCfg,
    input_structure: Path,
    output_pdb: Path,
    timeout: float | None,
) -> tuple[Path, StageResult]:
    """`reduce` from the Richardson lab: better at orienting rotatable groups.

    `-FLIP` lets it optimise Asn/Gln/His flips and rotatable hydrogens, which
    is exactly why bonds on rotatable donors are scored at the acceptor
    instead: otherwise the angle term would partly measure this program's
    choices.
    """
    raise StageError(
        "protonation.engine='reduce' is not implemented. `reduce` writes the "
        "whole protonated PDB to stdout, and base.run_command keeps only the "
        "tail, so wiring it up naively would write a truncated structure and "
        "every angle measured on it would be wrong. Use engine='pdbfixer', or "
        "extend base.run_command to stream stdout to a file first. Refusing "
        "rather than returning a structure that looks fine and is not."
    )
