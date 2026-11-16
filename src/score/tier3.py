"""Tier-3 scoring: expensive physics. DELIBERATELY STUBBED — see TODO.md.

    python -m src.score.tier3 --in results/run1/fold --out results/run1/tier3 ...

The brief's order of work puts these last and says to stub them and move on,
because they are the least likely to change which designs we pick and the most
likely to eat Saturday. Both functions below raise with exactly what is
missing.

**Do not implement either of these until Tier-1 and Tier-2 are green on real
designs on all three backends.** If there is spare time before then, it goes to
the novelty databases (owner 3), which change what we can *claim*, not to
physics that mostly re-ranks designs we already rank.

What each would add, if the time exists:

* ``minimise_and_rmsd`` — a short OpenMM minimisation, then RMSD between the
  predicted and the relaxed structure. A design whose structure moves a long
  way under a force field was being held up by the predictor's optimism.
  Missing: a decision on force field and solvent model, and whether we trust a
  100 ps result enough to filter on it. OpenMM is not in env/requirements.txt.

* ``per_residue_energies`` — Rosetta-style per-residue energies, to find the
  strained residue in an otherwise good design.
  Missing: **PyRosetta needs a licence** (free for academics, but it is a
  decision, not an install). Nobody has one set up. Ask before adding it:
  adding a licensed dependency is one of the three things the brief says to
  stop and ask about.
"""
from __future__ import annotations

import sys

TOOL = "score.tier3"
VERSION = "0.0.0-stub"


def minimise_and_rmsd(structure_path, out_dir, steps: int = 1000, force_field: str = "amber14"):
    """Short OpenMM minimisation, then relaxed-state RMSD. Not implemented."""
    raise NotImplementedError(
        "OpenMM minimisation not implemented: openmm is not a pinned dependency and no "
        "force-field/solvent decision has been made. Tier-2 self-consistency RMSD covers the "
        "same question more cheaply. See TODO.md."
    )


def per_residue_energies(structure_path, out_dir):
    """Rosetta-style per-residue energies. Not implemented: PyRosetta needs a licence."""
    raise NotImplementedError(
        "PyRosetta per-residue energies not implemented: PyRosetta requires a licence that "
        "nobody on the team has set up, and adding a licensed dependency needs Max's sign-off. "
        "See TODO.md."
    )


def main(argv: list[str] | None = None) -> int:
    raise SystemExit(
        "score.tier3 is deliberately not implemented (OpenMM minimisation, PyRosetta energies).\n"
        "Missing: an OpenMM force-field/solvent decision, and a PyRosetta licence.\n"
        "Tier-1 and Tier-2 cover what we actually pitch. See TODO.md and the module docstring."
    )


if __name__ == "__main__":
    sys.exit(main())
