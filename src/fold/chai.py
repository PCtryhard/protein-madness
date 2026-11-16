"""Chai-1 co-folding. NOT IMPLEMENTED — see TODO.md.

    python -m src.fold.chai --in designs.fasta --out results/run/fold ...

What is missing, exactly: nobody on this team has run Chai-1, and its Python
API and output layout have not been verified against the installed version.
Writing a wrapper from memory risks mis-reading the confidence arrays, which
would put wrong pLDDT and PAE numbers into the leaderboard while looking
entirely normal.

Boltz-2 (``src/fold/boltz.py``) already covers the ligand co-folding case and
its CLI and output layout are verified. Implement this only if Boltz-2 is
unavailable or Chai-1 turns out to be materially better on the Saturday target
— and if so, verify the output file names and array keys against a real run
before trusting a single number.
"""
from __future__ import annotations

import sys

from .. import cli

TOOL = "fold.chai"
VERSION = "0.0.0-stub"
BACKEND = "chai"


def fold_batch(fasta_path, out_dir, smiles: str | None = None, device: str = "cuda"):
    """Fold a FASTA with Chai-1. Not implemented: see module docstring."""
    raise NotImplementedError(
        "Chai-1 wrapper not implemented: its API and output layout have not been verified "
        "against a real run. Use src/fold/boltz.py for ligand co-folding. See TODO.md."
    )


def main(argv: list[str] | None = None) -> int:
    ap = cli.base_parser(__doc__)
    args = ap.parse_args(argv)
    cli.setup_logging(args.verbose)
    raise SystemExit(
        "fold.chai is not implemented. Missing: a verified Chai-1 API and output layout.\n"
        "Use --backend boltz for ligand co-folding, or --backend esmfold. See TODO.md."
    )


if __name__ == "__main__":
    sys.exit(main())
