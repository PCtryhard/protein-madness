"""ColabFold (AlphaFold2) folding. NOT IMPLEMENTED — see TODO.md.

    python -m src.fold.colabfold --in designs.fasta --out results/run/fold ...

What is missing, exactly: a decision about where MSAs come from. ColabFold
needs them, and the MMseqs2 web API that ``--msa-mode server`` would use is
unreachable from Euler compute nodes (restricted outbound internet) even though
it works on the login node. See the MSA gotcha in MANIFEST.md.

Before this can be written honestly, one of these has to be true:
  1. MSAs for the Saturday target are precomputed on the Euler LOGIN node (or a
     laptop) and cached in ``results/msa/<design_id>.a3m``, in which case this
     wrapper reads them and never touches the network; or
  2. we accept running ColabFold only on Modal, where outbound internet works.

Until then this raises rather than silently producing structures from a failed
or truncated MSA, which would make every downstream confidence number wrong in
a way that looks fine. ESMFold (``src/fold/esmfold.py``) is the single-sequence
default and needs no MSA at all.
"""
from __future__ import annotations

import sys

from .. import cli

TOOL = "fold.colabfold"
VERSION = "0.0.0-stub"
BACKEND = "colabfold"


def fold_batch(fasta_path, out_dir, msa_dir=None, num_recycles: int = 1, device: str = "cuda"):
    """Fold a FASTA with ColabFold. Not implemented: see module docstring."""
    raise NotImplementedError(
        "ColabFold wrapper not implemented: no decision yet on where MSAs come from "
        "(the MMseqs2 web API fails on Euler compute nodes). Use src/fold/esmfold.py, "
        "or precompute MSAs into results/msa/ first. See TODO.md."
    )


def main(argv: list[str] | None = None) -> int:
    ap = cli.base_parser(__doc__)
    ap.add_argument("--msa-dir", default=None, help="directory of precomputed .a3m files")
    args = ap.parse_args(argv)
    cli.setup_logging(args.verbose)
    raise SystemExit(
        "fold.colabfold is not implemented. Missing: a precomputed-MSA source "
        "(Euler compute nodes cannot reach the MMseqs2 web API).\n"
        "Use --backend esmfold (single-sequence, no MSA), or precompute MSAs on the login "
        "node into results/msa/ and implement fold_batch(). See TODO.md."
    )


if __name__ == "__main__":
    sys.exit(main())
