"""Run every available scoring tier and join the results into the leaderboard.

    python -m src.score.run_all --in results/run1 --out results/run1 \
        --config configs/pipeline.yaml --target configs/target.yaml

``--in`` is a run directory. This finds what is there and scores it:

* a ``designs.fasta`` anywhere under it   -> Tier-1 (sequence)
* a directory of ``<design_id>.pdb``      -> Tier-2 (structure)

then rebuilds ``<out>/leaderboard.csv``. Stages that already ran are cached, so
this is the safe thing to re-run after any new designs land — which is exactly
what Sunday morning looks like.

Tier-2 is skipped, loudly, when nothing has been folded: that is a true
statement about the run, not an error.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from ..cli import setup_logging
from ..io.target import REPO_ROOT
from ..report import leaderboard as lb

log = logging.getLogger("pipeline.score.run_all")


def find_designs_fasta(root: Path) -> list[Path]:
    return sorted(root.rglob("designs.fasta"))


def find_structure_dirs(root: Path) -> list[Path]:
    """Directories holding per-design structures, deduplicated."""
    dirs = {p.parent for p in root.rglob("*.pdb") if p.name != "input_backbone.pdb"
            and p.name != "motif_input.pdb"}
    return sorted(dirs)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="run directory")
    ap.add_argument("--out", default=None, help="default: same as --in")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "pipeline.yaml"))
    ap.add_argument("--target", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-tier2", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    setup_logging(args.verbose)

    root = Path(args.inp)
    out = Path(args.out or args.inp)
    if not root.exists():
        raise SystemExit(f"--in directory does not exist: {root}")
    common = ["--config", args.config, "--target", args.target]
    if args.force:
        common.append("--force")
    if args.verbose:
        common.append("-v")

    fastas = find_designs_fasta(root)
    if not fastas:
        raise SystemExit(f"no designs.fasta under {root}: run a generate stage first")
    for fasta in fastas:
        stage_out = out / f"tier1_{fasta.parent.name}"
        log.info("Tier-1 on %s", fasta)
        r = subprocess.run([sys.executable, "-m", "src.score.tier1", "--in", str(fasta),
                            "--out", str(stage_out), *common], cwd=REPO_ROOT)
        if r.returncode != 0:
            raise SystemExit(f"Tier-1 failed on {fasta} (exit {r.returncode})")

    struct_dirs = [] if args.skip_tier2 else find_structure_dirs(root)
    if not struct_dirs:
        log.warning("Tier-2 skipped: no folded structures under %s. "
                    "pLDDT, PAE, catalytic geometry and self-consistency will be absent from "
                    "the leaderboard, and ranking will have no terms. Run a fold stage.", root)
    for sdir in struct_dirs:
        stage_out = out / f"tier2_{sdir.name}"
        log.info("Tier-2 on %s", sdir)
        r = subprocess.run([sys.executable, "-m", "src.score.tier2", "--in", str(sdir),
                            "--out", str(stage_out), *common], cwd=REPO_ROOT)
        if r.returncode != 0:
            raise SystemExit(f"Tier-2 failed on {sdir} (exit {r.returncode})")

    df = lb.build(out, out, args.config)
    log.info("run_all: %d designs in the leaderboard (%d pass hard filters)",
             len(df), int(df["pass_hard"].sum()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
