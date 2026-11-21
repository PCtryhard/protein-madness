"""Screen backbones or designs against published structures by TM-score.

    # once, on a node with internet (head/login node, never inside a job):
    python scripts/tm_screen.py --build-set --set-dir /mnt/work/tools/kemp_pdb

    # then, as often as you like:
    python scripts/tm_screen.py --designs '/mnt/work/mpnn_batch1/*/backbone.pdb' \
                                --set-dir /mnt/work/tools/kemp_pdb --out tm.csv

Why this exists: the eligibility rule disqualifies anything with TM-score > 0.7
to a published structure, and we were blind to it for most of the first day.
Measured on the first batch of 8 de novo backbones, **two failed** - at 0.703
and 0.705, both against an engineered Diels-Alderase (3I1C). They were the two
backbones with the *best* burial, so this check actively opposes the metric we
are otherwise selecting on. It has to run before folding, not at submission.

This is the targeted screen and therefore a **lower bound**: a hundred-odd
Kemp-related entries is not the PDB. A pass means "not obviously
disqualified". The full search needs the Foldseek PDB database.
"""
from __future__ import annotations

import argparse
import csv
import glob
import logging
from pathlib import Path

from src.score.eligibility import (
    KEMP_SEARCH_TERMS,
    MAX_TM_SCORE,
    build_reference_set,
    max_tm_vs_set,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set-dir", type=Path, required=True,
                        help="where the reference structures live")
    parser.add_argument("--build-set", action="store_true",
                        help="download the reference set first (needs internet)")
    parser.add_argument("--designs", help="glob of design .pdb files to screen")
    parser.add_argument("--out", type=Path, help="write a CSV here")
    parser.add_argument("--tmalign", default="/mnt/work/tools/bin/TMalign")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.build_set:
        paths = build_reference_set(args.set_dir, KEMP_SEARCH_TERMS)
        print(f"reference set: {len(paths)} structures in {args.set_dir}")
        if not args.designs:
            return 0

    targets = sorted(args.set_dir.glob("*.pdb"))
    if not targets:
        print(f"no reference structures in {args.set_dir}; run with --build-set first")
        return 1
    if not args.designs:
        print(f"{len(targets)} reference structures available")
        return 0

    designs = sorted(Path(p) for p in glob.glob(args.designs))
    if not designs:
        print(f"no designs matched {args.designs!r}")
        return 1

    rows = []
    print(f"{'design':28s} {'maxTM':>6s}  {'closest':>8s}   verdict")
    print("-" * 62)
    for design in designs:
        score, closest = max_tm_vs_set(design, targets, args.tmalign)
        eligible = score <= MAX_TM_SCORE
        name = design.parent.name if design.name == "backbone.pdb" else design.stem
        rows.append({"design": name, "tm_score": round(score, 4),
                     "closest": closest, "eligible": eligible})
        print(f"{name:28s} {score:6.3f}  {closest:>8s}   "
              f"{'ok' if eligible else f'FAIL >{MAX_TM_SCORE}'}")

    failed = [r for r in rows if not r["eligible"]]
    print(f"\n{len(rows) - len(failed)}/{len(rows)} eligible; "
          f"highest TM-score {max(r['tm_score'] for r in rows):.3f} "
          f"(limit {MAX_TM_SCORE})")
    print("NOTE: targeted screen, so these are LOWER BOUNDS on similarity to "
          "all of the PDB.")

    if args.out:
        with args.out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["design", "tm_score",
                                                        "closest", "eligible"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
