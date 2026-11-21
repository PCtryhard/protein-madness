"""Is the score signal, or is it Boltz-2 prediction noise?

    python scripts/noise_control.py --from-csv          # CPU, from existing data, now
    python scripts/noise_control.py --target config/targets/01_his_asp.yaml \
                                    --seeds 5 --sequences 3     # needs a GPU

WHAT THIS MEASURES AND WHY IT DECIDES THE RUN
---------------------------------------------

The search is a hill-climb on one number. If Boltz-2's own seed-to-seed
variation on the SAME sequence is comparable to the variation between
genuinely different designs, then the score cannot tell designs apart and the
search is climbing noise. Worse, `patience` then terminates on noise rather
than on convergence.

So: two spreads.

    within-design SD    re-predict one sequence with several seeds, take the
                        SD of each score term. This is the noise floor.
    between-design SD   the SD of per-design means across different designs.
                        This is the signal you hope to climb.

The ratio within/between is the verdict:

    < 0.35   signal dominates; eval_replicates 1 is fine
    0.35-0.5 marginal; consider eval_replicates 3
    > 0.5    mostly noise. Raise eval_replicates to 3 and patience to 4, and
             do not trust a term whose ratio is near or above 1.

WHY 5 SEEDS, NOT 10
-------------------

PIPELINE.md asks for ten seeds on one sequence. Five across several sequences
is a better use of the same compute, for two reasons.

The precision of an SD estimate from n samples goes as sigma/sqrt(2(n-1)):
about 35% relative error at n=5 and 24% at n=10. That looks like an argument
for 10 -- but only if you measure one sequence. The quantity that actually
matters here is the MEAN within-design SD over many designs, and averaging
that over m designs tightens it by sqrt(m). Five seeds on three sequences (15
calls) estimates the noise floor better than ten seeds on one (10 calls), and
it also tells you whether the noise is sequence-dependent, which ten seeds on
one sequence cannot.

Five is also this project's existing convention: RUNLOG.md section 4 calls it
"the five-seed rule", and data/No_mutation_run/ holds exactly five seeds per
design. Staying at five keeps the new numbers comparable with the old ones.
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_CSV = ROOT / "data" / "No_mutation_run" / "scores" / "per_seed.csv"

# Thresholds for the verdict. Not tunables in pipeline.yaml because they are
# properties of this analysis, not of the pipeline.
RATIO_SIGNAL = 0.35
RATIO_NOISE = 0.50


def verdict(ratio: float) -> str:
    if ratio > RATIO_NOISE:
        return "MOSTLY NOISE"
    if ratio > RATIO_SIGNAL:
        return "marginal"
    return "signal dominates"


def summarise(
    label: str, per_design: dict[str, list[float]]
) -> tuple[float, float, float] | None:
    """Mean within-design SD, between-design SD, and their ratio."""
    withins = [st.stdev(v) for v in per_design.values() if len(v) > 1]
    means = [st.fmean(v) for v in per_design.values() if v]
    if len(withins) < 2 or len(means) < 2:
        print(f"  {label:<22} not enough data "
              f"({len(withins)} designs with >1 seed, {len(means)} with any)")
        return None
    within = st.fmean(withins)
    between = st.stdev(means)
    ratio = within / between if between else float("inf")
    print(f"  {label:<22} {within:>9.3f} {between:>11.3f} {ratio:>7.2f}  "
          f"{verdict(ratio)}")
    return within, between, ratio


def header() -> None:
    print(f"  {'quantity':<22} {'within':>9} {'between':>11} {'ratio':>7}  verdict")
    print("  " + "-" * 68)


# ---------------------------------------------------------------------------
# mode A: the existing five-seed data, on CPU, right now
# ---------------------------------------------------------------------------

CSV_COLUMNS = {
    "base_to_ligand_a": "base distance (A)",
    "donor_to_ligand_a": "donor distance (A)",
    "base_angle_deg": "base angle (deg)",
    "donor_angle_deg": "donor angle (deg)",
    "ipde": "iPDE (confidence)",
}


def from_csv(path: Path) -> int:
    """Analyse an existing per-seed CSV. No GPU, no model, no new compute.

    data/No_mutation_run/scores/per_seed.csv holds 32 designs x 5 seeds with
    both catalytic distances, both angles and the iPDE, which is exactly what
    this analysis needs. It is the previous pipeline's run, so the geometry
    conventions differ from the current scorer's -- but a noise ratio is
    scale-free, so the verdict carries over.
    """
    if not path.exists():
        print(f"error: {path} not found", file=sys.stderr)
        return 2

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        print(f"error: {path} is empty", file=sys.stderr)
        return 2

    designs = {r["design"] for r in rows if r.get("design")}
    seeds = sorted({r["seed"] for r in rows if r.get("seed")})
    print()
    print(f"Source : {path.relative_to(ROOT)}")
    print(f"Data   : {len(rows)} predictions, {len(designs)} designs, "
          f"{len(seeds)} seeds {seeds}")
    print()
    print("Within-design SD is the noise floor: the same sequence re-predicted.")
    print("Between-design SD is the signal: different designs compared.")
    print()
    header()

    ratios: dict[str, float] = {}
    for column, label in CSV_COLUMNS.items():
        per_design: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            try:
                per_design[row["design"]].append(float(row[column]))
            except (KeyError, ValueError, TypeError):
                continue
        result = summarise(label, per_design)
        if result:
            ratios[label] = result[2]

    # The combined geometry proxy: the mean of the two catalytic distances is
    # the closest thing in this old CSV to the current scorer's distance term.
    combined: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        try:
            combined[row["design"]].append(
                (float(row["base_to_ligand_a"]) + float(row["donor_to_ligand_a"])) / 2
            )
        except (KeyError, ValueError, TypeError):
            continue
    print()
    header()
    result = summarise("geometry proxy (A)", combined)
    if result:
        ratios["geometry proxy"] = result[2]

    _recommend(ratios)
    return 0


def _recommend(ratios: dict[str, float]) -> None:
    print()
    worst = max(ratios.values()) if ratios else 0.0
    noisy = sorted(k for k, v in ratios.items() if v > RATIO_NOISE)

    print("VERDICT")
    print()
    if not ratios:
        print("  Nothing measurable.")
        return

    if worst > RATIO_NOISE:
        print(f"  {len(noisy)} term(s) are mostly noise: {', '.join(noisy)}")
        print()
        print("  Set, in your pipeline config:")
        print("      run.eval_replicates: 3      # average 3 Boltz seeds per design")
        print("      run.patience: 4             # do not stop on noise")
        print()
        above_one = sorted(k for k, v in ratios.items() if v >= 0.9)
        if above_one:
            print(f"  Ratio at or above ~1 for: {', '.join(above_one)}.")
            print("  A term whose seed-to-seed spread equals its between-design")
            print("  spread carries no information about design quality. Consider")
            print("  lowering scoring.geometry_weights for it, and read the caveat")
            print("  below before doing so.")
    elif worst > RATIO_SIGNAL:
        print("  Marginal. eval_replicates 1 will work but 3 is safer;")
        print("  raise patience to 4 either way.")
    else:
        print("  Signal dominates. eval_replicates 1 and patience 2 are fine.")

    print()
    print("CAVEAT")
    print()
    print("  If this came from --from-csv, it is the PREVIOUS pipeline's run:")
    print("  full-length partial-diffusion designs with severe ligand drift")
    print("  (RUNLOG.md section 1: 2.53 A in the design, 11.18 A after")
    print("  re-prediction). That drift inflates the between-design spread, so")
    print("  the ratios here are, if anything, optimistic. The absolute within-")
    print("  design spreads are the thing to look at, and they are large.")
    print()
    print("  Re-run with --target on the real designs once the cluster is up.")


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--from-csv", nargs="?", const=str(DEFAULT_CSV), metavar="PATH",
        help="analyse an existing per-seed CSV instead of running anything "
             f"(default: {DEFAULT_CSV.relative_to(ROOT)})",
    )
    parser.add_argument("--pipeline", default="config/pipeline.yaml")
    parser.add_argument("--target")
    parser.add_argument("--sequences-file",
                        help="plain text, one sequence per line")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Boltz-2 seeds per sequence (default 5; see the "
                             "docstring for why not 10)")
    parser.add_argument("--sequences", type=int, default=3,
                        help="how many sequences from the file (default 3)")
    parser.add_argument("--name", default="noise-control")
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()

    if args.from_csv:
        return from_csv(Path(args.from_csv))

    if not args.target:
        parser.error("either --from-csv or --target is required")
    if args.seeds < 2:
        parser.error("--seeds must be at least 2 to have a spread at all")

    # GPU mode needs the provenance maps for the sequences being re-predicted,
    # which come from the run that produced them. Not yet wired up; refuse
    # rather than invent a mapping.
    print(
        "GPU mode is not wired up yet, and refuses rather than guessing.\n"
        "\n"
        "To score a re-predicted sequence this needs that sequence's provenance\n"
        "maps -- the seed->backbone and backbone->Boltz residue maps from the\n"
        "run that produced it. They are stored in designs.db, so the clean way\n"
        "is to read them back from the run rather than reconstruct them here,\n"
        "and that plumbing is not written.\n"
        "\n"
        "What works today, and answers the same question:\n"
        "\n"
        "  python scripts/noise_control.py --from-csv\n"
        "\n"
        "That uses data/No_mutation_run/, which already holds 32 designs x 5\n"
        "seeds with both catalytic distances, both angles and the iPDE.\n"
        "\n"
        "After a seed-only run, the equivalent on new data is to set\n"
        "run.eval_replicates to 5 for one short run: the runner already\n"
        "averages replicates per design, and designs.db keeps every per-bond\n"
        "measurement in hbond_raw, so the same analysis can be done on that.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
