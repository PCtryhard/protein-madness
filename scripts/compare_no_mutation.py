"""Compare the four 5FQJ baseline arms using the repository fitness score."""

import argparse
from collections import Counter
import csv
from math import isfinite
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
COST = "repo_cost_with_mapped_contacts"
ARMS = ("5A", "5A_noligand", "10A", "10A_noligand")


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def summarize(label, designs):
    best = min(designs, key=lambda row: row[COST])
    typical = median(row[COST] for row in designs)
    print(f"{label:18} {len(designs):>2}  {typical:>9.3f}  "
          f"{best[COST]:>9.3f}  {best['design']}")
    return typical


def compare(title, groups):
    print(f"\n{title}")
    print(f"{'Group':18}  n     Median       Best  Best design")
    medians = {name: summarize(name, designs) for name, designs in groups.items()}
    lowest = min(medians.values())
    winners = [name for name, value in medians.items() if value == lowest]
    print(f"Lowest median cost: {', '.join(winners)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path,
                        default=ROOT / "data/No_mutation_run/scores")
    args = parser.parse_args()
    designs = read_csv(args.scores / "leaderboard.csv")
    seed_rows = read_csv(args.scores / "per_seed.csv")
    expected_seeds = Counter({str(seed): 1 for seed in range(1, 6)})
    observed = {}
    for row in seed_rows:
        observed.setdefault(row["design"], Counter())[row["seed"]] += 1

    if len(designs) != 32 or len({row["design"] for row in designs}) != 32:
        parser.error("expected 32 distinct designs")
    if set(observed) != {row["design"] for row in designs}:
        parser.error("the two CSVs contain different design IDs")
    if any(observed[row["design"]] != expected_seeds or row["n_seeds"] != "5"
           for row in designs):
        parser.error("every design must have one result for each of seeds 1–5")


    for row in designs:
        row[COST] = float(row[COST])
        if not isfinite(row[COST]):
            parser.error(f"{row['design']} has a non-finite fitness cost")

    arms = {arm: [row for row in designs if row["arm"] == arm] for arm in ARMS}
    if any(len(group) != 8 for group in arms.values()):
        parser.error("expected eight designs per arm")

    print("Lower cost is better. Group comparisons use median design cost.")
    compare("Radius × ligand condition", arms)
    compare("Radius (both ligand conditions)", {
        "5 Å": arms["5A"] + arms["5A_noligand"],
        "10 Å": arms["10A"] + arms["10A_noligand"],
    })
    compare("Ligand during design (both radii)", {
        "With 6NT": arms["5A"] + arms["10A"],
        "Without 6NT": arms["5A_noligand"] + arms["10A_noligand"],
    })
    print("\nTop five designs")
    for rank, row in enumerate(sorted(designs, key=lambda row: row[COST])[:5], 1):
        print(f"{rank}. {row['design']:18} {row[COST]:.3f}")
    print("\nEach arm has one RFD3 scaffold. These comparisons describe this run.")


if __name__ == "__main__":
    main()
