"""Compare the 5FQJ baseline arms or summarize a pipeline run."""

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


def summarize_run(path, limit):
    csv_path = path / "designs.csv" if path.is_dir() else path
    with csv_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"design_id", "cycle", "branch", "status", "gated", "score"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{csv_path}: missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)

    print(f"Run results: {csv_path}")
    print(f"Designs: {len(rows)}")
    groups = {}
    gated_bonds = Counter()
    errors = Counter()
    ranked = []
    for row in rows:
        key = (int(row["cycle"]), row["branch"])
        counts, scores = groups.setdefault(key, (Counter(), []))
        status = row["status"]
        if status == "scored" and row["gated"] == "0":
            score = float(row["score"])
            if not isfinite(score):
                raise ValueError(f"{row['design_id']}: scored design has a non-finite score")
            scores.append(score)
            ranked.append((score, row))
        elif status == "gated" or row["gated"] == "1":
            status = "gated"
            gated_bonds.update((row.get("gated_bonds") or "").split())
        elif status == "errored":
            errors[row.get("error") or "(no error text recorded)"] += 1
        counts[status] += 1

    print("\nBy cycle and branch (higher score is better)")
    print(f"{'Cycle':>5}  {'Branch':12} {'Scored':>6} {'Gated':>6} {'Errors':>6} {'Median':>8} {'Best':>8}")
    for (cycle, branch), (counts, scores) in sorted(groups.items()):
        typical = f"{median(scores):.4f}" if scores else "-"
        best = f"{max(scores):.4f}" if scores else "-"
        print(f"{cycle:>5}  {branch:12} {counts['scored']:>6} {counts['gated']:>6} "
              f"{counts['errored']:>6} {typical:>8} {best:>8}")

    if gated_bonds:
        print("\nFailed scoring gates")
        for bond, count in gated_bonds.most_common():
            print(f"  {count:>3}  {bond}")
    if errors:
        print("\nDesign errors")
        for error, count in errors.most_common(5):
            print(f"  {count:>3}  {error}")
    if not ranked:
        print("\nNo designs passed the scoring gates.")
        return

    ranked.sort(key=lambda pair: (-pair[0], pair[1]["design_id"]))
    print(f"\nTop {min(limit, len(ranked))} designs")
    for rank, (score, row) in enumerate(ranked[:limit], 1):
        mutation = f", {row['mutation']}" if row.get("mutation") else ""
        print(f"{rank:>3}. {row['design_id']}  {score:.4f}  "
              f"(cycle {row['cycle']}, {row['branch']}{mutation})")

    score, best = ranked[0]
    print(f"\nBest: {best['design_id']} ({score:.4f})")
    seed_best = next((value for value, row in ranked if row["cycle"] == "0"), None)
    if seed_best is not None:
        print(f"Improvement over best cycle 0 design: {score - seed_best:+.4f}")
    for label, field in (("Predicted structure", "structure_path"),
                         ("Backbone", "backbone_path"), ("Parent", "parent_id")):
        if best.get(field):
            print(f"  {label}: {best[field]}")
    for field, value in best.items():
        if field.endswith("__distance") and value:
            bond = field.removesuffix("__distance")
            angle = best.get(f"{bond}__angle")
            print(f"  {bond}: {value} Å" + (f", {angle}°" if angle else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path,
                        default=ROOT / "data/No_mutation_run/scores")
    parser.add_argument("--run", type=Path,
                        help="pipeline run directory, exported scores directory, or designs.csv")
    parser.add_argument("--top", type=int, default=5, help="number of designs to show for --run")
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be at least 1")
    if args.run is not None:
        try:
            summarize_run(args.run, args.top)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        return

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
