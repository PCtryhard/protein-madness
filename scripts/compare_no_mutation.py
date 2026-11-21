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


def plot_run(directory, groups, ranked, run_name):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:
        raise ValueError("install matplotlib and seaborn to use --plots") from exc

    def save(fig, name):
        for extension in ("png", "pdf"):
            fig.savefig(directory / f"{name}.{extension}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    def measured_contacts(best, metric):
        suffix = f"__{metric}"
        return {
            field.removesuffix(suffix).replace("_", " "): float(value)
            for field, value in best.items()
            if field.endswith(suffix) and value and isfinite(float(value))
        }

    directory.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.82)
    plt.rcParams["pdf.fonttype"] = 42
    ordered = sorted(groups)
    branch_names = {"seed": "Initial", "wiggle": "Wiggle", "mutate": "Point mutation"}
    labels = [f"Cycle {cycle}\n{branch_names.get(branch, branch)}" for cycle, branch in ordered]

    fig, (scores_ax, outcomes_ax) = plt.subplots(1, 2, figsize=(10.8, 4.3), layout="constrained")
    fig.suptitle(f"Design outcomes · {run_name}", fontsize=14)
    points = [(label, score) for key, label in zip(ordered, labels)
              for score in groups[key][1]]
    if points:
        locations, values = zip(*points)
        sns.swarmplot(x=list(locations), y=list(values), order=labels,
                      color="#23668C", size=6, ax=scores_ax)
        for index, key in enumerate(ordered):
            scores = groups[key][1]
            if scores:
                scores_ax.hlines(median(scores), index - 0.25, index + 0.25,
                                 color="#20272E", linewidth=1.5, zorder=4)
        scores_ax.set_title("Scores of designs that passed")
    else:
        scores_ax.set_xticks(range(len(labels)), labels)
        scores_ax.set_title("No designs passed scoring")
    scores_ax.set_ylabel("Pipeline score (higher is better)")
    scores_ax.set_xlabel("")

    from matplotlib.ticker import MaxNLocator

    baseline = [0] * len(ordered)
    for status, label, color in (
        ("scored", "Scored", "#23668C"),
        ("gated", "Gated", "#D98C34"),
        ("errored", "Errored", "#A33D54"),
    ):
        counts = [groups[key][0][status] for key in ordered]
        outcomes_ax.bar(range(len(ordered)), counts, bottom=baseline, color=color,
                        label=label, width=0.62)
        baseline = [total + n for total, n in zip(baseline, counts)]
    outcomes_ax.set_title("Scoring outcomes")
    outcomes_ax.set_ylabel("Design count")
    outcomes_ax.set_xticks(range(len(labels)), labels)
    outcomes_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    outcomes_ax.legend(frameon=False, fontsize=10, loc="lower center",
                       bbox_to_anchor=(0.5, -0.34), ncols=3)
    sns.despine(fig=fig)
    save(fig, "search_outcomes")

    if ranked:
        score, best = ranked[0]
        panels = [(heading, measured_contacts(best, key), axis_label)
                  for key, heading, axis_label in (
                      ("distance", "Contact distances", "Distance (Å)"),
                      ("angle", "Contact angles", "Angle (°)"),
                  )]
        panels = [(heading, values, axis_label) for heading, values, axis_label in panels if values]
        if panels:
            fig, axes = plt.subplots(1, len(panels), squeeze=False,
                                     figsize=(5.6 * len(panels), 3.8), layout="constrained")
            fig.suptitle(f"Top design · {best['design_id']} · score {score:.4f}", fontsize=14)
            for ax, (heading, values, axis_label) in zip(axes[0], panels):
                bars = ax.barh(list(values), list(values.values()), color="#23668C", height=0.52)
                ax.bar_label(bars, fmt="%.2f" if "Å" in axis_label else "%.1f", padding=4)
                ax.invert_yaxis()
                ax.margins(x=0.18)
                ax.set_title(heading)
                ax.set_xlabel(axis_label)
                ax.grid(axis="y", visible=False)
            sns.despine(fig=fig)
            save(fig, "best_design_geometry")
    print(f"\nSaved presentation plots to {directory} (PNG and PDF)")


def summarize_run(path, limit, plots=None):
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
    ranked.sort(key=lambda pair: (-pair[0], pair[1]["design_id"]))
    if plots is not None:
        if not rows:
            print("\nNo design rows to plot.")
        else:
            plot_run(plots, groups, ranked, csv_path.parent.name)
    if not ranked:
        print("\nNo designs passed the scoring gates.")
        return

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
    parser.add_argument("--plots", type=Path, help="save presentation PNG and PDF figures here")
    args = parser.parse_args()
    if args.plots is not None and args.run is None:
        parser.error("--plots requires --run")
    if args.top < 1:
        parser.error("--top must be at least 1")
    if args.run is not None:
        try:
            summarize_run(args.run, args.top, args.plots)
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
