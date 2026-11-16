"""Regenerate every number and figure in PITCH.md from the current leaderboard.

    python -m src.report.pitch_assets --in results/leaderboard.csv --out results/pitch \
        --target configs/target.yaml

Writes ``results/pitch/numbers.json`` and ``numbers.md``, the three figures, and
fills the ``<!-- auto:key -->`` placeholders in PITCH.md. Nothing in the pitch
is typed in by hand: if a number is not in the leaderboard it appears as
``[NOT COMPUTED]`` in the pitch, which is the honest statement and an obvious
thing to fix before 17:30.

Refuses to run on a table containing fake rows unless ``--allow-fake``.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd

from ..cli import setup_logging
from ..ids import git_sha
from ..io.target import REPO_ROOT, load_target
from . import figures as figmod

log = logging.getLogger("pipeline.pitch")

PLACEHOLDER = re.compile(r"(<!--\s*auto:([a-z0-9_]+)\s*-->)(.*?)(<!--\s*/auto\s*-->)", re.S)
MISSING = "[NOT COMPUTED]"


def _fmt(v, spec: str = "{:.2f}") -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return MISSING
    try:
        return spec.format(float(v))
    except (TypeError, ValueError):
        return str(v)


def collect_numbers(df: pd.DataFrame, target_path: str | Path | None = None) -> dict:
    """Every number the pitch quotes, in one dict. Missing metrics stay missing."""
    designs = figmod._designs(df)
    wt = figmod._wildtype_row(df)
    passing = designs[designs["pass_hard"].fillna(False).astype(bool)] \
        if "pass_hard" in designs.columns else designs

    nums: dict = {
        "git_sha": git_sha(),
        "n_designs_total": int(len(designs)),
        "n_pass_hard": int(len(passing)),
        "n_fail_hard": int(len(designs) - len(passing)),
        "methods": {k: int(v) for k, v in designs.get(
            "method", pd.Series(dtype=str)).value_counts().items()},
        "stages_run": sorted({s for row in df.get("stages", pd.Series(dtype=str)).dropna()
                              for s in str(row).split(",") if s}),
    }
    if target_path:
        t = load_target(target_path)
        nums["target"] = {"name": t.name, "config_hash": t.hash,
                          "reference_structure": str(t.reference_structure.name),
                          "n_catalytic": len(t.catalytic_residues),
                          "objective": t.objective.strip(),
                          "success_metrics": t.success_metrics}

    for metric in ["mean_plddt", "active_site_plddt", "catalytic_geometry_worst_dev",
                   "self_consistency_rmsd", "self_consistency_tmscore", "seq_novelty",
                   "novelty_vs_parent", "n_mutations", "sequence_identity"]:
        if metric not in designs.columns:
            nums[metric] = None
            continue
        s = pd.to_numeric(passing[metric], errors="coerce").dropna()
        if s.empty:
            nums[metric] = None
            continue
        entry = {"n": int(len(s)), "mean": float(s.mean()), "median": float(s.median()),
                 "min": float(s.min()), "max": float(s.max())}
        if wt is not None and metric in wt.index and pd.notna(wt[metric]):
            entry["wildtype"] = float(wt[metric])
            entry["n_beating_wildtype_higher"] = int((s > float(wt[metric])).sum())
            entry["n_beating_wildtype_lower"] = int((s < float(wt[metric])).sum())
        nums[metric] = entry

    top_cols = [c for c in ["design_id", "rank", "score", "method", "mutations", "n_mutations",
                            "mean_plddt", "active_site_plddt", "catalytic_geometry_worst_dev",
                            "self_consistency_rmsd", "seq_novelty", "catalytic_integrity"]
                if c in df.columns]
    top = df.sort_values("rank").head(3)[top_cols] if "rank" in df.columns else df.head(3)[top_cols]
    nums["top3"] = json.loads(top.to_json(orient="records"))
    return nums


def numbers_markdown(nums: dict) -> str:
    lines = [f"# Pitch numbers\n", f"Generated from the leaderboard at git `{nums['git_sha']}`.\n"]
    if "target" in nums:
        t = nums["target"]
        lines += [f"**Target:** {t['name']} (config hash `{t['config_hash']}`, "
                  f"{t['n_catalytic']} catalytic residues)\n"]
    lines += [f"**Designs:** {nums['n_designs_total']} total, {nums['n_pass_hard']} pass hard "
              f"filters, {nums['n_fail_hard']} fail.\n",
              f"**Methods:** " + (", ".join(f"{k} ({v})" for k, v in nums["methods"].items())
                                  or MISSING) + "\n",
              f"**Stages run:** " + (", ".join(nums["stages_run"]) or MISSING) + "\n",
              "\n| metric | n | mean | median | best | wild-type | beat WT |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for metric, better in [("mean_plddt", "higher"), ("active_site_plddt", "higher"),
                           ("catalytic_geometry_worst_dev", "lower"),
                           ("self_consistency_rmsd", "lower"), ("seq_novelty", "higher")]:
        e = nums.get(metric)
        if not e:
            lines.append(f"| {metric} | {MISSING} | | | | | |")
            continue
        best = e["max"] if better == "higher" else e["min"]
        beat = e.get(f"n_beating_wildtype_{better}")
        lines.append(f"| {metric} | {e['n']} | {_fmt(e['mean'])} | {_fmt(e['median'])} | "
                     f"{_fmt(best)} | {_fmt(e.get('wildtype'))} | "
                     f"{beat if beat is not None else MISSING} |")
    lines.append("\n## Top 3 designs\n")
    for d in nums["top3"]:
        bits = [f"`{d.get('design_id')}`"]
        for k, spec in [("score", "{:.2f}"), ("mean_plddt", "{:.1f}"),
                        ("active_site_plddt", "{:.1f}"), ("seq_novelty", "{:.2f}")]:
            if k in d:
                bits.append(f"{k} {_fmt(d[k], spec)}")
        lines.append("- " + ", ".join(bits))
    return "\n".join(lines) + "\n"


def fill_pitch(pitch_path: Path, nums: dict) -> int:
    """Replace ``<!-- auto:key -->...<!-- /auto -->`` blocks in PITCH.md. Returns count."""
    if not pitch_path.exists():
        log.warning("%s not found: skipping placeholder fill", pitch_path)
        return 0
    text = pitch_path.read_text(encoding="utf-8")
    rendered = {
        "headline_counts": (
            f"**{nums['n_designs_total']}** designs generated, "
            f"**{nums['n_pass_hard']}** pass every hard filter "
            f"({', '.join(f'{k}: {v}' for k, v in nums['methods'].items()) or MISSING})."),
        "scoring_stack": "Stages run: " + (", ".join(f"`{s}`" for s in nums["stages_run"]) or MISSING),
        "numbers_table": "| metric" + numbers_markdown(nums).split("\n| metric", 1)[-1]
        .split("\n## Top 3")[0],
        "top3": "\n".join(
            f"{i + 1}. `{d.get('design_id')}` — score {_fmt(d.get('score'))}, "
            f"mean pLDDT {_fmt(d.get('mean_plddt'), '{:.1f}')}, "
            f"active-site pLDDT {_fmt(d.get('active_site_plddt'), '{:.1f}')}, "
            f"novelty {_fmt(d.get('seq_novelty'))}"
            for i, d in enumerate(nums["top3"])) or MISSING,
        "novelty": (
            f"Median sequence novelty {_fmt(nums['seq_novelty']['median'])} "
            f"(max {_fmt(nums['seq_novelty']['max'])}) against the reference database."
            if nums.get("seq_novelty") else
            f"{MISSING} — database novelty was not run (see TODO.md)."),
        "provenance": f"All numbers regenerated from `results/leaderboard.csv` at git "
                      f"`{nums['git_sha']}` by `python -m src.report.pitch_assets`.",
        "target_objective": (
            f"> {nums['target']['objective']}\n\n"
            f"Target: **{nums['target']['name']}** "
            f"({nums['target']['n_catalytic']} catalytic residues, reference "
            f"`{nums['target']['reference_structure']}`, config hash "
            f"`{nums['target']['config_hash']}`).\n\n"
            + ("Success metrics from the brief: "
               + ", ".join(f"`{m}`" for m in nums["target"]["success_metrics"])
               if nums["target"].get("success_metrics")
               else f"Success metrics: {MISSING} — fill `success_metrics` in the target config.")
            if nums.get("target") else
            f"{MISSING} — pass --target configs/target.yaml to fill this section."),
    }
    n = 0

    def sub(m):
        nonlocal n
        key = m.group(2)
        if key not in rendered:
            log.warning("PITCH.md has placeholder 'auto:%s' with no generator", key)
            return m.group(0)
        n += 1
        return f"{m.group(1)}\n{rendered[key]}\n{m.group(4)}"

    pitch_path.write_text(PLACEHOLDER.sub(sub, text), encoding="utf-8")
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default=str(REPO_ROOT / "results" / "leaderboard.csv"))
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "pitch"))
    ap.add_argument("--target", default=None)
    ap.add_argument("--plddt-dir", default=None)
    ap.add_argument("--structure-dir", default=None)
    ap.add_argument("--pitch", default=str(REPO_ROOT / "PITCH.md"))
    ap.add_argument("--allow-fake", action="store_true",
                    help="allow a leaderboard containing is_fake rows (dry runs only)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    setup_logging(args.verbose)

    df = pd.read_csv(args.inp)
    if "is_fake" in df.columns and df["is_fake"].fillna(False).astype(bool).any() \
            and not args.allow_fake:
        raise SystemExit(
            f"{args.inp} contains fake rows (is_fake=True). Refusing to generate pitch assets "
            "from fabricated data. Pass --allow-fake for a dry run."
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    nums = collect_numbers(df, args.target)
    (out / "numbers.json").write_text(json.dumps(nums, indent=1), encoding="utf-8")
    (out / "numbers.md").write_text(numbers_markdown(nums), encoding="utf-8")
    figmod.build_all(args.inp, out / "figures", args.plddt_dir, args.target, args.structure_dir)
    n = fill_pitch(Path(args.pitch), nums)
    log.info("pitch assets -> %s (%d placeholder(s) filled in %s)", out, n, args.pitch)

    missing = [k for k in ("mean_plddt", "active_site_plddt", "seq_novelty",
                           "catalytic_geometry_worst_dev") if not nums.get(k)]
    if missing:
        log.warning("pitch will show %s for: %s", MISSING, ", ".join(missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
