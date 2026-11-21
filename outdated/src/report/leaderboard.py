"""Join every stage CSV into the one table this project presents from.

    python -m src.report.leaderboard --in results --out results \
        --config configs/pipeline.yaml

Rule 5 of the brief: every design that exists anywhere has a row here, keyed by
``design_id``. Filters and ranking weights come from ``configs/pipeline.yaml``,
never from code, so a judge's "why 70?" has exactly one answer.

Columns added here:
* ``pass_hard`` / ``pass_soft`` / ``n_filters_failed`` / ``filters_failed``
* ``score`` — weighted sum of z-scored metrics (``leaderboard.ranking``)
* ``rank`` — 1 = best, among designs passing the hard filters
Metrics that no stage produced are reported as missing, not as zero.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..cli import load_pipeline_config, setup_logging
from ..io.target import REPO_ROOT

log = logging.getLogger("pipeline.leaderboard")

OPS = {
    "==": lambda s, v: s == v,
    "!=": lambda s, v: s != v,
    ">=": lambda s, v: s >= v,
    "<=": lambda s, v: s <= v,
    ">": lambda s, v: s > v,
    "<": lambda s, v: s < v,
}
# Stage CSVs are found by this glob; the tool name is the file stem.
STAGE_GLOB = "*.csv"
SKIP_STEMS = {"leaderboard"}


def find_stage_csvs(root: str | Path) -> list[Path]:
    """Every ``<tool>.csv`` under ``root``, deepest-last for stable precedence."""
    root = Path(root)
    if root.is_file():
        return [root]
    out = [p for p in sorted(root.rglob(STAGE_GLOB))
           if p.stem not in SKIP_STEMS and not p.name.endswith(".meta.json")]
    return out


def join_stages(paths: list[Path]) -> pd.DataFrame:
    """Outer-join stage CSVs on ``design_id``.

    Later stages win on conflicting column names, but the losing column is kept
    as ``<col>__<tool>`` rather than dropped, so a disagreement between stages
    is visible instead of silently resolved.
    """
    merged: pd.DataFrame | None = None
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            log.warning("skipping %s: %s", p, e)
            continue
        if "design_id" not in df.columns:
            log.warning("skipping %s: no design_id column", p)
            continue
        tool = p.stem
        df = df.drop_duplicates(subset="design_id", keep="last")
        df["stages"] = tool
        if merged is None:
            merged = df
            continue
        overlap = (set(df.columns) & set(merged.columns)) - {"design_id", "stages"}
        df = df.rename(columns={c: f"{c}__{tool}" for c in overlap})
        merged = merged.merge(df.drop(columns=["stages"]), on="design_id", how="outer")
        merged["stages"] = merged["stages"].fillna("") + "," + tool
    if merged is None:
        raise SystemExit("no stage CSVs with a design_id column found")
    merged["stages"] = merged["stages"].fillna("").str.strip(",")
    for col in (set(merged.columns) & {"sequence"}):
        merged[col] = merged[col].astype("string")
    return merged


def apply_filters(df: pd.DataFrame, filters: list[dict]) -> pd.DataFrame:
    """Add pass/fail columns. A filter on a missing metric is skipped loudly."""
    df = df.copy()
    hard = pd.Series(True, index=df.index)
    soft = pd.Series(True, index=df.index)
    failed: list[list[str]] = [[] for _ in range(len(df))]
    for f in filters:
        metric, op, value = f["metric"], f["op"], f["value"]
        if metric not in df.columns:
            log.warning("filter on '%s' skipped: no stage produced that column", metric)
            continue
        if op not in OPS:
            raise ValueError(f"unknown filter op '{op}' (use {sorted(OPS)})")
        series = pd.to_numeric(df[metric], errors="coerce")
        ok = OPS[op](series, value)
        ok = ok.fillna(False) if series.isna().any() else ok
        if series.isna().any():
            log.warning("filter on '%s': %d design(s) have no value and fail this filter",
                        metric, int(series.isna().sum()))
        for i, good in enumerate(ok.to_numpy()):
            if not good:
                failed[i].append(f"{metric}{op}{value}")
        if f.get("hard"):
            hard &= ok
        else:
            soft &= ok
    df["pass_hard"] = hard
    df["pass_soft"] = hard & soft
    df["filters_failed"] = ["+".join(x) for x in failed]
    df["n_filters_failed"] = [len(x) for x in failed]
    return df


def rank(df: pd.DataFrame, ranking: list[dict]) -> pd.DataFrame:
    """Weighted sum of z-scored metrics, ranked among hard-filter survivors.

    z-scores use the mean/sd of the designs that pass the hard filters, so a
    flood of dead designs cannot distort the scale. A metric absent from the
    table contributes nothing and is logged. A design missing one metric's
    value gets 0 for that term (the population mean), and
    ``score_metrics_missing`` records how many terms that was.
    """
    df = df.copy()
    pool = df[df["pass_hard"]] if "pass_hard" in df.columns else df
    score = pd.Series(0.0, index=df.index)
    missing = pd.Series(0, index=df.index)
    used = []
    for r in ranking:
        metric, weight = r["metric"], float(r.get("weight", 1.0))
        if metric not in df.columns:
            log.warning("ranking term '%s' skipped: no stage produced that column", metric)
            continue
        series = pd.to_numeric(df[metric], errors="coerce")
        ref = pd.to_numeric(pool[metric], errors="coerce").dropna()
        if ref.empty:
            log.warning("ranking term '%s' skipped: no values among passing designs", metric)
            continue
        sd = ref.std(ddof=0)
        z = (series - ref.mean()) / sd if sd and np.isfinite(sd) and sd > 0 else series * 0.0
        if not sd:
            log.info("ranking term '%s': zero variance, contributes nothing", metric)
        if str(r.get("direction", "higher")).lower() == "lower":
            z = -z
        missing += series.isna().astype(int)
        score += weight * z.fillna(0.0)
        used.append(f"{metric}({'+' if r.get('direction','higher')=='higher' else '-'}{weight})")
    df["score"] = score
    df["score_metrics_missing"] = missing
    df["score_terms"] = ";".join(used)
    order = df.sort_values(["pass_hard", "score"], ascending=[False, False]).index
    df.loc[order, "rank"] = range(1, len(df) + 1)
    df["rank"] = df["rank"].astype(int)
    if not used:
        log.warning("no ranking terms available: 'score' is 0 and 'rank' is arbitrary")
    return df.sort_values("rank")


def markdown_table(df: pd.DataFrame, columns: list[str] | None = None, top: int = 20) -> str:
    cols = [c for c in (columns or [
        "rank", "design_id", "method", "mutations", "score", "catalytic_integrity",
        "mean_plddt", "active_site_plddt", "catalytic_geometry_worst_dev",
        "self_consistency_rmsd", "seq_novelty", "pass_hard",
    ]) if c in df.columns]
    sub = df.head(top)[cols].copy()
    for c in sub.columns:
        if pd.api.types.is_float_dtype(sub[c]):
            sub[c] = sub[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in sub.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def build(in_root: str | Path, out_dir: str | Path, config: str | Path,
          top: int = 20) -> pd.DataFrame:
    cfg = load_pipeline_config(config)
    lcfg = cfg.get("leaderboard") or {}
    paths = find_stage_csvs(in_root)
    if not paths:
        raise SystemExit(f"no stage CSVs under {in_root}: run a generate/score stage first")
    log.info("joining %d stage CSV(s): %s", len(paths), ", ".join(p.stem for p in paths))
    df = join_stages(paths)
    df = apply_filters(df, lcfg.get("filters") or [])
    df = rank(df, lcfg.get("ranking") or [])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "leaderboard.csv"
    df.to_csv(csv_path, index=False)
    (out_dir / "leaderboard.md").write_text(
        f"# Leaderboard\n\n{len(df)} designs, "
        f"{int(df['pass_hard'].sum())} pass hard filters.\n\n"
        f"Ranking terms: `{df['score_terms'].iloc[0] if len(df) else ''}`\n\n"
        + markdown_table(df, top=top) + "\n",
        encoding="utf-8",
    )
    (out_dir / "leaderboard.meta.json").write_text(json.dumps({
        "n_designs": len(df),
        "n_pass_hard": int(df["pass_hard"].sum()),
        "n_pass_soft": int(df["pass_soft"].sum()),
        "stages": sorted({s for row in df["stages"] for s in str(row).split(",") if s}),
        "columns": list(df.columns),
        "filters": lcfg.get("filters") or [],
        "ranking": lcfg.get("ranking") or [],
    }, indent=1), encoding="utf-8")
    log.info("leaderboard: %d designs (%d pass hard filters) -> %s",
             len(df), int(df["pass_hard"].sum()), csv_path)
    return df


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default=str(REPO_ROOT / "results"),
                    help="directory searched recursively for <tool>.csv (default: results/)")
    ap.add_argument("--out", default=str(REPO_ROOT / "results"))
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "pipeline.yaml"))
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    setup_logging(args.verbose)
    build(args.inp, args.out, args.config, args.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
