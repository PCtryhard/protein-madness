"""Exactly three figures, as functions returning matplotlib figures.

    python -m src.report.figures --in results/leaderboard.csv --out results/figures

1. ``fig_score_distributions`` — design scores vs. the wild-type baseline, one
   panel per key metric
2. ``fig_novelty_vs_quality`` — novelty against predicted quality, so we can
   point at the top-left quadrant and say "novel *and* confident"
3. ``fig_best_design`` — the best design: active site highlighted, plus its
   per-residue pLDDT trace

Each is saved as PNG (300 dpi) and PDF. Figures never invent data: a panel
whose metric is missing from the table says so on the panel instead of
plotting zeros, and a fake-data table is stamped FAKE DATA across every figure.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.figure import Figure                               # noqa: E402

from ..cli import setup_logging
from ..io.target import REPO_ROOT, load_reference, load_target

log = logging.getLogger("pipeline.figures")

PALETTE = {
    "design": "#4C72B0", "wildtype": "#C44E52", "pass": "#55A868",
    "fail": "#B0B0B0", "accent": "#8172B2", "grid": "#D9D9D9",
}
# Metrics shown in figure 1, in order, with the direction that is better.
DISTRIBUTION_METRICS = [
    ("mean_plddt", "mean pLDDT", "higher"),
    ("active_site_plddt", "active-site pLDDT", "higher"),
    ("catalytic_geometry_worst_dev", "worst catalytic geometry dev.", "lower"),
    ("self_consistency_rmsd", "self-consistency RMSD (A)", "lower"),
    ("sequence_identity", "identity to wild-type", "higher"),
    ("net_charge_ph7", "net charge at pH 7", "neutral"),
]


def _style(ax) -> None:
    ax.grid(True, color=PALETTE["grid"], linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _empty_panel(ax, metric: str) -> None:
    ax.text(0.5, 0.5, f"{metric}\nnot in leaderboard\n(stage not run)", ha="center", va="center",
            transform=ax.transAxes, fontsize=8, color="#888888")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _stamp_if_fake(fig: Figure, df: pd.DataFrame) -> None:
    if "is_fake" in df.columns and df["is_fake"].fillna(False).astype(bool).any():
        fig.text(0.5, 0.5, "FAKE DATA", fontsize=64, color="#FF000022", ha="center",
                 va="center", rotation=30, zorder=1000, weight="bold")


def _wildtype_row(df: pd.DataFrame) -> pd.Series | None:
    if "method" in df.columns:
        wt = df[df["method"].astype(str) == "wildtype"]
        if len(wt):
            return wt.iloc[0]
    return None


def _designs(df: pd.DataFrame) -> pd.DataFrame:
    if "method" in df.columns:
        return df[df["method"].astype(str) != "wildtype"]
    return df


# ------------------------------------------------------------------ figure 1

def fig_score_distributions(df: pd.DataFrame, metrics: list | None = None) -> Figure:
    """Score distribution of designs vs. the wild-type baseline, one panel per metric."""
    metrics = metrics or DISTRIBUTION_METRICS
    designs, wt = _designs(df), _wildtype_row(df)
    ncol = 3
    nrow = int(np.ceil(len(metrics) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.9 * nrow), squeeze=False)
    for ax, (metric, label, direction) in zip(axes.ravel(), metrics):
        if metric not in designs.columns:
            _empty_panel(ax, metric)
            continue
        vals = pd.to_numeric(designs[metric], errors="coerce").dropna()
        if vals.empty:
            _empty_panel(ax, metric)
            continue
        ax.hist(vals, bins=min(30, max(8, len(vals) // 8)), color=PALETTE["design"],
                alpha=0.85, label=f"designs (n={len(vals)})")
        if wt is not None and metric in wt.index and pd.notna(wt[metric]):
            ax.axvline(float(wt[metric]), color=PALETTE["wildtype"], linewidth=2,
                       label=f"wild-type ({float(wt[metric]):.2f})")
            better = (vals > float(wt[metric])) if direction == "higher" else (
                vals < float(wt[metric])) if direction == "lower" else None
            if better is not None:
                ax.set_title(f"{label}\n{int(better.sum())}/{len(vals)} beat wild-type", fontsize=9)
            else:
                ax.set_title(label, fontsize=9)
        else:
            ax.set_title(f"{label}\n(no wild-type baseline)", fontsize=9)
        ax.set_xlabel(label, fontsize=8)
        ax.set_ylabel("designs", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, frameon=False)
        _style(ax)
    for ax in axes.ravel()[len(metrics):]:
        ax.set_visible(False)
    n_miss = sum(1 for m, _, _ in metrics if m not in designs.columns)
    fig.suptitle("Designs vs. wild-type baseline" + (f"  ({n_miss} metric(s) not computed)"
                                                     if n_miss else ""), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _stamp_if_fake(fig, df)
    return fig


# ------------------------------------------------------------------ figure 2

def fig_novelty_vs_quality(df: pd.DataFrame, novelty: str | None = None,
                           quality: str = "active_site_plddt",
                           novelty_threshold: float = 0.3,
                           quality_threshold: float = 70.0) -> Figure:
    """Novelty against predicted quality: the top-right quadrant is novel AND confident."""
    designs = _designs(df)
    if novelty is None:
        novelty = "seq_novelty" if "seq_novelty" in designs.columns else "novelty_vs_parent"
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    if novelty not in designs.columns or quality not in designs.columns:
        _empty_panel(ax, f"{novelty} / {quality}")
        ax.set_title("Novelty vs. predicted quality", fontsize=12)
        fig.tight_layout()
        _stamp_if_fake(fig, df)
        return fig

    x = pd.to_numeric(designs[novelty], errors="coerce")
    y = pd.to_numeric(designs[quality], errors="coerce")
    ok = x.notna() & y.notna()
    passing = designs["pass_hard"].fillna(False).astype(bool) if "pass_hard" in designs.columns \
        else pd.Series(True, index=designs.index)

    ax.scatter(x[ok & ~passing], y[ok & ~passing], s=26, c=PALETTE["fail"], alpha=0.7,
               edgecolor="none", label=f"fails hard filters (n={int((ok & ~passing).sum())})")
    sizes = 30
    if "n_mutations" in designs.columns:
        nm = pd.to_numeric(designs["n_mutations"], errors="coerce").fillna(1)
        sizes = 18 + 5 * nm[ok & passing]
    ax.scatter(x[ok & passing], y[ok & passing], s=sizes, c=PALETTE["design"], alpha=0.85,
               edgecolor="white", linewidth=0.4,
               label=f"passes hard filters (n={int((ok & passing).sum())})")

    wt = _wildtype_row(df)
    if wt is not None and pd.notna(wt.get(novelty)) and pd.notna(wt.get(quality)):
        ax.scatter([float(wt[novelty])], [float(wt[quality])], marker="*", s=320,
                   c=PALETTE["wildtype"], edgecolor="black", linewidth=0.5, zorder=5,
                   label="wild-type")

    ax.axvline(novelty_threshold, color="#666666", linestyle="--", linewidth=1)
    ax.axhline(quality_threshold, color="#666666", linestyle="--", linewidth=1)
    quad = ok & passing & (x >= novelty_threshold) & (y >= quality_threshold)
    ax.text(0.98, 0.98, f"novel & confident\nn = {int(quad.sum())}", transform=ax.transAxes,
            ha="right", va="top", fontsize=10, color=PALETTE["pass"], weight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=PALETTE["pass"]))

    if "design_id" in designs.columns and quad.any():
        # stagger the labels so neighbouring top designs do not overprint
        for k, idx in enumerate(y[quad].nlargest(3).index):
            ax.annotate(str(designs.loc[idx, "design_id"]), (x[idx], y[idx]), fontsize=7,
                        xytext=(6, -10 - 9 * k), textcoords="offset points", color="#333333",
                        arrowprops=dict(arrowstyle="-", linewidth=0.5, color="#999999"))

    ax.set_xlabel(f"novelty  ({novelty}; 1 = nothing similar in the reference DB)")
    ax.set_ylabel(f"predicted quality  ({quality})")
    ax.set_title("Novelty vs. predicted quality", fontsize=12)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    _style(ax)
    fig.tight_layout()
    _stamp_if_fake(fig, df)
    return fig


# ------------------------------------------------------------------ figure 3

def fig_best_design(df: pd.DataFrame, plddt_dir: str | Path | None = None,
                    target_config: str | Path | None = None,
                    structure_dir: str | Path | None = None) -> Figure:
    """The best design: active site highlighted, plus its per-residue pLDDT trace.

    Per-residue pLDDT is read from ``<plddt_dir>/<design_id>.plddt.csv`` as
    written by the fold stage. The structure render needs a folded structure;
    when none exists the panel says so rather than drawing a placeholder.
    """
    designs = _designs(df)
    pool = designs[designs["pass_hard"].fillna(False).astype(bool)] \
        if "pass_hard" in designs.columns else designs
    if pool.empty:
        pool = designs
    if "rank" in pool.columns and pool["rank"].notna().any():
        best = pool.sort_values("rank").iloc[0]
    elif "score" in pool.columns and pool["score"].notna().any():
        best = pool.sort_values("score", ascending=False).iloc[0]
    else:
        best = pool.iloc[0]
    did = str(best.get("design_id", "?"))

    catalytic: list[int] = []
    if target_config:
        try:
            ref = load_reference(load_target(target_config))
            catalytic = ref.catalytic_positions
        except Exception as e:
            log.warning("figure 3: could not resolve catalytic residues: %s", e)

    # constrained layout: fig3 carries a colorbar, which tight_layout cannot place
    fig = plt.figure(figsize=(11.5, 4.6), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.45])
    ax_struct, ax_plddt = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # --- left: structure render (CA trace, active site highlighted)
    struct_path = None
    if structure_dir:
        for suffix in (".pdb", ".cif"):
            p = Path(structure_dir) / f"{did}{suffix}"
            if p.exists():
                struct_path = p
                break
    if struct_path is None:
        _empty_panel(ax_struct, f"no folded structure for\n{did}")
        ax_struct.set_title("best design: structure", fontsize=10)
    else:
        from ..io.structure import normalise, read_structure

        atoms, _ = normalise(read_structure(struct_path))
        ca = atoms[atoms.atom_name == "CA"]
        # Project onto the two principal axes of the CA cloud: a stable 2D view
        # without a 3D renderer. Honest about being a projection, not a cartoon.
        xyz = ca.coord - ca.coord.mean(axis=0)
        _, _, vt = np.linalg.svd(xyz, full_matrices=False)
        proj = xyz @ vt[:2].T
        ax_struct.plot(proj[:, 0], proj[:, 1], color="#999999", linewidth=1.0, zorder=1)
        sc = ax_struct.scatter(proj[:, 0], proj[:, 1], c=np.arange(len(proj)), cmap="viridis",
                               s=9, zorder=2)
        cb = fig.colorbar(sc, ax=ax_struct, fraction=0.045, pad=0.02)
        cb.set_label("residue index", fontsize=7)
        cb.ax.tick_params(labelsize=6)
        if catalytic:
            mask = np.isin(ca.res_id, catalytic)
            ax_struct.scatter(proj[mask, 0], proj[mask, 1], s=130, facecolor="none",
                              edgecolor=PALETTE["wildtype"], linewidth=1.8, zorder=3,
                              label="catalytic residues")
            for p, r in zip(proj[mask], ca.res_id[mask]):
                ax_struct.annotate(str(int(r)), p, fontsize=7, xytext=(4, 4),
                                   textcoords="offset points", color=PALETTE["wildtype"])
            ax_struct.legend(fontsize=7, frameon=False, loc="lower right")
        ax_struct.set_title(f"best design: {did}\nCA trace (principal-axis projection)", fontsize=9)
        ax_struct.set_xlabel("PC1 (A)", fontsize=8)
        ax_struct.set_ylabel("PC2 (A)", fontsize=8)
        ax_struct.set_aspect("equal")
        ax_struct.tick_params(labelsize=7)
        _style(ax_struct)

    # --- right: per-residue pLDDT trace
    plddt_path = Path(plddt_dir) / f"{did}.plddt.csv" if plddt_dir else None
    if plddt_path is not None and plddt_path.exists():
        pl = pd.read_csv(plddt_path)
        col = "plddt" if "plddt" in pl.columns else pl.columns[-1]
        pos = pl["resi"] if "resi" in pl.columns else np.arange(1, len(pl) + 1)
        ax_plddt.plot(pos, pl[col], color=PALETTE["design"], linewidth=1.3)
        ax_plddt.fill_between(pos, 0, pl[col], color=PALETTE["design"], alpha=0.18)
        ax_plddt.axhline(70, color="#666666", linestyle="--", linewidth=1)
        ax_plddt.text(pos.iloc[0] if hasattr(pos, "iloc") else 1, 71.5, "pLDDT 70",
                      fontsize=7, color="#666666")
        for c in catalytic:
            ax_plddt.axvline(c, color=PALETTE["wildtype"], alpha=0.55, linewidth=1.2)
        if catalytic:
            ax_plddt.plot([], [], color=PALETTE["wildtype"], label="catalytic residue")
            ax_plddt.legend(fontsize=7, frameon=False, loc="lower right")
        ax_plddt.set_ylim(0, 100)
        ax_plddt.set_xlabel("residue (normalised numbering)")
        ax_plddt.set_ylabel("pLDDT")
        mp, asp = best.get("mean_plddt"), best.get("active_site_plddt")
        bits = [f"mean {float(mp):.1f}" if pd.notna(mp) else "mean n/a",
                f"active site {float(asp):.1f}" if pd.notna(asp) else "active site n/a"]
        ax_plddt.set_title("per-residue confidence  (" + ", ".join(bits) + ")", fontsize=10)
        _style(ax_plddt)
    else:
        _empty_panel(ax_plddt, f"no per-residue pLDDT for\n{did}\n(fold stage not run)")
        ax_plddt.set_title("per-residue confidence", fontsize=10)

    summary = []
    for key, label, fmt in [("score", "score", "{:.2f}"), ("n_mutations", "mutations", "{:.0f}"),
                            ("seq_novelty", "novelty", "{:.2f}"),
                            ("catalytic_geometry_worst_dev", "worst geom. dev", "{:.2f}")]:
        v = best.get(key)
        if v is not None and pd.notna(v):
            summary.append(f"{label} {fmt.format(float(v))}")
    fig.suptitle(f"Best design: {did}" + ("   |   " + "   |   ".join(summary) if summary else ""),
                 fontsize=11)
    _stamp_if_fake(fig, df)
    return fig


# ------------------------------------------------------------------ driver

FIGURES = {
    "fig1_distributions": fig_score_distributions,
    "fig2_novelty_vs_quality": fig_novelty_vs_quality,
    "fig3_best_design": fig_best_design,
}


def save(fig: Figure, out_dir: str | Path, stem: str) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext, kwargs in ((".png", {"dpi": 300}), (".pdf", {})):
        p = out_dir / f"{stem}{ext}"
        fig.savefig(p, bbox_inches="tight", **kwargs)
        paths.append(p)
    plt.close(fig)
    return paths


def build_all(leaderboard_csv: str | Path, out_dir: str | Path,
              plddt_dir: str | Path | None = None, target_config: str | Path | None = None,
              structure_dir: str | Path | None = None) -> list[Path]:
    df = pd.read_csv(leaderboard_csv)
    log.info("figures from %s (%d designs)", leaderboard_csv, len(df))
    written: list[Path] = []
    written += save(fig_score_distributions(df), out_dir, "fig1_distributions")
    written += save(fig_novelty_vs_quality(df), out_dir, "fig2_novelty_vs_quality")
    written += save(fig_best_design(df, plddt_dir, target_config, structure_dir),
                    out_dir, "fig3_best_design")
    for p in written:
        log.info("wrote %s", p)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default=str(REPO_ROOT / "results" / "leaderboard.csv"))
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "figures"))
    ap.add_argument("--plddt-dir", default=None, help="directory of <design_id>.plddt.csv")
    ap.add_argument("--structure-dir", default=None, help="directory of folded <design_id>.pdb")
    ap.add_argument("--target", default=None, help="target config, to mark catalytic residues")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    setup_logging(args.verbose)
    build_all(args.inp, args.out, args.plddt_dir, args.target, args.structure_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
