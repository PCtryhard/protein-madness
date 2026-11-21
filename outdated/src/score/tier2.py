"""Tier-2 scoring: metrics that need a folded structure.

    python -m src.score.tier2 --in results/run/fold --out results/run/tier2 \
        --config configs/pipeline.yaml --target configs/target.yaml

``--in`` is the fold stage's output directory: ``<design_id>.pdb`` structures,
optionally with ``<design_id>.pae.npz`` beside them. Writes
``<out>/score.tier2.csv``, one row per design.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from .. import cli
from ..io.target import load_reference
from . import structure as st

log = logging.getLogger("pipeline.tier2")
TOOL = "score.tier2"
VERSION = "0.1.0"


def find_pae(structure_path: Path) -> Path | None:
    for suffix in (".pae.npz", ".pae.npy", ".pae.json"):
        p = structure_path.with_suffix("").with_suffix(suffix) \
            if structure_path.suffix else structure_path
        cand = structure_path.parent / (structure_path.stem + suffix)
        if cand.exists():
            return cand
    return None


def main(argv: list[str] | None = None) -> int:
    ap = cli.base_parser(__doc__)
    ap.add_argument("--ligand-resn", default=None,
                    help="residue name of the ligand to score (default: target.substrate)")
    args = ap.parse_args(argv)
    cli.setup_logging(args.verbose)

    cfg = cli.load_pipeline_config(args.config)
    tcfg = cli.tool_config(cfg, TOOL)
    target = cli.require_target(args)
    ref = load_reference(target)

    inputs = cli.load_inputs(args.inp, args.limit)
    if inputs.kind != "structures":
        raise SystemExit(f"{TOOL} scores structures: --in must be a structure or a directory "
                         f"of structures, got {inputs.path}")
    ligand_resn = args.ligand_resn or (target.substrate or {}).get("reference_ligand")

    with cli.stage(TOOL, args, tcfg, VERSION,
                   hash_extra={"n_inputs": len(inputs), "target": target.hash,
                               "ligand_resn": ligand_resn}) as stage:
        if stage is None:
            return 0

        rows, failed = [], []
        for path in inputs.structures:
            did = path.stem
            try:
                row = st.all_structure_metrics(
                    path, ref, pae_path=find_pae(path),
                    shell=float(tcfg.get("active_site_shell", 8.0)),
                    tolerance_scale=float(tcfg.get("geometry_tolerance_scale", 1.0)),
                    probe_radius=float(tcfg.get("sasa_probe_radius", 1.4)),
                    patch_min_area=float(tcfg.get("hydrophobic_patch_min_area", 100.0)),
                    ligand_resn=ligand_resn,
                )
            except Exception as e:
                log.error("%s: %s: %s", did, type(e).__name__, e)
                failed.append(did)
                continue
            row["design_id"] = did
            row["structure_path"] = str(path)
            rows.append(row)

        fcfg = dict(tcfg.get("foldseek") or {})
        if fcfg.get("enabled"):
            try:
                hits = st.structural_novelty(
                    inputs.path if inputs.path.is_dir() else inputs.path.parent,
                    fcfg.get("reference_db"),
                    threads=int((cfg.get("compute") or {}).get("threads", 4)))
            except st.FoldseekUnavailable as e:
                raise SystemExit(
                    f"score.tier2: structural novelty is enabled but unavailable: {e}\n"
                    "Fix the config/DB, or set score.tier2.foldseek.enabled: false. "
                    "Not writing a fabricated structural-novelty column."
                )
            for r in rows:
                r.update(hits.get(r["design_id"], {
                    "struct_novelty": 1.0, "max_tmscore": 0.0, "nearest_structure": ""}))
        else:
            log.info("structural novelty: Foldseek disabled in config")

        if not rows:
            raise SystemExit(f"{TOOL}: every structure failed to score ({len(failed)} of "
                             f"{len(inputs)}); see the errors above")
        if failed:
            log.warning("%d of %d structures failed to score: %s", len(failed), len(inputs),
                        ", ".join(failed[:8]) + (" ..." if len(failed) > 8 else ""))
        df = stage.write_csv(rows)
        if "catalytic_geometry_fraction_pass" in df.columns:
            good = int((df["catalytic_geometry_fraction_pass"] == 1.0).sum())
            log.info("catalytic geometry: %d/%d designs within tolerance on every measurement",
                     good, len(df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
