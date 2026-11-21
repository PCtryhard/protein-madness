"""Does the wiggle branch actually produce backbones? GPU job only.

    python cluster/euler/check_wiggle.py [RUN_DIR] [DESIGN_ID]

Runs the wiggle stage standalone against a real scored parent from an earlier
run, so the branch is exercised even when a short run's own frontier comes out
empty. With no arguments it picks the best-scoring design in runs/euler-01.

Writes under runs/wiggle-check/. Not /tmp -- that is node-local on Euler, so
anything you want to read back afterwards has to live in $HOME.

Exits non-zero if the wiggle produces no backbones.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

# repo root is two levels up from cluster/euler/
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import load as load_config  # noqa: E402
from src.stages import rfd3 as rfd_stage  # noqa: E402
from src.store import DesignStore  # noqa: E402
from src.tracking import Provenance  # noqa: E402

PREV_RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/euler-01")
PARENT = sys.argv[2] if len(sys.argv) > 2 else None
OUT = Path("runs/wiggle-check")


def best_scored_design(store) -> str:
    """The highest-scoring design in that run, as the wiggle parent.

    A gated design has no backbone worth re-noising, so an unscored run gives
    this check nothing to work with -- which is itself worth saying out loud
    rather than failing obscurely.
    """
    import csv

    csv_path = PREV_RUN / "designs.csv"
    if not csv_path.exists():
        sys.exit("no %s -- run the pipeline once first" % csv_path)
    best, best_score = None, -1.0
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["status"] != "scored":
                continue
            try:
                score = float(row["score"])
            except (TypeError, ValueError):
                continue
            if score > best_score:
                best, best_score = row["design_id"], score
    if best is None:
        sys.exit(
            "no scored design in %s, so there is no sensible wiggle parent. "
            "Run more backbones first -- roughly 1 design in 15 clears the "
            "gate." % csv_path
        )
    print("picked best-scoring parent: %s (%.4f)" % (best, best_score))
    return best


def main() -> int:
    cfg = load_config(
        Path("config/pipeline.yaml"), Path("config/targets/01_his_asp.yaml")
    )
    print("rfd3.partial_t = %r Angstroms" % cfg.pipeline.rfd3.partial_t, flush=True)

    store = DesignStore(PREV_RUN / "designs.db")
    parent = PARENT or best_scored_design(store)
    row = store.get(parent)
    if row is None:
        sys.exit("parent %s not found in %s" % (parent, PREV_RUN / "designs.db"))
    prov = Provenance.from_json(row.provenance)
    backbone = Path(row.backbone_path)
    print("parent          = %s" % parent, flush=True)
    print("parent backbone = %s (exists: %s)" % (backbone, backbone.exists()),
          flush=True)
    print("seed_to_hal     = %s" % sorted(prov.seed_to_hal.items()), flush=True)
    if not backbone.exists():
        sys.exit("parent backbone missing; cannot wiggle it")

    seed = cfg.target.seeds[0]
    spec = rfd_stage.spec_from_seed(
        seed=seed,
        motif=cfg.target.motif,
        ligand_id=cfg.target.ligand.id,
        repo_root=Path("."),
        key="%s_%s_wiggle" % (cfg.target.name, seed.id),
    )
    print("\nspec as built from the SEED (wrong numbering for a backbone):",
          flush=True)
    print("  unindex = %s" % spec.unindex, flush=True)
    print("  fixed   = %s" % spec.select_fixed_atoms, flush=True)

    # Same remap as runner._relabel_spec_for_backbone.
    ligand_id = cfg.target.ligand.id
    fixed = {}
    for label, atoms in spec.select_fixed_atoms.items():
        if label == ligand_id:
            fixed[label] = atoms
            continue
        chain, resnum = label[0], int(label[1:])
        if (chain, resnum) not in prov.seed_to_hal:
            sys.exit("seed residue %s not in seed_to_hal" % label)
        hal = prov.seed_to_hal[(chain, resnum)]
        fixed["%s%d" % (hal[0], hal[1])] = atoms
    unindex = ",".join(k for k in fixed if k != ligand_id)
    spec = replace(spec, input_pdb=backbone, unindex=unindex,
                   select_fixed_atoms=fixed)

    print("\nspec REMAPPED onto the parent backbone:", flush=True)
    print("  unindex = %s" % spec.unindex, flush=True)
    print("  fixed   = %s" % spec.select_fixed_atoms, flush=True)

    print("\n--- running the wiggle (partial=True) ---", flush=True)
    backbones, result = rfd_stage.run(
        cfg.pipeline.rfd3,
        specs=[spec],
        out_dir=OUT / "rfd3",
        n_designs=2,
        partial=True,
        gpu=0,
        seed=1,
    )
    print("command: %s" % result.as_log_dict().get("command"), flush=True)
    print("returncode %s in %.1fs"
          % (result.as_log_dict().get("returncode"), result.wall_seconds),
          flush=True)

    written = OUT / "rfd3" / "rfd3_inputs.json"
    if written.exists():
        spec_json = json.loads(written.read_text())
        print("\npartial_t actually written into the spec: %r"
              % list(spec_json.values())[0].get("partial_t"), flush=True)

    print("\nbackbones produced: %d" % len(backbones), flush=True)
    for b in backbones:
        rmsd = b.motif_fidelity()
        print("  %s  motif RMSD %s"
              % (b.structure.name,
                 {k: round(v, 3) for k, v in rmsd.items()} or "(none reported)"),
              flush=True)

    if not backbones:
        print("\nWIGGLE CHECK FAILED: no backbones", flush=True)
        return 1
    print("\nWIGGLE CHECK PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
