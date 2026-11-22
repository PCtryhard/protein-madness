"""Re-score the committed runs under the new benchmark, from stored geometry.

No GPU and no re-prediction: designs.csv already holds each bond's raw
distance and angle and the normalised confidence term, which is everything
combine() needs. So this is a pure re-score of the same structures.

    python scripts/rescore.py                 # the three runs in data/
    python scripts/rescore.py <designs.csv> <target.yaml>

Re-scoring is cheap because the benchmark is a function of stored numbers.
When the weights or the ideal angles change again, this says what that did to
the existing leaderboards without spending an allocation to find out.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_pipeline, load_target  # noqa: E402
from src.scoring.combine import combine  # noqa: E402
from src.scoring.hbond import BondMeasurement  # noqa: E402

RUNS = [
    ("01", "data/01/scores/fastrelax-full-21882963/designs.csv",
     "data/01/inputs/fastrelax-full-21882963/target.yaml"),
    ("09", "data/09/scores/fastrelax-09-21883493/designs.csv",
     "data/09/inputs/fastrelax-09-21883493/target.yaml"),
    ("12", "data/12/scores/fastrelax-12-21883495/designs.csv",
     "data/12/inputs/fastrelax-12-21883495/target.yaml"),
]


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main() -> int:
    runs = RUNS
    if len(sys.argv) == 3:
        runs = [("(given)", sys.argv[1], sys.argv[2])]
    elif len(sys.argv) != 1:
        sys.exit(__doc__)

    cfg = load_pipeline(ROOT / "config" / "pipeline.yaml")
    sc = cfg.scoring
    print("new benchmark: S = %.2f*confidence + %.2f*(%.2f*catalytic + "
          "%.2f*non_catalytic + %.2f*angle)"
          % (sc.weights.confidence, sc.weights.geometry,
             sc.geometry_weights.catalytic, sc.geometry_weights.non_catalytic,
             sc.geometry_weights.angle))

    for variant, csv_path, target_path in runs:
        target = load_target(ROOT / target_path)
        # The ideals in the stored target are the OLD ones; re-score against
        # the acceptor chemistry the new config/targets now carries.
        from scripts.build_targets import acceptor_ideal

        bonds = {}
        for hb in target.hbonds:
            if hb.acceptor.site == "ligand":
                key = "lig_%s" % hb.acceptor.atom.lower()
                atom = hb.acceptor.atom
            else:
                key = ""
                atom = hb.acceptor.atom
            got = acceptor_ideal(key, atom)
            if got is None:
                sys.exit("no chemistry for acceptor %s of %s" % (atom, hb.id))
            bonds[hb.id] = (got[0], hb.is_catalytic, got[1])

        rows = list(csv.DictReader(open(ROOT / csv_path, newline="")))
        rescored = []
        for r in rows:
            conf = num(r.get("term_confidence"))
            if conf is None:
                continue
            ms = []
            ok = True
            for bond_id, (ideal, catalytic, _kind) in bonds.items():
                d = num(r.get("%s__distance" % bond_id))
                a = num(r.get("%s__angle" % bond_id))
                if d is None or a is None:
                    ok = False
                    break
                ms.append(BondMeasurement(
                    bond_id=bond_id, distance=d, angle=a,
                    angle_kind="acceptor_side", angle_ideal=ideal,
                    required=catalytic, catalytic=catalytic, atoms={},
                ))
            if not ok or not ms:
                continue
            new = combine(conf, ms, sc)
            rescored.append((r["design_id"], num(r["score"]) or 0.0, new))

        old_scored = sum(1 for _, o, _ in rescored if o > 0)
        new_scored = sum(1 for _, _, n in rescored if not n.gated)
        print()
        print("=== variant %s : %d designs ===" % (variant, len(rescored)))
        print("  bonds: " + ", ".join(
            "%s (%s, %s, ideal %.1f)"
            % (b, "catalytic" if c else "non-catalytic", k, i)
            for b, (i, c, k) in bonds.items()))
        print("  scored under old benchmark: %d    under new: %d"
              % (old_scored, new_scored))
        best_old = max(rescored, key=lambda t: t[1])
        best_new = max(rescored, key=lambda t: t[2].total)
        print("  best old: %.4f  %s" % (best_old[1], best_old[0].split(".")[-1]))
        print("  best new: %.4f  %s" % (best_new[2].total,
                                        best_new[0].split(".")[-1]))
        print("  %-24s %8s %8s  %s" % ("design", "old", "new", "delta"))
        top = sorted(rescored, key=lambda t: -t[2].total)[:6]
        for design_id, old, new in top:
            if new.gated:
                continue
            print("  %-24s %8.4f %8.4f  %+.4f"
                  % (design_id.split(".")[-1], old, new.total, new.total - old))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
