"""Generate a fake leaderboard with the real column names.

    python -m src.report.fake_data --out results/fake

So the figures exist and look right before any real design is made. On Sunday
you change the input path, not the plotting code.

Every file this writes is prefixed ``FAKE_`` and every row carries
``is_fake=True``, so fake numbers can never be mistaken for results. The
pitch-asset generator refuses to run on a table with ``is_fake`` set.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..ids import config_hash, design_id
from ..io.target import REPO_ROOT

FAKE_PREFIX = "FAKE_"


def make_fake_leaderboard(n: int = 240, seed: int = 0, length: int = 129) -> pd.DataFrame:
    """A leaderboard-shaped table with plausible correlations and realistic gaps."""
    rng = np.random.default_rng(seed)
    aa = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
    parent = "".join(rng.choice(aa, size=length))
    cfg_h = config_hash({"fake": True, "seed": seed})

    rows = []
    for i in range(n):
        n_mut = int(rng.integers(1, 12))
        seq = list(parent)
        for p in rng.choice(length, size=n_mut, replace=False):
            seq[p] = rng.choice(aa)
        seq = "".join(seq)
        method = str(rng.choice(["generate.mutate", "generate.ligandmpnn", "generate.proteinmpnn"],
                                p=[0.5, 0.3, 0.2]))
        # more mutations -> lower confidence, with noise
        base = 92.0 - 1.9 * n_mut + rng.normal(0, 4.5)
        mean_plddt = float(np.clip(base, 25, 97))
        active = float(np.clip(mean_plddt - rng.normal(4, 7), 20, 98))
        intact = 0.0 if rng.random() < 0.06 else (0.5 if rng.random() < 0.05 else 1.0)
        rows.append({
            "design_id": design_id(seq, method, cfg_h),
            "is_fake": True,
            "method": method,
            "sequence": seq,
            "mutations": f"<{n_mut} fake mutations>",
            "n_mutations": n_mut,
            "length": length,
            "sequence_identity": 1 - n_mut / length,
            "net_charge_ph7": float(rng.normal(2, 3)),
            "isoelectric_point": float(rng.normal(8.5, 1.5)),
            "gravy": float(rng.normal(-0.4, 0.2)),
            "fraction_charged": float(rng.normal(0.22, 0.03)),
            "max_homopolymer_run": int(rng.integers(1, 6)),
            "low_complexity_fraction": float(rng.uniform(0.2, 0.7)),
            "cysteine_count": int(rng.integers(0, 9)),
            "unpaired_cys": int(rng.random() < 0.2),
            "unpaired_cys_delta": int(rng.choice([0, 0, 0, 1])),
            "catalytic_integrity": intact,
            "n_catalytic": 2,
            "mean_plddt": mean_plddt,
            "active_site_plddt": active,
            "catalytic_pae_max": float(np.clip(rng.normal(8, 3) + 0.2 * n_mut, 0.5, 30)),
            "catalytic_geometry_worst_dev": float(abs(rng.normal(0, 0.8)) + 0.03 * n_mut),
            "self_consistency_rmsd": float(abs(rng.normal(1.2, 0.6)) + 0.05 * n_mut),
            "self_consistency_tmscore": float(np.clip(rng.normal(0.85, 0.08), 0.2, 1.0)),
            "hydrophobic_patch_area": float(abs(rng.normal(120, 60))),
            "seq_novelty": float(np.clip(rng.beta(2, 5) + n_mut / 200, 0, 1)),
            "novelty_vs_parent": n_mut / length,
        })

    df = pd.DataFrame(rows).drop_duplicates(subset="design_id")
    # Realistic gaps: ~15% of designs never got folded, so Tier-2 is missing.
    unfolded = df.sample(frac=0.15, random_state=seed).index
    for col in ["mean_plddt", "active_site_plddt", "catalytic_pae_max",
                "catalytic_geometry_worst_dev", "self_consistency_rmsd",
                "self_consistency_tmscore", "hydrophobic_patch_area"]:
        df.loc[unfolded, col] = np.nan
    df["stages"] = np.where(df.index.isin(unfolded), "score.tier1", "score.tier1,score.tier2")

    # The wild-type baseline row, which the figures compare against.
    wt = {c: np.nan for c in df.columns}
    wt.update({
        "design_id": "d_wildtype", "is_fake": True, "method": "wildtype", "sequence": parent,
        "mutations": "", "n_mutations": 0, "length": length, "sequence_identity": 1.0,
        "catalytic_integrity": 1.0, "n_catalytic": 2, "mean_plddt": 93.5,
        "active_site_plddt": 95.1, "catalytic_pae_max": 3.4,
        "catalytic_geometry_worst_dev": 0.0, "self_consistency_rmsd": 0.4,
        "self_consistency_tmscore": 0.98, "seq_novelty": 0.02, "novelty_vs_parent": 0.0,
        "net_charge_ph7": 2.1, "isoelectric_point": 8.9, "gravy": -0.41,
        "fraction_charged": 0.22, "max_homopolymer_run": 3, "low_complexity_fraction": 0.4,
        "cysteine_count": 8, "unpaired_cys": 0, "unpaired_cys_delta": 0,
        "hydrophobic_patch_area": 110.0, "stages": "score.tier1,score.tier2",
    })
    return pd.concat([pd.DataFrame([wt]), df], ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "fake"))
    ap.add_argument("-n", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = make_fake_leaderboard(args.n, args.seed)
    path = out / f"{FAKE_PREFIX}score.tier1.csv"
    df.to_csv(path, index=False)
    print(f"wrote {len(df)} FAKE rows -> {path}")
    print("These are not results. Every row has is_fake=True; pitch_assets.py refuses them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
