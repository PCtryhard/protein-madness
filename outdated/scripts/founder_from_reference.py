"""Extract the de novo founder for the evolution loop from the reference run.

The founder is the LigandMPNN sequence from the organisers' verification run:
a 260-residue **de novo** design, scaffolded by RFD3 around nothing but the
three-atom catalytic motif. It is not a homologue of HG3.17 or of anything else
- the only thing inherited from HG3.17 is the geometry of the base, the donor
and the ligand. That distinction is what keeps us clear of both eligibility
filters (>70% sequence identity, TM > 0.7), and it is why the evolution loop
starts here rather than from a natural enzyme.

It is also already measured, which is what makes it a usable starting point:
iPDE 2.55, catalytic contact 11.18 A, burial 30. Every one of those is a number
to beat.

    python scripts/founder_from_reference.py --out configs/founder_denovo.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evolve.mutate import shell_positions_from_structure
from src.score.official import CatalyticSite, find_prediction, read_complex, score_run

REFERENCE = Path(__file__).resolve().parents[1] / "tests" / "data" / "boltz_reference"
BASE_POSITION = 177
DONOR_POSITION = 99


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--out", type=Path, default=Path("configs/founder_denovo.json"))
    parser.add_argument("--shell-radius", type=float, default=10.0)
    args = parser.parse_args(argv)

    summary = json.loads((args.reference / "summary.json").read_text())
    sequence = summary["ligandmpnn"]["sequence"]

    # Cross-check the positions rather than trusting the constants above.
    mapped = {e["original"]: e["position"] for e in summary["rfd3"]["fixed"]}
    assert mapped["A127"] == BASE_POSITION, mapped
    assert mapped["A50"] == DONOR_POSITION, mapped
    assert sequence[BASE_POSITION - 1] == "D", "catalytic base is not an aspartate"
    assert sequence[DONOR_POSITION - 1] == "Q", "H-bond donor is not a glutamine"

    atoms = read_complex(find_prediction(args.reference)[0])
    shell = sorted(
        p for p in shell_positions_from_structure(atoms, "6NT", args.shell_radius)
        if p not in (BASE_POSITION, DONOR_POSITION)
    )
    score = score_run(args.reference, CatalyticSite(BASE_POSITION, DONOR_POSITION))

    payload = {
        "provenance": "RFD3 de novo scaffold + LigandMPNN, organisers' verification run (job 1)",
        "sequence": sequence,
        "length": len(sequence),
        "base_position": BASE_POSITION,
        "donor_position": DONOR_POSITION,
        "catalytic": {str(BASE_POSITION): "D", str(DONOR_POSITION): "Q"},
        "second_shell": shell,
        "baseline": {
            "ipde": score.ipde,
            "base_to_ligand_a": score.base_to_ligand_a,
            "donor_to_ligand_a": score.donor_to_ligand_a,
            "base_burial_ca_count": score.base_burial_ca_count,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"founder written to {args.out}")
    print(f"  {len(sequence)} residues, {len(shell)} second-shell positions")
    print(f"  baseline to beat: contact {score.base_to_ligand_a:.2f} A, "
          f"iPDE {score.ipde:.3f}, burial {score.base_burial_ca_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
