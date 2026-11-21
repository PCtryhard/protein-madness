"""Generate the scoring fixtures from geometry derived by hand.

Run from the repo root to regenerate:

    python tests/fixtures/build_fixtures.py

The point of these fixtures is that their geometry is known independently of
the code under test. The derivation is here in full so the numbers in
tests/test_scoring.py can be checked by reading, not by trusting.

THE DERIVATION
--------------

Bond 1 is ligand N3 (donor) to ASP229 OD2 (acceptor), with CG as the acceptor
antecedent. Put the acceptor at the origin and the antecedent along -x:

    OD2 = (0, 0, 0)
    CG  = (-1.25, 0, 0)          a C-O bond, ~1.25 A

For a target distance d and a target acceptor-side angle t, place the donor in
the xy plane at

    N3 = d * (cos(180 - t), sin(180 - t), 0)

Then the angle CG-OD2...N3 is the angle between (CG - OD2) = (-1, 0, 0) and
(N3 - OD2) = (cos(180-t), sin(180-t), 0), which is 180 - (180 - t) = t. The
distance |N3 - OD2| is d by construction.

Bond 2 is HIS291 NE2 (donor) to ligand N1 (acceptor), with ligand N2 as the
acceptor antecedent. Same construction, translated to z = 10 so the two bonds
do not overlap, and with the antecedent along +x instead:

    N1  = (8.0, 0, 10)
    N2  = (9.3, 0, 10)           (N2 - N1) is +x
    NE2 = N1 + d * (cos t, sin t, 0)

The angle N2-N1...NE2 is then the angle between (1, 0, 0) and (cos t, sin t, 0),
which is t.

THE EXPECTED TERMS, with the defaults in config/pipeline.yaml
(distance ideal 2.9, tolerated 3.5, fail 4.5; angle tolerance 60) and the
fixture target's acceptor_ideal_angle of 180 for both bonds:

    d = clip(1 - |r - 2.9| / (3.5 - 2.9), 0, 1)
    a = clip(1 - |theta - 180| / 60,      0, 1)

    2.900 A, 180 deg ->  d = 1 - 0/0.6     = 1.0
                         a = 1 - 0/60      = 1.0
    3.400 A, 130 deg ->  d = 1 - 0.5/0.6   = 0.1666666...
                         a = 1 - 50/60     = 0.1666666...
    5.000 A, 150 deg ->  beyond fail=4.5, so the design is GATED and the
                         total is 0 regardless of the other terms.
"""

from __future__ import annotations

import math
from pathlib import Path

HERE = Path(__file__).parent

# ---- bond 1 frame -------------------------------------------------------
OD2 = (0.0, 0.0, 0.0)
CG = (-1.25, 0.0, 0.0)
OD1 = (-1.90, 1.05, 0.0)       # the other carboxylate oxygen, unused by the score

# ---- bond 2 frame -------------------------------------------------------
N1 = (8.0, 0.0, 10.0)
N2 = (9.3, 0.0, 10.0)
CD2 = (4.0, -1.2, 10.0)        # HIS ring carbon, the donor antecedent


def donor_for_bond1(distance: float, angle_deg: float) -> tuple[float, float, float]:
    """Ligand N3, placed to give exactly (distance, angle) at the acceptor."""
    a = math.radians(180.0 - angle_deg)
    return (distance * math.cos(a), distance * math.sin(a), 0.0)


def donor_for_bond2(distance: float, angle_deg: float) -> tuple[float, float, float]:
    """HIS291 NE2, placed to give exactly (distance, angle) at the acceptor."""
    t = math.radians(angle_deg)
    return (
        N1[0] + distance * math.cos(t),
        N1[1] + distance * math.sin(t),
        N1[2],
    )


def atom_line(
    serial: int,
    name: str,
    resname: str,
    chain: str,
    resnum: int,
    xyz: tuple[float, float, float],
    record: str = "ATOM",
    element: str | None = None,
) -> str:
    """One PDB coordinate line, in fixed columns.

    Atom names are left-justified from column 14 for names of four characters
    and padded from 14 otherwise, which is what the PDB spec says and what
    tracking.read_pdb expects.
    """
    element = element or name[0]
    padded = name.ljust(4) if len(name) >= 4 else f" {name:<3}"
    return (
        f"{record:<6}{serial:>5} {padded}{'':1}{resname:>3} {chain}{resnum:>4}{'':4}"
        f"{xyz[0]:>8.3f}{xyz[1]:>8.3f}{xyz[2]:>8.3f}"
        f"{1.00:>6.2f}{0.00:>6.2f}{'':10}{element:>2}"
    )


def build(name: str, bond1: tuple[float, float], bond2: tuple[float, float]) -> str:
    """One fixture: (distance, angle) for each of the two named bonds."""
    n3 = donor_for_bond1(*bond1)
    ne2 = donor_for_bond2(*bond2)

    lines = [
        f"REMARK   1 FIXTURE {name}",
        f"REMARK   1 bond lig_n3_to_asp229_od2 : {bond1[0]:.3f} A, {bond1[1]:.1f} deg",
        f"REMARK   1 bond his291_ne2_to_lig_n1 : {bond2[0]:.3f} A, {bond2[1]:.1f} deg",
        "REMARK   1 Geometry derived by hand; see build_fixtures.py.",
    ]
    serial = 1

    def add(*args, **kwargs):
        nonlocal serial
        lines.append(atom_line(serial, *args, **kwargs))
        serial += 1

    # ASP 229: enough of a residue to be parsed as one, with the two atoms the
    # score actually uses placed exactly.
    add("N", "ASP", "A", 229, (-3.0, -1.5, 0.0), element="N")
    add("CA", "ASP", "A", 229, (-2.6, -0.2, 0.0), element="C")
    add("C", "ASP", "A", 229, (-3.6, 0.9, 0.0), element="C")
    add("O", "ASP", "A", 229, (-4.8, 0.7, 0.0), element="O")
    add("CB", "ASP", "A", 229, (-1.9, -0.9, 0.0), element="C")
    add("CG", "ASP", "A", 229, CG, element="C")
    add("OD1", "ASP", "A", 229, OD1, element="O")
    add("OD2", "ASP", "A", 229, OD2, element="O")

    # HIS 291
    add("N", "HIS", "A", 291, (2.0, -3.0, 10.0), element="N")
    add("CA", "HIS", "A", 291, (2.6, -2.0, 10.0), element="C")
    add("C", "HIS", "A", 291, (1.8, -0.8, 10.0), element="C")
    add("O", "HIS", "A", 291, (0.6, -0.9, 10.0), element="O")
    add("CB", "HIS", "A", 291, (3.2, -2.4, 11.3), element="C")
    add("CG", "HIS", "A", 291, (3.6, -1.3, 12.2), element="C")
    add("CD2", "HIS", "A", 291, CD2, element="C")
    add("NE2", "HIS", "A", 291, ne2, element="N")

    # The ligand. Three atoms is enough: the score names N3, N1 and N2.
    add("N3", "6NT", "A", 400, n3, record="HETATM", element="N")
    add("N1", "6NT", "A", 400, N1, record="HETATM", element="N")
    add("N2", "6NT", "A", 400, N2, record="HETATM", element="N")

    lines.append("END")
    return "\n".join(lines) + "\n"


FIXTURES = {
    # name           bond 1 (d, angle)   bond 2 (d, angle)
    "ideal": ((2.900, 180.0), (2.900, 180.0)),
    "stretched": ((3.400, 130.0), (3.400, 130.0)),
    "mixed": ((2.900, 180.0), (3.400, 130.0)),
    "gated": ((5.000, 150.0), (2.900, 180.0)),
}


def main() -> None:
    for name, (bond1, bond2) in FIXTURES.items():
        path = HERE / f"{name}.pdb"
        path.write_text(build(name, bond1, bond2), encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
