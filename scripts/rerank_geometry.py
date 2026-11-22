"""Re-rank scored designs on catalytic geometry rather than on confidence.

The shipped benchmark (src/scoring/combine.py) is half confidence and half
geometry, and its geometry half only knows about three H-bonds: a distance and
an acceptor-side angle each. That leaves two things it cannot see.

  1.  WHERE the partner sits relative to the ring plane. All three specified
      contacts are to sp2 lone pairs that lie IN the plane of the ligand's
      aromatic system -- the triazole N1 and N3, and the conjugated nitro
      oxygens. A side chain parked over the ring FACE can sit at a perfect
      2.9 A from one of those atoms and still be unable to engage its lone
      pair. Distance alone cannot tell the two apart; an out-of-plane offset
      can.

  2.  WHAT lines the pocket. The ligand has a polar end (triazole nitrogens
      plus the nitro group) and a hydrophobic edge (the unsubstituted benzo
      carbons). A design that puts charge against the hydrophobic edge, or
      leaves the catalytic end lined with leucines, is wrong in a way no
      H-bond distance reports.

This script adds three terms for exactly those, weights geometry heavily, and
re-ranks. It reads only committed data -- the Boltz-predicted complex and the
target definition -- and writes one CSV plus one PDB per design.

    python scripts/rerank_geometry.py                  # designs scoring >= 0.80
    python scripts/rerank_geometry.py --min-score 0    # everything scored

THE FORMULA

    S_new = 0.15 * confidence        (as recorded by the pipeline)
          + 0.30 * hbond             (the shipped geometry composite)
          + 0.25 * planarity         (new, below)
          + 0.15 * cat_polarity      (new, below)
          + 0.15 * apolar_match      (new, below)

Geometry therefore carries 0.85 against the shipped benchmark's 0.50, which is
the point of this pass. Every weight is a module constant; edit WEIGHTS.

Nothing here re-predicts or re-folds. It is a pure re-score of structures that
are already committed, so it is reproducible and costs no GPU.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tracking import read_structure, write_pdb  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Weights. Geometry is 0.85 of the total, against 0.50 in the shipped score.
# ---------------------------------------------------------------------------
WEIGHTS = {
    "confidence": 0.15,
    "hbond": 0.30,
    "planarity": 0.25,
    "cat_polarity": 0.15,
    "apolar_match": 0.15,
}

# The shipped geometry composite, reproduced so the old score can be rebuilt
# from the stored terms and checked against the stored total.
GEOM_W = {"catalytic": 0.5, "non_catalytic": 0.2, "angle": 0.3}

# --- planarity ---
# An interacting atom more than this far off the ring plane is treated as
# sitting over the face rather than in the lone-pair direction.
OUT_OF_PLANE_FAIL = 2.0      # A
# The catalytic base does the chemistry, so its geometry counts for more than
# the two auxiliary donors.
CATALYTIC_PLANE_WEIGHT = 2.0

# --- polarity shells ---
# The contact zone is deliberately tight: a polar partner has to be close
# enough to actually be one. The surround shell starts where the contact zone
# ends and is wide enough (60-95 side-chain atoms) to give a stable fraction.
CONTACT_ZONE = 3.5           # A of a ligand polar atom
SURROUND_OUT = 8.0           # A of any ligand atom
# Polar side-chain atoms in the contact zone beyond one per designed bond.
# At this many excess donors the site is solvated rather than localised.
EXCESS_POLAR_FAIL = 6
APOLAR_SHELL = 5.0           # A of the hydrophobic edge
APOLAR_PACKED = 8            # side-chain atoms near the edge counted as packed

# ---------------------------------------------------------------------------
# Ligand chemistry: 6NT, nitro-benzotriazole, 12 heavy atoms.
#
#   triazole   N1 N2 N3   fused at C3A-C7A
#   benzo      C3A C4 C5 C6 C7 C7A
#   nitro      NO1 O11 O21, on C5
# ---------------------------------------------------------------------------
RING_ATOMS = ("N1", "N2", "N3", "C3A", "C4", "C5", "C6", "C7", "C7A")
LIGAND_POLAR = ("N1", "N2", "N3", "NO1", "O11", "O21")
# The genuinely hydrophobic edge: benzo carbons that carry no heteroatom and
# are not at the ring fusion. C5 bears the nitro; C3A and C7A belong to the
# triazole as well, so none of the three is a clean nonpolar probe.
LIGAND_APOLAR_EDGE = ("C4", "C6", "C7")

BACKBONE = {"N", "CA", "C", "O", "OXT"}

# Fallback functional-group planes, used only when the target file does not
# pin the residue's atoms itself.
GROUP_PLANE = {
    "GLU": ("OE1", "CD", "OE2"),
    "ASP": ("OD1", "CG", "OD2"),
    "ASN": ("OD1", "CG", "ND2"),
    "GLN": ("OE1", "CD", "NE2"),
    "HIS": ("CD2", "CE1", "NE2"),
}


def ext(path) -> str:
    """Absolute path that survives Windows' 260-character MAX_PATH.

    The committed Boltz output nests the design id three times, so these paths
    run past 340 characters. Without the extended-length prefix Python's
    pathlib reports the files as simply absent -- silently, which is worse than
    an error.
    """
    s = os.path.abspath(str(path))
    if os.name == "nt" and not s.startswith("\\\\?\\"):
        return "\\\\?\\" + s
    return s


def write_lf(path, text: str) -> None:
    """Write text with LF endings whatever the platform.

    Python's write_text translates "\\n" to "\\r\\n" on Windows, which would
    make this script emit different bytes here than on the cluster. The CSV and
    the PDBs are read by other tools and diffed against Euler, so they have to
    be byte-identical either way.
    """
    with open(ext(path), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def element_of(atom_name: str) -> str:
    for ch in atom_name:
        if ch.isalpha():
            return ch.upper()
    return "X"


def fit_plane(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Least-squares plane through `coords`.

    Returns (centroid, unit normal, RMS deviation). The normal is the singular
    vector with the smallest singular value, i.e. the direction of least
    spread.
    """
    centroid = coords.mean(axis=0)
    centred = coords - centroid
    _u, _s, vt = np.linalg.svd(centred, full_matrices=False)
    normal = vt[-1]
    normal = normal / np.linalg.norm(normal)
    rms = float(np.sqrt(np.mean((centred @ normal) ** 2)))
    return centroid, normal, rms


def ramp(value: float, good: float, bad: float) -> float:
    """1.0 at `good`, 0.0 at `bad`, linear between, clamped outside."""
    if bad == good:
        return 1.0 if value == good else 0.0
    t = (value - good) / (bad - good)
    return float(min(1.0, max(0.0, 1.0 - t)))


# ---------------------------------------------------------------------------


@dataclass
class Bond:
    """One target H-bond, reduced to what this script needs."""

    bond_id: str
    ligand_atom: str
    protein_atom: str
    seed_resnum: int
    seed_chain: str
    required: bool
    plane_atoms: tuple[str, ...] | None = None
    # filled in once the design residue is identified
    design_resnum: int | None = None
    design_resname: str | None = None
    recorded_distance: float | None = None
    found_distance: float | None = None


@dataclass
class SiteGeometry:
    planarity: float = 0.0
    cat_polarity: float = 0.0
    apolar_match: float = 0.0
    ring_rms: float = 0.0
    offsets: dict = field(default_factory=dict)
    coplanarity: dict = field(default_factory=dict)
    f_cat_polar: float = 0.0
    f_shell_polar: float = 0.0
    n_polar_near: int = 0
    n_apolar_near: int = 0
    n_polar_expected: int = 0
    n_polar_excess: int = 0
    f_apolar_edge: float = 0.0
    n_edge_contacts: int = 0
    site_residues: str = ""
    edge_residues: str = ""
    notes: list = field(default_factory=list)


def load_bonds(target_path: Path) -> list[Bond]:
    """Read the H-bond definitions and the pinned atom triples from a target."""
    spec = yaml.safe_load(Path(ext(target_path)).read_text(encoding="utf-8"))
    fixed_atoms = (spec.get("motif") or {}).get("fixed_atoms") or {}
    bonds: list[Bond] = []
    for hb in spec.get("hbonds") or []:
        prot = next((hb[side] for side in ("donor", "acceptor")
                     if hb[side].get("site") == "protein"), None)
        lig = next((hb[side] for side in ("donor", "acceptor")
                    if hb[side].get("site") == "ligand"), None)
        if prot is None or lig is None:
            continue
        key = "%s%s" % (prot["chain"], prot["resnum"])
        pinned = fixed_atoms.get(key)
        plane = tuple(a.strip() for a in pinned.split(",")) if pinned else None
        bonds.append(Bond(
            bond_id=hb["id"],
            ligand_atom=lig["atom"],
            protein_atom=prot["atom"],
            seed_resnum=int(prot["resnum"]),
            seed_chain=prot["chain"],
            required=bool(hb.get("required", False)),
            plane_atoms=plane,
        ))
    return bonds


def identify_residues(bonds: list[Bond], st, row: dict,
                      notes: list) -> None:
    """Locate each catalytic residue in the design, and check the result.

    The seed numbering (A229/A261/A291) does not survive into a 260-residue de
    novo scaffold, and the RFD3 .trb that records the mapping is not committed.
    So each residue is found geometrically: the one protein residue carrying an
    atom of the right name at the distance the pipeline already recorded for
    that bond in designs.csv.

    That makes the identification self-checking. If no residue sits at the
    recorded distance, or more than one does, the design is reported rather
    than guessed at.
    """
    for b in bonds:
        col = "%s__distance" % b.bond_id
        raw = (row.get(col) or "").strip()
        if not raw:
            notes.append("%s: no recorded distance" % b.bond_id)
            continue
        b.recorded_distance = float(raw)
        lig_xyz = st.ligand.get(b.ligand_atom)
        if lig_xyz is None:
            notes.append("%s: ligand atom %s absent" % (b.bond_id, b.ligand_atom))
            continue

        hits = []
        for (chain, resnum, name), xyz in st.atoms.items():
            if name != b.protein_atom:
                continue
            d = float(np.linalg.norm(xyz - lig_xyz))
            if abs(d - b.recorded_distance) <= 0.02:
                hits.append((chain, resnum, d))
        if len(hits) == 1:
            chain, resnum, d = hits[0]
            b.design_resnum = resnum
            b.design_resname = st.residues.get((chain, resnum))
            b.found_distance = d
        elif not hits:
            notes.append("%s: no %s atom at the recorded %.3f A"
                         % (b.bond_id, b.protein_atom, b.recorded_distance))
        else:
            notes.append("%s: %d residues at the recorded distance"
                         % (b.bond_id, len(hits)))


def score_planarity(bonds: list[Bond], st, geo: SiteGeometry) -> float:
    """How well the catalytic groups lie in the ligand's aromatic plane.

    Two things per bond, averaged:

      offset       perpendicular distance of the interacting protein atom from
                   the ring plane. 1.0 in plane, 0.0 at OUT_OF_PLANE_FAIL.
      coplanarity  |cos| of the angle between the ring normal and the normal of
                   the residue's own functional-group plane. 1.0 coplanar,
                   0.0 perpendicular. The absolute value because a group
                   flipped 180 degrees is still coplanar.

    The catalytic bond is weighted CATALYTIC_PLANE_WEIGHT times the others.
    """
    ring = [st.ligand[a] for a in RING_ATOMS if a in st.ligand]
    if len(ring) < 4:
        geo.notes.append("ring has only %d of %d atoms" % (len(ring), len(RING_ATOMS)))
        return 0.0
    centroid, normal, rms = fit_plane(np.array(ring))
    geo.ring_rms = rms
    if rms > 0.25:
        geo.notes.append("ligand ring is not planar (rms %.2f A)" % rms)

    total = weight_sum = 0.0
    for b in bonds:
        if b.design_resnum is None:
            continue
        key = (b.seed_chain, b.design_resnum, b.protein_atom)
        xyz = st.atoms.get(key)
        if xyz is None:
            continue
        offset = abs(float((xyz - centroid) @ normal))
        geo.offsets[b.bond_id] = offset
        s_offset = ramp(offset, 0.0, OUT_OF_PLANE_FAIL)

        plane_atoms = b.plane_atoms or GROUP_PLANE.get(b.design_resname or "")
        s_cop = None
        if plane_atoms:
            pts = [st.atoms.get((b.seed_chain, b.design_resnum, n))
                   for n in plane_atoms]
            pts = [p for p in pts if p is not None]
            if len(pts) >= 3:
                _c, gnormal, _r = fit_plane(np.array(pts))
                cop = abs(float(gnormal @ normal))
                geo.coplanarity[b.bond_id] = float(np.degrees(np.arccos(
                    min(1.0, max(0.0, cop)))))
                s_cop = cop
        parts = [s_offset] + ([s_cop] if s_cop is not None else [])
        w = CATALYTIC_PLANE_WEIGHT if b.required else 1.0
        total += w * (sum(parts) / len(parts))
        weight_sum += w

    if weight_sum == 0:
        return 0.0
    geo.planarity = total / weight_sum
    return geo.planarity


def sidechain_atoms(st):
    """Side-chain atoms only, as (chain, resnum, name, xyz, element).

    Backbone is excluded because it is present everywhere regardless of what
    the design chose; side chains are what LigandMPNN actually picks, so they
    are what a polarity term should be measuring.
    """
    out = []
    for (chain, resnum, name), xyz in st.atoms.items():
        if name in BACKBONE:
            continue
        out.append((chain, resnum, name, xyz, element_of(name)))
    return out


def score_polarity(st, geo: SiteGeometry, bonds: list[Bond]) -> float:
    """Polarity LOCALISED at the catalytic atoms, in an otherwise apolar pocket.

    The first version of this term rewarded a high polar fraction near the
    ligand's polar end. The wild-type enzyme scores 0.06 on that, because a
    real Kemp eliminase buries its carboxylate base in a hydrophobic pocket on
    purpose: desolvating the base is part of how it works. Rewarding polar
    density at the site therefore penalises the architecture that catalyses.
    What distinguishes the real enzyme is not more polarity but LESS, placed
    precisely -- two polar side-chain atoms serving the polar end where our
    designs pile up six to eight, inside a pocket at 0.12 polar fraction where
    our designs average 0.19.

    So two parts, both measured on well-sampled shells:

      surround   1 - (polar fraction of side-chain atoms in the
                 CONTACT_ZONE-to-SURROUND_OUT shell around the whole ligand).
                 Rewards a hydrophobic pocket. 60-95 atoms, so it is stable.
      parsimony  how close the number of polar side-chain atoms in the contact
                 zone is to one partner per designed bond. Excess polar atoms
                 are competing donors that solvate the site, which is the
                 opposite of localised.

    Sanity-checked against the one full-length reference the project ships
    (wild-type with 6NT bound), which this formulation rates above every
    design. One reference is a sanity check, not a validation -- the raw
    counts are all written to the CSV so the weights can be revisited.
    """
    polar_pts = [st.ligand[a] for a in LIGAND_POLAR if a in st.ligand]
    whole = list(st.ligand.values())
    if not polar_pts or not whole:
        geo.notes.append("no ligand polar atoms found")
        return 0.0
    polar_pts = np.array(polar_pts)
    whole = np.array(whole)

    contact_p = contact_a = surr_p = surr_a = 0
    near_res = {}
    for chain, resnum, name, xyz, el in sidechain_atoms(st):
        is_polar = el in ("N", "O")
        is_apolar = el in ("C", "S")
        d_pol = float(np.min(np.linalg.norm(polar_pts - xyz, axis=1)))
        if d_pol <= CONTACT_ZONE:
            if is_polar:
                contact_p += 1
            elif is_apolar:
                contact_a += 1
            near_res[(chain, resnum)] = st.residues.get((chain, resnum), "?")
        d_all = float(np.min(np.linalg.norm(whole - xyz, axis=1)))
        if CONTACT_ZONE <= d_all < SURROUND_OUT:
            if is_polar:
                surr_p += 1
            elif is_apolar:
                surr_a += 1

    geo.n_polar_near, geo.n_apolar_near = contact_p, contact_a
    contact_tot, surr_tot = contact_p + contact_a, surr_p + surr_a
    if surr_tot < 20:
        geo.notes.append("surround shell has only %d atoms" % surr_tot)
    f_contact = (contact_p / contact_tot) if contact_tot else 0.0
    f_surround = (surr_p / surr_tot) if surr_tot else 0.0
    geo.f_cat_polar, geo.f_shell_polar = f_contact, f_surround

    surround = 1.0 - f_surround
    # one polar partner atom per designed bond to a polar ligand atom
    expected = max(1, sum(1 for b in bonds if b.ligand_atom in LIGAND_POLAR))
    excess = max(0, contact_p - expected)
    parsimony = ramp(excess, 0.0, EXCESS_POLAR_FAIL)
    geo.n_polar_expected, geo.n_polar_excess = expected, excess

    geo.site_residues = " ".join(
        "%s%d" % (near_res[k], k[1]) for k in sorted(near_res))
    geo.cat_polarity = 0.5 * surround + 0.5 * parsimony
    return geo.cat_polarity


def score_apolar(st, geo: SiteGeometry) -> float:
    """Nonpolar side chains against the ligand's hydrophobic edge.

    Rewards the fraction of nearby side-chain atoms that are carbon or sulfur,
    and separately rewards there being contacts at all -- an edge with nothing
    packed against it is solvent-exposed, which is its own problem.
    """
    edge = [st.ligand[a] for a in LIGAND_APOLAR_EDGE if a in st.ligand]
    if not edge:
        geo.notes.append("no ligand apolar edge atoms found")
        return 0.0
    edge = np.array(edge)

    n_apolar = n_polar = 0
    near_res = {}
    for chain, resnum, name, xyz, el in sidechain_atoms(st):
        d = float(np.min(np.linalg.norm(edge - xyz, axis=1)))
        if d > APOLAR_SHELL:
            continue
        if el in ("C", "S"):
            n_apolar += 1
        elif el in ("N", "O"):
            n_polar += 1
        near_res[(chain, resnum)] = st.residues.get((chain, resnum), "?")

    total = n_apolar + n_polar
    geo.n_edge_contacts = total
    if total == 0:
        geo.notes.append("hydrophobic edge has no side chain within %.1f A"
                         % APOLAR_SHELL)
        geo.f_apolar_edge = 0.0
        return 0.0
    geo.f_apolar_edge = n_apolar / total
    packing = min(1.0, total / APOLAR_PACKED)
    geo.edge_residues = " ".join(
        "%s%d" % (near_res[k], k[1]) for k in sorted(near_res))
    geo.apolar_match = 0.7 * geo.f_apolar_edge + 0.3 * packing
    return geo.apolar_match


def num(row: dict, key: str):
    raw = (row.get(key) or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def old_geometry(row: dict) -> tuple[float, float]:
    """Rebuild the shipped geometry composite from the stored terms.

    Returned with the total it implies, so the caller can check that against
    the stored `score`. If the two agree, the stored terms are being read the
    way combine.py wrote them.
    """
    present = {}
    for name, col in (("catalytic", "term_catalytic"),
                      ("non_catalytic", "term_non_catalytic"),
                      ("angle", "term_angle")):
        v = num(row, col)
        if v is not None:
            present[name] = v
    if not present:
        return 0.0, 0.0
    live = sum(GEOM_W[k] for k in present)
    geometry = sum(GEOM_W[k] * v for k, v in present.items()) / live
    conf = num(row, "term_confidence") or 0.0
    return geometry, 0.5 * conf + 0.5 * geometry


def find_cif(design_dir: Path) -> Path | None:
    hits = list(Path(ext(design_dir)).rglob("*_model_0.cif"))
    return hits[0] if hits else None


# ---------------------------------------------------------------------------
# Calibration against the one real enzyme the repo ships.
# ---------------------------------------------------------------------------
REFERENCE_CIF = "data/01/inputs/fold_wildtype_ligand_model_0.cif"

# Residue-type-aware probes. Without the resname check an ASN OD1 gets picked
# up as a carboxylate base, which is how the first version of this calibration
# reported nonsense.
_BASE = {"GLU": ("OE1", "OE2"), "ASP": ("OD1", "OD2")}
_DONOR = {"HIS": ("ND1", "NE2"), "ASN": ("ND2",), "GLN": ("NE2",),
          "SER": ("OG",), "THR": ("OG1",), "TYR": ("OH",), "TRP": ("NE1",),
          "LYS": ("NZ",), "ARG": ("NE", "NH1", "NH2")}


def reference_calibration() -> dict | None:
    """Score the wild-type enzyme on the three new terms.

    The reference has no target file and no provenance, so its catalytic
    partners are found by nearest-atom search restricted to chemically capable
    residues. Recorded so the calibration travels with the output instead of
    living in somebody's notebook.

    Note what this is and is not: ONE structure, and the polarity term was
    revised after seeing it. So these numbers show the score is consistent with
    the only real enzyme on hand -- not that it has been validated.
    """
    path = REPO / REFERENCE_CIF
    if not Path(ext(path)).exists():
        return None
    st = read_structure(ext(path), "6NT")
    bonds = []
    for lig_atom, table, required in (("N3", _BASE, True),
                                      ("N1", _DONOR, False),
                                      ("O11", _DONOR, False)):
        lig = st.ligand.get(lig_atom)
        if lig is None:
            continue
        best = None
        for (chain, resnum, name), xyz in st.atoms.items():
            resname = st.residues.get((chain, resnum))
            if resname not in table or name not in table[resname]:
                continue
            d = float(np.linalg.norm(xyz - lig))
            if d <= 4.5 and (best is None or d < best[0]):
                best = (d, chain, resnum, name, resname)
        if best is None:
            continue
        d, chain, resnum, name, resname = best
        b = Bond(bond_id="to_%s" % lig_atom, ligand_atom=lig_atom,
                 protein_atom=name, seed_resnum=resnum, seed_chain=chain,
                 required=required, plane_atoms=GROUP_PLANE.get(resname))
        b.design_resnum, b.design_resname, b.found_distance = resnum, resname, d
        bonds.append(b)

    geo = SiteGeometry()
    score_planarity(bonds, st, geo)
    score_polarity(st, geo, bonds)
    score_apolar(st, geo)
    return {
        "structure": REFERENCE_CIF,
        "partners": ["%s -> %s%d.%s @ %.2f A"
                     % (b.bond_id, b.design_resname, b.design_resnum,
                        b.protein_atom, b.found_distance) for b in bonds],
        "planarity": round(geo.planarity, 4),
        "cat_polarity": round(geo.cat_polarity, 4),
        "apolar_match": round(geo.apolar_match, 4),
        "ring_rms_planarity": round(geo.ring_rms, 4),
        "offsets_A": {k: round(v, 2) for k, v in sorted(geo.offsets.items())},
        "f_polar_surround": round(geo.f_shell_polar, 3),
        "pocket": geo.site_residues,
        "caveat": "one reference structure, and cat_polarity was revised after "
                  "seeing it: a consistency check, not a validation",
    }


# Runs scored before the section-14 distance fix. Their stored `score` and
# their stored term_* columns were computed with the old symmetric distance
# metric, so "above 0.80" does not mean the same thing for them and their
# totals do not rebuild from their own terms. Excluded unless asked for.
LEGACY_BATCHES = ("fastrelax",)


def collect(min_score: float, include_legacy: bool = False):
    """Every scored design at or above `min_score`, with its structure."""
    out = []
    skipped_legacy = 0
    for scores_csv in sorted(Path(ext(REPO / "data")).rglob("scores/*/designs.csv")):
        parts = Path(str(scores_csv)).parts
        variant = parts[parts.index("data") + 1]
        run = Path(str(scores_csv)).parent.name
        batch = run.split("-")[0]
        with open(scores_csv, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                s = num(row, "score")
                if s is None or s < min_score or s <= 0:
                    continue
                if batch in LEGACY_BATCHES and not include_legacy:
                    skipped_legacy += 1
                    continue
                design = row["design_id"]
                ddir = REPO / "data" / variant / "boltz" / run / design / "boltz_r0"
                out.append((variant, batch, run, design, s, row, ddir))
    if skipped_legacy:
        print("skipping %d design(s) from pre-fix %s runs (different score "
              "scale; --include-legacy to keep them)"
              % (skipped_legacy, "/".join(LEGACY_BATCHES)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--min-score", type=float, default=0.80,
                    help="only re-rank designs scoring at least this "
                         "(default 0.80; pass 0 for everything scored)")
    ap.add_argument("--out-dir", default="data/rerank",
                    help="where the CSV and the PDBs go (default data/rerank)")
    ap.add_argument("--no-pdb", action="store_true",
                    help="skip writing the PDB copies")
    ap.add_argument("--include-legacy", action="store_true",
                    help="also include the pre-fix fastrelax-* runs, whose "
                         "stored scores are on the old distance metric")
    args = ap.parse_args()

    out_dir = REPO / args.out_dir
    pdb_dir = out_dir / "pdb"
    rows = collect(args.min_score, args.include_legacy)
    if not rows:
        print("no designs at or above %.2f" % args.min_score)
        return 1
    print("re-ranking %d designs scoring >= %.2f" % (len(rows), args.min_score))
    print("weights: " + "  ".join("%s %.2f" % (k, v) for k, v in WEIGHTS.items()))
    print()

    results = []
    score_mismatch = 0
    for variant, batch, run, design, old_score, row, ddir in rows:
        cif = find_cif(ddir)
        if cif is None:
            print("  SKIP %s: no predicted structure under %s" % (design, ddir))
            continue
        st = read_structure(str(cif), "6NT")

        target = REPO / "data" / variant / "inputs" / run / "target.yaml"
        if not Path(ext(target)).exists():
            print("  SKIP %s: no target.yaml at %s" % (design, target))
            continue
        bonds = load_bonds(target)

        geo = SiteGeometry()
        identify_residues(bonds, st, row, geo.notes)
        score_planarity(bonds, st, geo)
        score_polarity(st, geo, bonds)
        score_apolar(st, geo)

        hbond, rebuilt = old_geometry(row)
        if abs(rebuilt - old_score) > 0.002:
            score_mismatch += 1
            geo.notes.append("rebuilt old score %.4f != stored %.4f"
                             % (rebuilt, old_score))

        conf = num(row, "term_confidence") or 0.0
        new_score = (WEIGHTS["confidence"] * conf
                     + WEIGHTS["hbond"] * hbond
                     + WEIGHTS["planarity"] * geo.planarity
                     + WEIGHTS["cat_polarity"] * geo.cat_polarity
                     + WEIGHTS["apolar_match"] * geo.apolar_match)

        pdb_rel = ""
        if not args.no_pdb:
            pdb_path = pdb_dir / ("%s.pdb" % design)
            pdb_path.parent.mkdir(parents=True, exist_ok=True)
            write_pdb(st, ext(pdb_path))
            # write_pdb goes through write_text, so re-emit with LF endings
            write_lf(pdb_path, Path(ext(pdb_path)).read_text(encoding="utf-8"))
            pdb_rel = str(Path(args.out_dir) / "pdb" / ("%s.pdb" % design)).replace(
                os.sep, "/")

        cif_rel = str(Path(str(cif)).relative_to(
            Path(ext(REPO)) if str(cif).startswith("\\\\?\\") else REPO)
        ).replace(os.sep, "/")

        results.append({
            "design_id": design,
            "variant": variant,
            "batch": batch,
            "run": run,
            "branch": row.get("branch", ""),
            "cycle": row.get("cycle", ""),
            "new_score": round(new_score, 4),
            "old_score": round(old_score, 4),
            "term_confidence": round(conf, 4),
            "term_hbond": round(hbond, 4),
            "term_planarity": round(geo.planarity, 4),
            "term_cat_polarity": round(geo.cat_polarity, 4),
            "term_apolar_match": round(geo.apolar_match, 4),
            "ring_rms_planarity": round(geo.ring_rms, 3),
            "offsets_A": "; ".join("%s=%.2f" % (k, v)
                                   for k, v in sorted(geo.offsets.items())),
            "group_coplanarity_deg": "; ".join(
                "%s=%.0f" % (k, v) for k, v in sorted(geo.coplanarity.items())),
            "f_polar_contact": round(geo.f_cat_polar, 3),
            "f_polar_surround": round(geo.f_shell_polar, 3),
            "n_polar_contact": geo.n_polar_near,
            "n_apolar_contact": geo.n_apolar_near,
            "n_polar_expected": geo.n_polar_expected,
            "n_polar_excess": geo.n_polar_excess,
            "f_apolar_edge": round(geo.f_apolar_edge, 3),
            "n_edge_contacts": geo.n_edge_contacts,
            "catalytic_residues": "; ".join(
                "%s=%s%s" % (b.bond_id, b.design_resname, b.design_resnum)
                for b in bonds if b.design_resnum is not None),
            "site_residues": geo.site_residues,
            "edge_residues": geo.edge_residues,
            "notes": "; ".join(geo.notes),
            "pdb_path": pdb_rel,
            "cif_path": cif_rel,
        })

    if not results:
        print("nothing could be scored")
        return 1

    results.sort(key=lambda r: -r["new_score"])
    old_order = sorted(results, key=lambda r: -r["old_score"])
    old_rank = {r["design_id"]: i + 1 for i, r in enumerate(old_order)}
    for i, r in enumerate(results):
        r["rank"] = i + 1
        r["old_rank"] = old_rank[r["design_id"]]
        r["rank_change"] = old_rank[r["design_id"]] - (i + 1)

    cols = ["rank", "old_rank", "rank_change", "design_id", "variant", "batch",
            "run", "branch", "cycle", "new_score", "old_score",
            "term_confidence", "term_hbond", "term_planarity",
            "term_cat_polarity", "term_apolar_match",
            "ring_rms_planarity", "offsets_A", "group_coplanarity_deg",
            "f_polar_contact", "f_polar_surround", "n_polar_contact",
            "n_apolar_contact", "n_polar_expected", "n_polar_excess",
            "f_apolar_edge", "n_edge_contacts", "catalytic_residues",
            "site_residues", "edge_residues", "notes", "pdb_path", "cif_path"]

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "rerank_geometry.csv"
    with open(ext(csv_path), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in cols})

    reference = reference_calibration()

    meta = {
        "weights": WEIGHTS,
        "min_score": args.min_score,
        "designs": len(results),
        "reference": reference,
        "constants": {
            "out_of_plane_fail_A": OUT_OF_PLANE_FAIL,
            "catalytic_plane_weight": CATALYTIC_PLANE_WEIGHT,
            "contact_zone_A": CONTACT_ZONE,
            "surround_out_A": SURROUND_OUT,
            "excess_polar_fail": EXCESS_POLAR_FAIL,
            "apolar_shell_A": APOLAR_SHELL,
            "apolar_packed_atoms": APOLAR_PACKED,
        },
        "ligand": {
            "ring_atoms": list(RING_ATOMS),
            "polar_atoms": list(LIGAND_POLAR),
            "apolar_edge": list(LIGAND_APOLAR_EDGE),
        },
    }
    write_lf(out_dir / "rerank_geometry.json", json.dumps(meta, indent=2) + "\n")

    print("%-4s %-8s %-8s %-4s %-7s %-7s %-7s %-7s %s" % (
        "rank", "new", "old", "var", "plane", "polar", "apolar", "move",
        "design"))
    for r in results:
        print("%-4d %-8.4f %-8.4f %-4s %-7.3f %-7.3f %-7.3f %+-7d %s" % (
            r["rank"], r["new_score"], r["old_score"], r["variant"],
            r["term_planarity"], r["term_cat_polarity"],
            r["term_apolar_match"], r["rank_change"],
            r["design_id"].split(".", 1)[1]))
    if reference:
        print("calibration against the real enzyme (%s):" % reference["structure"])
        print("   %-14s %-8s %-8s %-8s %s"
              % ("term", "wild-type", "designs", "best", "beating wt"))
        for key, col in (("planarity", "term_planarity"),
                         ("cat_polarity", "term_cat_polarity"),
                         ("apolar_match", "term_apolar_match")):
            vals = [r[col] for r in results]
            beat = sum(1 for v in vals if v > reference[key])
            print("   %-14s %-9.3f %-8.3f %-8.3f %d/%d"
                  % (key, reference[key], sum(vals) / len(vals), max(vals),
                     beat, len(vals)))
        print("   partners: %s" % "  ".join(reference["partners"]))
        print("   NB %s" % reference["caveat"])
        print()
    if score_mismatch:
        print("WARNING: %d design(s) where the rebuilt old score did not match "
              "the stored one -- see the notes column" % score_mismatch)
    flagged = [r for r in results if r["notes"]]
    if flagged:
        print("%d design(s) carry notes:" % len(flagged))
        for r in flagged:
            print("   %s: %s" % (r["design_id"].split(".", 1)[1], r["notes"]))
    print()
    print("csv : %s" % csv_path)
    print("json: %s" % (out_dir / "rerank_geometry.json"))
    if not args.no_pdb:
        print("pdbs: %s (%d files)" % (pdb_dir, len(results)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
