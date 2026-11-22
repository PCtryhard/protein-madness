#!/usr/bin/env python3
"""Build the Kemp Pocket Inspector, a browsable 3D view of the rescored designs.

Reads only committed data -- `data/rescored/` -- and writes a page you open in
a browser. Nothing is refolded, nothing is re-scored, no GPU, no network, and
no dependencies beyond the Python standard library.

    python scripts/pocket_inspector/build_inspector.py
    open inspector/pocket_inspector.html

WHAT IT SHOWS

One entry per rescored design. The viewport draws the Boltz-predicted complex
with the ligand in gold, the catalytic residues in red, the rest of the pocket
in cyan, and every catalytic bond as a dashed line labelled with its distance
in angstroms. Those distances are computed here from the deposited coordinates
-- the atom pair comes from the re-rank's own `catalytic_residues` mapping, so
`lig_n3_to_asp229_od2=ASP232` is measured on residue 232 of that design, not on
a guess. The list can be ranked by either scoring pass and filtered by how many
adjustments a design's ancestry actually contains.

TWO OUTPUT SHAPES

  --single-file   (default) one self-contained .html with every structure
                  embedded. Large -- tens of MB -- but it opens by double
                  clicking, with no server and no network.

  --split         index.html plus designs.json and one bundle per variant.
                  Much smaller to move around, but browsers refuse fetch()
                  from file://, so it needs a server:

                      cd inspector && python3 -m http.server 8000

                  then open http://localhost:8000/

The page itself pulls 3Dmol.js from a CDN, so the browser needs network access
the first time even for the single file. Everything else is local.

OUTPUT IS NOT COMMITTED. Write it outside the repo, or add the output
directory to .gitignore. The structures are already in `data/rescored/cif/`;
a built page is a copy of them and does not belong in git history.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
RESCORED = os.path.join(REPO, "data", "rescored")

# Sidechain atoms a catalytic bond can be measured to. Backbone N and O are
# deliberately absent: the target definitions name sidechain partners only.
SIDECHAIN_POLAR = {
    "OD1", "OD2", "OE1", "OE2", "ND1", "ND2", "NE2",
    "OG", "OG1", "OH", "NZ", "NE", "NH1", "NH2",
}


def read_cif_atoms(path):
    """Return {(seq_id, comp_id): {atom_name: (x, y, z)}} from an mmCIF file.

    Boltz writes one `_atom_site` loop with a fixed column order, so a
    whitespace split is enough; this is not a general mmCIF parser.
    """
    header, start = [], None
    with open(path) as fh:
        lines = fh.read().splitlines()
    for i, line in enumerate(lines):
        if line.startswith("_atom_site."):
            header.append(line.strip().split(".", 1)[1])
            start = i
        elif header:
            start = i
            break
    if not header or start is None:
        raise ValueError(f"no _atom_site loop in {path}")
    col = {name: k for k, name in enumerate(header)}
    out = {}
    for line in lines[start:]:
        if line.startswith("#") or not line.strip():
            break
        parts = line.split()
        if len(parts) < len(header):
            break
        key = (parts[col["label_seq_id"]], parts[col["label_comp_id"]])
        out.setdefault(key, {})[parts[col["label_atom_id"]]] = (
            float(parts[col["Cartn_x"]]),
            float(parts[col["Cartn_y"]]),
            float(parts[col["Cartn_z"]]),
        )
    return out


def bond_atoms(bond_id):
    """`lig_n3_to_asp229_od2` -> ('N3', 'OD2');  `his291_ne2_to_lig_n1` -> ('N1', 'NE2')."""
    left, right = bond_id.split("_to_")
    if left.startswith("lig_"):
        return left[4:].upper(), right.split("_")[-1].upper()
    return right.split("_")[-1].upper(), left.split("_")[-1].upper()


def measure_bonds(atoms, catalytic_residues, ligand="6NT"):
    """Measure each named catalytic contact against the ligand, in place."""
    ligands = [v for k, v in atoms.items() if k[1] == ligand]
    if not ligands:
        return []
    lig = ligands[0]
    by_number = {k[0]: v for k, v in atoms.items() if k[1] != ligand}
    bonds = []
    for pair in (catalytic_residues or "").split(";"):
        pair = pair.strip()
        if "=" not in pair:
            continue
        bond_id, residue = (x.strip() for x in pair.split("=", 1))
        match = re.match(r"^([A-Z]{3})(\d+)$", residue)
        if not match:
            continue
        lig_atom, prot_atom = bond_atoms(bond_id)
        sidechain = by_number.get(match.group(2))
        if not sidechain or lig_atom not in lig or prot_atom not in sidechain:
            continue
        a, b = sidechain[prot_atom], lig[lig_atom]
        bonds.append({
            "lab": f"{residue} {prot_atom} · {lig_atom}",
            "d": round(math.dist(a, b), 2),
            "a": [round(x, 3) for x in a],
            "b": [round(x, 3) for x in b],
        })
    return bonds


def load_rows(rescored):
    manifest_path = os.path.join(rescored, "manifest.csv")
    rerank_path = os.path.join(rescored, "rerank_geometry.csv")
    for path in (manifest_path, rerank_path):
        if not os.path.exists(path):
            sys.exit(f"missing {path}\nRun this from a checkout that has data/rescored/.")
    with open(manifest_path) as fh:
        manifest = {r["design_id"]: r for r in csv.DictReader(fh)}
    with open(rerank_path) as fh:
        rerank = list(csv.DictReader(fh))
    orphans = [r["design_id"] for r in rerank if r["design_id"] not in manifest]
    if orphans:
        sys.exit(f"{len(orphans)} re-rank rows have no manifest entry, first: {orphans[0]}")
    return manifest, rerank


def build_index(rescored, verbose=True):
    """One record per design: scores, lineage, provenance and measured bonds."""
    manifest, rerank = load_rows(rescored)
    cif_dir = os.path.join(rescored, "cif")
    designs, unresolved = [], 0
    for row in rerank:
        man = manifest[row["design_id"]]
        stem = man["filename"][:-4] if man["filename"].endswith(".cif") else man["filename"]
        path = os.path.join(cif_dir, man["filename"])
        if not os.path.exists(path):
            sys.exit(f"missing structure {path}")
        bonds = measure_bonds(read_cif_atoms(path), row["catalytic_residues"])
        if not bonds:
            unresolved += 1
        designs.append({
            "f": stem,
            "v": row["variant"],
            "lin": man["lineage"],
            "adj": int(man["adjustments"]),
            "cyc": int(row["cycle"]),
            "br": row["branch"],
            "batch": row["batch"],
            "run": row["run"],
            "id": row["design_id"],
            "rk": int(row["rank"]),
            "ork": int(row["old_rank"]),
            "chg": int(row["rank_change"]),
            "new": row["new_score"],
            "old": row["old_score"],
            "t": [row["term_confidence"], row["term_hbond"], row["term_planarity"],
                  row["term_cat_polarity"], row["term_apolar_match"]],
            "off": row["offsets_A"],
            "cop": row["group_coplanarity_deg"],
            "cat": row["catalytic_residues"],
            "site": row["site_residues"],
            "npc": row["n_polar_contact"],
            "npe": row["n_polar_expected"],
            "bonds": bonds,
        })
    designs.sort(key=lambda d: d["rk"])
    if verbose:
        total = sum(len(d["bonds"]) for d in designs)
        print(f"  {len(designs)} designs, {total} catalytic bonds measured", end="")
        print(f", {unresolved} with none resolved" if unresolved else "")
    return designs


def read_structures(rescored, designs):
    cif_dir = os.path.join(rescored, "cif")
    bundles = {}
    for d in designs:
        with open(os.path.join(cif_dir, d["f"] + ".cif")) as fh:
            bundles.setdefault(d["v"], {})[d["f"]] = fh.read()
    return bundles


def write_single_file(template, designs, bundles, out_dir):
    payload = json.dumps({"designs": designs, "b": bundles}, separators=(",", ":"))
    # </script> inside embedded text would end the block early; nothing in a
    # .cif should contain it, but escape defensively rather than hope.
    payload = payload.replace("</", "<\\/")
    html = template.replace("const INLINE = null; /*__DATA__*/",
                            "const INLINE = " + payload + ";", 1)
    path = os.path.join(out_dir, "pocket_inspector.html")
    with open(path, "w") as fh:
        fh.write(html)
    return [path]


def write_split(template, designs, bundles, out_dir):
    written = []
    index = os.path.join(out_dir, "index.html")
    with open(index, "w") as fh:
        fh.write(template)
    written.append(index)
    path = os.path.join(out_dir, "designs.json")
    with open(path, "w") as fh:
        json.dump(designs, fh, separators=(",", ":"))
    written.append(path)
    bundle_dir = os.path.join(out_dir, "b")
    os.makedirs(bundle_dir, exist_ok=True)
    for variant, group in sorted(bundles.items()):
        path = os.path.join(bundle_dir, f"v{variant}.json")
        with open(path, "w") as fh:
            json.dump(group, fh, separators=(",", ":"))
        written.append(path)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rescored", default=RESCORED,
                    help="directory holding manifest.csv, rerank_geometry.csv and cif/")
    ap.add_argument("--out", default=os.path.join(os.getcwd(), "inspector"),
                    help="where to write the built page (default ./inspector)")
    shape = ap.add_mutually_exclusive_group()
    shape.add_argument("--single-file", dest="single", action="store_true", default=True,
                       help="one self-contained .html (default)")
    shape.add_argument("--split", dest="single", action="store_false",
                       help="index.html plus JSON; needs a local server")
    args = ap.parse_args()

    template_path = os.path.join(HERE, "template.html")
    if not os.path.exists(template_path):
        sys.exit(f"missing {template_path}")
    with open(template_path) as fh:
        template = fh.read()

    print(f"reading {args.rescored}")
    designs = build_index(args.rescored)
    bundles = read_structures(args.rescored, designs)

    os.makedirs(args.out, exist_ok=True)
    writer = write_single_file if args.single else write_split
    written = writer(template, designs, bundles, args.out)

    total = sum(os.path.getsize(p) for p in written)
    print(f"wrote {len(written)} file(s), {total / 1e6:.1f} MB, to {args.out}")
    if args.single:
        print(f"open {written[0]}")
    else:
        print(f"cd {args.out} && python3 -m http.server 8000   then open http://localhost:8000/")


if __name__ == "__main__":
    main()
