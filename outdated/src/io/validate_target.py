"""Validate a target config against its reference structure.

    python -m src.io.validate_target configs/target.yaml [--strict]

Run this FIRST on Saturday. Exit 0 = safe to run the pipeline; exit 1 = an
index, residue type or atom selection does not match the structure and every
downstream number would be wrong. ``--strict`` also fails on warnings.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .fasta import read_fasta
from .structure import chain_breaks, normalise, read_structure, sequence_of, get_atom
from .target import (GEOMETRY_KINDS, deviation, load_target, measure, parse_atom_spec,
                     resolve_design_region)


def validate(path: str | Path) -> tuple[list[str], list[str], list[str]]:
    """Return (errors, warnings, info). Collects every problem instead of stopping at the first."""
    errors: list[str] = []
    warnings: list[str] = []
    info: list[str] = []

    try:
        target = load_target(path)
    except Exception as e:  # config unreadable: nothing else can be checked
        return [f"cannot load config: {e}"], warnings, info

    if not target.reference_structure.exists():
        return [f"reference_structure not found: {target.reference_structure}"], warnings, info

    try:
        atoms, numbering = normalise(read_structure(target.reference_structure), chain=target.chain)
    except Exception as e:
        return [f"cannot read/normalise reference structure: {e}"], warnings, info

    seq = sequence_of(atoms)
    info.append(f"target '{target.name}': chain {numbering.entries[0]['chain']}, {len(seq)} residues")
    first, last = numbering.to_orig(1), numbering.to_orig(len(numbering))
    info.append(f"reference numbering {first} .. {last} -> normalised 1 .. {len(numbering)}")
    shifted = [e for e in numbering.entries if e["resi"] != e["new_resi"] or e["ins"]]
    if shifted:
        warnings.append(
            f"{len(shifted)} residues change number on normalisation (offset/insertion codes/gaps). "
            "Config indices are interpreted in REFERENCE numbering and mapped automatically."
        )
    if "X" in seq:
        pos = [i + 1 for i, c in enumerate(seq) if c == "X"]
        errors.append(f"non-standard residues at normalised positions {pos}; decide how to treat them")
    breaks = chain_breaks(atoms)
    if breaks:
        warnings.append(
            f"chain breaks after normalised positions {breaks}: the structure-derived sequence is "
            "missing residues. Designs/folds built from it will lack them."
        )

    if target.reference_sequence is not None:
        if not target.reference_sequence.exists():
            errors.append(f"reference_sequence not found: {target.reference_sequence}")
        else:
            recs = read_fasta(target.reference_sequence)
            fasta_seq = recs[0].seq
            if fasta_seq == seq:
                info.append("reference_sequence matches the structure sequence exactly")
            elif seq in fasta_seq:
                off = fasta_seq.index(seq)
                warnings.append(
                    f"structure sequence is a fragment of reference_sequence (offset {off}, "
                    f"{len(fasta_seq) - len(seq)} residues unresolved). The pipeline uses the "
                    "STRUCTURE sequence as parent; positions are structure positions."
                )
            else:
                errors.append(
                    f"reference_sequence ({len(fasta_seq)} aa) does not match the structure "
                    f"sequence ({len(seq)} aa) and does not contain it. Wrong chain, wrong file, "
                    "or internal missing residues."
                )

    if not target.catalytic_residues:
        warnings.append("no catalytic_residues given: active-site metrics will be skipped")
    cat_pos: list[int] = []
    for c in target.catalytic_residues:
        try:
            pos = numbering.to_new(int(c["resi"]), chain=target.chain)
        except KeyError as e:
            errors.append(f"catalytic residue {c.get('resi')}: {e.args[0]}")
            continue
        found, expected = numbering.resn(pos), str(c.get("resn", "")).upper()
        if found != expected:
            near = [
                f"{numbering.to_orig(p)}" for p in range(max(1, pos - 3), min(len(numbering), pos + 3) + 1)
                if numbering.resn(p) == expected
            ]
            hint = f" ({expected} found nearby at {near}: off-by-N?)" if near else ""
            errors.append(f"catalytic residue {c['resi']}: config says {expected}, structure has {found}{hint}")
        else:
            cat_pos.append(pos)
            info.append(f"catalytic {expected}{c['resi']} ok -> normalised position {pos}")

    for g in target.catalytic_geometry:
        try:
            specs = [parse_atom_spec(s) for s in g["atoms"]]
            if len(specs) not in GEOMETRY_KINDS:
                raise ValueError("need 2, 3 or 4 atoms")
            kind = GEOMETRY_KINDS[len(specs)]
            coords = [get_atom(atoms, numbering.to_new(r, chain=target.chain), a) for r, a in specs]
        except (KeyError, ValueError) as e:
            errors.append(f"geometry {g.get('atoms')}: {e.args[0]}")
            continue
        value = measure(coords)
        ideal = g.get(f"ideal_{kind}", g.get("ideal"))
        unit = "A" if kind == "distance" else "deg"
        if ideal is None:
            info.append(f"geometry {g['atoms']}: {kind} {value:.2f} {unit} in reference (used as ideal)")
        else:
            dev = deviation(kind, value, float(ideal))
            msg = f"geometry {g['atoms']}: {kind} {value:.2f} {unit} in reference, ideal {float(ideal):.2f}"
            tol = g.get("tol")
            if tol is not None and dev > float(tol):
                warnings.append(msg + f" -> the REFERENCE itself is outside tol {tol} (dev {dev:.2f}). "
                                "Wrong atoms, wrong ideal, or intended?")
            else:
                info.append(msg + " ok")
        if g.get("tol") is None:
            warnings.append(f"geometry {g['atoms']}: no tol given, pass/fail cannot be computed")

    if not errors:
        try:
            design = resolve_design_region(target, atoms, numbering, cat_pos)
            info.append(f"design region ({target.design_region.get('mode', 'all')}): {len(design)} positions")
            if not design:
                errors.append("design region is empty: nothing can be designed")
        except (KeyError, ValueError) as e:
            errors.append(f"design_region: {e.args[0]}")

    if not target.objective.strip() or "plain-English" in target.objective:
        warnings.append("objective is empty/placeholder: fill it from the challenge brief")
    if not target.success_metrics:
        warnings.append("success_metrics is empty: fill it from the challenge brief")
    return errors, warnings, info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target")
    ap.add_argument("--strict", action="store_true", help="warnings are fatal too")
    args = ap.parse_args(argv)
    errors, warnings, info = validate(args.target)
    for m in info:
        print(f"[ ok ] {m}")
    for m in warnings:
        print(f"[WARN] {m}")
    for m in errors:
        print(f"[FAIL] {m}")
    failed = bool(errors) or (args.strict and bool(warnings))
    print(f"\n{'INVALID' if failed else 'VALID'}: {len(errors)} error(s), {len(warnings)} warning(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
