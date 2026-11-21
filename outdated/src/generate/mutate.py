"""Cheap CPU baseline: systematic mutational scan around the active site.

    python -m src.generate.mutate --in data/toy/ref.fasta --out results/mutate \
        --config configs/pipeline.yaml --target configs/target.yaml

No GPU, no model weights, no network. A systematic scan of the design region is
a legitimate submission on its own and it is the fallback if every GPU path
jams. It is also what gives the smoke test designs to score.

Enumeration is deterministic and ordered (position, then substitution), so a
truncated run is a prefix of the full one rather than a random subset.
"""
from __future__ import annotations

import itertools
import logging
import sys

from .. import cli
from ..ids import Provenance, design_id, mutation_string
from ..io.fasta import SeqRecord, read_fasta, write_fasta
from ..io.target import load_reference
from ..score.sequence import CHEMICAL_GROUPS, cysteine_count

log = logging.getLogger("pipeline.mutate")
TOOL = "generate.mutate"
VERSION = "0.1.0"

ALL_AA = "ACDEFGHIKLMNPQRSTVWY"


def substitutions_for(wt: str, policy, exclude_cysteine: bool = True) -> list[str]:
    """Which residues to try at a position whose wild-type is ``wt``.

    ``policy``: ``"all"`` (all 19 others), ``"chemical_similar"`` (within the
    wild-type's chemical group, plus alanine as the standard knock-out probe),
    or an explicit string/list of one-letter codes.
    """
    if isinstance(policy, (list, tuple)):
        cand = set("".join(policy).upper())
    elif policy == "all":
        cand = set(ALL_AA)
    elif policy == "chemical_similar":
        group = next((g for g in CHEMICAL_GROUPS.values() if wt in g), "")
        cand = set(group) | {"A"}
    else:
        cand = set(str(policy).upper())
    cand &= set(ALL_AA)
    cand.discard(wt)
    if exclude_cysteine:
        cand.discard("C")
    return sorted(cand)


def single_mutants(parent: str, positions: list[int], policy, exclude_cysteine: bool = True):
    """Yield (sequence, mutation_string) for every allowed point mutation, ordered."""
    for pos in positions:
        wt = parent[pos - 1]
        for mut in substitutions_for(wt, policy, exclude_cysteine):
            seq = parent[: pos - 1] + mut + parent[pos:]
            yield seq, f"{wt}{pos}{mut}"


def double_mutants(parent: str, positions: list[int], policy, exclude_cysteine: bool = True,
                   max_pairs: int | None = None):
    """Yield double mutants over position pairs, ordered, optionally capped.

    The cap applies to position PAIRS, taken in order, not to a random sample:
    a capped run is a prefix of the full enumeration.
    """
    pairs = itertools.combinations(positions, 2)
    if max_pairs is not None:
        pairs = itertools.islice(pairs, max_pairs)
    for p1, p2 in pairs:
        wt1, wt2 = parent[p1 - 1], parent[p2 - 1]
        for m1 in substitutions_for(wt1, policy, exclude_cysteine):
            for m2 in substitutions_for(wt2, policy, exclude_cysteine):
                seq = list(parent)
                seq[p1 - 1], seq[p2 - 1] = m1, m2
                yield "".join(seq), f"{wt1}{p1}{m1}+{wt2}{p2}{m2}"


def main(argv: list[str] | None = None) -> int:
    ap = cli.base_parser(__doc__)
    ap.add_argument("--positions", default=None,
                    help="comma-separated positions to scan (reference numbering); "
                         "default: the target's design_region")
    ap.add_argument("--mode", default=None, choices=["single", "double", "both"])
    ap.add_argument("--max-designs", type=int, default=None)
    args = ap.parse_args(argv)
    cli.setup_logging(args.verbose)

    cfg = cli.load_pipeline_config(args.config)
    tcfg = cli.tool_config(cfg, TOOL)
    mode = args.mode or tcfg.get("mode", "single")
    max_designs = args.max_designs or int(tcfg.get("max_designs", 2000))
    policy = tcfg.get("substitutions", "chemical_similar")
    excl_cys = bool(tcfg.get("exclude_cysteine", True))

    target = cli.require_target(args)
    ref = load_reference(target)

    inputs = cli.load_inputs(args.inp)
    if inputs.kind != "fasta":
        raise SystemExit(f"{TOOL} mutates sequences: --in must be a FASTA file, got {inputs.path}")
    parent_rec = inputs.sequences[0]
    parent = parent_rec.seq
    if len(inputs.sequences) > 1:
        log.warning("--in has %d sequences; scanning the first (%s) only",
                    len(inputs.sequences), parent_rec.id)
    if parent != ref.sequence:
        log.warning("parent FASTA differs from the target's reference sequence "
                    "(%d vs %d aa): positions are interpreted in the PARENT",
                    len(parent), len(ref.sequence))

    if args.positions:
        positions = [ref.numbering.to_new(int(p), chain=target.chain)
                     for p in args.positions.split(",") if p.strip()]
    else:
        positions = ref.design_positions
    positions = sorted(p for p in positions if 1 <= p <= len(parent))
    if not positions:
        raise SystemExit("no designable positions: check design_region in the target config")
    log.info("scanning %d positions, mode=%s, substitutions=%s", len(positions), mode, policy)

    with cli.stage(TOOL, args, tcfg, VERSION, hash_extra={
        "parent": parent, "positions": positions, "mode": mode, "max_designs": max_designs,
    }) as st:
        if st is None:
            return 0

        rows: list[dict] = []
        records: list[SeqRecord] = []
        seen: set[str] = set()
        cfg_h = st.cfg_hash
        parent_cys = cysteine_count(parent)

        def add(seq: str, muts: str, method: str) -> bool:
            if len(rows) >= max_designs:
                return False
            did = design_id(seq, method, cfg_h)
            if did in seen:
                return True
            seen.add(did)
            prov = Provenance(design_id=did, method=method, parent_id=parent_rec.id,
                              parent_sequence_id=parent_rec.id, mutations=muts)
            row = prov.as_row()
            row["new_cysteines"] = max(0, cysteine_count(seq) - parent_cys)
            rows.append(row)
            records.append(SeqRecord(did, seq, {"method": method, "parent": parent_rec.id,
                                                "mutations": muts or "none"}))
            return True

        if tcfg.get("include_wildtype", True):
            add(parent, "", "wildtype")

        if mode in ("single", "both"):
            for seq, muts in single_mutants(parent, positions, policy, excl_cys):
                if not add(seq, muts, TOOL):
                    break
        if mode in ("double", "both") and len(rows) < max_designs:
            for seq, muts in double_mutants(parent, positions, policy, excl_cys,
                                            int(tcfg.get("double_max_pairs", 200))):
                if not add(seq, muts, TOOL):
                    break

        if len(rows) >= max_designs:
            log.warning("hit max_designs=%d: enumeration truncated (ordered, so it is a prefix)",
                        max_designs)

        fasta_path = st.out_dir / "designs.fasta"
        write_fasta(records, fasta_path)
        for row, rec in zip(rows, records):
            row["sequence_length"] = len(rec.seq)
        st.write_csv(rows)
        log.info("wrote %d designs -> %s", len(records), fasta_path)
        # sanity: every design is the parent with exactly its listed mutations
        for rec, row in zip(records, rows):
            if row["method"] != "wildtype":
                assert mutation_string(parent, rec.seq) == row["mutations"], rec.id
    return 0


if __name__ == "__main__":
    sys.exit(main())
