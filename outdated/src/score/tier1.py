"""Tier-1 scoring: cheap, CPU, no model weights.

    python -m src.score.tier1 --in designs.fasta --out results/score_tier1 \
        --config configs/pipeline.yaml --target configs/target.yaml

Writes ``<out>/score.tier1.csv`` (one row per design) and ``score.tier1.meta.json``.
Every metric here must work on a laptop with no GPU and no network.
"""
from __future__ import annotations

import logging
import sys

from .. import cli
from ..io.fasta import read_fasta
from ..io.target import load_reference
from . import novelty as nov
from .sequence import all_sequence_metrics

log = logging.getLogger("pipeline.tier1")
TOOL = "score.tier1"
VERSION = "0.1.0"


def main(argv: list[str] | None = None) -> int:
    ap = cli.base_parser(__doc__)
    ap.add_argument("--parent", default=None,
                    help="parent/wild-type FASTA (default: the target's reference_sequence)")
    args = ap.parse_args(argv)
    cli.setup_logging(args.verbose)

    cfg = cli.load_pipeline_config(args.config)
    tcfg = cli.tool_config(cfg, TOOL)

    inputs = cli.load_inputs(args.inp, args.limit)
    if inputs.kind != "fasta":
        raise SystemExit(f"{TOOL} scores sequences: --in must be a FASTA file, got {inputs.path}")

    catalytic: list[dict] = []
    parent_seq: str | None = None
    if args.target:
        ref = load_reference(cli.require_target(args))
        catalytic = ref.catalytic
        parent_seq = ref.sequence
    if args.parent:
        parent_seq = read_fasta(args.parent)[0].seq
    if parent_seq is None:
        log.warning("no --target and no --parent: identity/novelty-vs-parent will be blank")

    ncfg = dict(tcfg.get("novelty") or {})
    with cli.stage(TOOL, args, tcfg, VERSION,
                   hash_extra={"n_inputs": len(inputs), "parent": parent_seq or ""}) as st:
        if st is None:
            return 0

        rows = []
        for rec in inputs.sequences:
            row = {"design_id": rec.id, "sequence": rec.seq}
            row.update(all_sequence_metrics(
                rec.seq, parent=parent_seq, catalytic=catalytic,
                ph=float(tcfg.get("ph", 7.0)),
                window=int(tcfg.get("low_complexity_window", 10)),
            ))
            row["low_complexity_flag"] = int(
                row["low_complexity_fraction"] > float(tcfg.get("low_complexity_max_fraction", 0.6))
                or row["max_homopolymer_run"] > int(tcfg.get("max_homopolymer_run", 4))
            )
            if parent_seq is not None and len(rec.seq) == len(parent_seq):
                row["novelty_vs_parent"] = nov.novelty_vs_parent(rec.seq, parent_seq)
            rows.append(row)

        # Database-backed novelty: all designs in one MMseqs2 call, or none.
        if ncfg.get("enabled"):
            seqs = {r["design_id"]: r["sequence"] for r in rows}
            try:
                hits = nov.search(
                    seqs, ncfg.get("reference_db"),
                    sensitivity=float(ncfg.get("sensitivity", 5.7)),
                    max_evalue=float(ncfg.get("max_evalue", 1e-3)),
                    threads=int((cfg.get("compute") or {}).get("threads", 4)),
                )
            except nov.NoveltyUnavailable as e:
                raise SystemExit(
                    f"score.tier1: novelty is enabled but unavailable: {e}\n"
                    "Fix the config/DB, or set score.tier1.novelty.enabled: false to skip it. "
                    "Not writing a fabricated novelty column."
                )
            for r in rows:
                r.update(hits[r["design_id"]])
        else:
            log.info("novelty: database search disabled in config (novelty_vs_parent still reported)")

        df = st.write_csv(rows)
        log.info("tier1: %d designs, %d metrics", len(df), len(df.columns) - 1)
        if catalytic:
            dead = int((df["catalytic_integrity"] < 1.0).sum())
            log.info("catalytic integrity: %d/%d designs keep every catalytic residue",
                     len(df) - dead, len(df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
