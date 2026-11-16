"""Shared machinery for generation backends.

Every generator, whatever the model, writes the same things into ``--out``:

    designs.fasta              every design, ``>design_id method=... parent=...``
    generate.<tool>.csv        one row per design, with provenance
    generate.<tool>.meta.json  the stage contract

so the fold and scoring stages do not care which generator produced a design,
and the leaderboard join is trivial.
"""
from __future__ import annotations

import logging
from pathlib import Path

from ..ids import Provenance, design_id
from ..io.fasta import SeqRecord, check_canonical, write_fasta

log = logging.getLogger("pipeline.generate")


class ModelUnavailable(RuntimeError):
    """Model weights or the package implementing a generator are missing.

    Carries the exact fetch/install command.
    """


class DesignCollector:
    """Collects designs, assigns deterministic ids, drops duplicates, writes outputs.

    Deduplication is by ``design_id``, so the same sequence produced twice by
    the same method and config collides into one row instead of inflating our
    design count — which is a number we will have to defend on Sunday.
    """

    def __init__(self, method: str, cfg_hash: str, parent_id: str | None = None,
                 parent_sequence: str | None = None, max_designs: int | None = None):
        self.method = method
        self.cfg_hash = cfg_hash
        self.parent_id = parent_id
        self.parent_sequence = parent_sequence
        self.max_designs = max_designs
        self.rows: list[dict] = []
        self.records: list[SeqRecord] = []
        self._seen: set[str] = set()
        self.n_duplicates = 0
        self.n_rejected = 0

    def full(self) -> bool:
        return self.max_designs is not None and len(self.rows) >= self.max_designs

    def add(self, sequence: str, mutations: str = "", notes: str = "",
            method: str | None = None, **extra) -> str | None:
        """Add one design. Returns its id, or None if rejected or duplicate."""
        if self.full():
            return None
        sequence = "".join(sequence.split()).upper()
        bad = check_canonical(sequence)
        if bad:
            log.warning("rejecting a design with non-canonical residues at %s", bad[:5])
            self.n_rejected += 1
            return None
        method = method or self.method
        did = design_id(sequence, method, self.cfg_hash)
        if did in self._seen:
            self.n_duplicates += 1
            return None
        self._seen.add(did)
        if not mutations and self.parent_sequence and len(sequence) == len(self.parent_sequence):
            from ..ids import mutation_string
            mutations = mutation_string(self.parent_sequence, sequence)
        row = Provenance(design_id=did, method=method, parent_id=self.parent_id,
                         parent_sequence_id=self.parent_id, mutations=mutations,
                         notes=notes).as_row()
        row["sequence_length"] = len(sequence)
        row.update(extra)
        self.rows.append(row)
        self.records.append(SeqRecord(did, sequence, {
            "method": method, "parent": self.parent_id or "none",
            "mutations": mutations or "none"}))
        return did

    def write(self, out_dir: Path) -> Path:
        path = Path(out_dir) / "designs.fasta"
        write_fasta(self.records, path)
        log.info("%s: %d designs (%d duplicates collapsed, %d rejected) -> %s",
                 self.method, len(self.records), self.n_duplicates, self.n_rejected, path)
        return path
