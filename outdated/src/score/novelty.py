"""Sequence novelty: how far a design is from anything natural.

Metric: 1 - (max sequence identity to the nearest hit in a reference database),
via MMseqs2 ``easy-search``. Reported per design, plus the identity itself and
the hit it came from, because "novel" is only meaningful next to what it is
novel *with respect to*.

Requires the ``mmseqs`` binary and a reference database. Neither is in this
repo (see MANIFEST.md). When either is absent this module raises rather than
returning a plausible number: a fabricated novelty score is the single most
embarrassing thing we could put in front of a judge.
"""
from __future__ import annotations

import csv
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from ..io.fasta import SeqRecord, write_fasta

log = logging.getLogger("pipeline.novelty")

# MMseqs2 easy-search tab-separated output we ask for.
SEARCH_COLUMNS = "query,target,fident,alnlen,evalue,bits,qcov,tcov"


class NoveltyUnavailable(RuntimeError):
    """MMseqs2 or the reference DB is missing. Caller decides: skip or fail."""


def mmseqs_binary() -> str:
    """Path to the mmseqs binary, or raise NoveltyUnavailable."""
    exe = shutil.which("mmseqs")
    if exe is None:
        raise NoveltyUnavailable(
            "mmseqs not on PATH. Install MMseqs2 (conda: `mamba install -c bioconda mmseqs2`, "
            "or the static binary on Euler) and record the location in MANIFEST.md. "
            "No native Windows build: use WSL or Euler."
        )
    return exe


def check_available(reference_db: str | Path | None) -> str:
    """Raise NoveltyUnavailable unless both the binary and the DB are present."""
    exe = mmseqs_binary()
    if reference_db is None:
        raise NoveltyUnavailable(
            "no reference_db configured (score.tier1.novelty.reference_db). "
            "Build one with scripts/fetch_weights.sh --novelty-db and record it in MANIFEST.md."
        )
    db = Path(reference_db)
    # An MMseqs DB is a set of files sharing a prefix; a FASTA also works for easy-search.
    if not db.exists() and not any(db.parent.glob(db.name + "*")):
        raise NoveltyUnavailable(f"reference_db not found: {db}")
    return exe


def search(
    sequences: dict[str, str],
    reference_db: str | Path,
    sensitivity: float = 5.7,
    max_evalue: float = 1e-3,
    threads: int = 4,
    workdir: str | Path | None = None,
) -> dict[str, dict]:
    """Run ``mmseqs easy-search`` and return the best hit per design.

    Returns ``{design_id: {seq_novelty, max_identity, nearest_hit, nearest_evalue,
    nearest_coverage}}``. Designs with no hit above the E-value threshold get
    novelty 1.0 and an empty hit, which is the correct reading: nothing in the
    reference DB is detectably similar.
    """
    exe = check_available(reference_db)
    tmp = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="mmseqs_novelty_"))
    tmp.mkdir(parents=True, exist_ok=True)
    query = tmp / "query.fasta"
    out_tsv = tmp / "hits.tsv"
    write_fasta([SeqRecord(k, v) for k, v in sequences.items()], query)

    cmd = [
        exe, "easy-search", str(query), str(reference_db), str(out_tsv), str(tmp / "work"),
        "-s", str(sensitivity), "-e", str(max_evalue), "--threads", str(threads),
        "--format-output", SEARCH_COLUMNS, "--max-seqs", "300", "-v", "1",
    ]
    log.info("mmseqs easy-search on %d sequences against %s", len(sequences), reference_db)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"mmseqs easy-search failed (exit {proc.returncode}):\n{proc.stderr[-2000:]}"
        )

    best: dict[str, dict] = {}
    if out_tsv.exists():
        with open(out_tsv, encoding="utf-8") as fh:
            for row in csv.reader(fh, delimiter="\t"):
                if len(row) < 8:
                    continue
                qid, tid, fident, _alnlen, evalue, _bits, qcov, _tcov = row[:8]
                ident = float(fident)
                if qid not in best or ident > best[qid]["max_identity"]:
                    best[qid] = {
                        "max_identity": ident,
                        "nearest_hit": tid,
                        "nearest_evalue": float(evalue),
                        "nearest_coverage": float(qcov),
                    }

    out = {}
    for did in sequences:
        hit = best.get(did)
        if hit is None:
            out[did] = {"seq_novelty": 1.0, "max_identity": 0.0, "nearest_hit": "",
                        "nearest_evalue": float("nan"), "nearest_coverage": 0.0}
        else:
            out[did] = {"seq_novelty": 1.0 - hit["max_identity"], **hit}
    return out


def novelty_vs_parent(seq: str, parent: str) -> float:
    """Fallback novelty with no database: 1 - identity to the parent only.

    Honest but weak: it says how far the design is from the wild-type, NOT how
    far from anything natural. Reported under its own column name so it is never
    confused with the database-backed number.
    """
    from .sequence import sequence_identity

    return 1.0 - sequence_identity(seq, parent)
