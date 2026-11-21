"""FASTA reading and writing.

Header convention used throughout the repo::

    >design_id key=value key=value ...

The first whitespace-delimited token is the id; remaining ``key=value`` tokens
are provenance fields (method, parent, mutations, ...). Values never contain
whitespace.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class SeqRecord:
    id: str
    seq: str
    fields: dict[str, str] = field(default_factory=dict)

    def header(self) -> str:
        extra = " ".join(f"{k}={v}" for k, v in self.fields.items())
        return f"{self.id} {extra}".strip()


def read_fasta(path: str | Path) -> list[SeqRecord]:
    records: list[SeqRecord] = []
    cur: SeqRecord | None = None
    chunks: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur is not None:
                    cur.seq = "".join(chunks)
                    records.append(cur)
                tokens = line[1:].split()
                if not tokens:
                    raise ValueError(f"{path}: FASTA header with no id")
                fields = dict(t.split("=", 1) for t in tokens[1:] if "=" in t)
                cur = SeqRecord(id=tokens[0], seq="", fields=fields)
                chunks = []
            else:
                if cur is None:
                    raise ValueError(f"{path}: sequence data before first header")
                chunks.append(line.upper())
    if cur is not None:
        cur.seq = "".join(chunks)
        records.append(cur)
    ids = [r.id for r in records]
    if len(set(ids)) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"{path}: duplicate FASTA ids: {dup[:5]}")
    for r in records:
        if not r.seq:
            raise ValueError(f"{path}: empty sequence for {r.id}")
    return records


def write_fasta(records: list[SeqRecord], path: str | Path, width: int = 80) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(f">{r.header()}\n")
            for i in range(0, len(r.seq), width):
                fh.write(r.seq[i : i + width] + "\n")


def check_canonical(seq: str) -> list[int]:
    """Return 1-indexed positions holding anything but the 20 canonical residues."""
    return [i + 1 for i, c in enumerate(seq) if c not in VALID_AA]
