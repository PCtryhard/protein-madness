"""design_id generation and provenance.

``design_id`` is a short deterministic hash of (sequence, method, config hash).
The same design produced twice by the same method and config collides rather
than duplicating, which is what makes `results/leaderboard.csv` a table keyed
by design rather than an append-only log of runs.

Deliberately NOT part of the hash: wall time, host, git SHA, output paths.
Those are provenance and live in the stage ``.meta.json``.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ID_LEN = 8


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(config: dict, extra: Any = None) -> str:
    """Stable 12-char hash of a config dict (plus optional extra keys)."""
    payload = _canonical({"config": config, "extra": extra})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def design_id(sequence: str, method: str, cfg_hash: str) -> str:
    """``d_<8 hex>``. Sequence is upper-cased and whitespace-stripped first."""
    seq = "".join(sequence.split()).upper()
    if not seq:
        raise ValueError("design_id: empty sequence")
    h = hashlib.sha256(f"{seq}|{method}|{cfg_hash}".encode("utf-8")).hexdigest()
    return f"d_{h[:ID_LEN]}"


def git_sha(repo: str | Path | None = None) -> str:
    """Current commit, with ``-dirty`` when the tree is modified. 'unknown' outside a repo."""
    repo = Path(repo) if repo else Path(__file__).resolve().parent.parent
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True,
                             text=True, timeout=10, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True,
                               text=True, timeout=10, check=True).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.SubprocessError, OSError):
        return "unknown"


@dataclass
class Provenance:
    """How one design came to exist. Written to the generating stage's CSV."""

    design_id: str
    method: str
    parent_id: str | None = None
    parent_sequence_id: str | None = None
    mutations: str = ""          # e.g. "A1C+D3E", parent -> this, 1-indexed
    notes: str = ""

    def as_row(self) -> dict:
        return {
            "design_id": self.design_id,
            "method": self.method,
            "parent_id": self.parent_id or "",
            "parent_sequence_id": self.parent_sequence_id or "",
            "mutations": self.mutations,
            "n_mutations": 0 if not self.mutations else len(self.mutations.split("+")),
            "notes": self.notes,
        }


def mutation_string(parent: str, child: str) -> str:
    """'A1C+D3E' for two equal-length sequences. Raises on length mismatch."""
    if len(parent) != len(child):
        raise ValueError(f"cannot diff sequences of length {len(parent)} and {len(child)}")
    return "+".join(f"{p}{i + 1}{c}" for i, (p, c) in enumerate(zip(parent, child)) if p != c)


def apply_mutations(parent: str, mutations: str) -> str:
    """Inverse of :func:`mutation_string`. Verifies the wild-type letter at each position."""
    seq = list(parent)
    for m in filter(None, mutations.split("+")):
        wt, pos, mut = m[0], int(m[1:-1]), m[-1]
        if not 1 <= pos <= len(seq):
            raise ValueError(f"mutation {m}: position outside 1..{len(seq)}")
        if seq[pos - 1] != wt:
            raise ValueError(f"mutation {m}: parent has {seq[pos - 1]} at position {pos}, not {wt}")
        seq[pos - 1] = mut
    return "".join(seq)
