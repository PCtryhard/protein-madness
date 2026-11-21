"""One sqlite table carries the whole search: lineage, scores and the cache.

Two things this file is responsible for beyond storage:

* The `seq_hash` index is the Boltz-2 cache. Before submitting, look up the
  hash of sequence plus ligand plus Boltz settings; on a hit, reuse the stored
  result. A mutation search regenerates identical sequences constantly, and on
  two cards this cache is worth more than any other optimisation in the build.

* Ancestry is a recursive query on `parent_id`, which is why lineage needs no
  separate structure. The tree is never pruned, only the frontier is.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import BoltzCfg, Ligand
from .search.tree import Candidate

SCHEMA = """
CREATE TABLE IF NOT EXISTS designs (
  design_id        TEXT PRIMARY KEY,
  parent_id        TEXT,
  scaffold_id      TEXT NOT NULL,
  cycle            INTEGER NOT NULL,
  branch           TEXT NOT NULL,      -- seed | wiggle | mutate
  mutation         TEXT,               -- e.g. A72V, null outside the mutate branch
  sequence         TEXT NOT NULL,
  seq_hash         TEXT NOT NULL,
  score            REAL,
  term_confidence  REAL,
  term_distance    REAL,
  term_angle       REAL,
  gated            INTEGER DEFAULT 0,
  status           TEXT NOT NULL,      -- scored | gated | errored
  error            TEXT,
  hbond_raw        TEXT,               -- json: per-bond distance and angle
  extra_metrics    TEXT,               -- json: iptm, ptm, plddt, motif_pae, ...
  provenance       TEXT NOT NULL,      -- json: the three index maps
  structure_path   TEXT,
  backbone_path    TEXT,               -- RFdiffusion output, for the wiggle branch
  created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_seq_hash ON designs(seq_hash);
CREATE INDEX IF NOT EXISTS idx_scaffold_cycle ON designs(scaffold_id, cycle);

CREATE TABLE IF NOT EXISTS cache_stats (
  key   TEXT PRIMARY KEY,
  value INTEGER NOT NULL
);
"""

ANCESTRY_SQL = """
WITH RECURSIVE line(id) AS (
  SELECT ?  UNION ALL
  SELECT d.parent_id FROM designs d JOIN line ON d.design_id = line.id
  WHERE d.parent_id IS NOT NULL
)
SELECT * FROM designs WHERE design_id IN (SELECT id FROM line);
"""


def sequence_hash(sequence: str, ligand: Ligand, boltz: BoltzCfg) -> str:
    """The cache key: sequence plus ligand plus the Boltz settings that matter.

    Settings that change the prediction are in the key; settings that only
    change how it is computed are not. `no_kernels` and `precision` are
    performance and hardware-compatibility switches, but they can change the
    numbers slightly, so they are included: a cache hit must mean "this exact
    computation was already done".
    """
    payload = {
        "sequence": sequence,
        "ligand": {"id": ligand.id, "ccd": ligand.ccd, "smiles": ligand.smiles},
        "boltz": {
            "recycling_steps": boltz.recycling_steps,
            "diffusion_samples": boltz.diffusion_samples,
            "sampling_steps": boltz.sampling_steps,
            "use_potentials": boltz.use_potentials,
            "precision": boltz.precision,
            "no_kernels": boltz.no_kernels,
        },
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class DesignRow:
    """One row of `designs`, as python."""

    design_id: str
    scaffold_id: str
    cycle: int
    branch: str
    sequence: str
    seq_hash: str
    status: str
    provenance: str
    parent_id: str | None = None
    mutation: str | None = None
    score: float | None = None
    term_confidence: float | None = None
    term_distance: float | None = None
    term_angle: float | None = None
    gated: bool = False
    error: str | None = None
    hbond_raw: str | None = None
    extra_metrics: str | None = None
    structure_path: str | None = None
    backbone_path: str | None = None
    created_at: str | None = None

    def to_candidate(self) -> Candidate:
        if self.score is None:
            raise ValueError(
                f"design {self.design_id} has status {self.status!r} and no score; "
                "it cannot enter the search as a candidate"
            )
        return Candidate(
            design_id=self.design_id,
            scaffold_id=self.scaffold_id,
            cycle=self.cycle,
            branch=self.branch,
            sequence=self.sequence,
            score=self.score,
            gated=self.gated,
            parent_id=self.parent_id,
            backbone_path=self.backbone_path,
            structure_path=self.structure_path,
            mutation=self.mutation,
        )


class DesignStore:
    """The run's sqlite file. Safe to reopen: `resume` depends on it."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        # Two worker processes read and write this file; WAL lets a reader and a
        # writer proceed at once instead of one blocking the other.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> DesignStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- writing ----

    def insert(self, row: DesignRow) -> None:
        """Insert one design. The design_id is the primary key, so a repeat is
        an error rather than a silent overwrite of a scored design."""
        created = row.created_at or datetime.now(timezone.utc).isoformat()
        try:
            self._conn.execute(
                """
                INSERT INTO designs (
                  design_id, parent_id, scaffold_id, cycle, branch, mutation,
                  sequence, seq_hash, score, term_confidence, term_distance,
                  term_angle, gated, status, error, hbond_raw, extra_metrics,
                  provenance, structure_path, backbone_path, created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    row.design_id, row.parent_id, row.scaffold_id, row.cycle,
                    row.branch, row.mutation, row.sequence, row.seq_hash,
                    row.score, row.term_confidence, row.term_distance,
                    row.term_angle, int(row.gated), row.status, row.error,
                    row.hbond_raw, row.extra_metrics, row.provenance,
                    row.structure_path, row.backbone_path, created,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                f"design_id {row.design_id!r} is already in {self.path}"
            ) from exc
        self._conn.commit()

    # ---- reading ----

    def get(self, design_id: str) -> DesignRow | None:
        cur = self._conn.execute(
            "SELECT * FROM designs WHERE design_id = ?", (design_id,)
        )
        got = cur.fetchone()
        return _row_to_design(got) if got else None

    def by_scaffold(self, scaffold_id: str) -> list[DesignRow]:
        cur = self._conn.execute(
            "SELECT * FROM designs WHERE scaffold_id = ? ORDER BY cycle, design_id",
            (scaffold_id,),
        )
        return [_row_to_design(r) for r in cur.fetchall()]

    def scaffolds(self) -> list[str]:
        cur = self._conn.execute(
            "SELECT DISTINCT scaffold_id FROM designs ORDER BY scaffold_id"
        )
        return [r[0] for r in cur.fetchall()]

    def leaderboard(self, scaffold_id: str, limit: int = 20) -> list[DesignRow]:
        cur = self._conn.execute(
            """
            SELECT * FROM designs
            WHERE scaffold_id = ? AND status = 'scored'
            ORDER BY score DESC, design_id ASC
            LIMIT ?
            """,
            (scaffold_id, limit),
        )
        return [_row_to_design(r) for r in cur.fetchall()]

    def ancestry(self, design_id: str) -> list[DesignRow]:
        """The design and every ancestor, via the recursive query."""
        cur = self._conn.execute(ANCESTRY_SQL, (design_id,))
        rows = [_row_to_design(r) for r in cur.fetchall()]
        return sorted(rows, key=lambda r: r.cycle)

    def max_cycle(self, scaffold_id: str) -> int:
        cur = self._conn.execute(
            "SELECT MAX(cycle) FROM designs WHERE scaffold_id = ?", (scaffold_id,)
        )
        got = cur.fetchone()[0]
        return -1 if got is None else int(got)

    def frontier_at(self, scaffold_id: str, cycle: int, keep: int) -> list[DesignRow]:
        """Best `keep` non-gated designs at or before `cycle`. Used by `resume`."""
        cur = self._conn.execute(
            """
            SELECT * FROM designs
            WHERE scaffold_id = ? AND cycle <= ? AND status = 'scored' AND gated = 0
            ORDER BY score DESC, design_id ASC
            LIMIT ?
            """,
            (scaffold_id, cycle, keep),
        )
        return [_row_to_design(r) for r in cur.fetchall()]

    # ---- the Boltz-2 cache ----

    def cache_lookup(self, seq_hash: str) -> DesignRow | None:
        """A previously scored design with this exact hash, if any.

        Only `scored` rows are a hit. An errored row must be retried: the error
        may have been a transient OOM, and reusing it would make one failure
        permanent for every design that shares the sequence.
        """
        cur = self._conn.execute(
            """
            SELECT * FROM designs
            WHERE seq_hash = ? AND status IN ('scored', 'gated')
            ORDER BY created_at ASC LIMIT 1
            """,
            (seq_hash,),
        )
        got = cur.fetchone()
        if got is None:
            self.bump("cache_miss")
            return None
        self.bump("cache_hit")
        return _row_to_design(got)

    def bump(self, key: str, by: int = 1) -> None:
        self._conn.execute(
            """
            INSERT INTO cache_stats (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = value + ?
            """,
            (key, by, by),
        )
        self._conn.commit()

    def stats(self) -> dict[str, int]:
        cur = self._conn.execute("SELECT key, value FROM cache_stats")
        return {r[0]: int(r[1]) for r in cur.fetchall()}

    def counts_by_status(self) -> dict[str, int]:
        cur = self._conn.execute(
            "SELECT status, COUNT(*) FROM designs GROUP BY status"
        )
        return {r[0]: int(r[1]) for r in cur.fetchall()}


def _row_to_design(row: sqlite3.Row) -> DesignRow:
    data: dict[str, Any] = dict(row)
    data["gated"] = bool(data.get("gated", 0))
    return DesignRow(**data)


def new_design_id(
    scaffold_id: str, cycle: int, branch: str, discriminator: str
) -> str:
    """A readable, collision-free id.

    Readable matters: these ids end up in filenames, log lines and the
    leaderboard, and `5fqj_seed_5a.c03.mutate.a1f2c8` tells you where a design
    came from without a database lookup.
    """
    digest = hashlib.sha256(discriminator.encode("utf-8")).hexdigest()[:6]
    return f"{scaffold_id}.c{cycle:02d}.{branch}.{digest}"


def candidates_from_rows(rows: Iterable[DesignRow]) -> list[Candidate]:
    return [r.to_candidate() for r in rows if r.score is not None]
