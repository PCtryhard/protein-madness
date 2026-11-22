"""Live output: a CSV, a FASTA and a progress file, written as designs finish.

The run directory has to be useful while the run is still going, for two
reasons. A multi-hour search that only produces readable output at the end
tells you nothing about whether it is working, and anything you want on your
own machine has to be copyable mid-run.

What this module writes, all under `runs/<name>/`:

    designs.csv       one row per design, appended the moment it is scored
    sequences.fasta   the same designs as sequences, appended at the same time
    progress.txt      a few lines of counts, rewritten after every design

`designs.db` was already incremental -- every design is a committed INSERT, and
the connection runs in WAL mode so a reader can read it while a worker writes.
But sqlite is not something you can `tail -f`, and `report` renders markdown
rather than a table you can load. Hence the CSV.

CONCURRENCY

Two GPU workers finish designs at the same time, so every append is taken
under a lock. Each row is written and then flushed AND fsynced: without the
fsync a reader on another machine, or an rsync, can see a truncated row. A
design takes minutes, so the cost of an fsync per row is irrelevant.

The header is fixed when the file is created, from the target's bond ids, so
the per-bond columns are stable for the whole run.
"""

from __future__ import annotations

import csv
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from .config import Config
from .store import DesignRow

# Boltz metrics worth a column of their own. Logged for every design without
# being scored on, so the weighting can be revisited offline.
EXTRA_COLUMNS = (
    "iptm",
    "ptm",
    "ligand_iptm",
    "complex_plddt",
    "complex_ipde",
    "confidence_score",
)

BASE_COLUMNS = (
    "finished_at",
    "design_id",
    "scaffold_id",
    "cycle",
    "branch",
    "mutation",
    "status",
    "gated",
    "gated_bonds",
    "score",
    "term_confidence",
    "term_distance",
    "term_catalytic",
    "term_non_catalytic",
    "term_angle",
    "parent_id",
    "seq_len",
    "seq_hash",
    "error",
    "structure_path",
    "backbone_path",
    "sequence",
)


class LiveResults:
    """Append-only run output. One instance per run, shared by both workers."""

    def __init__(self, root: Path, cfg: Config) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.root / "designs.csv"
        self.fasta_path = self.root / "sequences.fasta"
        self.progress_path = self.root / "progress.txt"

        self.bond_ids = [bond.id for bond in cfg.target.hbonds]
        # Which bonds are gatable, and the thresholds, so the CSV can name the
        # bond that failed instead of only saying gated=1.
        self._required = {b.id for b in cfg.target.hbonds if b.required}
        self._distance = cfg.pipeline.scoring.distance
        self.columns = list(BASE_COLUMNS)
        for bond_id in self.bond_ids:
            self.columns += [f"{bond_id}__distance", f"{bond_id}__angle"]
        self.columns += list(EXTRA_COLUMNS)

        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}
        self._best: tuple[float, str] | None = None
        self._run_name = self.root.name
        self._target_name = cfg.target.name
        self._started = datetime.now(timezone.utc)

        self._write_header_if_new()
        self._write_progress()

    # ------------------------------------------------------------------

    def _write_header_if_new(self) -> None:
        """Write the header once.

        If the file exists with a different header -- a resumed or re-used run
        directory whose target changed -- the old file is kept under a dated
        name rather than appended to, because mixing two column sets in one CSV
        silently corrupts every row.
        """
        if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
            with self.csv_path.open(newline="", encoding="utf-8") as fh:
                existing = next(csv.reader(fh), [])
            if existing == self.columns:
                return
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            moved = self.csv_path.with_name(f"designs.{stamp}.csv")
            self.csv_path.rename(moved)
            print(
                f"note: {self.csv_path.name} had a different set of columns and "
                f"was moved to {moved.name}"
            )
        with self.csv_path.open("w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(self.columns)
            fh.flush()
            os.fsync(fh.fileno())

    # ------------------------------------------------------------------

    def append(self, row: DesignRow) -> None:
        """Record one finished design. Safe to call from either worker.

        Takes the database row rather than a score object so that scored,
        gated, cached and errored designs all travel the same path -- an
        errored design has no score to pass but still deserves a line.
        """
        values = self._row_values(row)
        with self._lock:
            with self.csv_path.open("a", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow([values.get(c, "") for c in self.columns])
                fh.flush()
                os.fsync(fh.fileno())

            if row.sequence and row.status != "errored":
                self._append_fasta(row)

            self._counts[row.status] = self._counts.get(row.status, 0) + 1
            if row.status == "scored" and row.score is not None:
                if self._best is None or row.score > self._best[0]:
                    self._best = (row.score, row.design_id)
            self._write_progress()

    def _row_values(self, row: DesignRow) -> dict[str, object]:
        values: dict[str, object] = {
            "finished_at": (row.created_at or datetime.now(timezone.utc).isoformat()),
            "design_id": row.design_id,
            "scaffold_id": row.scaffold_id,
            "cycle": row.cycle,
            "branch": row.branch,
            "mutation": row.mutation or "",
            "status": row.status,
            "gated": int(bool(row.gated)),
            "score": _fmt(row.score),
            "term_confidence": _fmt(row.term_confidence),
            "term_distance": _fmt(row.term_distance),
            "term_catalytic": _fmt(row.term_catalytic),
            "term_non_catalytic": _fmt(row.term_non_catalytic),
            "term_angle": _fmt(row.term_angle),
            "parent_id": row.parent_id or "",
            "seq_len": len(row.sequence) if row.sequence else "",
            "seq_hash": (row.seq_hash or "")[:12],
            # One line per row matters more than a full traceback here; the
            # whole message is in designs.db and in the run log.
            "error": (row.error or "").replace("\n", " ")[:300],
            "structure_path": row.structure_path or "",
            "backbone_path": row.backbone_path or "",
            "sequence": row.sequence or "",
        }

        gated_bonds: list[str] = []
        if row.hbond_raw:
            try:
                for bond in json.loads(row.hbond_raw):
                    bond_id = bond.get("bond_id")
                    if bond_id is None:
                        continue
                    distance = bond.get("distance")
                    values[f"{bond_id}__distance"] = _fmt(distance)
                    values[f"{bond_id}__angle"] = _fmt(bond.get("angle"), digits=1)
                    # Name the offender rather than only flagging gated=1: with
                    # several bonds per design, "which one" is the whole
                    # question when you are watching a run go wrong.
                    if (
                        bond_id in self._required
                        and isinstance(distance, (int, float))
                        and (distance > self._distance.fail
                             or distance < self._distance.clash)
                    ):
                        gated_bonds.append(bond_id)
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass

        if row.extra_metrics:
            try:
                extra = json.loads(row.extra_metrics)
                for key in EXTRA_COLUMNS:
                    if key in extra:
                        values[key] = _fmt(extra[key])
            except (json.JSONDecodeError, TypeError):
                pass

        values["gated_bonds"] = " ".join(gated_bonds)
        return values

    def _append_fasta(self, row: DesignRow) -> None:
        header = (
            f">{row.design_id} scaffold={row.scaffold_id} cycle={row.cycle} "
            f"branch={row.branch} score={_fmt(row.score)} status={row.status}"
        )
        if row.mutation:
            header += f" mutation={row.mutation}"
        with self.fasta_path.open("a", encoding="utf-8") as fh:
            fh.write(header + "\n")
            for start in range(0, len(row.sequence), 60):
                fh.write(row.sequence[start : start + 60] + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def _write_progress(self) -> None:
        """A tiny file that answers 'is it alive and how is it doing'.

        Rewritten rather than appended, and written via a temporary file and a
        rename so a reader never catches it half-written.
        """
        total = sum(self._counts.values())
        elapsed = datetime.now(timezone.utc) - self._started
        hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)

        lines = [
            f"run      : {self._run_name}",
            f"target   : {self._target_name}",
            f"started  : {self._started.isoformat(timespec='seconds')}",
            f"updated  : {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
            f"elapsed  : {hours:d}h {minutes:02d}m {seconds:02d}s",
            "",
            f"designs  : {total}",
        ]
        for status in ("scored", "gated", "errored"):
            if status in self._counts:
                lines.append(f"  {status:<8}: {self._counts[status]}")
        if self._best:
            lines.append("")
            lines.append(f"best     : {self._best[0]:.4f}  {self._best[1]}")
        # Only once there is enough wall time for the number to mean
        # something. A rate extrapolated from the first few seconds is a
        # fiction, and a fiction here would be used to plan the rest of the day.
        seconds = elapsed.total_seconds()
        if total and seconds >= 120:
            per = seconds / total
            lines.append("")
            lines.append(
                f"rate     : {per / 60:.1f} min/design "
                f"({3600 / per:.0f} designs/hour)"
            )
        elif total:
            lines.append("")
            lines.append("rate     : measuring (needs ~2 min of wall time)")
        lines.append("")

        temp = self.progress_path.with_suffix(".txt.tmp")
        temp.write_text("\n".join(lines), encoding="utf-8")
        os.replace(temp, self.progress_path)

    # ------------------------------------------------------------------

    def summary(self) -> str:
        with self._lock:
            parts = [f"{n} {status}" for status, n in sorted(self._counts.items())]
        return ", ".join(parts) if parts else "nothing recorded"


def _fmt(value: object, digits: int = 4) -> object:
    """Round floats for readability; leave everything else alone.

    A CSV of 17-significant-digit floats is unreadable, and the full precision
    is in designs.db for anyone who needs it.
    """
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return "" if value is None else value
