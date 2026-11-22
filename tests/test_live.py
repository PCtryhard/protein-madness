"""Live output, and the threading model that broke without it being tested.

The bug these tests exist for: `sqlite3.connect()` defaults to
`check_same_thread=True`, but the GPU worker pool calls `store.insert()` from
its worker threads. The real pipeline would have raised ProgrammingError on the
very first design, on the cluster, after the first RFdiffusion call. Nothing in
the suite exercised two threads writing, so nothing caught it.
"""

from __future__ import annotations

import csv
import json
import threading

import pytest

from src.config import load
from src.live import LiveResults
from src.store import DesignRow, DesignStore
from tests.paths import REAL_TARGET, ROOT


@pytest.fixture
def live_cfg(pipeline_path):
    return load(pipeline_path, ROOT / REAL_TARGET)


def mkrow(index: int, cfg, status: str = "scored", **kwargs) -> DesignRow:
    bonds = [b.id for b in cfg.target.hbonds]
    required = {b.id for b in cfg.target.hbonds if b.required}
    hbond_raw = json.dumps([
        {
            "bond_id": b, "distance": 2.9, "angle": 120.0, "angle_ideal": 120.0,
            "required": b in required, "angle_kind": "acceptor_side", "atoms": {},
        }
        for b in bonds
    ])
    defaults = dict(
        scaffold_id="t.01", cycle=0, branch="seed", sequence="MKVL" * 10,
        seq_hash=f"hash{index:04d}", status=status, score=0.5,
        term_confidence=0.6, term_distance=0.7, term_angle=0.8,
        hbond_raw=hbond_raw,
        extra_metrics=json.dumps({"iptm": 0.75, "ptm": 0.8, "complex_ipde": 2.1}),
        provenance="{}",
    )
    defaults.update(kwargs)
    return DesignRow(design_id=f"d{index:04d}", **defaults)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------
# the threading model
# --------------------------------------------------------------------------


def test_the_store_is_usable_from_another_thread(tmp_path, live_cfg):
    """The regression. sqlite connections are thread-bound by default, and the
    GPU pool inserts from worker threads."""
    store = DesignStore(tmp_path / "designs.db")
    errors: list[BaseException] = []

    def worker():
        try:
            store.insert(mkrow(1, live_cfg))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    store.close()
    assert not errors, f"insert from a worker thread raised {errors[0]!r}"


def test_two_threads_can_write_the_store_and_the_csv(tmp_path, live_cfg):
    """Two GPU workers finishing designs at once. Every design must land in
    both files exactly once, with no torn rows."""
    live = LiveResults(tmp_path, live_cfg)
    store = DesignStore(tmp_path / "designs.db")
    errors: list[BaseException] = []

    def worker(indices):
        for i in indices:
            try:
                row = mkrow(i, live_cfg)
                store.insert(row)
                live.append(row)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

    a = threading.Thread(target=worker, args=(range(0, 40, 2),))
    b = threading.Thread(target=worker, args=(range(1, 40, 2),))
    a.start(); b.start(); a.join(); b.join()

    assert not errors, f"concurrent writes raised {errors[0]!r}"
    assert store.counts_by_status() == {"scored": 40}
    rows = read_csv(live.csv_path)
    assert len(rows) == 40
    assert len({r["design_id"] for r in rows}) == 40
    # every row must have the full column set, i.e. no interleaved writes
    for row in rows:
        assert len(row) == len(live.columns)
        assert row["score"] == "0.5000"
    store.close()


# --------------------------------------------------------------------------
# the CSV is written as designs finish, not at the end
# --------------------------------------------------------------------------


def test_header_exists_before_any_design(tmp_path, live_cfg):
    live = LiveResults(tmp_path, live_cfg)
    rows = read_csv(live.csv_path)
    assert rows == []
    with live.csv_path.open(newline="", encoding="utf-8") as fh:
        assert next(csv.reader(fh)) == live.columns


def test_a_row_is_readable_immediately_after_one_design(tmp_path, live_cfg):
    """The whole point: a reader on another machine sees design N while design
    N+1 is still folding."""
    live = LiveResults(tmp_path, live_cfg)
    live.append(mkrow(1, live_cfg))
    assert len(read_csv(live.csv_path)) == 1
    live.append(mkrow(2, live_cfg))
    assert len(read_csv(live.csv_path)) == 2


def test_per_bond_columns_come_from_the_target(tmp_path, live_cfg):
    live = LiveResults(tmp_path, live_cfg)
    for bond in live_cfg.target.hbonds:
        assert f"{bond.id}__distance" in live.columns
        assert f"{bond.id}__angle" in live.columns


def test_per_bond_values_land_in_their_columns(tmp_path, live_cfg):
    live = LiveResults(tmp_path, live_cfg)
    live.append(mkrow(1, live_cfg))
    row = read_csv(live.csv_path)[0]
    for bond in live_cfg.target.hbonds:
        assert row[f"{bond.id}__distance"] == "2.9000"
        assert row[f"{bond.id}__angle"] == "120.0"


def test_extra_metrics_land_in_their_columns(tmp_path, live_cfg):
    live = LiveResults(tmp_path, live_cfg)
    live.append(mkrow(1, live_cfg))
    row = read_csv(live.csv_path)[0]
    assert row["iptm"] == "0.7500"
    assert row["complex_ipde"] == "2.1000"


# --------------------------------------------------------------------------
# every outcome gets a row
# --------------------------------------------------------------------------


def test_an_errored_design_still_gets_a_row(tmp_path, live_cfg):
    """Watching a run go wrong is most of the value of watching it."""
    live = LiveResults(tmp_path, live_cfg)
    live.append(mkrow(
        1, live_cfg, status="errored", score=None, hbond_raw=None,
        extra_metrics=None, error="BoltzTooLargeError: 512 tokens over 400",
    ))
    row = read_csv(live.csv_path)[0]
    assert row["status"] == "errored"
    assert row["score"] == ""
    assert "BoltzTooLargeError" in row["error"]


def test_a_multiline_error_stays_on_one_csv_row(tmp_path, live_cfg):
    live = LiveResults(tmp_path, live_cfg)
    live.append(mkrow(
        1, live_cfg, status="errored", score=None, hbond_raw=None,
        extra_metrics=None, error="line one\nline two\nline three",
    ))
    rows = read_csv(live.csv_path)
    assert len(rows) == 1
    assert "\n" not in rows[0]["error"]


def test_a_gated_design_names_the_bond_that_failed(tmp_path, live_cfg):
    """With several bonds per design, "which one" is the whole question."""
    bonds = [b.id for b in live_cfg.target.hbonds]
    required = {b.id for b in live_cfg.target.hbonds if b.required}
    raw = json.dumps([
        {
            "bond_id": b,
            # the first required bond is driven far too long
            "distance": 6.0 if b == bonds[0] else 2.9,
            "angle": 120.0, "angle_ideal": 120.0,
            "required": b in required, "angle_kind": "acceptor_side", "atoms": {},
        }
        for b in bonds
    ])
    live = LiveResults(tmp_path, live_cfg)
    live.append(mkrow(1, live_cfg, status="gated", gated=True, score=0.0,
                      hbond_raw=raw))
    row = read_csv(live.csv_path)[0]
    assert row["gated"] == "1"
    assert bonds[0] in row["gated_bonds"]


def test_a_clash_is_named_too(tmp_path, live_cfg):
    bonds = [b.id for b in live_cfg.target.hbonds]
    required = {b.id for b in live_cfg.target.hbonds if b.required}
    raw = json.dumps([
        {
            # Below the 0.5 A clash gate: 1.5 A is no longer an overlap.
            "bond_id": b, "distance": 0.3 if b == bonds[0] else 2.9,
            "angle": 120.0, "angle_ideal": 120.0,
            "required": b in required, "angle_kind": "acceptor_side", "atoms": {},
        }
        for b in bonds
    ])
    live = LiveResults(tmp_path, live_cfg)
    live.append(mkrow(1, live_cfg, status="gated", gated=True, score=0.0,
                      hbond_raw=raw))
    assert bonds[0] in read_csv(live.csv_path)[0]["gated_bonds"]


def test_a_non_required_bond_is_never_named_as_gating(tmp_path, live_cfg):
    """`required: false` contributes to the score but must never gate, however
    bad the contact is."""
    bonds = list(live_cfg.target.hbonds)
    relaxed = [bonds[0].model_copy(update={"required": False})] + bonds[1:]
    cfg = live_cfg.model_copy(update={
        "target": live_cfg.target.model_copy(update={"hbonds": relaxed})
    })
    optional = [b.id for b in cfg.target.hbonds if not b.required]
    raw = json.dumps([
        {
            "bond_id": b.id, "distance": 9.0 if b.id in optional else 2.9,
            "angle": 120.0, "angle_ideal": 120.0, "required": b.required,
            "angle_kind": "acceptor_side", "atoms": {},
        }
        for b in cfg.target.hbonds
    ])
    live = LiveResults(tmp_path, cfg)
    live.append(mkrow(1, cfg, hbond_raw=raw))
    assert read_csv(live.csv_path)[0]["gated_bonds"] == ""

def test_sequences_are_appended_as_fasta(tmp_path, live_cfg):
    live = LiveResults(tmp_path, live_cfg)
    live.append(mkrow(1, live_cfg))
    live.append(mkrow(2, live_cfg))
    text = live.fasta_path.read_text(encoding="utf-8")
    assert text.count(">") == 2
    assert "d0001" in text
    # wrapped, and the sequence survives the wrapping
    body = "".join(
        line for line in text.splitlines() if not line.startswith(">")
    )
    assert body == ("MKVL" * 10) * 2


def test_errored_designs_are_not_written_to_the_fasta(tmp_path, live_cfg):
    """A sequence that was never successfully folded is not a result."""
    live = LiveResults(tmp_path, live_cfg)
    live.append(mkrow(1, live_cfg, status="errored", score=None,
                      hbond_raw=None, extra_metrics=None, error="boom"))
    assert not live.fasta_path.exists() or ">" not in live.fasta_path.read_text(
        encoding="utf-8"
    )


def test_progress_file_tracks_counts_and_best(tmp_path, live_cfg):
    live = LiveResults(tmp_path, live_cfg)
    live.append(mkrow(1, live_cfg, score=0.30))
    live.append(mkrow(2, live_cfg, score=0.90))
    live.append(mkrow(3, live_cfg, status="gated", gated=True, score=0.0))
    text = live.progress_path.read_text(encoding="utf-8")
    assert "designs  : 3" in text
    assert "scored  : 2" in text
    assert "gated   : 1" in text
    assert "0.9000" in text and "d0002" in text


def test_progress_file_exists_before_any_design(tmp_path, live_cfg):
    live = LiveResults(tmp_path, live_cfg)
    assert live.progress_path.exists()
    assert "designs  : 0" in live.progress_path.read_text(encoding="utf-8")


def test_summary_counts_every_status(tmp_path, live_cfg):
    live = LiveResults(tmp_path, live_cfg)
    live.append(mkrow(1, live_cfg))
    live.append(mkrow(2, live_cfg, status="gated", gated=True, score=0.0))
    assert "1 gated" in live.summary()
    assert "1 scored" in live.summary()


# --------------------------------------------------------------------------
# a reused run directory
# --------------------------------------------------------------------------


def test_a_csv_with_different_columns_is_moved_aside(tmp_path, live_cfg, capsys):
    """Appending a different column set into one CSV silently corrupts every
    row, so the old file is kept under a dated name instead."""
    live = LiveResults(tmp_path, live_cfg)
    live.append(mkrow(1, live_cfg))

    # A target with a different bond set, built here rather than requiring a
    # second target file to exist -- which variant is available changes with
    # the data.
    first = live_cfg.target.hbonds[0]
    other_target = live_cfg.target.model_copy(update={
        "hbonds": [first.model_copy(update={"id": "a_different_bond_id"})]
    })
    other = live_cfg.model_copy(update={"target": other_target})

    again = LiveResults(tmp_path, other)
    assert "different set of columns" in capsys.readouterr().out
    assert read_csv(again.csv_path) == []
    assert list(tmp_path.glob("designs.*.csv")), "the old CSV should be kept"

def test_reopening_with_the_same_columns_appends(tmp_path, live_cfg):
    live = LiveResults(tmp_path, live_cfg)
    live.append(mkrow(1, live_cfg))
    again = LiveResults(tmp_path, live_cfg)
    again.append(mkrow(2, live_cfg))
    assert len(read_csv(again.csv_path)) == 2
