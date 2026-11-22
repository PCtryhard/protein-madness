"""CLI argument handling.

The regression here cost a submitted cluster job: `-v` was only accepted
before the subcommand, so `run --name x -v` -- the order anyone actually types
-- failed with an argparse error two seconds into a GPU allocation.
"""

from __future__ import annotations

import pytest

from src.cli import build_parser
from tests.paths import REAL_TARGET


def test_verbose_accepted_before_the_subcommand():
    args = build_parser().parse_args(["-v", "smoke", "--target", REAL_TARGET])
    assert args.verbose


def test_verbose_accepted_after_the_subcommand():
    """The order people type. Used to be `error: unrecognized arguments: -v`."""
    args = build_parser().parse_args(["smoke", "--target", REAL_TARGET, "-v"])
    assert args.verbose


def test_verbose_defaults_off():
    args = build_parser().parse_args(["smoke", "--target", REAL_TARGET])
    assert not args.verbose


@pytest.mark.parametrize(
    "argv",
    [
        ["smoke", "--target", REAL_TARGET],
        ["measure", "--target", REAL_TARGET, "--pdb", "x.pdb"],
        ["run", "--target", REAL_TARGET, "--name", "n"],
        ["report", "--run", "runs/x"],
        ["watch", "--run", "runs/x"],
        ["resume", "--run", "runs/x"],
    ],
)
def test_every_subcommand_parses_and_takes_verbose(argv):
    """Every subcommand, with -v appended, since that is where it gets typed."""
    parser = build_parser()
    assert parser.parse_args(argv).func is not None
    assert parser.parse_args(argv + ["-v"]).verbose


# --------------------------------------------------------------------------
# smoke actually RUNNING, not just parsing
# --------------------------------------------------------------------------


def test_smoke_runs_and_prints_the_live_scoring_settings(capsys):
    """Execute cmd_smoke, do not merely parse its arguments.

    This file only ever checked argument handling, so when the scoring config
    was restructured and `cmd_smoke` kept printing a field that no longer
    existed, every test passed and the failure surfaced as an
    `AttributeError` 45 seconds into a submitted job -- which is the exact
    class of regression this module was written to stop.

    Asserting on the printed values rather than just a clean exit, because a
    smoke check that prints stale settings is worse than one that crashes:
    it tells you the run is configured in a way it is not.
    """
    from src.cli import build_parser, cmd_smoke

    args = build_parser().parse_args(["smoke", "--target", REAL_TARGET])
    assert cmd_smoke(args) == 0

    out = capsys.readouterr().out
    assert "SMOKE TEST PASSED" in out
    # the three geometry weights, by their current names
    assert "catalytic" in out
    assert "non-catalytic" in out
    # both ends of the gate, so a one-sided report cannot creep back
    assert "4.5 A" in out
    assert "0.5 A" in out


def test_smoke_reports_both_sides_of_the_distance_term(capsys):
    """The term is asymmetric; a reader needs both spans to interpret a score."""
    from src.cli import build_parser, cmd_smoke

    args = build_parser().parse_args(["smoke", "--target", REAL_TARGET])
    cmd_smoke(args)
    out = capsys.readouterr().out

    line = next(l for l in out.splitlines() if l.startswith("distance term"))
    assert "2.9 A" in line      # full marks
    assert "0.5 A" in line      # zero below
    assert "4.5 A" in line      # zero above
