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
