"""Where the test suite looks for real repo data.

One module so a data reshuffle is a one-line change rather than fifteen. The
`data/` layout has already moved once -- from 5A/10A/15A full-length seeds to
the 01..07 motif folders -- and the tests that hard-coded the old paths went
quietly to SKIPPED rather than failing, which is the worst outcome.

Kept out of conftest.py because pytest treats conftest specially; a plain
module is importable from test files without surprises.
"""

from __future__ import annotations

import pathlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
TARGETS_DIR = ROOT / "config" / "targets"

# The reference real target and its seed: the variant whose ASP229 contact is
# still genuine 5FQJ crystal geometry, 2.734 A at 106.6 deg.
#
# This was variant 01 until its pocket was replaced with a refold whose ligand
# sits 25 A from the motif. Variant 04 kept the crystal base contact, so the
# geometry assertions moved there. If 04 is ever refolded too, move this and
# re-measure rather than loosening the assertions.
REAL_TARGET = "config/targets/04_gln_asp_his.yaml"
REAL_SEED = (
    ROOT / "data" / "04" / "inputs" / "5fqj-6cf4761" / "04_gln_asp_his.pdb"
)

EXAMPLE_TARGET = "config/targets/example_target.yaml"

# A real Boltz-2 prediction kept in the repo, for the CIF reader.
BOLTZ_CIF = (
    ROOT / "data" / "No_mutation_run" / "boltz" / "seed-1" / "5A_01"
    / "5A_01_model_0.cif"
)
BOLTZ_CONFIDENCE = (
    ROOT / "data" / "No_mutation_run" / "boltz" / "seed-1" / "5A_01"
    / "confidence_5A_01_model_0.json"
)


def all_shipped_targets() -> list[str]:
    """Every target file in config/targets/, repo-relative.

    Globbed rather than listed so a new variant is covered by the
    loads-under-both-rules tests the moment it is generated.
    """
    return sorted(
        p.relative_to(ROOT).as_posix() for p in TARGETS_DIR.glob("*.yaml")
    )


def variant_targets() -> list[str]:
    """The numbered variants, without the schema example.

    Matched by "starts with two digits" rather than by a `0*` glob, so a tenth
    variant is included rather than silently skipped.
    """
    import re

    return [
        t for t in all_shipped_targets()
        if re.match(r"^\d{2}_", pathlib.Path(t).name)
    ]
