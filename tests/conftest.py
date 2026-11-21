"""Shared fixtures. No GPU, no model weights, no network."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import load, load_pipeline
from src.tracking import identity_provenance, read_structure

from tests.paths import FIXTURES, ROOT


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def pipeline_path() -> Path:
    return ROOT / "config" / "pipeline.yaml"


@pytest.fixture(scope="session")
def fixture_target_path() -> Path:
    return FIXTURES / "fixture_target.yaml"


@pytest.fixture
def cfg(pipeline_path, fixture_target_path):
    """The real pipeline.yaml plus the fixture target.

    Deliberately the real config file: a test that invented its own weights
    would not catch a change to the shipped defaults, which is exactly the
    thing a scoring regression would come from.
    """
    return load(pipeline_path, fixture_target_path)


@pytest.fixture
def pipeline(pipeline_path):
    return load_pipeline(pipeline_path)


def load_fixture(cfg, name: str):
    """A fixture structure plus an identity Provenance for it.

    The fixtures are already in seed numbering, so both residue maps are the
    identity -- but resolution still goes through the real `Provenance.resolve`,
    so the fixtures exercise the production code path.
    """
    path = FIXTURES / f"{name}.pdb"
    structure = read_structure(path, cfg.target.ligand.ccd)
    ligand_atoms = sorted(
        {
            spec.atom
            for bond in cfg.target.hbonds
            for spec in (bond.donor, bond.acceptor,
                         bond.donor_antecedent, bond.acceptor_antecedent)
            if spec is not None and spec.site == "ligand"
        }
    )
    provenance = identity_provenance(
        design_id=f"fixture:{name}",
        scaffold_id="fixture",
        structure=structure,
        motif=cfg.target.motif,
        ligand_atoms=ligand_atoms,
    )
    return structure, provenance


@pytest.fixture
def ideal(cfg):
    return load_fixture(cfg, "ideal")


@pytest.fixture
def stretched(cfg):
    return load_fixture(cfg, "stretched")


@pytest.fixture
def mixed(cfg):
    return load_fixture(cfg, "mixed")


@pytest.fixture
def gated(cfg):
    return load_fixture(cfg, "gated")


@pytest.fixture
def clashing(cfg):
    return load_fixture(cfg, "clashing")
