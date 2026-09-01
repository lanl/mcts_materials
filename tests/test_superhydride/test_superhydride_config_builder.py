"""
Config validation and end-to-end wiring for the superhydride material type.

Covers the Pydantic section, the CLI builder, and one short search that runs
the whole loop - expansion, descriptor lookup, Tc reward - against a small
descriptor table.

(c) 2026. Triad National Security, LLC. All rights reserved.
"""

import warnings

import pytest

pytest.importorskip("ase", reason="the superhydride builder needs ASE")
pytest.importorskip("spglib", reason="structure identifiers need spglib")

from ase.io import write as ase_write  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from mcts_framework.cli.builders import build_mcts  # noqa: E402
from mcts_framework.core.config import Config, SuperhydrideConfig  # noqa: E402
from mcts_framework.superhydride import (  # noqa: E402
    DescriptorTableEvaluator,
    HostSubstitutionMoves,
    SuperhydrideStructure,
    TcReward,
)

#: Descriptors for the template and everything one or two host swaps away that
#: this test needs. LaMgH8 is the intended winner: phi* sits on the 2/3
#: optimum with the best phi and H_DOS.
DESCRIPTOR_TABLE = """formula,phi,phi_star,h_dos
BeLaH8,0.527,0.738,0.724
BeYH8,0.500,0.720,0.600
BaBeH8,0.450,0.900,0.500
BeCeH8,0.520,0.750,0.700
LaLiH8,0.480,0.800,0.550
LaBH8,0.510,0.950,0.640
LaMgH8,0.700,0.667,0.800
"""


# Local aliases for the shared superhydride fixtures in tests/conftest.py.
# (They carry a `superhydride_` prefix there because that conftest is global.)


@pytest.fixture
def template(superhydride_template):
    return superhydride_template


@pytest.fixture
def make_structure(make_superhydride_structure):
    return make_superhydride_structure


@pytest.fixture
def make_template(make_superhydride_template):
    return make_superhydride_template


@pytest.fixture
def template_cif(tmp_path, make_template):
    path = tmp_path / "template.cif"
    ase_write(str(path), make_template())
    return str(path)


@pytest.fixture
def table_path(tmp_path):
    path = tmp_path / "descriptors.csv"
    path.write_text(DESCRIPTOR_TABLE)
    return str(path)


def make_config(template_cif, table_path=None, **overrides):
    section = {"structure_path": template_cif, "descriptor_table_path": table_path}
    section.update(overrides)
    return Config(
        material_type="superhydride",
        mcts={"iterations": 60, "seed": 0, "search_mode": "thorough"},
        superhydride=section,
    )


# --- Config ---------------------------------------------------------------


def test_defaults():
    section = SuperhydrideConfig(structure_path="t.cif")
    assert section.host_palette == "high_tc"
    assert section.preserve_distinct_hosts is True
    assert section.normalize_reward is True
    assert section.descriptor_table_path is None


def test_structure_path_is_required():
    with pytest.raises(ValidationError):
        SuperhydrideConfig()


def test_unknown_palette_is_rejected():
    with pytest.raises(ValidationError):
        SuperhydrideConfig(structure_path="t.cif", host_palette="noble_gases")


def test_unknown_field_is_rejected():
    with pytest.raises(ValidationError):
        SuperhydrideConfig(structure_path="t.cif", rollout_method="ehull")


def test_material_type_requires_its_section():
    with pytest.raises(ValidationError, match="requires a 'superhydride' section"):
        Config(material_type="superhydride")


def test_config_round_trips_through_yaml(tmp_path, template_cif):
    config = make_config(template_cif, host_palette="electropositive")
    path = tmp_path / "config.yaml"
    config.dump_yaml(str(path))
    reloaded = Config.from_yaml(str(path))
    assert reloaded.material_type == "superhydride"
    assert reloaded.superhydride.host_palette == "electropositive"
    assert reloaded.superhydride.structure_path == template_cif


# --- Builder --------------------------------------------------------------


def test_builder_assembles_the_right_components(template_cif, table_path):
    mcts = build_mcts(make_config(template_cif, table_path))
    assert isinstance(mcts.root.material, SuperhydrideStructure)
    assert isinstance(mcts.move_generator, HostSubstitutionMoves)
    assert isinstance(mcts.property_evaluator, DescriptorTableEvaluator)
    assert isinstance(mcts.reward_function, TcReward)
    assert mcts.root.material.get_formula() == "BeLaH8"


def test_builder_passes_the_palette_through(template_cif, table_path):
    mcts = build_mcts(make_config(template_cif, table_path, host_palette="electropositive"))
    assert mcts.move_generator.palette == "electropositive"


def test_builder_rejects_a_template_without_hydrogen(tmp_path, make_template):
    from ase import Atoms

    path = tmp_path / "no_h.cif"
    ase_write(str(path), Atoms("LaBe", positions=[(0, 0, 0), (2, 2, 2)], cell=[5, 5, 5], pbc=True))
    with pytest.raises(ValueError, match="no hydrogen"):
        build_mcts(make_config(str(path)))


def test_builder_warns_about_a_frozen_host(tmp_path, make_template):
    path = tmp_path / "lapd.cif"
    ase_write(str(path), make_template("La", "Pd"))
    with pytest.warns(UserWarning, match="stay fixed"):
        build_mcts(make_config(str(path)))


# --- End to end -----------------------------------------------------------


async def test_search_finds_the_best_tabulated_candidate(template_cif, table_path):
    """
    The whole loop: expand hosts, look up descriptors, score with Eq. 2.
    LaMgH8 has the best descriptors in the table, so the search must surface it.
    """
    mcts = build_mcts(make_config(template_cif, table_path))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        await mcts.run(iterations=60)

    best = mcts.get_best_materials(n=1)[0]
    assert best.material.get_formula() == "LaMgH8"
    assert 0.0 < mcts.best_reward <= 1.0


async def test_search_only_visits_hydrides_of_the_template(template_cif, table_path):
    mcts = build_mcts(make_config(template_cif, table_path))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        await mcts.run(iterations=40)

    assert len(mcts.visited_materials) > 1
    for node in mcts.get_best_materials(n=50):
        material = node.material
        assert material.get_hydrogen_fraction() == pytest.approx(0.8)
        assert len(material.get_host_elements()) == 2


async def test_search_runs_without_a_descriptor_table(template_cif):
    """
    A dry run: every reward is 0.0, but the search still enumerates the space
    so the compositions worth computing can be collected.
    """
    mcts = build_mcts(make_config(template_cif, table_path=None))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        await mcts.run(iterations=20)
    assert len(mcts.visited_materials) > 1
    assert mcts.best_reward == 0.0
