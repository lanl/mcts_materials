"""
Unit tests for the descriptor-table evaluator.

(c) 2026. Triad National Security, LLC. All rights reserved.
"""

import math

import pytest

pytest.importorskip("ase", reason="superhydride structures need ASE")
pytest.importorskip("spglib", reason="structure identifiers need spglib")

from mcts_framework.superhydride import DescriptorTableEvaluator, TcReward  # noqa: E402
from mcts_framework.superhydride.evaluator import normalize_formula  # noqa: E402

# Local aliases for the shared superhydride fixtures in tests/conftest.py.
# (They carry a `superhydride_` prefix there because that conftest is global.)


@pytest.fixture
def make_structure(make_superhydride_structure):
    return make_superhydride_structure

TABLE = """formula,phi,phi_star,h_dos
BeLaH8,0.527,0.738,0.724
BeYH8,0.610,0.700,0.660
"""


@pytest.fixture
def table_path(tmp_path):
    path = tmp_path / "descriptors.csv"
    path.write_text(TABLE)
    return str(path)


# --- Formula normalisation ------------------------------------------------


def test_normalize_formula_is_order_insensitive():
    assert normalize_formula("LaBeH8") == normalize_formula("BeLaH8")
    assert normalize_formula("H8LaBe") == normalize_formula("BeLaH8")


def test_normalize_formula_keeps_counts():
    assert normalize_formula("BeLaH8") == "BeH8La"
    assert normalize_formula("Ce2H8") == "Ce2H8"
    assert normalize_formula("CaH6") != normalize_formula("CaH8")


# --- Lookups --------------------------------------------------------------


async def test_lookup_returns_the_tabulated_descriptors(table_path, make_structure):
    evaluator = DescriptorTableEvaluator(table_path)
    properties = await evaluator.evaluate(make_structure())
    assert properties["phi"] == pytest.approx(0.527)
    assert properties["phi_star"] == pytest.approx(0.738)
    assert properties["h_dos"] == pytest.approx(0.724)
    assert properties["formula"] == "BeLaH8"


async def test_hydrogen_fraction_comes_from_the_structure_not_the_table(
    table_path, make_structure
):
    """
    H_f is exact from the composition, and it enters the fit as a cube - a
    stale table column would quietly poison every estimate.
    """
    evaluator = DescriptorTableEvaluator(table_path)
    properties = await evaluator.evaluate(make_structure())
    assert properties["h_f"] == pytest.approx(0.8)
    assert "h_f" not in TABLE.splitlines()[0].split(",")


async def test_lookup_matches_regardless_of_formula_ordering(tmp_path, make_structure):
    path = tmp_path / "t.csv"
    path.write_text("formula,phi,phi_star,h_dos\nLaBeH8,0.5,0.7,0.6\n")
    properties = await DescriptorTableEvaluator(str(path)).evaluate(make_structure())
    assert properties["phi"] == pytest.approx(0.5)


async def test_unknown_composition_gives_nan_descriptors_and_zero_reward(
    table_path, make_structure
):
    evaluator = DescriptorTableEvaluator(table_path)
    properties = await evaluator.evaluate(make_structure("La", "Mg"))  # not in the table
    assert math.isnan(properties["phi"])
    assert math.isnan(properties["phi_star"])
    assert math.isnan(properties["h_dos"])
    assert properties["h_f"] == pytest.approx(0.8)   # still known
    assert TcReward().compute_reward(properties) == 0.0


async def test_screened_candidates_outrank_unscreened_ones(table_path, make_structure):
    evaluator = DescriptorTableEvaluator(table_path)
    reward = TcReward()
    screened = reward.compute_reward(await evaluator.evaluate(make_structure()))
    unscreened = reward.compute_reward(await evaluator.evaluate(make_structure("La", "Mg")))
    assert screened > unscreened == 0.0


async def test_results_are_cached_by_identifier(table_path, make_structure):
    evaluator = DescriptorTableEvaluator(table_path)
    material = make_structure()
    first = await evaluator.evaluate(material)
    assert len(evaluator) == 1
    assert evaluator.get_cached_result(material.get_identifier()) == first
    assert await evaluator.evaluate(material) is first


def test_membership_check(table_path, make_structure):
    evaluator = DescriptorTableEvaluator(table_path)
    assert "LaBeH8" in evaluator
    assert "BeLaH8" in evaluator
    assert "LaMgH8" not in evaluator


# --- Degraded modes -------------------------------------------------------


async def test_no_table_scores_everything_zero(make_structure):
    """A dry run that only enumerates which compositions the search wants."""
    evaluator = DescriptorTableEvaluator(None)
    properties = await evaluator.evaluate(make_structure())
    assert math.isnan(properties["phi"])
    assert TcReward().compute_reward(properties) == 0.0


async def test_missing_table_file_is_not_fatal(tmp_path, make_structure):
    evaluator = DescriptorTableEvaluator(str(tmp_path / "absent.csv"))
    assert TcReward().compute_reward(await evaluator.evaluate(make_structure())) == 0.0


def test_table_missing_a_required_column_is_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("formula,phi,h_dos\nBeLaH8,0.5,0.6\n")
    with pytest.raises(ValueError, match="missing required column"):
        DescriptorTableEvaluator(str(path))
