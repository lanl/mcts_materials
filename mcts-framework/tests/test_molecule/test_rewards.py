"""
Tests for molecule reward functions (pure logic; no RDKit/molecule-modifier).

© 2025. Triad National Security, LLC. All rights reserved.
"""

import pytest

from mcts_framework.molecule.rewards import (
    MeltingPointReward,
    H2CapacityReward,
    SynthesizabilityReward,
    MultiObjectiveReward,
    create_molecule_reward,
)


# --- MeltingPointReward (favors HIGH, linear, unclamped) -----------------


def test_melting_point_linear_endpoints():
    r = MeltingPointReward(min_temp=0.0, max_temp=700.0)
    assert r.compute_reward({"melting_point": 0.0}) == pytest.approx(0.0)
    assert r.compute_reward({"melting_point": 700.0}) == pytest.approx(1.0)
    # midpoint
    assert r.compute_reward({"melting_point": 350.0}) == pytest.approx(0.5)


def test_melting_point_favors_high():
    r = MeltingPointReward()
    low = r.compute_reward({"melting_point": 300.0})
    high = r.compute_reward({"melting_point": 600.0})
    assert high > low  # higher melting point -> higher reward


def test_melting_point_unclamped():
    r = MeltingPointReward(min_temp=0.0, max_temp=700.0)
    # Above max -> > 1
    assert r.compute_reward({"melting_point": 950.0}) > 1.0
    # Below min -> < 0
    assert r.compute_reward({"melting_point": -10.0}) < 0.0


def test_melting_point_invalid_range():
    with pytest.raises(ValueError):
        MeltingPointReward(min_temp=700.0, max_temp=200.0)


def test_melting_point_property_names():
    assert MeltingPointReward().get_property_names() == ["melting_point"]


# --- H2CapacityReward ----------------------------------------------------


def test_h2_capacity_linear():
    r = H2CapacityReward(scale=10.0)
    assert r.compute_reward({"h2_capacity": 5.0}) == pytest.approx(0.5)
    assert r.compute_reward({"h2_capacity": 10.0}) == pytest.approx(1.0)


def test_h2_capacity_invalid_scale():
    with pytest.raises(ValueError):
        H2CapacityReward(scale=0.0)


# --- SynthesizabilityReward (favors EASY = low score) --------------------


def test_synthesizability_easy_high_reward():
    r = SynthesizabilityReward()
    easy = r.compute_reward({"synthesizability": 1.0})
    hard = r.compute_reward({"synthesizability": 10.0})
    assert easy > hard
    assert easy == pytest.approx(1.0)   # (5.5-1)/4.5 = 1.0
    assert hard == pytest.approx(-1.0)  # (5.5-10)/4.5 = -1.0


# --- MultiObjectiveReward ------------------------------------------------


def test_multi_objective_weighted_sum():
    r = MultiObjectiveReward({"h2_capacity": 1.0, "melting_point": 0.5})
    props = {"h2_capacity": 10.0, "melting_point": 700.0}
    # h2: 10/10=1.0 * 1.0; mp: (700-200)/500=1.0 * 0.5 -> 1.5
    assert r.compute_reward(props) == pytest.approx(1.5)


def test_multi_objective_property_names():
    r = MultiObjectiveReward({"h2_capacity": 1.0, "synthesizability": 0.3})
    assert set(r.get_property_names()) == {"h2_capacity", "synthesizability"}


def test_multi_objective_rejects_empty():
    with pytest.raises(ValueError):
        MultiObjectiveReward({})


def test_multi_objective_rejects_unknown():
    with pytest.raises(ValueError):
        MultiObjectiveReward({"bogus_property": 1.0})


# --- Factory -------------------------------------------------------------


def test_create_single_objectives():
    assert isinstance(create_molecule_reward("melting_point"), MeltingPointReward)
    assert isinstance(create_molecule_reward("h2_capacity"), H2CapacityReward)
    assert isinstance(
        create_molecule_reward("synthesizability"), SynthesizabilityReward
    )


def test_create_multi_objective():
    r = create_molecule_reward("multi_objective", {"h2_capacity": 1.0})
    assert isinstance(r, MultiObjectiveReward)


def test_create_multi_objective_requires_weights():
    with pytest.raises(ValueError):
        create_molecule_reward("multi_objective")


def test_create_unknown_objective():
    with pytest.raises(ValueError):
        create_molecule_reward("bogus")