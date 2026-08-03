"""
Unit tests for selection strategies.

© 2025. Triad National Security, LLC. All rights reserved.
"""

import random

import pytest

from mcts_framework.core.material import Material
from mcts_framework.core.search_node import SearchNode
from mcts_framework.core.selection import (
    UCB1,
    PUCT,
    EpsilonGreedy,
    Boltzmann,
    AllChildrenTerminated,
    create_selection_strategy,
)


class SimpleMaterial(Material):
    """Simple material for testing."""

    def __init__(self, name: str):
        self.name = name

    def get_identifier(self) -> str:
        return self.name

    def copy(self) -> "SimpleMaterial":
        return SimpleMaterial(self.name)


def make_parent_with_children(n_children: int) -> SearchNode:
    """Build a visited parent with n unvisited children."""
    parent = SearchNode(SimpleMaterial("parent"))
    parent.update(1.0)  # give parent at least one visit
    for i in range(n_children):
        child = SearchNode(SimpleMaterial(f"child_{i}"))
        parent.add_child(child)
    return parent


# --- UCB1 -----------------------------------------------------------------


def test_ucb1_prefers_unvisited():
    """UCB1 should pick an unvisited child (UCB=+inf) over a visited one."""
    parent = make_parent_with_children(2)
    # Visit only child_0
    parent.children[0].update(0.9)

    strategy = UCB1()
    selected = strategy.select_child(parent)
    # child_1 is unvisited -> +inf -> must be selected
    assert selected.material.get_identifier() == "child_1"


def test_ucb1_argmax_among_visited():
    """With all children visited, UCB1 picks the highest score."""
    parent = make_parent_with_children(2)
    parent.update(1.0)
    parent.update(1.0)  # parent N=3 total

    # Both children visited once; child_0 has higher reward
    parent.children[0].update(0.9)
    parent.children[1].update(0.1)

    strategy = UCB1()
    selected = strategy.select_child(parent)
    assert selected.material.get_identifier() == "child_0"


# --- PUCT -----------------------------------------------------------------


def test_puct_selects_child():
    """PUCT returns a valid child."""
    parent = make_parent_with_children(3)
    strategy = PUCT()
    selected = strategy.select_child(parent)
    assert selected in parent.children


def test_puct_prefers_higher_q_when_visited():
    """When all children visited equally, PUCT prefers higher Q."""
    parent = make_parent_with_children(2)
    parent.update(1.0)
    parent.children[0].update(0.8)
    parent.children[1].update(0.2)

    strategy = PUCT()
    selected = strategy.select_child(parent)
    assert selected.material.get_identifier() == "child_0"


# --- EpsilonGreedy --------------------------------------------------------


def test_epsilon_greedy_exploits_when_epsilon_zero():
    """With epsilon=0, always argmax UCB1."""
    parent = make_parent_with_children(2)
    parent.update(1.0)
    parent.children[0].update(0.9)
    parent.children[1].update(0.1)

    strategy = EpsilonGreedy(epsilon=0.0)
    selected = strategy.select_child(parent)
    assert selected.material.get_identifier() == "child_0"


def test_epsilon_greedy_explores_when_epsilon_one():
    """With epsilon=1, always uniform random (deterministic via seeded rng)."""
    parent = make_parent_with_children(3)
    for c in parent.children:
        c.update(0.5)

    rng = random.Random(42)
    strategy = EpsilonGreedy(epsilon=1.0, rng=rng)
    # Should not crash and should return a valid child
    selected = strategy.select_child(parent)
    assert selected in parent.children


def test_epsilon_greedy_reproducible():
    """Seeded rng gives reproducible exploratory picks."""
    parent = make_parent_with_children(4)
    for c in parent.children:
        c.update(0.5)

    picks_a = []
    rng_a = random.Random(123)
    strat_a = EpsilonGreedy(epsilon=1.0, rng=rng_a)
    for _ in range(10):
        picks_a.append(strat_a.select_child(parent).material.get_identifier())

    picks_b = []
    rng_b = random.Random(123)
    strat_b = EpsilonGreedy(epsilon=1.0, rng=rng_b)
    for _ in range(10):
        picks_b.append(strat_b.select_child(parent).material.get_identifier())

    assert picks_a == picks_b


# --- Boltzmann ------------------------------------------------------------


def test_boltzmann_rejects_nonpositive_temperature():
    with pytest.raises(ValueError):
        Boltzmann(temperature=0.0)


def test_boltzmann_prefers_unvisited():
    """Boltzmann explores unvisited children first."""
    parent = make_parent_with_children(2)
    parent.children[0].update(0.9)  # visited
    # child_1 unvisited

    strategy = Boltzmann(temperature=1.0, rng=random.Random(0))
    selected = strategy.select_child(parent)
    assert selected.material.get_identifier() == "child_1"


def test_boltzmann_returns_valid_child():
    """With all visited, Boltzmann returns a valid child."""
    parent = make_parent_with_children(3)
    parent.update(1.0)
    for i, c in enumerate(parent.children):
        c.update(0.1 * (i + 1))

    strategy = Boltzmann(temperature=1.0, rng=random.Random(7))
    selected = strategy.select_child(parent)
    assert selected in parent.children


def test_boltzmann_low_temperature_is_greedy():
    """Very low temperature should concentrate on the best child."""
    parent = make_parent_with_children(2)
    parent.update(1.0)
    parent.children[0].update(0.9)
    parent.children[1].update(0.1)

    strategy = Boltzmann(temperature=0.01, rng=random.Random(1))
    # Over several draws, near-greedy should almost always pick child_0
    picks = [strategy.select_child(parent).material.get_identifier() for _ in range(20)]
    assert picks.count("child_0") >= 18


# --- All-terminated handling ---------------------------------------------


def test_all_children_terminated_raises():
    """Every strategy raises AllChildrenTerminated when no valid children."""
    parent = make_parent_with_children(2)
    for c in parent.children:
        c.terminated = True

    for strategy in (UCB1(), PUCT(), EpsilonGreedy(), Boltzmann()):
        with pytest.raises(AllChildrenTerminated):
            strategy.select_child(parent)


def test_terminated_children_excluded():
    """Terminated children are never selected."""
    parent = make_parent_with_children(2)
    parent.update(1.0)
    # child_0 would win on UCB but is terminated
    parent.children[0].update(0.99)
    parent.children[0].terminated = True
    parent.children[1].update(0.01)

    strategy = UCB1()
    selected = strategy.select_child(parent)
    assert selected.material.get_identifier() == "child_1"


# --- Factory --------------------------------------------------------------


def test_create_selection_strategy():
    assert isinstance(create_selection_strategy("ucb1"), UCB1)
    assert isinstance(create_selection_strategy("puct"), PUCT)
    assert isinstance(create_selection_strategy("epsilon_greedy"), EpsilonGreedy)
    assert isinstance(create_selection_strategy("boltzmann"), Boltzmann)


def test_create_selection_strategy_passes_params():
    strat = create_selection_strategy("epsilon_greedy", epsilon=0.5)
    assert isinstance(strat, EpsilonGreedy)
    assert strat.epsilon == 0.5

    strat2 = create_selection_strategy("boltzmann", temperature=2.0)
    assert isinstance(strat2, Boltzmann)
    assert strat2.temperature == 2.0


def test_create_selection_strategy_unknown():
    with pytest.raises(ValueError):
        create_selection_strategy("not_a_mode")
