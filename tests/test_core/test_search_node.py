"""
Unit tests for SearchNode.

© 2025. Triad National Security, LLC. All rights reserved.
"""

import pytest
import numpy as np
from mcts_framework.core.material import Material
from mcts_framework.core.search_node import SearchNode


class SimpleMaterial(Material):
    """Simple material for testing."""

    def __init__(self, name: str):
        self.name = name

    def get_identifier(self) -> str:
        return self.name

    def copy(self) -> 'SimpleMaterial':
        return SimpleMaterial(self.name)


def test_search_node_initialization():
    """Test SearchNode initialization."""
    material = SimpleMaterial("A")
    node = SearchNode(material, exploration_constant=0.5, termination_limit=100)

    assert node.material == material
    assert node.parent is None
    assert node.children == []
    assert node.expandable is True
    assert node.visits == 0
    assert node.total_reward == 0.0
    assert node.subtree_best == -np.inf
    assert node.terminated is False
    assert node.exploration_constant == 0.5
    assert node.termination_limit == 100


def test_ucb_unvisited_node():
    """Test UCB1 returns infinity for unvisited nodes."""
    material = SimpleMaterial("A")
    node = SearchNode(material)

    assert node.get_ucb() == float('inf')


def test_ucb_visited_node():
    """Test UCB1 calculation for visited nodes."""
    parent = SimpleMaterial("parent")
    child = SimpleMaterial("child")

    parent_node = SearchNode(parent)
    child_node = SearchNode(child, exploration_constant=0.1)

    # Simulate visits
    parent_node.update(1.0)
    parent_node.update(1.0)
    parent_node.update(1.0)  # parent: N=3

    child_node.parent = parent_node
    child_node.update(0.5)  # child: N=1, total=0.5

    ucb = child_node.get_ucb()

    # UCB = 0.5/1 + 0.1 * sqrt(ln(3) / 1)
    expected = 0.5 + 0.1 * np.sqrt(np.log(3) / 1)
    assert np.isclose(ucb, expected)


def test_puct_unvisited_node():
    """Test PUCT for unvisited nodes (Q=0, not inf)."""
    parent = SimpleMaterial("parent")
    child = SimpleMaterial("child")

    parent_node = SearchNode(parent)
    child_node = SearchNode(child, exploration_constant=0.1)

    parent_node.update(1.0)
    parent_node.update(1.0)  # parent: N=2

    child_node.parent = parent_node

    puct = child_node.get_puct(prior=1.0)

    # PUCT = 0 + 0.1 * 1.0 * sqrt(2) / (1 + 0)
    expected = 0.1 * 1.0 * np.sqrt(2) / 1
    assert np.isclose(puct, expected)


def test_puct_visited_node():
    """Test PUCT calculation for visited nodes."""
    parent = SimpleMaterial("parent")
    child = SimpleMaterial("child")

    parent_node = SearchNode(parent)
    child_node = SearchNode(child, exploration_constant=0.1)

    parent_node.update(1.0)
    parent_node.update(1.0)  # parent: N=2

    child_node.parent = parent_node
    child_node.update(0.8)  # child: N=1, Q=0.8

    puct = child_node.get_puct(prior=1.0)

    # PUCT = 0.8 + 0.1 * 1.0 * sqrt(2) / (1 + 1)
    expected = 0.8 + 0.1 * 1.0 * np.sqrt(2) / 2
    assert np.isclose(puct, expected)


def test_update_improves_subtree_best():
    """Test that update tracks the running subtree-best reward."""
    material = SimpleMaterial("A")
    node = SearchNode(material)

    node.update(0.5)
    assert node.visits == 1
    assert node.total_reward == 0.5
    assert node.subtree_best == 0.5
    assert node.visits_since_improvement == 0

    node.update(0.8)
    assert node.visits == 2
    assert node.total_reward == 1.3
    assert node.subtree_best == 0.8
    assert node.visits_since_improvement == 0

    node.update(0.3)  # Worse than best
    assert node.visits == 3
    assert node.total_reward == 1.6
    assert node.subtree_best == 0.8
    assert node.visits_since_improvement == 1


def test_tie_counts_as_improvement():
    """A visit tying the subtree best resets the no-improvement countdown
    (non-strict >= comparison)."""
    node = SearchNode(SimpleMaterial("A"))

    node.update(1.0)
    assert node.visits_since_improvement == 0

    node.update(0.5)  # below best -> counts as no improvement
    assert node.visits_since_improvement == 1

    node.update(1.0)  # ties best -> resets countdown
    assert node.subtree_best == 1.0
    assert node.visits_since_improvement == 0


def test_ties_prevent_termination():
    """Repeatedly tying the best keeps a node alive past termination_limit;
    strictly-below rewards would have terminated it."""
    node = SearchNode(SimpleMaterial("A"), termination_limit=3)
    node.update(1.0)
    for _ in range(10):
        node.update(1.0)  # every visit ties -> countdown never fires
    assert not node.terminated


def test_termination_after_limit():
    """Test node terminates after limit visits without improvement."""
    material = SimpleMaterial("A")
    node = SearchNode(material, termination_limit=3)

    node.update(1.0)  # Best: 1.0, visits_since_improvement: 0
    assert not node.terminated

    node.update(0.5)  # No improvement, visits_since_improvement: 1
    assert not node.terminated

    node.update(0.5)  # No improvement, visits_since_improvement: 2
    assert not node.terminated

    node.update(0.5)  # No improvement, visits_since_improvement: 3
    assert node.terminated


def test_add_child():
    """Test adding child nodes."""
    parent_material = SimpleMaterial("parent")
    child_material = SimpleMaterial("child")

    parent_node = SearchNode(parent_material)
    child_node = SearchNode(child_material)

    parent_node.add_child(child_node)

    assert child_node.parent == parent_node
    assert child_node in parent_node.children
    assert len(parent_node.children) == 1


def test_is_leaf():
    """Test is_leaf()."""
    parent_material = SimpleMaterial("parent")
    child_material = SimpleMaterial("child")

    parent_node = SearchNode(parent_material)
    child_node = SearchNode(child_material)

    assert parent_node.is_leaf()

    parent_node.add_child(child_node)
    assert not parent_node.is_leaf()


def test_is_fully_expanded():
    """Test is_fully_expanded()."""
    material = SimpleMaterial("A")
    node = SearchNode(material)

    assert not node.is_fully_expanded()  # expandable=True

    node.expandable = False
    assert node.is_fully_expanded()


def test_get_mean_reward():
    """Test mean reward calculation."""
    material = SimpleMaterial("A")
    node = SearchNode(material)

    assert node.get_mean_reward() == 0.0  # No visits

    node.update(0.5)
    node.update(1.0)
    node.update(0.7)

    mean = node.get_mean_reward()
    expected = (0.5 + 1.0 + 0.7) / 3
    assert np.isclose(mean, expected)


def test_get_subtree_size():
    """Test subtree size calculation."""
    root = SearchNode(SimpleMaterial("root"))
    child1 = SearchNode(SimpleMaterial("child1"))
    child2 = SearchNode(SimpleMaterial("child2"))
    grandchild = SearchNode(SimpleMaterial("grandchild"))

    root.add_child(child1)
    root.add_child(child2)
    child1.add_child(grandchild)

    assert root.get_subtree_size() == 4
    assert child1.get_subtree_size() == 2
    assert child2.get_subtree_size() == 1
    assert grandchild.get_subtree_size() == 1


def test_get_depth():
    """Test depth calculation."""
    root = SearchNode(SimpleMaterial("root"))
    child = SearchNode(SimpleMaterial("child"))
    grandchild = SearchNode(SimpleMaterial("grandchild"))

    root.add_child(child)
    child.add_child(grandchild)

    assert root.get_depth() == 0
    assert child.get_depth() == 1
    assert grandchild.get_depth() == 2


def test_repr_and_str():
    """Test string representations."""
    material = SimpleMaterial("TestMaterial")
    node = SearchNode(material)
    node.update(0.5)
    node.update(0.8)

    repr_str = repr(node)
    assert "TestMaterial" in repr_str
    assert "visits=2" in repr_str

    str_str = str(node)
    assert "TestMaterial" in str_str
    assert "N=2" in str_str
