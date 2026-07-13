"""Tests for mcts_crystal.mcts.MCTS, using stub energy/doscar calculators (no real MACE/MP calls)."""

import pytest

from mcts_crystal.mcts import MCTS
from mcts_crystal.node import MCTSTreeNode, ehull_reward


class StubEnergyCalculator:
    """Deterministic stand-in for MaceEnergyCalculator - no MACE/Materials Project needed."""

    def calculate_energies(self, atoms):
        # Stable-ish, deterministic values so reward calculations are predictable
        return -0.5, 0.02


class StubDoscarLookup:
    def __init__(self, reward=0.3):
        self.reward = reward

    def get_reward(self, formula):
        return self.reward


@pytest.fixture
def root_node(u_w_pb_atoms):
    return MCTSTreeNode(u_w_pb_atoms, f_block_mode='u_only', termination_limit=10)


def _make_child(t_of_visit, total_reward, exploration_constant=1.0, terminated=False, formula="X"):
    """A bare MCTSTreeNode with only the fields select_node()/get_ucb()/get_puct() touch -
    no atoms/CIF needed, so selection-mode behavior can be tested in isolation."""
    node = MCTSTreeNode.__new__(MCTSTreeNode)
    node.t_of_visit = t_of_visit
    node.total_reward = total_reward
    node.best_reward = total_reward
    node.exploration_constant = exploration_constant
    node.terminated = terminated
    node.children = []
    node.expandable = True
    node.parent = None
    node.get_chemical_formula = lambda: formula
    return node


def _make_root_with_children(children, root_visits=10):
    root = MCTSTreeNode.__new__(MCTSTreeNode)
    root.t_of_visit = root_visits
    root.expandable = False
    root.children = children
    root.metal_move = [1]  # only read by MCTS.__init__ to size t_warmup
    root.g_iv_move = [1]
    for child in children:
        child.parent = root
    return root


class TestUCBAndSelection:
    def test_root_ucb_is_infinite_before_any_visits(self, root_node):
        assert root_node.get_ucb() == float('inf')

    def test_select_node_returns_root_when_not_expanded(self, root_node):
        mcts = MCTS(root_node)
        chain = mcts.select_node()
        assert chain == [root_node]


class TestSelectionModes:
    def test_ucb1_is_deterministic_argmax(self):
        low = _make_child(t_of_visit=5, total_reward=1.0, formula="low")
        high = _make_child(t_of_visit=5, total_reward=4.0, formula="high")
        root = _make_root_with_children([low, high])
        mcts = MCTS(root)

        for _ in range(10):
            chain = mcts.select_node(mode='ucb1')
            assert chain == [root, high]

    def test_puct_unvisited_child_has_zero_q_not_infinity(self):
        unvisited = _make_child(t_of_visit=0, total_reward=0.0, formula="unvisited")
        assert unvisited.get_ucb() == float('inf')
        root = _make_root_with_children([unvisited])
        assert unvisited.get_puct(prior=1.0) != float('inf')

    def test_puct_is_deterministic_argmax(self):
        low = _make_child(t_of_visit=5, total_reward=1.0, formula="low")
        high = _make_child(t_of_visit=5, total_reward=4.0, formula="high")
        root = _make_root_with_children([low, high])
        mcts = MCTS(root)

        for _ in range(10):
            chain = mcts.select_node(mode='puct')
            assert chain == [root, high]

    def test_epsilon_greedy_is_argmax_when_epsilon_zero(self):
        low = _make_child(t_of_visit=5, total_reward=1.0, formula="low")
        high = _make_child(t_of_visit=5, total_reward=4.0, formula="high")
        root = _make_root_with_children([low, high])
        mcts = MCTS(root, epsilon=0.0)

        for _ in range(10):
            chain = mcts.select_node(mode='epsilon_greedy')
            assert chain == [root, high]

    def test_epsilon_greedy_explores_uniformly_when_epsilon_one(self):
        low = _make_child(t_of_visit=5, total_reward=1.0, formula="low")
        high = _make_child(t_of_visit=5, total_reward=4.0, formula="high")
        root = _make_root_with_children([low, high])
        mcts = MCTS(root, epsilon=1.0)

        selected = {mcts.select_node(mode='epsilon_greedy')[-1].get_chemical_formula() for _ in range(50)}
        # Uniform random exploration should visit both children, not just the argmax
        assert selected == {"low", "high"}

    def test_boltzmann_low_temperature_behaves_like_argmax(self):
        low = _make_child(t_of_visit=5, total_reward=1.0, formula="low")
        high = _make_child(t_of_visit=5, total_reward=4.0, formula="high")
        root = _make_root_with_children([low, high])
        mcts = MCTS(root, temperature=0.01)

        selected = [mcts.select_node(mode='boltzmann')[-1].get_chemical_formula() for _ in range(20)]
        assert selected == ["high"] * 20

    def test_boltzmann_never_visited_children_are_explored_first(self):
        unvisited = _make_child(t_of_visit=0, total_reward=0.0, formula="unvisited")
        visited = _make_child(t_of_visit=5, total_reward=4.0, formula="visited")
        root = _make_root_with_children([unvisited, visited])
        mcts = MCTS(root, temperature=0.01)

        chain = mcts.select_node(mode='boltzmann')
        assert chain == [root, unvisited]

    def test_hybrid_is_argmax_when_epsilon_zero(self):
        low = _make_child(t_of_visit=5, total_reward=1.0, formula="low")
        high = _make_child(t_of_visit=5, total_reward=4.0, formula="high")
        root = _make_root_with_children([low, high])
        mcts = MCTS(root, epsilon=0.0)

        chain = mcts.select_node(mode='hybrid')
        assert chain == [root, high]

    def test_terminated_children_are_never_selected(self):
        terminated = _make_child(t_of_visit=5, total_reward=100.0, terminated=True, formula="terminated")
        alive = _make_child(t_of_visit=5, total_reward=1.0, formula="alive")
        root = _make_root_with_children([terminated, alive])
        mcts = MCTS(root, epsilon=1.0)

        for mode in ['ucb1', 'epsilon_greedy', 'boltzmann', 'puct', 'hybrid']:
            chain = mcts.select_node(mode=mode)
            assert chain == [root, alive], f"mode={mode} selected a terminated node"

    def test_unknown_selection_mode_raises(self):
        low = _make_child(t_of_visit=5, total_reward=1.0, formula="low")
        root = _make_root_with_children([low])
        mcts = MCTS(root)

        with pytest.raises(ValueError):
            mcts.select_node(mode='not_a_real_mode')


class TestExpansionSimulation:
    def test_ehull_reward_matches_node_formula(self, root_node):
        mcts = MCTS(root_node)
        reward, _ = mcts.expansion_simulation(
            rollout_depth=0, n_rollout=1,
            energy_calculator=StubEnergyCalculator(),
            rollout_method='ehull',
        )
        assert reward == pytest.approx(ehull_reward(0.02))

    def test_ehull_rdos_combines_both_terms(self, root_node):
        mcts = MCTS(root_node)
        reward, _ = mcts.expansion_simulation(
            rollout_depth=0, n_rollout=1,
            energy_calculator=StubEnergyCalculator(),
            rollout_method='ehull_rdos',
            beta=1.0, gamma=0.0001,
            doscar_lookup=StubDoscarLookup(reward=0.3),
        )
        assert reward == pytest.approx(1.0 * ehull_reward(0.02) + 0.0001 * 0.3)

    def test_rdos_only_ignores_energy_calculator(self, root_node):
        mcts = MCTS(root_node)
        reward, _ = mcts.expansion_simulation(
            rollout_depth=0, n_rollout=1,
            energy_calculator=None,
            rollout_method='rdos',
            doscar_lookup=StubDoscarLookup(reward=0.9),
        )
        assert reward == pytest.approx(0.9)

    def test_unknown_rollout_method_raises(self, root_node):
        mcts = MCTS(root_node)
        with pytest.raises(ValueError):
            mcts.expansion_simulation(
                rollout_depth=0, n_rollout=1,
                energy_calculator=StubEnergyCalculator(),
                rollout_method='not_a_real_method',
            )


class TestRunIntegration:
    @pytest.mark.parametrize("rollout_method", ["ehull", "ehull_rdos", "rdos"])
    def test_run_completes_for_each_rollout_method(self, root_node, rollout_method):
        mcts = MCTS(root_node)
        results = mcts.run(
            n_iterations=5,
            energy_calculator=StubEnergyCalculator(),
            rollout_depth=0,
            n_rollout=1,
            rollout_method=rollout_method,
            beta=1.0,
            gamma=0.0001,
            doscar_lookup=StubDoscarLookup(),
        )
        assert results['iterations_completed'] >= 1
        assert len(results['stat_dict']) >= 1

    @pytest.mark.parametrize("selection_mode", ["ucb1", "epsilon_greedy", "boltzmann", "puct", "hybrid"])
    def test_run_completes_for_each_selection_mode(self, root_node, selection_mode):
        mcts = MCTS(root_node, epsilon=0.2, temperature=1.0)
        results = mcts.run(
            n_iterations=5,
            energy_calculator=StubEnergyCalculator(),
            rollout_depth=0,
            n_rollout=1,
            selection_mode=selection_mode,
            rollout_method='ehull',
        )
        assert results['iterations_completed'] >= 1
        assert len(results['stat_dict']) >= 1
