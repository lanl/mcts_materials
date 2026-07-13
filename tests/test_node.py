"""Tests for mcts_crystal.node: reward functions and substitution/expansion logic."""

import math

import pytest

from mcts_crystal.node import MCTSTreeNode, ehull_reward


class TestEhullReward:
    def test_at_threshold_is_zero(self):
        assert ehull_reward(0.05) == pytest.approx(0.0, abs=1e-9)

    def test_stable_compound_is_strongly_positive(self):
        # E_hull = 0 (on the hull) should sit close to the +1 asymptote
        assert ehull_reward(0.0) == pytest.approx(1.0, abs=1e-3)

    def test_unstable_compound_is_strongly_negative(self):
        # E_hull = 0.1 (50 meV above the stability threshold) should sit close to -1
        assert ehull_reward(0.1) == pytest.approx(-1.0, abs=1e-3)

    def test_symmetric_around_threshold(self):
        below = ehull_reward(0.05 - 0.01)
        above = ehull_reward(0.05 + 0.01)
        assert below == pytest.approx(-above, abs=1e-9)


class TestFBlockModes:
    def test_u_only_restricts_to_uranium(self, u_w_pb_atoms):
        node = MCTSTreeNode(u_w_pb_atoms, f_block_mode='u_only')
        assert node.f_block_move == [92]

    def test_lanthanides_u_extended_includes_uranium_and_lanthanides(self, u_w_pb_atoms):
        node = MCTSTreeNode(u_w_pb_atoms, f_block_mode='lanthanides_u_extended')
        # U (92) plus several lanthanide moves should be reachable
        assert 92 in node.f_block_move
        lanthanides = set(range(58, 72))
        assert len(set(node.f_block_move) & lanthanides) > 0

    def test_experimental_mode_reaches_uranium_from_nd(self, u_w_pb_atoms):
        node = MCTSTreeNode(u_w_pb_atoms, f_block_mode='experimental')
        # Starting from U (92), experimental mode adds an explicit move to Nd (60)
        assert node.f_block_move == [60, 92]


class TestExpansion:
    def test_expand_produces_candidates(self, u_w_pb_atoms):
        node = MCTSTreeNode(u_w_pb_atoms, f_block_mode='u_only')
        node.expand()
        assert len(node.expansion_list) > 0

    def test_expansion_never_reproduces_parent_formula(self, u_w_pb_atoms):
        root = MCTSTreeNode(u_w_pb_atoms, f_block_mode='u_only')
        root.expand()
        child = root.expansion_list[0]
        child.add_parent(root)
        child.expand()
        parent_formula = root.get_chemical_formula()
        assert all(
            n.get_chemical_formula() != parent_formula
            for n in child.expansion_list
        )

    def test_rollout_rdos_uses_doscar_lookup_without_energy_calculator(self, u_w_pb_atoms):
        node = MCTSTreeNode(u_w_pb_atoms, f_block_mode='u_only')

        class StubDoscarLookup:
            def get_reward(self, formula):
                return 0.77

        reward = node.rollout(depth=0, energy_calculator=None, mode='rdos',
                              doscar_lookup=StubDoscarLookup())
        assert reward == pytest.approx(0.77)

    def test_rollout_unknown_mode_raises(self, u_w_pb_atoms):
        node = MCTSTreeNode(u_w_pb_atoms, f_block_mode='u_only')

        class StubEnergyCalculator:
            def calculate_energies(self, atoms):
                return -0.5, 0.02

        with pytest.raises(ValueError):
            node.rollout(depth=0, energy_calculator=StubEnergyCalculator(), mode='not_a_real_mode')
