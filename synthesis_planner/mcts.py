"""A small PUCT-style MCTS engine for synthesis routes."""

from __future__ import annotations

import math
import random

from .grammar import apply_action, expand_state, rollout_completion
from .schema import Action, PlanningState
from .scoring import evaluate_state


class TreeNode:
    def __init__(self, state: PlanningState, prior: float = 1.0, parent: "TreeNode | None" = None):
        self.state = state
        self.prior = prior
        self.parent = parent
        self.children: list[TreeNode] = []
        self.visit_count = 0
        self.total_value = 0.0
        self.expanded = False
        self.terminal_routes = []

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


class MonteCarloTreeSearch:
    def __init__(self, exploration_constant: float = 1.4, rollout_count: int = 8, seed: int | None = None):
        self.exploration_constant = exploration_constant
        self.rollout_count = rollout_count
        self.rng = random.Random(seed)

    def run(self, root_state: PlanningState, analogs, candidate_precursor_sets, iterations: int):
        root = TreeNode(root_state)

        for _ in range(iterations):
            node = self._select(root)
            if not node.state.is_terminal:
                self._expand(node, analogs, candidate_precursor_sets)
                if node.children:
                    node = self.rng.choice(node.children)

            value, route = self._simulate(node.state, analogs, candidate_precursor_sets)
            self._backup(node, value, route)

        return root

    def _select(self, node: TreeNode) -> TreeNode:
        current = node
        while current.children:
            current = max(current.children, key=lambda child: self._puct_score(current, child))
        return current

    def _expand(self, node: TreeNode, analogs, candidate_precursor_sets) -> None:
        if node.expanded or node.state.is_terminal:
            return
        actions = expand_state(node.state, analogs, candidate_precursor_sets)
        for action in actions:
            child_state = apply_action(node.state, action, analogs)
            node.children.append(TreeNode(child_state, prior=action.prior, parent=node))
        node.expanded = True

    def _simulate(self, state: PlanningState, analogs, candidate_precursor_sets):
        best_route = None
        best_value = float("-inf")
        for _ in range(self.rollout_count):
            terminal_state = rollout_completion(state, analogs, candidate_precursor_sets, self.rng)
            route = evaluate_state(terminal_state, analogs)
            if route.mcts_value > best_value:
                best_value = route.mcts_value
                best_route = route
        return best_value, best_route

    def _backup(self, node: TreeNode, value: float, route) -> None:
        current = node
        while current is not None:
            current.visit_count += 1
            current.total_value += value
            if route is not None:
                current.terminal_routes.append(route)
            current = current.parent

    def _puct_score(self, parent: TreeNode, child: TreeNode) -> float:
        exploration = self.exploration_constant * child.prior * math.sqrt(parent.visit_count + 1) / (1 + child.visit_count)
        return child.q_value + exploration
