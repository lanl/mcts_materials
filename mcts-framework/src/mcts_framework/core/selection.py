"""
Child-selection strategies for the MCTS selection phase.

Each strategy decides which child to descend into at a node during the
selection walk from root to leaf. All strategies operate on SearchNode
statistics (visits, rewards) via get_ucb()/get_puct() and never touch the
underlying material - they are fully material-agnostic.

Formulas preserved from the validated mcts_crystal implementation:
- UCB1:      Q/N + c * sqrt(ln(N_parent) / N)
- PUCT:      Q + c * prior * sqrt(N_parent) / (1 + N)
- Boltzmann: P(i) proportional to exp(UCB1_i / T)

© 2025. Triad National Security, LLC. All rights reserved.
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Optional
import random

import numpy as np

from .material import Material
from .search_node import SearchNode

M = TypeVar('M', bound=Material)


class AllChildrenTerminated(Exception):
    """Raised when every child of a node is terminated (dead branch)."""


def _valid_children(parent: SearchNode[M]) -> List[SearchNode[M]]:
    """Return non-terminated children, or raise if none remain."""
    valid = [c for c in parent.children if not c.terminated]
    if not valid:
        raise AllChildrenTerminated(
            f"All {len(parent.children)} children of "
            f"{parent.material.get_identifier()} are terminated"
        )
    return valid


class SelectionStrategy(ABC, Generic[M]):
    """
    Abstract base class for child-selection strategies.

    A strategy picks exactly one child from a parent node's children during
    the MCTS selection phase. Terminated children are always excluded.
    """

    @abstractmethod
    def select_child(self, parent: SearchNode[M]) -> SearchNode[M]:
        """
        Select one child from parent's children.

        Args:
            parent: Parent node with at least one child.

        Returns:
            The selected child node.

        Raises:
            AllChildrenTerminated: If every child is terminated.
        """
        raise NotImplementedError


class UCB1(SelectionStrategy[M]):
    """
    Classic UCB1 selection - deterministic argmax of the UCB1 score.

    Unvisited children have UCB1 = +inf, so they are always explored before
    any visited child. The exploration/exploitation tradeoff comes entirely
    from the exploration constant baked into SearchNode.get_ucb().
    """

    def select_child(self, parent: SearchNode[M]) -> SearchNode[M]:
        valid = _valid_children(parent)
        return max(valid, key=lambda c: c.get_ucb())


class PUCT(SelectionStrategy[M]):
    """
    AlphaZero-style PUCT selection - deterministic argmax of the PUCT score.

    Uses a uniform prior (1/num_children) since there is no learned policy
    network. Unlike UCB1, an unvisited child has Q=0 rather than +inf; its
    exploration bonus is largest when N=0 (via the 1/(1+N) term).
    """

    def select_child(self, parent: SearchNode[M]) -> SearchNode[M]:
        valid = _valid_children(parent)
        # Uniform prior over ALL children (matches mcts_crystal semantics).
        prior = 1.0 / len(parent.children) if parent.children else 1.0
        return max(valid, key=lambda c: c.get_puct(prior))


class EpsilonGreedy(SelectionStrategy[M]):
    """
    Textbook epsilon-greedy selection.

    With probability (1 - epsilon), picks argmax(UCB1) (identical to UCB1);
    with probability epsilon, picks a child uniformly at random, ignoring the
    UCB1 magnitude entirely.
    """

    def __init__(self, epsilon: float = 0.2, rng: Optional[random.Random] = None):
        """
        Args:
            epsilon: Probability of taking a uniform-random exploratory step.
            rng: Optional random.Random for reproducibility. Defaults to the
                global random module state.
        """
        self.epsilon = epsilon
        self._rng = rng or random

    def select_child(self, parent: SearchNode[M]) -> SearchNode[M]:
        valid = _valid_children(parent)
        if self._rng.random() < self.epsilon:
            return self._rng.choice(valid)
        return max(valid, key=lambda c: c.get_ucb())


class Boltzmann(SelectionStrategy[M]):
    """
    Softmax/Boltzmann exploration - always stochastic.

    Picks child i with probability proportional to exp(UCB1_i / T). Lower T
    biases toward greedy argmax; higher T biases toward uniform. Unvisited
    children (UCB1 = +inf) are always explored first: if any are present, one
    is chosen uniformly among them.
    """

    def __init__(self, temperature: float = 1.0, rng: Optional[random.Random] = None):
        """
        Args:
            temperature: Softmax temperature T (must be > 0).
            rng: Optional random.Random for reproducibility.
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = temperature
        self._rng = rng or random

    def select_child(self, parent: SearchNode[M]) -> SearchNode[M]:
        valid = _valid_children(parent)

        scores = np.array([c.get_ucb() for c in valid], dtype=float)

        # Unvisited children have +inf UCB - explore one of them first.
        inf_mask = np.isinf(scores)
        if inf_mask.any():
            unvisited = [c for c, is_inf in zip(valid, inf_mask) if is_inf]
            return self._rng.choice(unvisited)

        # Numerically stable softmax over UCB1 / T.
        logits = scores / self.temperature
        logits -= logits.max()
        weights = np.exp(logits)
        probs = weights / weights.sum()

        # random.Random has no weighted choice; use cumulative sampling.
        r = self._rng.random()
        cumulative = np.cumsum(probs)
        idx = int(np.searchsorted(cumulative, r, side="right"))
        idx = min(idx, len(valid) - 1)  # guard floating-point edge
        return valid[idx]


# Registry mapping config strings to strategy factories.
_STRATEGY_FACTORIES = {
    "ucb1": lambda **kw: UCB1(),
    "puct": lambda **kw: PUCT(),
    "epsilon_greedy": lambda **kw: EpsilonGreedy(epsilon=kw.get("epsilon", 0.2)),
    "boltzmann": lambda **kw: Boltzmann(temperature=kw.get("temperature", 1.0)),
}


def create_selection_strategy(
    mode: str,
    epsilon: float = 0.2,
    temperature: float = 1.0,
) -> SelectionStrategy:
    """
    Factory: build a SelectionStrategy from a config-style mode string.

    Args:
        mode: One of 'ucb1', 'puct', 'epsilon_greedy', 'boltzmann'.
        epsilon: Exploration rate for epsilon_greedy.
        temperature: Softmax temperature for boltzmann.

    Returns:
        The corresponding SelectionStrategy instance.

    Raises:
        ValueError: If mode is not recognized.
    """
    if mode not in _STRATEGY_FACTORIES:
        raise ValueError(
            f"Unknown selection mode: {mode!r}. "
            f"Valid modes: {sorted(_STRATEGY_FACTORIES)}"
        )
    return _STRATEGY_FACTORIES[mode](epsilon=epsilon, temperature=temperature)
