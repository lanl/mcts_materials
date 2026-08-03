"""
Generic MCTS tree node (material-agnostic).

© 2025. Triad National Security, LLC. All rights reserved.
"""

from typing import TypeVar, Generic, List, Optional, Dict
import numpy as np

from .material import Material

M = TypeVar('M', bound=Material)


class SearchNode(Generic[M]):
    """
    Generic MCTS tree node.

    Completely material-agnostic - just tree structure + MCTS statistics.
    The material being evaluated is stored in the 'material' attribute.

    This class handles:
    - Tree structure (parent, children)
    - MCTS statistics (visits, rewards)
    - UCB/PUCT score calculations
    - Termination tracking
    - Expansion state management

    Type Parameters:
        M: Material type being searched

    Attributes:
        material: The material this node represents
        parent: Parent node (None for root)
        children: List of child nodes
        expandable: Whether node can still be expanded
        pending_children: Materials waiting to be added as children
        visits: Number of times this node has been visited
        total_reward: Sum of all rewards received (mean = total_reward/visits)
        own_reward: This node's OWN evaluated reward (its material's value),
            set once when the node is simulated; None until then.
        subtree_best: Best reward seen anywhere in this node's subtree,
            accumulated during backpropagation. Drives the termination
            heuristic only - do NOT use it as the node's own value (use
            own_reward for that).
        terminated: Whether node is marked as terminated
        visits_since_improvement: Visits since subtree_best last improved
        properties: Cached property values from PropertyEvaluator
    """

    def __init__(
        self,
        material: M,
        exploration_constant: float = 0.1,
        termination_limit: int = 60,
    ):
        """
        Initialize search node.

        Args:
            material: Material this node represents
            exploration_constant: UCB/PUCT exploration weight (c parameter)
            termination_limit: Number of visits without improvement before termination
        """
        # Material being evaluated
        self.material: M = material

        # Tree structure
        self.parent: Optional[SearchNode[M]] = None
        self.children: List[SearchNode[M]] = []

        # Expansion state
        self.expandable: bool = True
        self.pending_children: List[M] = []

        # MCTS statistics
        self.visits: int = 0
        self.total_reward: float = 0.0
        # subtree_best tracks the best reward seen anywhere in this node's
        # SUBTREE (updated during backpropagation, so on internal nodes it
        # reflects the best descendant, not this node's own evaluation). Its
        # sole job is the termination heuristic below.
        self.subtree_best: float = -np.inf
        # own_reward is THIS node's own simulated reward (set once by the
        # search when the node is evaluated); None until evaluated. Use this
        # to rank actual candidate materials, not subtree bests.
        self.own_reward: Optional[float] = None

        # Termination tracking
        self.terminated: bool = False
        self.visits_since_improvement: int = 0
        self.termination_limit: int = termination_limit

        # Hyperparameters
        self.exploration_constant: float = exploration_constant

        # Cached properties (filled by PropertyEvaluator)
        self.properties: Dict[str, float] = {}

    def get_ucb(self) -> float:
        """
        Calculate UCB1 score for selection.

        Formula: Q/N + c * sqrt(ln(N_parent) / N)

        Returns:
            UCB1 score. Returns +inf for unvisited nodes (ensures exploration).

        Notes:
            - Unvisited nodes (N=0) return +inf to prioritize exploration
            - Root node (no parent) returns 0.0
            - Terminated nodes should be filtered before calling this
        """
        if self.visits == 0:
            return float('inf')

        if self.parent is None:
            return 0.0

        exploitation = self.total_reward / self.visits
        exploration = self.exploration_constant * np.sqrt(
            np.log(self.parent.visits) / self.visits
        )

        return exploitation + exploration

    def get_puct(self, prior: float = 1.0) -> float:
        """
        Calculate PUCT score (AlphaZero-style selection).

        Formula: Q + c * prior * sqrt(N_parent) / (1 + N)

        Args:
            prior: Prior probability (default 1.0 for uniform prior).
                   In AlphaZero, this comes from a policy network.

        Returns:
            PUCT score

        Notes:
            - Unvisited nodes have Q=0 (not +inf like UCB1)
            - Exploration bonus decreases as 1/(1+N) instead of 1/sqrt(N)
            - Root node returns 0.0
        """
        if self.parent is None:
            return 0.0

        q_value = self.total_reward / self.visits if self.visits > 0 else 0.0
        u_value = (
            self.exploration_constant
            * prior
            * np.sqrt(self.parent.visits)
            / (1 + self.visits)
        )

        return q_value + u_value

    def update(self, reward: float) -> None:
        """
        Update node statistics after receiving a reward.

        Updates:
        - Increments visit count
        - Adds reward to total
        - Updates subtree_best if reward >= subtree_best (non-strict)
        - Tracks visits_since_improvement
        - Checks termination condition

        The improvement test is non-strict (reward >= subtree_best) so that a
        visit tying the current subtree best resets the no-improvement
        countdown. The comparison is per-node (this node's own subtree_best),
        not a global maximum.

        Args:
            reward: Reward value to incorporate
        """
        self.total_reward += reward
        self.visits += 1

        if reward >= self.subtree_best:
            self.subtree_best = reward
            self.visits_since_improvement = 0
        else:
            self.visits_since_improvement += 1

            # Check termination condition
            if self.visits_since_improvement >= self.termination_limit:
                self.terminated = True

    def add_child(self, child: 'SearchNode[M]') -> None:
        """
        Add a child node.

        Args:
            child: Child node to add

        Side effects:
            - Sets child.parent to self
            - Appends child to self.children
        """
        child.parent = self
        self.children.append(child)

    def is_leaf(self) -> bool:
        """
        Check if this is a leaf node (no children yet).

        Returns:
            True if node has no children
        """
        return len(self.children) == 0

    def is_fully_expanded(self) -> bool:
        """
        Check if all pending children have been added to the tree.

        Returns:
            True if no more children can be added
        """
        return not self.expandable

    def get_mean_reward(self) -> float:
        """
        Calculate mean reward (Q value).

        Returns:
            Mean reward, or 0.0 if never visited
        """
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits

    def get_subtree_size(self) -> int:
        """
        Count total nodes in subtree (including self).

        Returns:
            Number of nodes in subtree
        """
        return 1 + sum(child.get_subtree_size() for child in self.children)

    def get_depth(self) -> int:
        """
        Calculate depth of this node (distance from root).

        Returns:
            Depth (root has depth 0)
        """
        depth = 0
        current = self.parent
        while current is not None:
            depth += 1
            current = current.parent
        return depth

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"SearchNode("
            f"material={self.material.get_identifier()}, "
            f"visits={self.visits}, "
            f"reward={self.total_reward:.3f}, "
            f"children={len(self.children)})"
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        mean_reward = self.get_mean_reward()
        return (
            f"{self.material.get_identifier()}: "
            f"N={self.visits}, "
            f"Q={mean_reward:.3f}, "
            f"subtree_best={self.subtree_best:.3f}"
        )
