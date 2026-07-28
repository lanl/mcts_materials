"""
Abstract base class for reward functions.

© 2025. Triad National Security, LLC. All rights reserved.
"""

from abc import ABC, abstractmethod
from typing import Dict, List


class RewardFunction(ABC):
    """
    Abstract base class for reward functions.

    Transforms material properties into a scalar reward signal for MCTS.
    The MCTS algorithm maximizes this reward.

    Examples:
        - Minimize energy: reward = -e_form
        - Maximize stability: reward = -tanh(120*(e_hull - 0.05))
        - Multi-objective: reward = w1*f1(p1) + w2*f2(p2) + ...
    """

    @abstractmethod
    def compute_reward(self, properties: Dict[str, float]) -> float:
        """
        Compute reward from material properties.

        Args:
            properties: Dictionary of property_name -> value from PropertyEvaluator

        Returns:
            Scalar reward (higher = better). The MCTS algorithm maximizes
            this value.

        Examples:
            >>> properties = {"e_form": -1.2, "e_above_hull": 0.03}
            >>> reward_fn.compute_reward(properties)
            0.85

        Notes:
            - Return -inf for invalid materials to exclude them
            - Rewards should be roughly normalized (e.g., [-1, 1] range)
            - For multi-objective, use weighted sum or Pareto ranking
        """
        pass

    def get_property_names(self) -> List[str]:
        """
        Return list of property names required by this reward function.

        Used by PropertyEvaluator to know which properties to compute.
        Override if your reward function needs specific properties.

        Returns:
            List of required property names. Empty list means no specific
            requirements (reward function will handle missing properties).

        Examples:
            >>> reward_fn.get_property_names()
            ['e_above_hull']
        """
        return []
