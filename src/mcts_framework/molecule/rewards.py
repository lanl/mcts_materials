"""
Reward functions for molecular search.

Single-objective rewards target one predicted property; MultiObjectiveReward
combines several via a weighted sum. All operate purely on the property dict
produced by MoleculeEvaluator, so they need neither RDKit nor
molecule-modifier and are fully unit-testable.

© 2026. Triad National Security, LLC. All rights reserved.
"""

from typing import Dict, List, Optional

from ..core.reward import RewardFunction


class MeltingPointReward(RewardFunction):
    """
    Reward that favors HIGH melting points, linear in melting point.

    The melting point is linearly rescaled from the reference window
    [min_temp, max_temp] onto [0, 1]:

        reward = (melting_point - min_temp) / (max_temp - min_temp)

    The result is NOT clamped: melting points above max_temp score > 1 and
    those below min_temp score < 0, preserving a strict linear signal
    everywhere. Reward increases with melting point, so higher-melting
    molecules are preferred. Defaults span a broad organic-solid range
    (min_temp=200 K, max_temp=700 K); tune to the dataset.
    """

    def __init__(self, min_temp: float = 0.0, max_temp: float = 700.0):
        if max_temp <= min_temp:
            raise ValueError(
                f"max_temp ({max_temp}) must exceed min_temp ({min_temp})"
            )
        self.min_temp = min_temp
        self.max_temp = max_temp

    def compute_reward(self, properties: Dict[str, float]) -> float:
        mp = properties["melting_point"]
        return float((mp - self.min_temp) / (self.max_temp - self.min_temp))

    def get_property_names(self) -> List[str]:
        return ["melting_point"]


class H2CapacityReward(RewardFunction):
    """
    Reward that favors HIGH H2 storage capacity, linearly normalized.

        reward = h2_capacity / scale
    """

    def __init__(self, scale: float = 10.0):
        if scale <= 0:
            raise ValueError(f"scale must be > 0, got {scale}")
        self.scale = scale

    def compute_reward(self, properties: Dict[str, float]) -> float:
        return float(properties["h2_capacity"] / self.scale)

    def get_property_names(self) -> List[str]:
        return ["h2_capacity"]


class SynthesizabilityReward(RewardFunction):
    """
    Reward that favors EASILY synthesizable molecules.

    molecule-modifier's synthesizability score runs 1 (easy) to 10 (hard);
    this maps it linearly to roughly [+1 (easy) .. -1 (hard)]:

        reward = (5.5 - synthesizability) / 4.5
    """

    def compute_reward(self, properties: Dict[str, float]) -> float:
        score = properties["synthesizability"]
        return float((5.5 - score) / 4.5)

    def get_property_names(self) -> List[str]:
        return ["synthesizability"]


class MultiObjectiveReward(RewardFunction):
    """
    Weighted sum of the single-objective rewards above.

    weights maps property names to coefficients, e.g.
        {"h2_capacity": 1.0, "melting_point": 0.5, "synthesizability": 0.3}
    A property's sub-reward is computed by its dedicated reward class, then
    scaled by the weight and summed. Unknown property names raise ValueError.
    """

    def __init__(self, weights: Dict[str, float]):
        if not weights:
            raise ValueError("MultiObjectiveReward requires at least one weight")
        self._sub_rewards: Dict[str, RewardFunction] = {
            "melting_point": MeltingPointReward(),
            "h2_capacity": H2CapacityReward(),
            "synthesizability": SynthesizabilityReward(),
        }
        unknown = set(weights) - set(self._sub_rewards)
        if unknown:
            raise ValueError(
                f"Unknown objective(s) {sorted(unknown)}; "
                f"valid: {sorted(self._sub_rewards)}"
            )
        self.weights = weights

    def compute_reward(self, properties: Dict[str, float]) -> float:
        total = 0.0
        for prop, weight in self.weights.items():
            total += weight * self._sub_rewards[prop].compute_reward(properties)
        return total

    def get_property_names(self) -> List[str]:
        return list(self.weights.keys())


def create_molecule_reward(
    objective: str,
    weights: Optional[Dict[str, float]] = None,
) -> RewardFunction:
    """
    Factory: build a molecule reward from a config objective string.

    Args:
        objective: 'melting_point', 'h2_capacity', 'synthesizability', or
            'multi_objective'.
        weights: Required for 'multi_objective'.

    Raises:
        ValueError: on unknown objective or missing weights.
    """
    if objective == "melting_point":
        return MeltingPointReward()
    if objective == "h2_capacity":
        return H2CapacityReward()
    if objective == "synthesizability":
        return SynthesizabilityReward()
    if objective == "multi_objective":
        if not weights:
            raise ValueError("objective='multi_objective' requires weights")
        return MultiObjectiveReward(weights)
    raise ValueError(f"Unknown objective: {objective!r}")