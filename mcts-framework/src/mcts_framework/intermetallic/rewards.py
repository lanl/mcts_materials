"""
Reward functions for intermetallic search.

Three formulations, matching the validated mcts_crystal rollout methods:

    EhullReward       : -tanh(120 * (e_hull - 0.05))          ('ehull')
    EhullRdosReward   : beta * ehull_term + gamma * r_DOS      ('ehull_rdos')
    RdosReward        : r_DOS                                  ('rdos')

The ehull sharpness (120) and stability threshold (0.05 eV/atom) are
physics-informed constants, not tunable hyperparameters. beta and gamma are
the composite-score weights (defaults 1.0 / 0.0001 from the published study).

Properties consumed:
    e_above_hull : from the MACE/Materials Project evaluator.
    formula      : plain composition string, used to look up r_DOS.
    rdos         : optional precomputed r_DOS; if absent, looked up from the
                   DoscarRewardLookup using 'formula'.

© 2025. Triad National Security, LLC. All rights reserved.
"""

from typing import Dict, List, Optional

import numpy as np

from ..core.reward import RewardFunction
from .doscar import DoscarRewardLookup

# Physics-informed constants (do NOT sweep).
_EHULL_SHARPNESS = 120.0
_EHULL_THRESHOLD = 0.05  # eV/atom


def ehull_reward(e_hull: float) -> float:
    """
    Sharp tanh reward for energy above hull.

    f(E_hull) = -tanh(120 * (E_hull - 0.05)); ~+1 for stable (E_hull~0),
    0 at the 0.05 eV/atom boundary, ~-1 for unstable (E_hull>=0.1).
    """
    return float(-np.tanh(_EHULL_SHARPNESS * (e_hull - _EHULL_THRESHOLD)))


def _resolve_rdos(
    properties: Dict[str, float],
    doscar_lookup: DoscarRewardLookup,
) -> float:
    """
    Resolve the rDOS value for a set of properties.

    Prefers a precomputed 'rdos' entry; otherwise looks it up from the DOSCAR
    data by 'formula'. Returns 0.0 when neither is available.
    """
    if "rdos" in properties:
        return properties["rdos"]
    formula = properties.get("formula")
    if formula is None:
        return 0.0
    return doscar_lookup.get_reward(str(formula))


class EhullReward(RewardFunction):
    """Energy-above-hull reward only ('ehull')."""

    def compute_reward(self, properties: Dict[str, float]) -> float:
        return ehull_reward(properties["e_above_hull"])

    def get_property_names(self) -> List[str]:
        return ["e_above_hull"]


class RdosReward(RewardFunction):
    """rDOS-only reward ('rdos'); no MACE/Materials Project needed."""

    def __init__(self, doscar_lookup: DoscarRewardLookup):
        self.doscar_lookup = doscar_lookup

    def compute_reward(self, properties: Dict[str, float]) -> float:
        return _resolve_rdos(properties, self.doscar_lookup)

    def get_property_names(self) -> List[str]:
        return ["formula"]


class EhullRdosReward(RewardFunction):
    """
    Composite reward ('ehull_rdos'): beta * ehull_reward + gamma * r_DOS.

    Defaults beta=1.0, gamma=0.0001 reproduce the published study.
    """

    def __init__(
        self,
        doscar_lookup: DoscarRewardLookup,
        beta: float = 1.0,
        gamma: float = 0.0001,
    ):
        self.doscar_lookup = doscar_lookup
        self.beta = beta
        self.gamma = gamma

    def compute_reward(self, properties: Dict[str, float]) -> float:
        ehull_term = ehull_reward(properties["e_above_hull"])
        rdos = _resolve_rdos(properties, self.doscar_lookup)
        return self.beta * ehull_term + self.gamma * rdos

    def get_property_names(self) -> List[str]:
        return ["e_above_hull", "formula"]


def create_intermetallic_reward(
    rollout_method: str,
    doscar_lookup: Optional[DoscarRewardLookup] = None,
    beta: float = 1.0,
    gamma: float = 0.0001,
) -> RewardFunction:
    """
    Factory: build the reward function for a given rollout method.

    Args:
        rollout_method: One of 'ehull', 'ehull_rdos', 'rdos'.
        doscar_lookup: Required for 'rdos' and 'ehull_rdos'.
        beta, gamma: Composite weights for 'ehull_rdos'.

    Raises:
        ValueError: On unknown method or missing doscar_lookup.
    """
    if rollout_method == "ehull":
        return EhullReward()
    if rollout_method == "rdos":
        if doscar_lookup is None:
            raise ValueError("rollout_method='rdos' requires a doscar_lookup")
        return RdosReward(doscar_lookup)
    if rollout_method == "ehull_rdos":
        if doscar_lookup is None:
            raise ValueError("rollout_method='ehull_rdos' requires a doscar_lookup")
        return EhullRdosReward(doscar_lookup, beta=beta, gamma=gamma)
    raise ValueError(f"Unknown rollout_method: {rollout_method!r}")
