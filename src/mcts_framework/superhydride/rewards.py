"""
Reward function for the superhydride search: the ELF-based Tc estimator.

The reward is the symbolic-regression fit of Belli, Torres, Contreras-Garcia
and Zurek, Ann. Phys. (Berlin) 2025, 537, e00280, Equation (2):

    Tc = 422.2 * (27/4) * (phi*^2 - phi*^3) * H_f^3 * (phi * H_DOS)^(1/3) + 5.5

with

    phi     networking value - highest ELF isovalue whose isosurface spans the
            crystal in all three directions.
    phi*    molecularity index - highest ELF isovalue at which two hydrogen
            atoms connect.
    H_f     hydrogen fraction, N_H / N_total.
    H_DOS   hydrogen-projected share of the DOS at the Fermi level.

Fitted on 244 hydrides (119 binary, 125 ternary): RMSE 41 K, mean absolute
error 31 K, largest deviation 108 K.

Stability is deliberately not part of this reward. The search ranks candidates
by estimated Tc alone; screening the survivors for thermodynamic and dynamic
stability is a separate, later step.

Three properties of the fit that matter for using it as a reward:

    * The (phi*^2 - phi*^3) factor peaks at phi* = 2/3 (the paper quotes the
      empirical optimum at 0.68) and the 27/4 normalises that peak to 1. So
      422.2 K is the whole dynamic range, and TC_MAX_K below is exact, not a
      guess.
    * Tc is therefore bounded to [5.5, 427.7] K. The fit cannot express a
      lower or a higher value, whatever the descriptors.
    * It is monotone in phi, H_f and H_DOS but NOT in phi*: pushing phi* past
      2/3 towards 1 lowers the estimate. That is the physics - intact H2
      molecules put their states away from E_F - and the search must be
      allowed to move phi* downwards.

(c) 2026. Triad National Security, LLC. All rights reserved.
"""

import math
from typing import Dict, List

from ..core.reward import RewardFunction

# --- Fit constants (Ann. Phys. 2025, 537, e00280, Eq. 2) ------------------

#: Prefactor of the fit, in kelvin.
BELLI2025_PREFACTOR_K = 422.2
#: Intercept of the fit, in kelvin.
BELLI2025_INTERCEPT_K = 5.5
#: Normalisation of the molecularity factor, 1 / max(phi*^2 - phi*^3) = 27/4.
MOLECULARITY_NORM = 27.0 / 4.0

#: Where (phi*^2 - phi*^3) peaks. Exactly 2/3; the paper quotes 0.68.
PHI_STAR_OPTIMUM = 2.0 / 3.0

#: Published error of the fit on its own 244-compound dataset.
BELLI2025_RMSE_K = 41.0
BELLI2025_MAE_K = 31.0
BELLI2025_MAX_ERROR_K = 108.0

#: Analytic bounds of Eq. 2, attained at phi* = 2/3 and phi = H_f = H_DOS = 1
#: (max) and at any vanishing factor (min).
TC_MIN_K = BELLI2025_INTERCEPT_K
TC_MAX_K = BELLI2025_PREFACTOR_K + BELLI2025_INTERCEPT_K  # 427.7 K

#: Descriptors the fit consumes.
DESCRIPTOR_NAMES = ("phi", "phi_star", "h_f", "h_dos")


def _check_unit_interval(**values: float) -> None:
    """Every descriptor is an ELF value or a fraction, so all live in [0, 1]."""
    for name, value in sorted(values.items()):
        if value is None or not math.isfinite(value):
            raise ValueError(f"{name} is missing or not finite (got {value!r})")
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{name} must lie in [0, 1] (an ELF value or a fraction), got {value!r}"
            )


def belli2025_tc(phi: float, phi_star: float, h_f: float, h_dos: float) -> float:
    """
    Estimated Tc in kelvin from Eq. 2 of Ann. Phys. 2025, 537, e00280.

    Args:
        phi: networking value, in [0, 1].
        phi_star: molecularity index, in [0, 1].
        h_f: hydrogen fraction, in [0, 1].
        h_dos: hydrogen share of the DOS at E_F, in [0, 1].

    Returns:
        Tc in kelvin, in [5.5, 427.7]. Carries a published RMSE of 41 K, which
        is the fit's error on its own training set and therefore a floor: it
        says nothing about how far these descriptors sit from the fitted ones.
        A 250 K estimate is a candidate worth a phonon calculation, not a 250 K
        material.

    Raises:
        ValueError: if any descriptor is missing, non-finite or outside [0, 1].
    """
    _check_unit_interval(phi=phi, phi_star=phi_star, h_f=h_f, h_dos=h_dos)
    molecularity = phi_star**2 - phi_star**3
    return (
        BELLI2025_PREFACTOR_K
        * MOLECULARITY_NORM
        * molecularity
        * h_f**3
        * (phi * h_dos) ** (1.0 / 3.0)
        + BELLI2025_INTERCEPT_K
    )


class TcReward(RewardFunction):
    """
    Reward = estimated Tc from Eq. 2, optionally normalised to (0, 1].

    Normalising divides by TC_MAX_K = 427.7 K, the analytic maximum of the fit.
    That is a monotone rescaling, so the ranking is untouched, but it keeps
    rewards O(1) - which UCB1's exploration constant (0.1 by default) assumes.
    With raw kelvin the exploration term is swamped by reward differences of
    tens of kelvin and the search degenerates to greedy.

    A candidate whose descriptors are missing or unusable (a DFT run that did
    not converge, a compound absent from the descriptor table) scores 0.0.
    That is below the reward of any real candidate, since Eq. 2 cannot return
    less than 5.5 K, so failures sink without needing a sentinel value.
    """

    def __init__(self, normalize: bool = True):
        """
        Args:
            normalize: Divide the Tc estimate by TC_MAX_K so rewards land in
                (0, 1]. Set False to reward in raw kelvin, and retune
                exploration_constant accordingly.
        """
        self.normalize = normalize

    def compute_reward(self, properties: Dict[str, float]) -> float:
        try:
            tc = belli2025_tc(
                phi=properties["phi"],
                phi_star=properties["phi_star"],
                h_f=properties["h_f"],
                h_dos=properties["h_dos"],
            )
        except (KeyError, TypeError, ValueError):
            # Missing, non-numeric or out-of-range descriptors: not a candidate
            # we can score. Rank it below every scorable one.
            return 0.0
        return tc / TC_MAX_K if self.normalize else tc

    def get_property_names(self) -> List[str]:
        return list(DESCRIPTOR_NAMES)


def create_superhydride_reward(normalize: bool = True) -> RewardFunction:
    """
    Factory for the superhydride reward, mirroring the other material packages.

    Args:
        normalize: see :class:`TcReward`.
    """
    return TcReward(normalize=normalize)
