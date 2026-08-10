"""
DOSCAR-derived electronic-structure reward (rDOS) lookup.

Ported from mcts_crystal.doscar_utils. rDOS is computed in real time from raw
DOSCAR peak data as a Gaussian-weighted sum of peak intensity near the Fermi
level:

    r_DOS = sum_peaks (PEAK_HEIGHT / PEAK_WIDTH) * exp(-0.5 * (PEAK_ENERGY/sigma)^2)

with sigma = 0.5 eV (physics-informed, not a tunable hyperparameter).

© 2026. Triad National Security, LLC. All rights reserved.
"""

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Gaussian width (eV) for weighting peaks by proximity to the Fermi level.
# Physics-informed constant - do NOT sweep as a hyperparameter.
_SIGMA = 0.5

# Element categories for the ternary DOSCAR formula conversion.
_GROUP_IV = {"Si", "Ge", "Sn", "Pb"}
_LANTHANIDES = [
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
]
_ACTINIDES = ["U"]
_F_BLOCK = set(_LANTHANIDES + _ACTINIDES)
_TRANSITION_METALS = {
    "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
}


class DoscarRewardLookup:
    """Loads raw DOSCAR peak data and computes rDOS rewards by formula."""

    def __init__(self, peaks_file: Optional[str] = None):
        """
        Args:
            peaks_file: Path to the raw DOSCAR peaks CSV. If None or missing,
                all rewards default to 0.0 (with a warning).
        """
        self.rewards_dict: dict[str, float] = {}

        if peaks_file is None:
            logger.warning("No DOSCAR peaks file provided; rDOS rewards will be 0.0")
            return

        peaks_path = Path(peaks_file)
        if not peaks_path.exists():
            logger.warning("DOSCAR peaks file not found: %s; rDOS = 0.0", peaks_path)
            return

        try:
            self.rewards_dict = self._compute_rewards(pd.read_csv(peaks_path))
            logger.info("Computed %d DOSCAR rewards", len(self.rewards_dict))
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Error computing DOSCAR rewards: %s; rDOS = 0.0", exc)

    @staticmethod
    def _compute_rewards(peaks_df: pd.DataFrame) -> dict[str, float]:
        """
        Compute the rDOS sum per compound from raw peak rows.

        Prefers core compounds (no '_valence' suffix), including a
        valence-only compound only when its core counterpart is absent.

        The '_valence' suffix is stripped from the reward-dict key so that a
        plain formula lookup (e.g. 'U-Pb-W') resolves to a valence-only
        compound's reward. (The original mcts_crystal code kept the suffix in
        the key, which made the "valence fallback" path unreachable - a latent
        bug fixed here. The published U-only study never hit this path since
        all its compounds had core entries, so results are unaffected.)
        """
        core = peaks_df[~peaks_df["COMPOUND_NAME"].str.endswith("_valence")]
        valence = peaks_df[peaks_df["COMPOUND_NAME"].str.endswith("_valence")]

        valence_bases = valence["COMPOUND_NAME"].str.replace("_valence", "").unique()
        core_names = core["COMPOUND_NAME"].unique()
        missing_bases = set(valence_bases) - set(core_names)
        valence_to_include = valence[
            valence["COMPOUND_NAME"].str.replace("_valence", "").isin(missing_bases)
        ]
        filtered = pd.concat([core, valence_to_include])

        results: dict[str, float] = {}
        for cname, group in filtered.groupby("COMPOUND_NAME"):
            exp_factor = np.exp(-0.5 * (group["PEAK_ENERGY"] / _SIGMA) ** 2)
            contrib = (group["PEAK_HEIGHT"] / group["PEAK_WIDTH"]) * exp_factor
            # Normalize key: strip the '_valence' suffix so plain-formula
            # lookups resolve to it.
            key = str(cname).replace("_valence", "")
            results[key] = float(contrib.sum())
        return results

    @staticmethod
    def convert_formula_to_doscar_format(formula: str) -> Optional[str]:
        """
        Convert an MCTS formula (e.g. 'Ti6Si6Ce') to DOSCAR format
        ('Ce-Si-Ti' = fblock-groupIV-metal). Returns None if the formula is
        not a recognizable f-block / Group IV / transition-metal ternary.
        """
        matches = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
        elements = {el: (int(c) if c else 1) for el, c in matches if el}

        if len(elements) != 3:
            return None

        f_elem = g_iv_elem = metal_elem = None
        for el in elements:
            if el in _F_BLOCK:
                f_elem = el
            elif el in _GROUP_IV:
                g_iv_elem = el
            elif el in _TRANSITION_METALS:
                metal_elem = el

        if f_elem is None or g_iv_elem is None or metal_elem is None:
            return None
        return f"{f_elem}-{g_iv_elem}-{metal_elem}"

    def get_reward(self, formula: str) -> float:
        """Return the rDOS reward for a formula, or 0.0 if unavailable."""
        doscar_formula = self.convert_formula_to_doscar_format(formula)
        if doscar_formula is None:
            return 0.0
        return self.rewards_dict.get(doscar_formula, 0.0)
