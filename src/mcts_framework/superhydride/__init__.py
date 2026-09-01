"""
Ternary superhydride support for MCTS materials search.

Searches the host sublattice of a fixed hydride template, scoring candidates by
the ELF-based Tc estimator of Belli et al., Ann. Phys. (Berlin) 2025, 537,
e00280 (Equation 2). Stability is not part of the reward at this stage.

    SuperhydrideStructure     the material (ASE Atoms + crystallographic id)
    HostSubstitutionMoves     the expansion rule (one host swap per move)
    DescriptorTableEvaluator  phi / phi* / H_DOS from a precomputed table
    QuantumEspressoEvaluator  the same descriptors computed with QE (qe/)
    TcReward                  Eq. 2, optionally normalised to (0, 1]

(c) 2026. Triad National Security, LLC. All rights reserved.
"""

from . import elements
from .descriptors import (
    ELFDescriptors,
    compute_descriptors,
    descriptors_from_cube,
    hydrogen_fraction,
    molecularity_index,
    networking_value,
    read_elf_cube,
)
from .evaluator import DescriptorTableEvaluator
from .moves import HostSubstitutionMoves
from .rewards import (
    BELLI2025_MAE_K,
    BELLI2025_MAX_ERROR_K,
    BELLI2025_RMSE_K,
    PHI_STAR_OPTIMUM,
    TC_MAX_K,
    TC_MIN_K,
    TcReward,
    belli2025_tc,
    create_superhydride_reward,
)
from .structure import SuperhydrideStructure

__all__ = [
    "elements",
    "SuperhydrideStructure",
    "HostSubstitutionMoves",
    "DescriptorTableEvaluator",
    "TcReward",
    "create_superhydride_reward",
    "belli2025_tc",
    "ELFDescriptors",
    "compute_descriptors",
    "descriptors_from_cube",
    "hydrogen_fraction",
    "networking_value",
    "molecularity_index",
    "read_elf_cube",
    "BELLI2025_RMSE_K",
    "BELLI2025_MAE_K",
    "BELLI2025_MAX_ERROR_K",
    "PHI_STAR_OPTIMUM",
    "TC_MAX_K",
    "TC_MIN_K",
]
