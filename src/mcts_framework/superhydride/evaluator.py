"""
Property evaluator backed by a table of precomputed ELF descriptors.

Of the four descriptors the Tc fit needs, only H_f is free: it follows from the
composition. phi, phi* and H_DOS each require a converged ground-state DFT run
(an ELF cube from ``pp.x`` and a projected DOS from ``projwfc.x``), which is
far too slow to sit inside an MCTS iteration without a cache.

This evaluator therefore reads them from a CSV keyed by composition:

    formula,phi,phi_star,h_dos
    LaBeH8,0.527,0.738,0.724
    CaH6,0.811,0.811,0.793

A compound absent from the table gets NaN descriptors, which :class:`TcReward`
scores as 0.0 - so an unscreened candidate is ranked below every screened one
rather than crashing the search. Run the search, collect the compositions it
asked for, compute those with DFT, extend the table, and run again.

Computing the descriptors on demand from Quantum ESPRESSO is a separate
evaluator; this one is what makes the search runnable and testable without a
DFT stack.

(c) 2026. Triad National Security, LLC. All rights reserved.
"""

import logging
import math
import re
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from ..core.evaluator import PropertyEvaluator
from .structure import SuperhydrideStructure

logger = logging.getLogger(__name__)

#: Columns the table must provide, beyond the 'formula' key.
DESCRIPTOR_COLUMNS = ("phi", "phi_star", "h_dos")


def normalize_formula(formula: str) -> str:
    """
    Alphabetical element-count normalisation, for order-insensitive matching.

    'LaBeH8', 'BeLaH8' and 'H8LaBe' all normalise to 'BeH8La', so a table
    written in one convention still matches formulas produced in another.
    """
    counts: Dict[str, int] = {}
    for element, count in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if element:
            counts[element] = counts.get(element, 0) + (int(count) if count else 1)
    return "".join(
        element if counts[element] == 1 else f"{element}{counts[element]}"
        for element in sorted(counts)
    )


class DescriptorTableEvaluator(PropertyEvaluator):
    """
    Looks up phi, phi* and H_DOS by composition; computes H_f from the structure.

    H_f always comes from the structure rather than from the table, because the
    structure knows it exactly and a stale table column would silently poison
    the H_f^3 term in the fit.
    """

    def __init__(self, table_path: Optional[str] = None):
        """
        Args:
            table_path: CSV with columns formula, phi, phi_star, h_dos. If None
                or missing, every lookup returns NaN descriptors (and every
                reward is 0.0) - useful for a dry run that only enumerates
                which compositions the search wants.
        """
        super().__init__()
        self.table_path = table_path
        self._descriptors: Dict[str, Dict[str, float]] = {}

        if table_path is None:
            logger.warning(
                "No descriptor table provided; all ELF descriptors will be NaN "
                "and all rewards 0.0"
            )
            return

        path = Path(table_path)
        if not path.exists():
            logger.warning(
                "Descriptor table not found: %s; all rewards will be 0.0", path
            )
            return

        self._descriptors = self._load(pd.read_csv(path))
        logger.info("Loaded ELF descriptors for %d compounds from %s",
                    len(self._descriptors), path)

    @staticmethod
    def _load(table: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Index a descriptor table by normalised formula."""
        missing = [c for c in ("formula",) + DESCRIPTOR_COLUMNS if c not in table.columns]
        if missing:
            raise ValueError(
                f"Descriptor table is missing required column(s): {missing}. "
                f"Expected formula, {', '.join(DESCRIPTOR_COLUMNS)}."
            )

        indexed: Dict[str, Dict[str, float]] = {}
        for _, row in table.iterrows():
            key = normalize_formula(str(row["formula"]))
            indexed[key] = {name: float(row[name]) for name in DESCRIPTOR_COLUMNS}
        return indexed

    async def _compute(self, material: SuperhydrideStructure) -> Dict[str, float]:
        """
        Return {phi, phi_star, h_f, h_dos, formula} for a candidate.

        The lookup is a dict hit, so there is no blocking work to offload to a
        thread here - unlike the DFT-backed evaluator.
        """
        formula = material.get_formula()
        descriptors = self._descriptors.get(normalize_formula(formula))

        if descriptors is None:
            logger.debug("No ELF descriptors for %s; reward will be 0.0", formula)
            descriptors = {name: math.nan for name in DESCRIPTOR_COLUMNS}

        return {
            **descriptors,
            "h_f": material.get_hydrogen_fraction(),
            "formula": formula,
        }

    def __contains__(self, formula: str) -> bool:
        """True if the table carries descriptors for this composition."""
        return normalize_formula(formula) in self._descriptors
