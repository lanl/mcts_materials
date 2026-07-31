"""
Property evaluator for intermetallic structures.

Wraps the validated MACE + Materials Project energy pipeline from
mcts_crystal.energy_calculator as a framework PropertyEvaluator. Computes:

    e_form       : formation energy (eV/atom) via MACE relaxation
    e_above_hull : energy above the Materials Project convex hull (eV/atom)
    formula      : composition string (added so rDOS rewards can be looked up)

A CSV cache (schema: name, e_form, e_above_hull, e_decomp, data_quality) is
consulted first; only unseen compositions trigger a live MACE/MP calculation,
and results are appended back. Heavy deps (mace-torch, pymatgen,
matbench-discovery) are imported lazily so cache-only / rdos-only workflows
don't need them.

The synchronous heavy work runs in a thread-pool executor via the async
PropertyEvaluator interface, keeping the MCTS event loop responsive.

© 2025. Triad National Security, LLC. All rights reserved.
"""

import asyncio
import logging
import re
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from ..core.evaluator import PropertyEvaluator
from .structure import IntermetallicStructure

logger = logging.getLogger(__name__)

# Penalty e_above_hull (eV/atom) for compounds with missing/invalid MP data.
UnstablePenalty = 10.0

_CACHE_COLUMNS = ["name", "e_form", "e_above_hull", "e_decomp", "data_quality"]


class MaceEvaluator(PropertyEvaluator):
    """
    MACE + Materials Project property evaluator with CSV caching.

    Note on identity vs. cache key: the framework caches by
    material.get_identifier() ('<formula>|SG..|..'), while the on-disk CSV
    cache is keyed by composition formula (matching mcts_crystal). Both are
    maintained; the formula cache is what persists across runs.
    """

    def __init__(
        self,
        cache_path: Optional[str] = None,
        mp_api_key: Optional[str] = None,
    ):
        super().__init__()
        self.cache_path = cache_path
        self.mp_api_key = mp_api_key
        self.last_e_decomp = 0.0

        # MACE calculators are stateful; keep one per thread.
        self._thread_local = threading.local()
        self._main_calculator = None
        # Guards cache_df reads/writes and CSV persistence.
        self._cache_lock = threading.Lock()

        # Load or initialize the formula-keyed CSV cache.
        if cache_path and Path(cache_path).exists():
            self.cache_df = pd.read_csv(cache_path)
            if "data_quality" not in self.cache_df.columns:
                self.cache_df["data_quality"] = self.cache_df.apply(
                    lambda r: "no_mp_data"
                    if abs(r["e_above_hull"] - r["e_form"]) < 1e-9
                    else "valid",
                    axis=1,
                )
                self.cache_df["e_decomp"] = (
                    self.cache_df["e_form"] - self.cache_df["e_above_hull"]
                )
            logger.info("Loaded %d cached calculations from %s",
                        len(self.cache_df), cache_path)
        else:
            self.cache_df = pd.DataFrame(columns=_CACHE_COLUMNS)

    # ------------------------------------------------------------------ #
    # PropertyEvaluator interface
    # ------------------------------------------------------------------ #

    async def _compute(self, material: IntermetallicStructure) -> Dict[str, float]:
        """
        Compute properties, offloading the CPU-bound MACE/MP work to a thread.
        """
        loop = asyncio.get_event_loop()
        e_form, e_hull, formula = await loop.run_in_executor(
            None, self._compute_sync, material.atoms
        )
        return {"e_form": e_form, "e_above_hull": e_hull, "formula": formula}

    # ------------------------------------------------------------------ #
    # Synchronous core (ported from mcts_crystal.energy_calculator)
    # ------------------------------------------------------------------ #

    def _compute_sync(self, atoms) -> Tuple[float, float, str]:
        """Return (e_form, e_above_hull, formula) for an ASE Atoms object."""
        formula = atoms.get_chemical_formula(mode="metal")

        # Cache first.
        with self._cache_lock:
            cached = self._get_cached_result(formula)
            if cached is not None:
                e_form, e_hull = cached
                return e_form, e_hull, formula

        calculator = self._get_calculator()
        if calculator is None:
            logger.warning("No MACE calculator; returning zeros for %s", formula)
            return 0.0, 0.0, formula

        try:
            from ase.optimize import FIRE
            from ase.filters import ExpCellFilter
            from matbench_discovery.energy import get_e_form_per_atom

            atoms_copy = atoms.copy()
            atoms_copy.calc = calculator

            optimizer = FIRE(ExpCellFilter(atoms_copy))
            optimizer.run(fmax=0.05)

            e_form = get_e_form_per_atom(dict(
                energy=atoms_copy.get_total_energy(),
                composition=atoms_copy.get_chemical_formula(),
            ))

            e_decomp, data_quality = self._get_decomposition_energy(atoms_copy)
            if data_quality in ("no_mp_data", "error"):
                e_hull = UnstablePenalty
            else:
                e_hull = e_form - e_decomp

            self.last_e_decomp = e_decomp
            self._cache_result(formula, e_form, e_hull, e_decomp, data_quality)
            return e_form, e_hull, formula

        except Exception as exc:  # pragma: no cover - live-calc failure path
            logger.error("Error computing energies for %s: %s", formula, exc)
            self._cache_result(formula, 0.0, UnstablePenalty, 0.0, "error")
            return 0.0, UnstablePenalty, formula

    # --- MACE calculator management --------------------------------------

    def _init_calculator(self):
        try:
            from mace.calculators import mace_mp
            return mace_mp(
                model="large", dispersion=False,
                default_dtype="float64", device="cpu",
            )
        except Exception as exc:
            logger.warning("Could not initialize MACE calculator: %s", exc)
            return None

    def _get_calculator(self):
        """One MACE instance per thread (they are stateful)."""
        if threading.current_thread() is threading.main_thread():
            if self._main_calculator is None:
                self._main_calculator = self._init_calculator()
            return self._main_calculator
        if not hasattr(self._thread_local, "calculator"):
            self._thread_local.calculator = self._init_calculator()
        return self._thread_local.calculator

    # --- Materials Project decomposition energy --------------------------

    def _get_decomposition_energy(self, atoms) -> Tuple[float, str]:
        """Compute e_decomp via MP phase diagram; returns (e_decomp, quality)."""
        if not self.mp_api_key:
            return 0.0, "no_api_key"

        chemical_formula = atoms.get_chemical_formula()
        element_set = set(atoms.get_chemical_symbols())

        try:
            from pymatgen.core import Composition
            from pymatgen.io.ase import AseAtomsAdaptor
            from pymatgen.ext.matproj import MPRester
            from pymatgen.analysis.phase_diagram import PhaseDiagram
            from matbench_discovery.energy import get_e_form_per_atom

            with MPRester(self.mp_api_key) as mpr:
                try:
                    entries = mpr.get_entries_in_chemsys(
                        elements=element_set,
                        additional_criteria={"thermo_types": ["GGA_GGA+U"]},
                    )
                except TypeError:
                    entries = mpr.get_entries_in_chemsys(elements=element_set)

            if not entries:
                return 0.0, "no_mp_data"

            pd_obj = PhaseDiagram(entries)
            decomp = pd_obj.get_decomposition(Composition(chemical_formula))

            calculator = self._get_calculator()
            total_e_decomp = 0.0
            for entry, fraction in decomp.items():
                try:
                    decomp_atoms = AseAtomsAdaptor.get_atoms(
                        entry.structure, msonable=False
                    )
                    if calculator is None:
                        e_form = entry.energy_per_atom
                    else:
                        decomp_atoms.calc = calculator
                        e_form = get_e_form_per_atom(dict(
                            energy=decomp_atoms.get_total_energy(),
                            composition=decomp_atoms.get_chemical_formula(),
                        ))
                    total_e_decomp += e_form * fraction
                except Exception as phase_exc:
                    logger.error("Decomp phase %s failed: %s",
                                 entry.composition, phase_exc)
                    total_e_decomp += entry.energy_per_atom * fraction

            return total_e_decomp, "valid"

        except Exception as exc:  # pragma: no cover - network/API path
            logger.error("Decomposition energy failed for %s: %s",
                         chemical_formula, exc)
            return 0.0, "error"

    # --- CSV cache helpers -----------------------------------------------

    def _get_cached_result(self, formula: str) -> Optional[Tuple[float, float]]:
        """Look up (e_form, e_above_hull) by formula, with flexible matching."""
        if self.cache_df is None or self.cache_df.empty:
            return None

        matches = self.cache_df[self.cache_df["name"] == formula]
        if not matches.empty:
            return self._row_result(matches.iloc[0])

        normalized = self._normalize_formula(formula)
        for _, row in self.cache_df.iterrows():
            if self._normalize_formula(str(row["name"])) == normalized:
                return self._row_result(row)
        return None

    @staticmethod
    def _row_result(row) -> Tuple[float, float]:
        e_form = float(row["e_form"])
        e_hull = float(row["e_above_hull"])
        if "data_quality" in row and row["data_quality"] in ("no_mp_data", "error"):
            e_hull = UnstablePenalty
        return e_form, e_hull

    @staticmethod
    def _normalize_formula(formula: str) -> str:
        """Alphabetical element-count normalization for flexible matching."""
        try:
            matches = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
            counts: Dict[str, int] = {}
            for el, c in matches:
                if el:
                    counts[el] = counts.get(el, 0) + (int(c) if c else 1)
            return "".join(
                el if counts[el] == 1 else f"{el}{counts[el]}"
                for el in sorted(counts)
            )
        except Exception:
            return formula

    def _cache_result(
        self,
        formula: str,
        e_form: float,
        e_above_hull: float,
        e_decomp: float = 0.0,
        data_quality: str = "valid",
    ) -> None:
        new_row = pd.DataFrame([{
            "name": formula,
            "e_form": e_form,
            "e_above_hull": e_above_hull,
            "e_decomp": e_decomp,
            "data_quality": data_quality,
        }])
        with self._cache_lock:
            self.cache_df = pd.concat([self.cache_df, new_row], ignore_index=True)
            if self.cache_path:
                self.cache_df.to_csv(self.cache_path, index=False)
