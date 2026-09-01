"""
Property evaluator that computes the ELF descriptors with Quantum ESPRESSO.

Same contract as :class:`DescriptorTableEvaluator` - phi, phi*, H_f, H_DOS per
candidate - but it runs the ground-state funnel instead of reading a table.
A candidate whose run fails scores NaN, which :class:`TcReward` turns into 0.0,
so one bad SCF costs one candidate rather than the search.

The CSV cache is written in the descriptor-table schema, so a finished campaign
IS a descriptor table: point :class:`DescriptorTableEvaluator` at it and rerun
the search for free.

(c) 2026. Triad National Security, LLC. All rights reserved.
"""

import asyncio
import logging
import math
import re
import threading
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from ...core.evaluator import PropertyEvaluator
from ..evaluator import normalize_formula
from ..structure import SuperhydrideStructure
from .inputs import QESettings
from .pipeline import clean_scratch, run_ground_state
from .runner import QEError, QERunner

logger = logging.getLogger(__name__)

#: Cache columns. The first four are the descriptor-table schema.
CACHE_COLUMNS = [
    "formula",
    "phi",
    "phi_star",
    "h_dos",
    "h_f",
    "fermi_ev",
    "pressure_gpa",
    "status",
]


def _safe_dirname(formula: str) -> str:
    """A filesystem-safe directory name for a composition."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", formula)


class QuantumEspressoEvaluator(PropertyEvaluator):
    """
    Computes phi, phi*, H_f and H_DOS by running QE for each candidate.

    Every candidate gets its own working directory under ``work_root``, because
    pw.x resolves ``outdir`` relative to the process working directory and two
    concurrent runs sharing one would silently read each other's wavefunctions.
    """

    def __init__(
        self,
        settings: QESettings,
        runner: QERunner,
        work_root: str,
        *,
        pressure_gpa: Optional[float] = None,
        relax: bool = True,
        relax_passes: int = 2,
        cache_path: Optional[str] = None,
        keep_scratch: bool = False,
        keep_cube: bool = False,
    ):
        """
        Args:
            settings: the numerical protocol, fixed across the campaign.
            runner: how to invoke the QE binaries.
            work_root: parent directory for the per-candidate run directories.
                Put this on scratch - the funnel writes wavefunctions and cubes.
            pressure_gpa: target pressure for the relaxation. Required when
                ``relax`` is True.
            relax: vc-relax each candidate before the SCF. Leave True unless the
                template is already relaxed at this pressure with this protocol
                - and note substituting a host changes the equilibrium cell, so
                a substituted candidate is generally not relaxed even if its
                parent was.
            relax_passes: vc-relax passes; 2 is the minimum that sheds the
                Pulay error in the stress.
            cache_path: CSV of results, in the descriptor-table schema. Loaded
                on start and appended to as candidates complete, so an
                interrupted campaign resumes instead of recomputing. Failures
                are cached too, under a 'status' column that says why - which
                is right for the deterministic ones (a missing pseudopotential,
                an SCF that will not converge) and wrong for a transient one
                (a node died, a step timed out). Delete those rows to retry
                them.
            keep_scratch: keep each candidate's QE scratch directory.
                Wavefunctions dominate the footprint and nothing downstream
                reads them once the descriptors are out.
            keep_cube: keep the ELF cubes. They are tens of megabytes each.
        """
        super().__init__()
        if relax and pressure_gpa is None:
            raise ValueError(
                "QuantumEspressoEvaluator(relax=True) needs pressure_gpa: a "
                "pressure-stabilised hydride relaxed to 0 GPa is a different "
                "material."
            )
        self.settings = settings
        self.runner = runner
        self.work_root = Path(work_root)
        self.pressure_gpa = pressure_gpa
        self.relax = relax
        self.relax_passes = relax_passes
        self.cache_path = cache_path
        self.keep_scratch = keep_scratch
        self.keep_cube = keep_cube

        self._cache_lock = threading.Lock()
        self.cache_df = self._load_cache()

    # --- cache ------------------------------------------------------------

    def _load_cache(self) -> pd.DataFrame:
        if self.cache_path and Path(self.cache_path).exists():
            frame = pd.read_csv(self.cache_path)
            logger.info("Loaded %d cached QE results from %s", len(frame), self.cache_path)
            return frame
        return pd.DataFrame(columns=CACHE_COLUMNS)

    def _cached(self, formula: str) -> Optional[Dict[str, float]]:
        with self._cache_lock:
            if self.cache_df.empty:
                return None
            key = normalize_formula(formula)
            for _, row in self.cache_df.iterrows():
                if normalize_formula(str(row["formula"])) == key:
                    return {
                        "phi": float(row["phi"]),
                        "phi_star": float(row["phi_star"]),
                        "h_dos": float(row["h_dos"]),
                    }
        return None

    def _record(self, row: Dict[str, object]) -> None:
        with self._cache_lock:
            self.cache_df = pd.concat(
                [self.cache_df, pd.DataFrame([row])], ignore_index=True
            )
            if self.cache_path:
                Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
                self.cache_df.to_csv(self.cache_path, index=False)

    # --- PropertyEvaluator ------------------------------------------------

    async def _compute(self, material: SuperhydrideStructure) -> Dict[str, float]:
        """Run the funnel off the event loop; QE takes minutes to hours."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._compute_sync, material)

    def _compute_sync(self, material: SuperhydrideStructure) -> Dict[str, float]:
        formula = material.get_formula()
        h_f = material.get_hydrogen_fraction()

        cached = self._cached(formula)
        if cached is not None:
            logger.debug("QE cache hit for %s", formula)
            return {**cached, "h_f": h_f, "formula": formula}

        workdir = self.work_root / _safe_dirname(formula)
        try:
            result = run_ground_state(
                material.atoms,
                self.settings,
                self.runner,
                str(workdir),
                pressure_gpa=self.pressure_gpa,
                relax=self.relax,
                relax_passes=self.relax_passes,
                keep_cube=self.keep_cube,
            )
        except (QEError, ValueError, FileNotFoundError) as exc:
            logger.error("QE funnel failed for %s: %s", formula, exc)
            self._record({
                "formula": formula,
                "phi": math.nan,
                "phi_star": math.nan,
                "h_dos": math.nan,
                "h_f": h_f,
                "fermi_ev": math.nan,
                "pressure_gpa": math.nan,
                "status": f"failed: {type(exc).__name__}",
            })
            return {
                "phi": math.nan,
                "phi_star": math.nan,
                "h_dos": math.nan,
                "h_f": h_f,
                "formula": formula,
            }
        finally:
            if not self.keep_scratch:
                clean_scratch(str(workdir))

        logger.info(
            "%s: phi=%.3f phi*=%.3f H_f=%.3f H_DOS=%.3f (grid %s, P=%s GPa)",
            formula, result.phi, result.phi_star, result.h_f, result.h_dos,
            "x".join(str(n) for n in result.grid_shape),
            None if result.pressure_gpa is None else round(result.pressure_gpa, 1),
        )
        self._record({
            "formula": formula,
            "phi": result.phi,
            "phi_star": result.phi_star,
            "h_dos": result.h_dos,
            "h_f": result.h_f,
            "fermi_ev": result.fermi_ev,
            "pressure_gpa": result.pressure_gpa,
            "status": "ok",
        })
        return {**result.as_properties(), "formula": formula}

    def preflight(self) -> Dict[str, bool]:
        """
        Check the QE binaries are reachable before committing to a campaign.

        A search whose every reward is 0.0 usually means this would have failed.
        """
        return self.runner.check_available()
