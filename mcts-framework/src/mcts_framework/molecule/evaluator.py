"""
Property evaluator for molecules, backed by molecule-modifier's predictors.

Computes any subset of:
    melting_point    : mean of Chemprop (MPNN) and XGBoost predictions
    h2_capacity      : H2 storage capacity (cc/g)
    synthesizability : synthesizability score (1 easy .. 10 hard)

Models are preloaded once (Chemprop checkpoint, XGBoost artifacts) and reused
across predictions. Heavy work runs in a thread-pool executor via the async
PropertyEvaluator interface. RDKit and molecule-modifier are imported lazily.

© 2026. Triad National Security, LLC. All rights reserved.
"""

import asyncio
import logging
from typing import Dict, List, Optional

from ..core.evaluator import PropertyEvaluator
from .structure import MolecularStructure

logger = logging.getLogger(__name__)

_VALID_PROPERTIES = ("melting_point", "h2_capacity", "synthesizability")


class MoleculeEvaluator(PropertyEvaluator):
    """Predicts molecular properties via molecule-modifier, with caching."""

    def __init__(
        self,
        properties: Optional[List[str]] = None,
        chemprop_model_dir: Optional[str] = None,
        xgboost_model_dir: Optional[str] = None,
    ):
        """
        Args:
            properties: Which properties to predict; subset of
                ('melting_point', 'h2_capacity', 'synthesizability').
                Defaults to ['melting_point'].
            chemprop_model_dir: Optional Chemprop model directory (else
                molecule-modifier auto-detects).
            xgboost_model_dir: Optional XGBoost model directory.
        """
        super().__init__()
        self.properties_to_predict = list(properties or ["melting_point"])

        unknown = set(self.properties_to_predict) - set(_VALID_PROPERTIES)
        if unknown:
            raise ValueError(
                f"Unknown propert(ies) {sorted(unknown)}; "
                f"valid: {list(_VALID_PROPERTIES)}"
            )

        self.chemprop_model_dir = chemprop_model_dir
        self.xgboost_model_dir = xgboost_model_dir

        # Preloaded model handles (lazy, filled on first use).
        self._chemprop_model = None
        self._xgb_artifacts = None
        self._models_loaded = False

    # ------------------------------------------------------------------ #
    # PropertyEvaluator interface
    # ------------------------------------------------------------------ #

    async def _compute(self, material: MolecularStructure) -> Dict[str, float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._compute_sync, material.get_smiles()
        )

    # ------------------------------------------------------------------ #
    # Synchronous prediction
    # ------------------------------------------------------------------ #

    def _ensure_models(self) -> None:
        """Preload Chemprop/XGBoost handles once, if melting_point is needed."""
        if self._models_loaded:
            return
        if "melting_point" in self.properties_to_predict:
            from molecule_modifier.prediction import (
                load_chemprop_model,
                load_xgboost_artifacts,
            )
            self._chemprop_model = load_chemprop_model(self.chemprop_model_dir)
            self._xgb_artifacts = load_xgboost_artifacts(self.xgboost_model_dir)
        self._models_loaded = True

    def _compute_sync(self, smiles: str) -> Dict[str, float]:
        """Run the configured predictors for a single SMILES string."""
        from molecule_modifier.prediction import (
            predict_chemprop,
            predict_xgboost,
            predict_h2_capacity,
            predict_synthesizability,
        )

        self._ensure_models()
        props: Dict[str, float] = {}

        if "melting_point" in self.properties_to_predict:
            df_cp = predict_chemprop([smiles], model=self._chemprop_model)
            df_xgb = predict_xgboost([smiles], artifacts=self._xgb_artifacts)
            props["melting_point"] = float(
                (df_cp["melting_temp"].iloc[0] + df_xgb["melting_temp"].iloc[0]) / 2.0
            )

        if "h2_capacity" in self.properties_to_predict:
            df_h2 = predict_h2_capacity([smiles])
            props["h2_capacity"] = float(df_h2["h2_capacity"].iloc[0])

        if "synthesizability" in self.properties_to_predict:
            df_s = predict_synthesizability([smiles])
            props["synthesizability"] = float(df_s["synthesizability"].iloc[0])

        return props