"""
Tests for MaceEvaluator's no-Materials-Project-key behavior.

Without an MP API key the convex hull cannot be built, so the decomposition
energy is undefined (NaN) rather than a misleading 0.0, and E_hull falls back to
the formation energy (ranking is by formation energy only). These paths are pure
(no MACE/MP calls), so they are testable without the optional heavy deps.

© 2026. Triad National Security, LLC. All rights reserved.
"""

import math

import pandas as pd

from mcts_framework.intermetallic import UnstablePenalty
from mcts_framework.intermetallic.evaluator import MaceEvaluator


def test_decomposition_energy_is_nan_without_api_key():
    ev = MaceEvaluator(mp_api_key=None)
    e_decomp, quality = ev._get_decomposition_energy(None)
    assert quality == "no_api_key"
    assert math.isnan(e_decomp)  # undefined, not a misleading 0.0


def test_row_result_no_api_key_falls_back_to_formation_energy():
    # A cached no_api_key row keeps E_hull = e_form (formation energy only) -
    # it is NOT force-penalized like no_mp_data/error, and NOT NaN.
    row = pd.Series({
        "name": "X", "e_form": -0.5, "e_above_hull": -0.5,
        "e_decomp": float("nan"), "data_quality": "no_api_key",
    })
    e_form, e_hull = MaceEvaluator._row_result(row)
    assert e_form == -0.5
    assert e_hull == -0.5  # == e_form, not UnstablePenalty


def test_row_result_no_mp_data_is_penalized():
    # Contrast: no_mp_data/error rows ARE forced to the penalty on reload.
    row = pd.Series({
        "name": "Y", "e_form": -0.5, "e_above_hull": -0.5,
        "e_decomp": 0.0, "data_quality": "no_mp_data",
    })
    _, e_hull = MaceEvaluator._row_result(row)
    assert e_hull == UnstablePenalty
