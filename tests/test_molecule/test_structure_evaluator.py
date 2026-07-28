"""
Tests for MolecularStructure identity and MoleculeEvaluator.

MolecularStructure needs RDKit (skipped if absent). MoleculeEvaluator's
predictors are mocked via a fake molecule_modifier.prediction module so we can
verify computation/caching logic without model files.

© 2026. Triad National Security, LLC. All rights reserved.
"""

import sys
import types

import pandas as pd
import pytest

pytest.importorskip("rdkit")

from rdkit import Chem  # noqa: E402

from mcts_framework.molecule.structure import MolecularStructure  # noqa: E402
from mcts_framework.molecule.evaluator import MoleculeEvaluator  # noqa: E402


# --- MolecularStructure identity -----------------------------------------


def test_canonical_smiles_identifier():
    s = MolecularStructure.from_smiles("OCC")  # ethanol, non-canonical order
    # RDKit canonicalizes to 'CCO'
    assert s.get_identifier() == "CCO"


def test_identifier_order_independent():
    a = MolecularStructure.from_smiles("OCC")
    b = MolecularStructure.from_smiles("CCO")
    assert a.get_identifier() == b.get_identifier()


def test_from_smiles_invalid_raises():
    with pytest.raises(ValueError):
        MolecularStructure.from_smiles("not_a_valid_smiles((")


def test_copy_independent():
    s = MolecularStructure.from_smiles("CCO")
    s2 = s.copy()
    assert s2.get_identifier() == s.get_identifier()
    assert s2.mol is not s.mol


# --- MoleculeEvaluator (mocked predictions) ------------------------------


@pytest.fixture
def fake_prediction(monkeypatch):
    """Install a fake molecule_modifier.prediction module."""
    pkg = types.ModuleType("molecule_modifier")
    pred = types.ModuleType("molecule_modifier.prediction")

    pred.load_chemprop_model = lambda d=None: "CHEMPROP_MODEL"
    pred.load_xgboost_artifacts = lambda d=None: "XGB_ARTIFACTS"

    def predict_chemprop(smiles_list, model=None):
        return pd.DataFrame({"melting_temp": [300.0]})

    def predict_xgboost(smiles_list, artifacts=None):
        return pd.DataFrame({"melting_temp": [320.0]})

    def predict_h2_capacity(smiles_list):
        return pd.DataFrame({"h2_capacity": [5.0]})

    def predict_synthesizability(smiles_list):
        return pd.DataFrame({"synthesizability": [3.0]})

    pred.predict_chemprop = predict_chemprop
    pred.predict_xgboost = predict_xgboost
    pred.predict_h2_capacity = predict_h2_capacity
    pred.predict_synthesizability = predict_synthesizability

    monkeypatch.setitem(sys.modules, "molecule_modifier", pkg)
    monkeypatch.setitem(sys.modules, "molecule_modifier.prediction", pred)
    return pred


@pytest.mark.asyncio
async def test_evaluator_melting_point_averages(fake_prediction):
    evaluator = MoleculeEvaluator(properties=["melting_point"])
    mat = MolecularStructure.from_smiles("CCO")
    props = await evaluator.evaluate(mat)
    # mean of 300 and 320
    assert props["melting_point"] == pytest.approx(310.0)


@pytest.mark.asyncio
async def test_evaluator_all_properties(fake_prediction):
    evaluator = MoleculeEvaluator(
        properties=["melting_point", "h2_capacity", "synthesizability"]
    )
    mat = MolecularStructure.from_smiles("CCO")
    props = await evaluator.evaluate(mat)
    assert props["melting_point"] == pytest.approx(310.0)
    assert props["h2_capacity"] == pytest.approx(5.0)
    assert props["synthesizability"] == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_evaluator_caches(fake_prediction):
    evaluator = MoleculeEvaluator(properties=["h2_capacity"])
    mat = MolecularStructure.from_smiles("CCO")
    await evaluator.evaluate(mat)
    assert "CCO" in evaluator.cache
    # Second call served from cache (identifier present).
    props = await evaluator.evaluate(mat)
    assert props["h2_capacity"] == pytest.approx(5.0)


def test_evaluator_rejects_unknown_property():
    with pytest.raises(ValueError):
        MoleculeEvaluator(properties=["not_a_property"])