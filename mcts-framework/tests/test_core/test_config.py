"""
Unit tests for Pydantic configuration models.

© 2025. Triad National Security, LLC. All rights reserved.
"""

import json

import pytest
from pydantic import ValidationError

from mcts_framework.core.config import (
    MCTSConfig,
    IntermetallicConfig,
    MoleculeConfig,
    Config,
)


# --- MCTSConfig ----------------------------------------------------------


def test_mcts_config_defaults():
    cfg = MCTSConfig()
    assert cfg.iterations == 1000
    assert cfg.exploration_constant == 0.1
    assert cfg.selection_mode == "ucb1"
    assert cfg.n_rollout == 5
    assert cfg.rollout_aggregation == "max"  # matches mcts_crystal default


def test_mcts_config_rollout_aggregation():
    assert MCTSConfig(rollout_aggregation="mean").rollout_aggregation == "mean"
    with pytest.raises(ValidationError):
        MCTSConfig(rollout_aggregation="median")


def test_mcts_config_rejects_unknown_field():
    with pytest.raises(ValidationError):
        MCTSConfig(not_a_field=123)


def test_mcts_config_bounds():
    with pytest.raises(ValidationError):
        MCTSConfig(iterations=0)  # ge=1
    with pytest.raises(ValidationError):
        MCTSConfig(epsilon=1.5)  # le=1.0
    with pytest.raises(ValidationError):
        MCTSConfig(temperature=0.0)  # gt=0.0


def test_mcts_config_invalid_selection_mode():
    with pytest.raises(ValidationError):
        MCTSConfig(selection_mode="greedy")


# --- IntermetallicConfig -------------------------------------------------


def test_intermetallic_rdos_needs_no_api_key():
    """rdos does not require mp_api_key but does require doscar path."""
    cfg = IntermetallicConfig(
        structure_path="foo.cif",
        rollout_method="rdos",
        doscar_data_path="doscar.csv",
    )
    assert cfg.rollout_method == "rdos"


def test_intermetallic_move_step():
    """move_step defaults to 1 and must be >= 1."""
    cfg = IntermetallicConfig(
        structure_path="foo.cif", rollout_method="rdos",
        doscar_data_path="d.csv",
    )
    assert cfg.move_step == 1
    cfg3 = IntermetallicConfig(
        structure_path="foo.cif", rollout_method="rdos",
        doscar_data_path="d.csv", move_step=3,
    )
    assert cfg3.move_step == 3
    with pytest.raises(ValidationError):
        IntermetallicConfig(
            structure_path="foo.cif", rollout_method="rdos",
            doscar_data_path="d.csv", move_step=0,
        )


def test_intermetallic_ehull_requires_api_key():
    with pytest.raises(ValidationError):
        IntermetallicConfig(structure_path="foo.cif", rollout_method="ehull")


def test_intermetallic_ehull_rdos_requires_both():
    # Missing doscar path
    with pytest.raises(ValidationError):
        IntermetallicConfig(
            structure_path="foo.cif",
            rollout_method="ehull_rdos",
            mp_api_key="KEY",
        )
    # With both -> ok
    cfg = IntermetallicConfig(
        structure_path="foo.cif",
        rollout_method="ehull_rdos",
        mp_api_key="KEY",
        doscar_data_path="doscar.csv",
    )
    assert cfg.beta == 1.0
    assert cfg.gamma == 0.0001


def test_intermetallic_ehull_rdos_product_requires_both():
    # ehull_rdos_product needs BOTH mp_api_key and doscar_data_path.
    with pytest.raises(ValidationError):
        IntermetallicConfig(
            structure_path="foo.cif",
            rollout_method="ehull_rdos_product",
            mp_api_key="KEY",  # missing doscar
        )
    with pytest.raises(ValidationError):
        IntermetallicConfig(
            structure_path="foo.cif",
            rollout_method="ehull_rdos_product",
            doscar_data_path="doscar.csv",  # missing key
        )
    cfg = IntermetallicConfig(
        structure_path="foo.cif",
        rollout_method="ehull_rdos_product",
        mp_api_key="KEY",
        doscar_data_path="doscar.csv",
    )
    assert cfg.rollout_method == "ehull_rdos_product"


def test_f_block_mode_canonical_values():
    for mode in [
        "u_only",
        "lanthanides_u",
        "lanthanides_u_extended",
        "lanthanides_u_no_wrap",
        "full_f_block",
    ]:
        cfg = IntermetallicConfig(
            structure_path="foo.cif",
            rollout_method="rdos",
            doscar_data_path="d.csv",
            f_block_mode=mode,
        )
        assert cfg.f_block_mode == mode


def test_f_block_mode_experimental_alias():
    """Deprecated 'experimental' normalizes to 'lanthanides_u_no_wrap'."""
    cfg = IntermetallicConfig(
        structure_path="foo.cif",
        rollout_method="rdos",
        doscar_data_path="d.csv",
        f_block_mode="experimental",
    )
    assert cfg.f_block_mode == "lanthanides_u_no_wrap"


def test_f_block_mode_invalid():
    with pytest.raises(ValidationError):
        IntermetallicConfig(
            structure_path="foo.cif",
            rollout_method="rdos",
            doscar_data_path="d.csv",
            f_block_mode="actinides_only",
        )


# --- MoleculeConfig ------------------------------------------------------


def test_molecule_config_defaults():
    cfg = MoleculeConfig(starting_smiles="CCO")
    assert cfg.objective == "melting_point"
    assert "C" in cfg.functional_groups


def test_molecule_multi_objective_requires_weights():
    with pytest.raises(ValidationError):
        MoleculeConfig(starting_smiles="CCO", objective="multi_objective")

    cfg = MoleculeConfig(
        starting_smiles="CCO",
        objective="multi_objective",
        objective_weights={"h2_capacity": 1.0, "melting_point": -0.5},
    )
    assert cfg.objective_weights["h2_capacity"] == 1.0


# --- Top-level Config ----------------------------------------------------


def test_config_requires_matching_section():
    with pytest.raises(ValidationError):
        Config(material_type="intermetallic")  # no intermetallic section

    with pytest.raises(ValidationError):
        Config(material_type="molecule")  # no molecule section


def test_config_intermetallic_ok():
    cfg = Config(
        material_type="intermetallic",
        mcts=MCTSConfig(iterations=10),
        intermetallic=IntermetallicConfig(
            structure_path="foo.cif",
            rollout_method="rdos",
            doscar_data_path="d.csv",
        ),
    )
    assert cfg.material_type == "intermetallic"
    assert cfg.mcts.iterations == 10


def test_config_from_dict_molecule():
    data = {
        "material_type": "molecule",
        "mcts": {"iterations": 50},
        "molecule": {"starting_smiles": "CCO", "objective": "h2_capacity"},
    }
    cfg = Config.from_dict(data)
    assert cfg.molecule.starting_smiles == "CCO"
    assert cfg.mcts.iterations == 50


def test_config_from_json(tmp_path):
    data = {
        "material_type": "molecule",
        "molecule": {"starting_smiles": "CCO"},
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data))

    cfg = Config.from_json(str(p))
    assert cfg.material_type == "molecule"
    assert cfg.molecule.starting_smiles == "CCO"
    # mcts uses defaults
    assert cfg.mcts.iterations == 1000


def test_config_from_yaml(tmp_path):
    yaml = pytest.importorskip("yaml")
    data = {
        "material_type": "intermetallic",
        "mcts": {"iterations": 25, "selection_mode": "puct"},
        "intermetallic": {
            "structure_path": "foo.cif",
            "rollout_method": "rdos",
            "doscar_data_path": "d.csv",
            "f_block_mode": "experimental",
        },
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(data))

    cfg = Config.from_yaml(str(p))
    assert cfg.mcts.selection_mode == "puct"
    # alias normalized
    assert cfg.intermetallic.f_block_mode == "lanthanides_u_no_wrap"
