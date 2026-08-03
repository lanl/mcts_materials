"""
Tests for the CLI: config loading, validate command, and results saving.

The `run` command's heavy builders (ase/rdkit) aren't exercised here - those
need optional deps and are covered by builder/integration tests. We test the
CLI plumbing: suffix dispatch, validation, error handling, and result files.

© 2026. Triad National Security, LLC. All rights reserved.
"""

import json

import pytest
from typer.testing import CliRunner

from mcts_framework.cli.main import app, _load_config
from mcts_framework.cli.results import save_results

runner = CliRunner()


# --- Fixtures ------------------------------------------------------------


def _molecule_config_dict():
    return {
        "material_type": "molecule",
        "mcts": {"iterations": 5},
        "molecule": {"starting_smiles": "CCO", "objective": "melting_point"},
    }


def _write(tmp_path, name, data):
    p = tmp_path / name
    if name.endswith(".json"):
        p.write_text(json.dumps(data))
    else:
        yaml = pytest.importorskip("yaml")
        p.write_text(yaml.safe_dump(data))
    return str(p)


# --- _load_config suffix dispatch ----------------------------------------


def test_load_config_json(tmp_path):
    path = _write(tmp_path, "c.json", _molecule_config_dict())
    cfg = _load_config(path)
    assert cfg.material_type == "molecule"


def test_load_config_yaml(tmp_path):
    path = _write(tmp_path, "c.yaml", _molecule_config_dict())
    cfg = _load_config(path)
    assert cfg.mcts.iterations == 5


def test_load_config_bad_suffix(tmp_path):
    p = tmp_path / "c.txt"
    p.write_text("nope")
    with pytest.raises(Exception):
        _load_config(str(p))


# --- validate command ----------------------------------------------------


def test_validate_ok(tmp_path):
    path = _write(tmp_path, "c.json", _molecule_config_dict())
    result = runner.invoke(app, ["validate", "--config", path])
    assert result.exit_code == 0
    assert "Config is valid" in result.stdout
    assert "molecule" in result.stdout


def test_validate_invalid_config(tmp_path):
    # material_type=intermetallic but no intermetallic section -> invalid
    bad = {"material_type": "intermetallic", "mcts": {"iterations": 5}}
    path = _write(tmp_path, "bad.json", bad)
    result = runner.invoke(app, ["validate", "--config", path])
    assert result.exit_code == 1
    # Error text is written to stderr; result.output captures both streams.
    assert "Invalid config" in result.output


def test_validate_missing_file():
    result = runner.invoke(app, ["validate", "--config", "does_not_exist.json"])
    assert result.exit_code == 1


# --- results saving (uses a completed toy MCTS) --------------------------


@pytest.mark.asyncio
async def test_save_results_writes_files(tmp_path):
    """Run a tiny toy MCTS and confirm all three result files are written."""
    from mcts_framework.core.mcts import MCTS
    from mcts_framework.core.selection import UCB1
    from conftest import (  # shared toy classes (see tests/conftest.py)
        IntMaterial,
        LineMoves,
        DistanceEvaluator,
        NegDistanceReward,
    )

    mcts = MCTS(
        root_material=IntMaterial(0),
        move_generator=LineMoves(),
        property_evaluator=DistanceEvaluator(target=5),
        reward_function=NegDistanceReward(),
        selection_strategy=UCB1(),
        n_rollout=1,
        rollout_depth=0,
        seed=0,
    )
    await mcts.run(iterations=50)

    out_dir = tmp_path / "results"
    paths = save_results(mcts, str(out_dir), top_n=5)

    for key in ("summary", "best_materials", "convergence"):
        assert key in paths
        assert (out_dir / f"{key}.json" if key == "summary"
                else out_dir / f"{key}.csv").exists() or \
               __import__("pathlib").Path(paths[key]).exists()

    # summary.json is valid JSON with expected keys
    summary = json.loads((out_dir / "summary.json").read_text())
    assert "best_material" in summary
    assert "unique_materials" in summary

    # A human-readable report is also written.
    assert "report" in paths
    assert (out_dir / "report.txt").exists()
    assert "MCTS Materials Search Report" in (out_dir / "report.txt").read_text()

    # The explored tree is persisted by default for offline figure regeneration.
    assert "tree" in paths
    assert (out_dir / "tree.json").exists()
    tree = json.loads((out_dir / "tree.json").read_text())
    assert tree["root_id"] == 0 and tree["nodes"]


@pytest.mark.asyncio
async def test_convergence_rows_match_iterations(tmp_path):
    import pandas as pd
    from mcts_framework.core.mcts import MCTS
    from mcts_framework.core.selection import UCB1
    from conftest import (
        IntMaterial,
        LineMoves,
        DistanceEvaluator,
        NegDistanceReward,
    )

    mcts = MCTS(
        root_material=IntMaterial(0),
        move_generator=LineMoves(),
        property_evaluator=DistanceEvaluator(target=5),
        reward_function=NegDistanceReward(),
        selection_strategy=UCB1(),
        n_rollout=1,
        rollout_depth=0,
        seed=0,
    )
    await mcts.run(iterations=30)

    paths = save_results(mcts, str(tmp_path / "r"))
    df = pd.read_csv(paths["convergence"])
    assert len(df) == len(mcts.reward_history)
    assert list(df.columns) == ["iteration", "best_reward", "unique_materials"]
