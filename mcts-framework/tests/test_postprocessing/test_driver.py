"""
Tests for the postprocessing driver (generate_study_outputs) and the
`mcts-run figures` CLI command.

Build a fake finished run directory (config.yaml + tree.json), then confirm the
driver reconstructs the run DataFrame, writes the table + both figures into
figures/, and that the CLI command does the same and errors cleanly on a run
missing its persisted artifacts.

© 2026. Triad National Security, LLC. All rights reserved.
"""

import json
from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
pytest.importorskip("networkx")

from mcts_framework.core.config import Config
from mcts_framework.postprocessing import (
    generate_study_outputs,
    load_run_config,
    load_run_dataframe,
)

EXAMPLES = Path(__file__).resolve().parents[1].parent / "examples"
MACE_CACHE = str(EXAMPLES / "high_throughput_mace_results.full.csv")
DOSCAR_PEAKS = str(EXAMPLES / "doscar_peaks_data_with_U.csv")

_have_data = Path(MACE_CACHE).exists() and Path(DOSCAR_PEAKS).exists()
requires_data = pytest.mark.skipif(not _have_data, reason="example data files absent")


def _write_run(run_dir: Path):
    """A minimal finished-run directory: config.yaml + tree.json."""
    cfg = Config(
        material_type="intermetallic",
        intermetallic={
            "structure_path": str(EXAMPLES / "mat_Pb6U1W6_sg191.cif"),
            "rollout_method": "ehull_rdos_product",
            "gamma": 0.0001,
            "beta": 1.0,
            "mp_api_key": "X",
            "doscar_data_path": DOSCAR_PEAKS,
            "cache_path": MACE_CACHE,
        },
    )
    cfg.dump_yaml(str(run_dir / "config.yaml"))

    formulas = ["Pb6U1W6", "Pb6U1Ta6", "Sn6U1W6", "Ge6U1W6", "Fe6Ge6U", "Co6Sn6U"]
    nodes = [{
        "id": 0, "parent": None, "identifier": formulas[0],
        "own_reward": 1.0, "visits": 12, "total_reward": 9.0,
        "subtree_best": 2.0, "terminated": False,
        "properties": {"e_above_hull": 0.02, "e_form": -0.5},
    }]
    for i, f in enumerate(formulas[1:], start=1):
        nodes.append({
            "id": i, "parent": 0, "identifier": f,
            "own_reward": 0.5, "visits": 2 + i, "total_reward": 1.0 * i,
            "subtree_best": 0.5, "terminated": False,
            "properties": {"e_above_hull": 0.02 + 0.01 * i, "e_form": -0.4},
        })
    (run_dir / "tree.json").write_text(json.dumps({"root_id": 0, "nodes": nodes}))


class TestLoadRunDataframe:
    def test_unique_evaluated_rows_with_properties(self, tmp_path):
        _write_run(tmp_path)
        df = load_run_dataframe(str(tmp_path / "tree.json"))
        assert set(df["name"]) == {
            "Pb6U1W6", "Pb6U1Ta6", "Sn6U1W6", "Ge6U1W6", "Fe6Ge6U", "Co6Sn6U"
        }
        assert "e_above_hull" in df.columns
        assert "own_reward" in df.columns

    def test_skips_unevaluated_nodes(self, tmp_path):
        (tmp_path / "tree.json").write_text(json.dumps({
            "root_id": 0,
            "nodes": [
                {"id": 0, "parent": None, "identifier": "A",
                 "own_reward": None, "visits": 0, "total_reward": 0.0,
                 "subtree_best": None, "terminated": False, "properties": {}},
                {"id": 1, "parent": 0, "identifier": "B",
                 "own_reward": 0.3, "visits": 1, "total_reward": 0.3,
                 "subtree_best": 0.3, "terminated": False,
                 "properties": {"e_above_hull": 0.01}},
            ],
        }))
        df = load_run_dataframe(str(tmp_path / "tree.json"))
        assert list(df["name"]) == ["B"]


class TestLoadRunConfig:
    def test_reads_persisted_config(self, tmp_path):
        _write_run(tmp_path)
        cfg = load_run_config(str(tmp_path))
        assert cfg.material_type == "intermetallic"
        assert cfg.intermetallic.rollout_method == "ehull_rdos_product"

    def test_missing_config_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="config.yaml"):
            load_run_config(str(tmp_path))


@requires_data
class TestGenerateStudyOutputs:
    def test_writes_table_and_figures(self, tmp_path):
        _write_run(tmp_path)
        produced = generate_study_outputs(str(tmp_path), top_n=5)
        assert set(produced) == {"table", "scatter", "radial_tree"}
        figs = tmp_path / "figures"
        assert (figs / "top5_table.tex").exists()
        assert (figs / "ehull_vs_rdos.png").exists()
        assert (figs / "radial_tree.png").exists()

    def test_custom_out_dir(self, tmp_path):
        _write_run(tmp_path)
        out = tmp_path / "custom"
        produced = generate_study_outputs(str(tmp_path), out_dir=str(out), top_n=5)
        assert Path(produced["table"]).parent == out

    def test_missing_tree_raises(self, tmp_path):
        _write_run(tmp_path)
        (tmp_path / "tree.json").unlink()
        with pytest.raises(FileNotFoundError, match="tree.json"):
            generate_study_outputs(str(tmp_path), top_n=5)


@requires_data
class TestFiguresCLI:
    def test_cli_generates_figures(self, tmp_path):
        from typer.testing import CliRunner
        from mcts_framework.cli.main import app

        _write_run(tmp_path)
        result = CliRunner().invoke(
            app, ["figures", "--run-dir", str(tmp_path), "--top-n", "5"]
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "figures" / "top5_table.tex").exists()

    def test_cli_errors_on_missing_run(self, tmp_path):
        from typer.testing import CliRunner
        from mcts_framework.cli.main import app

        result = CliRunner().invoke(app, ["figures", "--run-dir", str(tmp_path)])
        assert result.exit_code == 1
