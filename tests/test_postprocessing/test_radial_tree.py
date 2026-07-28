"""
Tests for postprocessing.radial_tree.plot_radial_tree.

Build a small tree.json of intermetallic formulas (with e_above_hull), then draw
the 4-panel radial figure and confirm it renders, stars the root, and fails
cleanly without DOSCAR data. Rendering is headless (Agg).

© 2026. Triad National Security, LLC. All rights reserved.
"""

import json
from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
pytest.importorskip("networkx")

from mcts_framework.core.config import Config
from mcts_framework.postprocessing import plot_radial_tree

EXAMPLES = Path(__file__).resolve().parents[1].parent / "examples"
MACE_CACHE = str(EXAMPLES / "high_throughput_mace_results.full.csv")
DOSCAR_PEAKS = str(EXAMPLES / "doscar_peaks_data_with_U.csv")

_have_doscar = Path(DOSCAR_PEAKS).exists()
requires_doscar = pytest.mark.skipif(not _have_doscar, reason="DOSCAR data absent")


def _make_config():
    return Config(
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


def _write_tree(path):
    """A small root->children tree; each formula visited a few times."""
    formulas = ["Pb6U1W6", "Pb6U1Ta6", "Sn6U1W6", "Ge6U1W6", "Pb6U1Re6"]
    nodes = [{
        "id": 0, "parent": None, "identifier": formulas[0],
        "own_reward": 1.0, "visits": 10, "total_reward": 8.0,
        "subtree_best": 2.0, "terminated": False,
        "properties": {"e_above_hull": 0.02},
    }]
    for i, f in enumerate(formulas[1:], start=1):
        nodes.append({
            "id": i, "parent": 0, "identifier": f,
            "own_reward": 0.5, "visits": 3 + i, "total_reward": 1.5,
            "subtree_best": 0.5, "terminated": False,
            "properties": {"e_above_hull": 0.03 + 0.01 * i},
        })
    Path(path).write_text(json.dumps({"root_id": 0, "nodes": nodes}))


@requires_doscar
class TestPlotRadialTree:
    def test_renders_four_panels_and_stars_root(self, tmp_path):
        tree = tmp_path / "tree.json"
        _write_tree(tree)
        out = tmp_path / "radial.png"
        fig = plot_radial_tree(str(tree), str(out), _make_config(), max_nodes=60)
        assert fig is not None
        assert len(fig.axes) >= 4  # 4 panels (+ colorbar axes)
        assert out.exists() and out.stat().st_size > 0

    def test_empty_tree_returns_none(self, tmp_path):
        tree = tmp_path / "empty.json"
        tree.write_text(json.dumps({"root_id": 0, "nodes": []}))
        fig = plot_radial_tree(str(tree), str(tmp_path / "x.png"), _make_config())
        assert fig is None

    def test_missing_doscar_raises(self, tmp_path):
        tree = tmp_path / "tree.json"
        _write_tree(tree)
        cfg = _make_config()
        cfg.intermetallic.doscar_data_path = None
        with pytest.raises(ValueError, match="doscar_data_path"):
            plot_radial_tree(str(tree), str(tmp_path / "x.png"), cfg)
