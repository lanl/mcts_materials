"""
Tests for postprocessing.scatter.plot_ehull_vs_rdos.

Verify the plot builds from the run's Config (gamma + data paths), overlays the
run's top-N onto the design-space backdrop, honors a space_filter, and fails
cleanly when data paths are missing. Rendering is done headlessly (Agg).

© 2026. Triad National Security, LLC. All rights reserved.
"""

from pathlib import Path

import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from mcts_framework.core.config import Config
from mcts_framework.postprocessing import plot_ehull_vs_rdos

EXAMPLES = Path(__file__).resolve().parents[1].parent / "examples"
MACE_CACHE = str(EXAMPLES / "high_throughput_mace_results.full.csv")
DOSCAR_PEAKS = str(EXAMPLES / "doscar_peaks_data_with_U.csv")

_have_data = Path(MACE_CACHE).exists() and Path(DOSCAR_PEAKS).exists()
requires_data = pytest.mark.skipif(not _have_data, reason="example data files absent")


def _make_config(gamma=0.0001):
    return Config(
        material_type="intermetallic",
        intermetallic={
            "structure_path": str(EXAMPLES / "mat_Pb6U1W6_sg191.cif"),
            "rollout_method": "ehull_rdos",
            "gamma": gamma,
            "beta": 1.0,
            "mp_api_key": "X",
            "doscar_data_path": DOSCAR_PEAKS,
            "cache_path": MACE_CACHE,
        },
    )


def _run_df():
    return pd.DataFrame({
        "formula": ["Fe6Ge6U", "Co6Sn6U", "Ni6Si6U"],
        "e_above_hull": [0.01, 0.02, 0.08],
        "dos_reward": [500.0, 400.0, 300.0],
    })


@requires_data
class TestPlotEhullVsRdos:
    def test_writes_png_and_returns_figure(self, tmp_path):
        out = str(tmp_path / "scatter.png")
        fig = plot_ehull_vs_rdos(_run_df(), out, _make_config(), top_n=3)
        assert fig is not None
        assert Path(out).exists() and Path(out).stat().st_size > 0

    def test_overlay_has_topn_points(self, tmp_path):
        fig = plot_ehull_vs_rdos(_run_df(), str(tmp_path / "s.png"),
                                 _make_config(), top_n=3)
        # The 'Top 3 (MCTS)' overlay collection should carry 3 matched points.
        ax = fig.axes[0]
        top_coll = [c for c in ax.collections
                    if c.get_label().startswith("Top ")]
        assert top_coll and top_coll[0].get_offsets().shape[0] == 3

    def test_space_filter_shrinks_backdrop(self, tmp_path):
        import re

        def u_only(name):
            s = set(re.findall(r"[A-Z][a-z]?", str(name)))
            other_f = {"Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
                       "Ho", "Er", "Tm", "Yb", "Lu", "Th", "Pa", "Np", "Pu"}
            return "U" in s and not (s & other_f)

        full = plot_ehull_vs_rdos(_run_df(), str(tmp_path / "full.png"),
                                  _make_config(), top_n=3)
        filt = plot_ehull_vs_rdos(_run_df(), str(tmp_path / "filt.png"),
                                  _make_config(), top_n=3, space_filter=u_only)
        n_full = full.axes[0].collections[0].get_offsets().shape[0]
        n_filt = filt.axes[0].collections[0].get_offsets().shape[0]
        assert n_filt < n_full

    def test_missing_data_paths_raise(self, tmp_path):
        cfg = _make_config()
        cfg.intermetallic.cache_path = None
        with pytest.raises(ValueError, match="cache_path"):
            plot_ehull_vs_rdos(_run_df(), str(tmp_path / "x.png"), cfg, top_n=3)
