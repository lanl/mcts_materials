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


def _make_config(gamma=0.0001, rollout_method="ehull_rdos"):
    return Config(
        material_type="intermetallic",
        intermetallic={
            "structure_path": str(EXAMPLES / "mat_Pb6U1W6_sg191.cif"),
            "rollout_method": rollout_method,
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

    def test_publication_styling_defaults(self, tmp_path):
        fig = plot_ehull_vs_rdos(_run_df(), str(tmp_path / "s.png"),
                                 _make_config(), top_n=3)
        ax = fig.axes[0]
        # 3x3 figure, top/right spines hidden (publication default).
        assert tuple(fig.get_size_inches()) == (3.0, 3.0)
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()

    def test_ymax_caps_axis_else_shows_outliers(self, tmp_path):
        # Default: full range shows the E_hull~10 penalty outliers.
        full = plot_ehull_vs_rdos(_run_df(), str(tmp_path / "full.png"),
                                  _make_config(), top_n=3)
        assert full.axes[0].get_ylim()[1] > 5  # autoscaled above the outlier
        # ymax caps the view for the tight publication look.
        capped = plot_ehull_vs_rdos(_run_df(), str(tmp_path / "cap.png"),
                                    _make_config(), top_n=3, ymax=1.5)
        assert capped.axes[0].get_ylim()[1] == 1.5

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

    def test_overlay_ranking_respects_rollout_method(self, tmp_path):
        # Regression: the top-N overlay must be ranked by the run's own reward
        # (score_by_method dispatching on rollout_method), not a fixed additive
        # composite. Build a run_df where additive and product disagree on the
        # single best pick, then confirm the product-config overlay highlights
        # the product winner.
        from mcts_framework.postprocessing import (
            full_formula_key,
            load_design_space,
            score_by_method,
        )

        # This set disagrees: additive ranks Zr6Pb6U first (stable, ehull<0),
        # product ranks Ti6Sn6U first (Zr6Pb6U's rDOS bonus loses once the
        # reward is multiplicative rather than an additive gamma*rDOS nudge).
        names = ["Ti6Sn6U", "Zr6Pb6U", "Mn6Pb6U"]
        df = pd.DataFrame({"formula": names})

        # Use the real design-space r_DOS/e_hull (same source the plot uses).
        df_mace, lookup = load_design_space(MACE_CACHE, DOSCAR_PEAKS)
        space = {full_formula_key(r["name"] if "name" in r else r["formula"]):
                 (lookup.get_reward(r.get("name", r.get("formula"))),
                  float(r["e_above_hull"]))
                 for _, r in df_mace.assign(
                     name=df_mace.get("name", df_mace.get("formula"))).iterrows()}

        def best_by(method):
            scored = []
            for n in names:
                hit = space.get(full_formula_key(n))
                if hit is None:
                    continue
                r_dos, e_hull = hit
                scored.append((n, score_by_method(method, e_hull, r_dos, 1.0, 0.0001)))
            return max(scored, key=lambda t: t[1])[0]

        additive_best = best_by("ehull_rdos")
        product_best = best_by("ehull_rdos_product")
        # Only a meaningful regression test if the two methods actually differ.
        if additive_best == product_best:
            pytest.skip("additive and product agree on this set; test is vacuous")

        # Plot with the PRODUCT config; the top-1 overlay point must be the
        # product winner's (r_DOS, e_hull), not the additive winner's.
        fig = plot_ehull_vs_rdos(
            df, str(tmp_path / "prod.png"),
            _make_config(rollout_method="ehull_rdos_product"), top_n=1,
        )
        top_coll = [c for c in fig.axes[0].collections
                    if c.get_label().startswith("Top ")][0]
        plotted = tuple(top_coll.get_offsets()[0])
        assert plotted == pytest.approx(space[full_formula_key(product_best)])
        assert plotted != pytest.approx(space[full_formula_key(additive_best)])

    def test_missing_data_paths_raise(self, tmp_path):
        cfg = _make_config()
        cfg.intermetallic.cache_path = None
        with pytest.raises(ValueError, match="cache_path"):
            plot_ehull_vs_rdos(_run_df(), str(tmp_path / "x.png"), cfg, top_n=3)
