"""
Tests for postprocessing.tables.write_top_n_table.

Verify that: N controls table length; gamma/beta/data paths are read from the
run's Config (not passed separately); the default ranks against the full design
space (no U-only default); and config.dump_yaml round-trips with the API key
redacted.

© 2026. Triad National Security, LLC. All rights reserved.
"""

from pathlib import Path

import pandas as pd
import pytest

from mcts_framework.core.config import Config
from mcts_framework.postprocessing import write_top_n_table

EXAMPLES = Path(__file__).resolve().parents[1].parent / "examples"
MACE_CACHE = str(EXAMPLES / "high_throughput_mace_results.full.csv")
DOSCAR_PEAKS = str(EXAMPLES / "doscar_peaks_data_with_U.csv")

_have_data = Path(MACE_CACHE).exists() and Path(DOSCAR_PEAKS).exists()
requires_data = pytest.mark.skipif(not _have_data, reason="example data files absent")


def _make_config(gamma=0.0001, beta=1.0, mp_api_key="SECRET"):
    """A minimal intermetallic Config wired to the example data files."""
    return Config(
        material_type="intermetallic",
        intermetallic={
            "structure_path": str(EXAMPLES / "mat_Pb6U1W6_sg191.cif"),
            "rollout_method": "ehull_rdos_product",
            "gamma": gamma,
            "beta": beta,
            "mp_api_key": mp_api_key,
            "doscar_data_path": DOSCAR_PEAKS,
            "cache_path": MACE_CACHE,
        },
    )


def _sample_df():
    return pd.DataFrame({
        "formula": ["Fe6Ge6U", "Co6Sn6U", "Ni6Si6U", "Cr6Pb6U", "V6Ge6U"],
        "e_above_hull": [0.01, 0.02, 0.08, 0.03, 0.15],
        "dos_reward": [500.0, 400.0, 300.0, 200.0, 100.0],
    })


@requires_data
class TestWriteTopNTable:
    def test_n_controls_row_count(self, tmp_path):
        cfg = _make_config()
        for n in (2, 3, 5):
            out = write_top_n_table(_sample_df(), str(tmp_path / f"t{n}.tex"), cfg, n=n)
            body = [ln for ln in Path(out).read_text().splitlines()
                    if "&" in ln and "MCTS Rank" not in ln]
            assert len(body) == n

    def test_gamma_read_from_config_appears_in_header(self, tmp_path):
        cfg = _make_config(gamma=0.005)
        out = write_top_n_table(_sample_df(), str(tmp_path / "g.tex"), cfg, n=3)
        assert "gamma=0.005" in Path(out).read_text()

    def test_default_space_is_full_not_u_only(self, tmp_path):
        cfg = _make_config()
        out = write_top_n_table(_sample_df(), str(tmp_path / "f.tex"), cfg, n=3)
        assert "full design space" in Path(out).read_text()

    def test_header_records_rollout_method(self, tmp_path):
        cfg = _make_config()  # ehull_rdos_product
        out = write_top_n_table(_sample_df(), str(tmp_path / "m.tex"), cfg, n=3)
        text = Path(out).read_text()
        assert "rollout_method=ehull_rdos_product" in text
        # The product reward has no gamma weighting, so the old gamma*rDOS
        # column framing must be gone.
        assert "gamma \\cdot" not in text and "\\gamma \\cdot" not in text

    def test_reward_column_matches_score_by_method_for_product(self, tmp_path):
        from mcts_framework.postprocessing import score_by_method

        cfg = _make_config()  # ehull_rdos_product
        df = _sample_df()
        out = write_top_n_table(df, str(tmp_path / "p.tex"), cfg, n=1)
        # Top row's Reward column must equal the product reward of the best-by-
        # product compound, not the additive composite.
        best = max(
            df.itertuples(),
            key=lambda r: score_by_method("ehull_rdos_product", r.e_above_hull, r.dos_reward),
        )
        expected = score_by_method("ehull_rdos_product", best.e_above_hull, best.dos_reward)
        body = [ln for ln in Path(out).read_text().splitlines()
                if "&" in ln and "MCTS Rank" not in ln]
        reward_cell = float(body[0].split("&")[5])
        assert reward_cell == pytest.approx(expected, abs=1e-4)

    def test_missing_data_paths_raise(self, tmp_path):
        cfg = _make_config()
        cfg.intermetallic.cache_path = None
        with pytest.raises(ValueError, match="cache_path"):
            write_top_n_table(_sample_df(), str(tmp_path / "x.tex"), cfg, n=3)

    def test_rdos_recovered_from_doscar_when_column_absent(self, tmp_path):
        # Regression: a tree-derived df has no r_DOS/dos_reward column and its
        # 'name' may be a full 'formula|SG|Wyckoff' identifier. The table must
        # recover r_DOS from the run's DOSCAR data (splitting on '|'), not treat
        # it as 0 (which would zero every rDOS-dependent reward). Uses U-only
        # compounds known to have DOSCAR entries.
        from mcts_framework.postprocessing import load_design_space

        cfg = _make_config()  # ehull_rdos_product
        _, lookup = load_design_space(MACE_CACHE, DOSCAR_PEAKS)
        # A df with NO r_DOS column and identifier-style names.
        df = pd.DataFrame({
            "name": ["Cr6Sn6U|SG191|a", "Fe6Ge6U|SG191|b"],
            "e_above_hull": [0.02, 0.01],
        })
        out = write_top_n_table(df, str(tmp_path / "r.tex"), cfg, n=2)
        body = [ln for ln in Path(out).read_text().splitlines()
                if "&" in ln and "MCTS Rank" not in ln]
        # r_DOS column (index 3) must be the real per-compound value, not 0.
        rdos_cells = [float(ln.split("&")[3]) for ln in body]
        assert all(v > 0 for v in rdos_cells), f"r_DOS not recovered: {rdos_cells}"
        # And it must match the DOSCAR lookup on the plain formula.
        assert rdos_cells[0] == pytest.approx(
            max(lookup.get_reward("Cr6Sn6U"), lookup.get_reward("Fe6Ge6U")), abs=1e-3
        )


class TestConfigDumpYaml:
    def test_redacts_api_key_and_roundtrips(self, tmp_path):
        cfg = _make_config(mp_api_key="TOPSECRET")
        path = str(tmp_path / "config.yaml")
        cfg.dump_yaml(path)

        text = Path(path).read_text()
        assert "TOPSECRET" not in text

        reloaded = Config.from_yaml(path)
        assert reloaded.intermetallic.gamma == cfg.intermetallic.gamma
        assert reloaded.intermetallic.mp_api_key == Config.REDACTED_PLACEHOLDER
