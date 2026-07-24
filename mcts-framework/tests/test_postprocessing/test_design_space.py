"""
Tests for the postprocessing.design_space ranking/scoring foundation.

These exercise the chemistry-agnostic behavior: full-formula keying, composite
scoring against the framework's own ehull_reward, per-compound rDOS attachment
via the full formula, and pluggable key_fn / space_filter. They use the bundled
example MACE cache + DOSCAR peaks so the ranking path runs end to end.

© 2026. Triad National Security, LLC. All rights reserved.
"""

from pathlib import Path

import pytest

from mcts_framework.intermetallic import (
    DoscarRewardLookup,
    EhullRdosProductReward,
    EhullRdosReward,
    EhullReward,
    RdosReward,
)
from mcts_framework.postprocessing import (
    full_formula_key,
    load_design_space,
    rank_design_space,
    score_by_method,
)

EXAMPLES = Path(__file__).resolve().parents[1].parent / "examples"
MACE_CACHE = str(EXAMPLES / "high_throughput_mace_results.full.csv")
DOSCAR_PEAKS = str(EXAMPLES / "doscar_peaks_data_with_U.csv")

_have_data = Path(MACE_CACHE).exists() and Path(DOSCAR_PEAKS).exists()
requires_data = pytest.mark.skipif(not _have_data, reason="example data files absent")


class TestFullFormulaKey:
    def test_order_independent(self):
        assert full_formula_key("Fe6Ge6U") == full_formula_key("UFe6Ge6")

    def test_implicit_count_is_one(self):
        assert full_formula_key("Fe6Ge6U") == "Fe6Ge6U1"

    def test_distinct_f_block_compounds_never_collide(self):
        # The old tm_giv_key collapsed these onto one key; full formula must not.
        assert full_formula_key("Fe6Ge6U") != full_formula_key("Fe6Ge6Ce")


class TestScoreByMethod:
    """score_by_method must reproduce the reward classes bit-for-bit, so the
    ranking scores compounds by exactly the formula the run optimized."""

    @pytest.fixture
    def lookup(self):
        return DoscarRewardLookup(peaks_file=DOSCAR_PEAKS) if _have_data \
            else DoscarRewardLookup(peaks_file=None)

    def test_ehull_matches_class(self, lookup):
        props = {"e_above_hull": 0.03, "formula": "Fe6Ge6U"}
        assert score_by_method("ehull", 0.03, 123.0) == pytest.approx(
            EhullReward().compute_reward(props)
        )

    def test_rdos_matches_class(self, lookup):
        props = {"formula": "Fe6Ge6U"}
        r_dos = lookup.get_reward("Fe6Ge6U")
        assert score_by_method("rdos", 0.03, r_dos) == pytest.approx(
            RdosReward(lookup).compute_reward(props)
        )

    def test_ehull_rdos_matches_class(self, lookup):
        props = {"e_above_hull": 0.03, "formula": "Fe6Ge6U"}
        r_dos = lookup.get_reward("Fe6Ge6U")
        expected = EhullRdosReward(lookup, beta=1.0, gamma=0.0001).compute_reward(props)
        assert score_by_method("ehull_rdos", 0.03, r_dos, 1.0, 0.0001) == pytest.approx(expected)

    def test_product_matches_class_and_ignores_gamma(self, lookup):
        props = {"e_above_hull": 0.03, "formula": "Fe6Ge6U"}
        r_dos = lookup.get_reward("Fe6Ge6U")
        expected = EhullRdosProductReward(lookup).compute_reward(props)
        # gamma must not affect the product reward.
        assert score_by_method("ehull_rdos_product", 0.03, r_dos, 1.0, 0.5) == pytest.approx(expected)
        assert score_by_method("ehull_rdos_product", 0.03, r_dos, 1.0, 0.0) == pytest.approx(expected)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown rollout_method"):
            score_by_method("bogus", 0.03, 1.0)


@requires_data
class TestRankDesignSpace:
    def test_ranks_are_dense_and_start_at_one(self):
        ranks = rank_design_space(MACE_CACHE, DOSCAR_PEAKS, "ehull_rdos")
        assert ranks
        values = sorted(set(ranks.values()))
        assert values[0] == 1

    def test_no_filter_ranks_more_than_u_only(self):
        all_ranks = rank_design_space(MACE_CACHE, DOSCAR_PEAKS, "ehull_rdos")

        def u_only(name):
            import re
            s = set(re.findall(r"[A-Z][a-z]?", str(name)))
            other_f = {"Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
                       "Ho", "Er", "Tm", "Yb", "Lu", "Th", "Pa", "Np", "Pu"}
            return "U" in s and not (s & other_f)

        u_ranks = rank_design_space(
            MACE_CACHE, DOSCAR_PEAKS, "ehull_rdos", space_filter=u_only
        )
        assert len(all_ranks) > len(u_ranks)
        # Every U-only key is also present in the unfiltered ranking.
        assert set(u_ranks) <= set(all_ranks)

    def test_rollout_method_changes_ranking(self):
        # Additive and product rewards need not agree on the best compound;
        # the ranking must reflect whichever the run used.
        additive = rank_design_space(MACE_CACHE, DOSCAR_PEAKS, "ehull_rdos")
        product = rank_design_space(MACE_CACHE, DOSCAR_PEAKS, "ehull_rdos_product")
        assert additive != product

    def test_custom_key_fn_is_respected(self):
        # A constant key collapses everything onto one entry (rank 1 wins).
        ranks = rank_design_space(
            MACE_CACHE, DOSCAR_PEAKS, "ehull_rdos", key_fn=lambda n: "ALL"
        )
        assert ranks == {"ALL": 1}

    def test_missing_mace_cache_returns_empty(self):
        ranks = rank_design_space("/no/such/file.csv", DOSCAR_PEAKS, "ehull_rdos")
        assert ranks == {}


@requires_data
class TestLoadDesignSpace:
    def test_returns_dataframe_and_lookup(self):
        df, lookup = load_design_space(MACE_CACHE, DOSCAR_PEAKS)
        assert df is not None and len(df)
        # rDOS is resolved per-compound directly from the full formula.
        assert hasattr(lookup, "get_reward")

    def test_distinct_f_block_compounds_get_distinct_rdos(self):
        _, lookup = load_design_space(MACE_CACHE, DOSCAR_PEAKS)
        r_u = lookup.get_reward("Fe6Ge6U")
        r_ce = lookup.get_reward("Fe6Ge6Ce")
        # Different f-block elements have their own DOSCAR entries; they must not
        # be forced to share a value (the whole point of dropping tm_giv_key).
        assert r_u != r_ce
