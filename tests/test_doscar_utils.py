"""Tests for mcts_crystal.doscar_utils.DoscarRewardLookup."""

import pandas as pd
import pytest

from mcts_crystal.doscar_utils import DoscarRewardLookup


class TestFormulaConversion:
    def test_converts_mcts_formula_to_doscar_format(self):
        lookup = DoscarRewardLookup(peaks_file="/nonexistent.csv")
        assert lookup.convert_formula_to_doscar_format("Ti6Si6Ce") == "Ce-Si-Ti"

    def test_order_independent(self):
        lookup = DoscarRewardLookup(peaks_file="/nonexistent.csv")
        assert lookup.convert_formula_to_doscar_format("U1Pb6W6") == "U-Pb-W"

    def test_returns_none_for_wrong_element_count(self):
        lookup = DoscarRewardLookup(peaks_file="/nonexistent.csv")
        assert lookup.convert_formula_to_doscar_format("Ti6Si6") is None


class TestGetReward:
    def test_loads_and_finds_known_compound(self, doscar_peaks_csv):
        lookup = DoscarRewardLookup(peaks_file=str(doscar_peaks_csv))
        assert lookup.get_reward("Pb6UW6") == pytest.approx(0.42)
        assert lookup.get_reward("Ti6Si6Ce") == pytest.approx(1.23)

    def test_unknown_compound_returns_zero(self, doscar_peaks_csv):
        lookup = DoscarRewardLookup(peaks_file=str(doscar_peaks_csv))
        assert lookup.get_reward("Ti6Si6Nd") == 0.0

    def test_missing_peaks_file_degrades_to_zero_rewards(self):
        lookup = DoscarRewardLookup(peaks_file="/definitely/not/a/real/path.csv")
        assert lookup.rewards_dict == {}
        assert lookup.get_reward("Pb6UW6") == 0.0

    def test_reward_decays_with_distance_from_fermi_level(self, tmp_path):
        """Peaks farther from E_Fermi (PEAK_ENERGY=0) should contribute less."""
        csv_path = tmp_path / "doscar_peaks_data_with_U.csv"
        pd.DataFrame({
            "COMPOUND_NAME": ["U-Pb-W", "Ce-Si-Ti"],
            "PEAK_ENERGY": [0.0, 5.0],
            "PEAK_WIDTH": [1.0, 1.0],
            "PEAK_HEIGHT": [1.0, 1.0],
        }).to_csv(csv_path, index=False)

        lookup = DoscarRewardLookup(peaks_file=str(csv_path))
        assert lookup.get_reward("Pb6UW6") == pytest.approx(1.0)
        assert lookup.get_reward("Ti6Si6Ce") < lookup.get_reward("Pb6UW6")
