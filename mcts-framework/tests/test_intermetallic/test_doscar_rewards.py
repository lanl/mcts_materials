"""
Tests for DOSCAR rDOS lookup and intermetallic reward functions.

These exercise pure logic (formula parsing, Gaussian sum, reward formulas)
without MACE or Materials Project.

© 2025. Triad National Security, LLC. All rights reserved.
"""

import numpy as np
import pandas as pd
import pytest

from mcts_framework.intermetallic.doscar import DoscarRewardLookup
from mcts_framework.intermetallic.rewards import (
    ehull_reward,
    EhullReward,
    RdosReward,
    EhullRdosReward,
    create_intermetallic_reward,
)


# --- ehull_reward formula ------------------------------------------------


def test_ehull_reward_boundary():
    # At threshold 0.05, reward is 0
    assert abs(ehull_reward(0.05)) < 1e-9


def test_ehull_reward_stable_positive():
    # At E_hull = 0 (stable), reward ~ +1
    assert ehull_reward(0.0) > 0.99


def test_ehull_reward_unstable_negative():
    # At E_hull = 0.1 (unstable), reward ~ -1
    assert ehull_reward(0.1) < -0.99


def test_ehull_reward_monotonic_decreasing():
    vals = [ehull_reward(e) for e in [0.0, 0.03, 0.05, 0.07, 0.1]]
    assert vals == sorted(vals, reverse=True)


# --- EhullReward ---------------------------------------------------------


def test_ehull_reward_class():
    r = EhullReward()
    assert r.get_property_names() == ["e_above_hull"]
    assert abs(r.compute_reward({"e_above_hull": 0.05})) < 1e-9


# --- DoscarRewardLookup: formula conversion ------------------------------


def test_convert_formula_ternary():
    lookup = DoscarRewardLookup(peaks_file=None)  # empty rewards
    # metal6 groupIV6 fblock -> fblock-groupIV-metal
    assert lookup.convert_formula_to_doscar_format("Ti6Si6Ce") == "Ce-Si-Ti"
    assert lookup.convert_formula_to_doscar_format("W6Pb6U") == "U-Pb-W"


def test_convert_formula_rejects_non_ternary():
    lookup = DoscarRewardLookup(peaks_file=None)
    assert lookup.convert_formula_to_doscar_format("Fe2O3") is None  # only 2 elems
    assert lookup.convert_formula_to_doscar_format("NaClKBr") is None  # 4 elems


def test_convert_formula_rejects_wrong_categories():
    lookup = DoscarRewardLookup(peaks_file=None)
    # No f-block element present
    assert lookup.convert_formula_to_doscar_format("Ti6Si6Fe") is None


def test_get_reward_missing_file_returns_zero():
    lookup = DoscarRewardLookup(peaks_file=None)
    assert lookup.get_reward("Ti6Si6Ce") == 0.0


def test_get_reward_nonexistent_path_returns_zero(tmp_path):
    lookup = DoscarRewardLookup(peaks_file=str(tmp_path / "missing.csv"))
    assert lookup.get_reward("Ti6Si6Ce") == 0.0


# --- DoscarRewardLookup: Gaussian sum from peaks -------------------------


def _write_peaks_csv(path):
    """Two compounds, simple peaks, so we can hand-check the Gaussian sum."""
    df = pd.DataFrame({
        "COMPOUND_NAME": ["Ce-Si-Ti", "Ce-Si-Ti", "U-Pb-W"],
        "PEAK_ENERGY": [0.0, 0.5, 0.0],
        "PEAK_WIDTH": [1.0, 1.0, 2.0],
        "PEAK_HEIGHT": [10.0, 4.0, 6.0],
    })
    df.to_csv(path, index=False)


def test_rdos_gaussian_sum(tmp_path):
    p = tmp_path / "peaks.csv"
    _write_peaks_csv(p)
    lookup = DoscarRewardLookup(peaks_file=str(p))

    sigma = 0.5
    # Ce-Si-Ti: peak0 (E=0): 10/1 * exp(0) = 10
    #           peak1 (E=0.5): 4/1 * exp(-0.5*(0.5/0.5)^2)=4*exp(-0.5)
    expected_cesiti = 10.0 + 4.0 * np.exp(-0.5)
    assert abs(lookup.get_reward("Ti6Si6Ce") - expected_cesiti) < 1e-6

    # U-Pb-W: 6/2 * exp(0) = 3
    assert abs(lookup.get_reward("W6Pb6U") - 3.0) < 1e-6


def test_rdos_prefers_core_over_valence(tmp_path):
    """A '_valence' compound is only used when its core counterpart is absent."""
    df = pd.DataFrame({
        "COMPOUND_NAME": ["Ce-Si-Ti", "Ce-Si-Ti_valence", "U-Pb-W_valence"],
        "PEAK_ENERGY": [0.0, 0.0, 0.0],
        "PEAK_WIDTH": [1.0, 1.0, 1.0],
        "PEAK_HEIGHT": [10.0, 999.0, 5.0],
    })
    p = tmp_path / "peaks.csv"
    df.to_csv(p, index=False)
    lookup = DoscarRewardLookup(peaks_file=str(p))

    # Core Ce-Si-Ti present -> valence ignored -> reward 10, not 999
    assert abs(lookup.get_reward("Ti6Si6Ce") - 10.0) < 1e-6
    # U-Pb-W has only valence -> included -> reward 5
    assert abs(lookup.get_reward("W6Pb6U") - 5.0) < 1e-6


# --- RdosReward / EhullRdosReward ----------------------------------------


def test_rdos_reward_uses_precomputed():
    lookup = DoscarRewardLookup(peaks_file=None)
    r = RdosReward(lookup)
    # If 'rdos' present in properties, use it directly.
    assert r.compute_reward({"rdos": 42.0}) == 42.0


def test_rdos_reward_looks_up_by_formula(tmp_path):
    p = tmp_path / "peaks.csv"
    _write_peaks_csv(p)
    lookup = DoscarRewardLookup(peaks_file=str(p))
    r = RdosReward(lookup)
    assert abs(r.compute_reward({"formula": "W6Pb6U"}) - 3.0) < 1e-6


def test_ehull_rdos_composite():
    lookup = DoscarRewardLookup(peaks_file=None)
    r = EhullRdosReward(lookup, beta=1.0, gamma=0.001)
    # e_hull=0 -> ehull_term ~ +1; rdos supplied = 100 -> +0.001*100=0.1
    props = {"e_above_hull": 0.0, "rdos": 100.0}
    reward = r.compute_reward(props)
    expected = ehull_reward(0.0) + 0.001 * 100.0
    assert abs(reward - expected) < 1e-9


def test_ehull_rdos_default_weights():
    lookup = DoscarRewardLookup(peaks_file=None)
    r = EhullRdosReward(lookup)
    assert r.beta == 1.0
    assert r.gamma == 0.0001


# --- Factory -------------------------------------------------------------


def test_create_reward_ehull():
    r = create_intermetallic_reward("ehull")
    assert isinstance(r, EhullReward)


def test_create_reward_rdos_requires_lookup():
    with pytest.raises(ValueError):
        create_intermetallic_reward("rdos")
    r = create_intermetallic_reward("rdos", DoscarRewardLookup(peaks_file=None))
    assert isinstance(r, RdosReward)


def test_create_reward_ehull_rdos():
    lookup = DoscarRewardLookup(peaks_file=None)
    r = create_intermetallic_reward("ehull_rdos", lookup, beta=2.0, gamma=0.01)
    assert isinstance(r, EhullRdosReward)
    assert r.beta == 2.0
    assert r.gamma == 0.01


def test_create_reward_unknown():
    with pytest.raises(ValueError):
        create_intermetallic_reward("bogus")
