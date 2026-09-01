"""
Unit tests for the ELF-based Tc fit and the reward built on it.

The fit is Equation 2 of Belli et al., Ann. Phys. (Berlin) 2025, 537, e00280.
These tests pin its algebra (bounds, the phi* optimum, monotonicity), check it
against published descriptor/Tc pairs, and check the reward's handling of
candidates whose descriptors are missing.

(c) 2026. Triad National Security, LLC. All rights reserved.
"""

import math

import pytest

from mcts_framework.superhydride.rewards import (
    BELLI2025_MAE_K,
    BELLI2025_MAX_ERROR_K,
    BELLI2025_RMSE_K,
    PHI_STAR_OPTIMUM,
    TC_MAX_K,
    TC_MIN_K,
    TcReward,
    belli2025_tc,
    create_superhydride_reward,
)

#: Descriptors and measured Tc for six hydrides, as recorded in the
#: elf-descriptors reference set. phi/phi*/H_f/H_DOS are DFT values; the last
#: column is experiment. Used to check the fit reproduces published estimates.
REFERENCE_SET = [
    # name,     phi,   phi*,  H_f,   H_DOS, Tc_expt, Tc_fit
    ("CaH6",   0.811, 0.811, 0.857, 0.793, 215, 197.971),
    ("YH6",    0.781, 0.781, 0.857, 0.459, 224, 175.715),
    ("LaH10",  0.758, 0.836, 0.909, 0.667, 250, 200.950),
    ("LaBeH8", 0.527, 0.738, 0.800, 0.724, 110, 156.516),
    ("PdCuH2", 0.396, 0.404, 0.500, 0.431,  17,  24.722),
    ("H3S",    0.873, 0.873, 0.750, 0.429, 203,  89.381),
]


# --- The fit --------------------------------------------------------------


def test_published_error_statistics():
    """The fit's own reported error on its 244-compound dataset."""
    assert BELLI2025_RMSE_K == 41.0
    assert BELLI2025_MAE_K == 31.0
    assert BELLI2025_MAX_ERROR_K == 108.0


@pytest.mark.parametrize("name,phi,phi_star,h_f,h_dos,_expt,expected", REFERENCE_SET)
def test_fit_reproduces_published_estimates(name, phi, phi_star, h_f, h_dos, _expt, expected):
    assert belli2025_tc(phi, phi_star, h_f, h_dos) == pytest.approx(expected, abs=1e-3)


def test_fit_lands_within_the_papers_outlier_criterion_except_for_h3s():
    """
    The paper counts a compound as an outlier when the fit misses by more than
    60 K (42 of its 244 compounds do). Five of these six are inside that.

    H3S is the exception, and it is not the fit's fault: its phi here was
    computed with a semicore-free sulphur pseudopotential, giving 0.873 against
    a published 0.68. Feed a wrong descriptor to a right formula and you get a
    wrong Tc - which is why this test pins the outlier to H3S by name rather
    than just counting.

    Note this is a six-compound spot check on the transcription of Eq. 2, not a
    measurement of the fit's accuracy; the published RMSE of 41 K comes from
    244 compounds.
    """
    outliers = [
        name
        for name, phi, ps, hf, hd, expt, _ in REFERENCE_SET
        if abs(belli2025_tc(phi, ps, hf, hd) - expt) > 60.0
    ]
    assert outliers == ["H3S"]


def test_maximum_is_attained_at_the_analytic_optimum():
    assert belli2025_tc(1.0, PHI_STAR_OPTIMUM, 1.0, 1.0) == pytest.approx(TC_MAX_K)


def test_the_27_over_4_normalises_the_molecularity_factor_to_one():
    """
    27/4 is exactly 1/max(phi*^2 - phi*^3), so the prefactor is the whole
    dynamic range: 422.2 K above the 5.5 K intercept.
    """
    peak = belli2025_tc(1.0, PHI_STAR_OPTIMUM, 1.0, 1.0) - TC_MIN_K
    assert peak == pytest.approx(422.2)


@pytest.mark.parametrize(
    "phi,phi_star,h_f,h_dos",
    [
        (1.0, 0.0, 1.0, 1.0),  # no H-H connection at any ELF value
        (1.0, 1.0, 1.0, 1.0),  # intact H2 molecules
        (1.0, PHI_STAR_OPTIMUM, 0.0, 1.0),  # no hydrogen
        (0.0, PHI_STAR_OPTIMUM, 1.0, 1.0),  # no percolating network
        (1.0, PHI_STAR_OPTIMUM, 1.0, 0.0),  # no H states at E_F
    ],
)
def test_vanishing_factors_give_the_intercept(phi, phi_star, h_f, h_dos):
    assert belli2025_tc(phi, phi_star, h_f, h_dos) == pytest.approx(TC_MIN_K)


def test_fit_is_bounded():
    """Eq. 2 cannot express a Tc below 5.5 K or above 427.7 K."""
    step = 0.05
    values = [i * step for i in range(int(1 / step) + 1)]
    for phi in values:
        for phi_star in values:
            tc = belli2025_tc(phi, phi_star, 1.0, 1.0)
            assert TC_MIN_K - 1e-9 <= tc <= TC_MAX_K + 1e-9


def test_molecularity_factor_peaks_at_two_thirds_and_falls_towards_one():
    """
    The physics in the formula: hydrogen stretched but still bonded is what
    couples, and intact H2 (phi* -> 1) contributes nothing. The search must
    therefore be able to gain by LOWERING phi*, unlike every other descriptor.
    """
    at_peak = belli2025_tc(0.8, PHI_STAR_OPTIMUM, 0.9, 0.7)
    assert at_peak > belli2025_tc(0.8, 0.50, 0.9, 0.7)
    assert at_peak > belli2025_tc(0.8, 0.80, 0.9, 0.7)
    assert at_peak > belli2025_tc(0.8, 0.95, 0.9, 0.7)
    # Monotone decreasing on the molecular side.
    assert (
        belli2025_tc(0.8, 0.75, 0.9, 0.7)
        > belli2025_tc(0.8, 0.85, 0.9, 0.7)
        > belli2025_tc(0.8, 0.95, 0.9, 0.7)
    )


@pytest.mark.parametrize("index,name", [(0, "phi"), (2, "h_f"), (3, "h_dos")])
def test_fit_increases_with_phi_h_f_and_h_dos(index, name):
    base = [0.5, PHI_STAR_OPTIMUM, 0.6, 0.5]
    lower, higher = list(base), list(base)
    lower[index], higher[index] = 0.3, 0.9
    assert belli2025_tc(*higher) > belli2025_tc(*lower), name


def test_h_f_enters_as_a_cube():
    """Doubling H_f multiplies the Tc excess over the intercept by 8."""
    small = belli2025_tc(0.8, PHI_STAR_OPTIMUM, 0.4, 0.7) - TC_MIN_K
    large = belli2025_tc(0.8, PHI_STAR_OPTIMUM, 0.8, 0.7) - TC_MIN_K
    assert large == pytest.approx(8.0 * small)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"phi": -0.1, "phi_star": 0.5, "h_f": 0.5, "h_dos": 0.5},
        {"phi": 1.5, "phi_star": 0.5, "h_f": 0.5, "h_dos": 0.5},
        {"phi": 0.5, "phi_star": 0.5, "h_f": 0.5, "h_dos": float("nan")},
        {"phi": 0.5, "phi_star": 0.5, "h_f": 0.5, "h_dos": None},
    ],
)
def test_fit_rejects_descriptors_outside_the_unit_interval(kwargs):
    with pytest.raises(ValueError):
        belli2025_tc(**kwargs)


# --- The reward -----------------------------------------------------------


def _properties(phi=0.8, phi_star=PHI_STAR_OPTIMUM, h_f=0.9, h_dos=0.7):
    return {"phi": phi, "phi_star": phi_star, "h_f": h_f, "h_dos": h_dos}


def test_reward_normalises_to_the_unit_interval_by_default():
    reward = TcReward()
    value = reward.compute_reward(_properties())
    assert 0.0 < value <= 1.0
    assert value == pytest.approx(belli2025_tc(0.8, PHI_STAR_OPTIMUM, 0.9, 0.7) / TC_MAX_K)


def test_normalised_reward_reaches_exactly_one_at_the_optimum():
    assert TcReward().compute_reward(
        _properties(phi=1.0, h_f=1.0, h_dos=1.0)
    ) == pytest.approx(1.0)


def test_raw_reward_is_kelvin():
    properties = _properties()
    assert TcReward(normalize=False).compute_reward(properties) == pytest.approx(
        belli2025_tc(**properties)
    )


def test_normalisation_preserves_the_ranking():
    """Dividing by a constant cannot reorder candidates."""
    candidates = [
        _properties(phi=0.4, phi_star=0.95, h_f=0.5, h_dos=0.3),
        _properties(phi=0.8, phi_star=0.70, h_f=0.9, h_dos=0.7),
        _properties(phi=0.6, phi_star=0.50, h_f=0.8, h_dos=0.5),
    ]
    normalized = TcReward(normalize=True)
    raw = TcReward(normalize=False)
    order_norm = sorted(range(3), key=lambda i: normalized.compute_reward(candidates[i]))
    order_raw = sorted(range(3), key=lambda i: raw.compute_reward(candidates[i]))
    assert order_norm == order_raw


@pytest.mark.parametrize(
    "properties",
    [
        {},                                                # nothing computed
        {"phi": 0.8, "phi_star": 0.7, "h_f": 0.9},         # h_dos missing
        _properties(phi=math.nan),                         # DFT did not converge
        _properties(h_dos=math.nan),
        _properties(phi=1.4),                              # out of range
        {"phi": "n/a", "phi_star": 0.7, "h_f": 0.9, "h_dos": 0.7},  # non-numeric
    ],
)
def test_unscorable_candidates_get_zero_not_an_exception(properties):
    assert TcReward().compute_reward(properties) == 0.0


def test_zero_ranks_below_every_scorable_candidate():
    """
    Eq. 2 cannot return less than 5.5 K, so 0.0 is strictly below any real
    estimate - no sentinel value needed for failed or unscreened candidates.
    """
    worst_real = TcReward().compute_reward(
        _properties(phi=1e-6, phi_star=1e-6, h_f=1e-6, h_dos=1e-6)
    )
    assert worst_real > 0.0
    assert TcReward().compute_reward({}) < worst_real


def test_reward_declares_the_descriptors_it_needs():
    assert set(TcReward().get_property_names()) == {"phi", "phi_star", "h_f", "h_dos"}


def test_factory_builds_a_working_reward():
    assert create_superhydride_reward().compute_reward(_properties()) > 0.0
    assert create_superhydride_reward(normalize=False).compute_reward(
        _properties()
    ) > 1.0  # kelvin, not a fraction
