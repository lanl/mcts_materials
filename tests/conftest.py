"""Shared pytest fixtures for the mcts_crystal test suite."""

from pathlib import Path

import pandas as pd
import pytest
from ase.io import read

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def u_w_pb_atoms():
    """The default starting structure (Pb6UW6, space group 191) - no heavy deps needed to read a CIF."""
    return read(str(REPO_ROOT / "examples" / "mat_Pb6U1W6_sg191.cif"))


@pytest.fixture
def doscar_peaks_csv(tmp_path):
    """A small synthetic doscar_peaks_data_with_U.csv for DoscarRewardLookup tests.

    Rewards are always computed in real time from raw peaks - there is no
    precomputed rewards cache - so this fixture supplies peak rows instead of
    a rewards table. One peak each for "U-Pb-W" and "Ce-Si-Ti".
    """
    csv_path = tmp_path / "doscar_peaks_data_with_U.csv"
    df = pd.DataFrame({
        "COMPOUND_NAME": ["U-Pb-W", "Ce-Si-Ti"],
        "PEAK_ENERGY": [0.0, 0.0],
        "PEAK_WIDTH": [1.0, 1.0],
        "PEAK_HEIGHT": [0.42, 1.23],
    })
    df.to_csv(csv_path, index=False)
    return csv_path
