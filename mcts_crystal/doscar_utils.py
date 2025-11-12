"""
Utility functions for DOSCAR reward lookup and formula conversion.
"""

import pandas as pd
import re
from pathlib import Path
from typing import Optional


class DoscarRewardLookup:
    """
    Handles loading and looking up DOSCAR rewards for compounds.
    """

    def __init__(self, csv_file: Optional[str] = None):
        """
        Initialize DOSCAR reward lookup.

        Args:
            csv_file: Path to doscar_rewards.csv file
        """
        self.rewards_dict = {}

        if csv_file is None:
            # Try default location
            csv_file = Path(__file__).parent.parent / "doscar_rewards.csv"

        if Path(csv_file).exists():
            df = pd.read_csv(csv_file)
            # Create dictionary mapping compound_name -> reward_normalized
            self.rewards_dict = dict(zip(df['compound_name'], df['reward_normalized']))
            print(f"   ✓ Loaded {len(self.rewards_dict)} DOSCAR rewards from {Path(csv_file).name}")
        else:
            print(f"   ⚠ DOSCAR rewards file not found: {csv_file}")
            print(f"   ⚠ DOSCAR rewards will be set to 0.0")

    def convert_formula_to_doscar_format(self, formula: str) -> Optional[str]:
        """
        Convert MCTS formula (e.g., Ti6Si6Ce) to DOSCAR format (e.g., Ce-Si-Ti).

        The DOSCAR format is: fblock-groupIV-metal
        The MCTS format is: metal6groupIV6fblock (with counts)

        Args:
            formula: Chemical formula in MCTS format (e.g., "Ti6Si6Ce")

        Returns:
            Formula in DOSCAR format (e.g., "Ce-Si-Ti") or None if cannot convert
        """
        # Parse formula to extract elements and counts
        pattern = r'([A-Z][a-z]?)(\d*)'
        matches = re.findall(pattern, formula)

        elements = {}
        for element, count in matches:
            if element:  # Skip empty matches
                count = int(count) if count else 1
                elements[element] = count

        if len(elements) != 3:
            return None  # DOSCAR format expects exactly 3 elements

        # Define element categories
        group_iv = {'Si', 'Ge', 'Sn', 'Pb'}
        f_block = set()
        # Lanthanides: Ce (58) to Lu (71)
        lanthanides = ['Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
        # Actinides: Th (90) to Pu (94)
        actinides = ['Th', 'Pa', 'U', 'Np', 'Pu']
        f_block = set(lanthanides + actinides)

        # Transition metals: 3d, 4d, 5d
        transition_metals = {
            'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',  # 3d
            'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',  # 4d
            'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg'    # 5d
        }

        # Identify element types
        f_elem = None
        g_iv_elem = None
        metal_elem = None

        for elem in elements.keys():
            if elem in f_block:
                f_elem = elem
            elif elem in group_iv:
                g_iv_elem = elem
            elif elem in transition_metals:
                metal_elem = elem

        # Check if we found all three types
        if f_elem is None or g_iv_elem is None or metal_elem is None:
            return None

        # Format as fblock-groupIV-metal
        doscar_format = f"{f_elem}-{g_iv_elem}-{metal_elem}"
        return doscar_format

    def get_reward(self, formula: str) -> float:
        """
        Get DOSCAR reward for a given compound formula.

        Args:
            formula: Chemical formula in MCTS format (e.g., "Ti6Si6Ce")

        Returns:
            Normalized DOSCAR reward, or 0.0 if not found
        """
        # Convert to DOSCAR format
        doscar_formula = self.convert_formula_to_doscar_format(formula)

        if doscar_formula is None:
            return 0.0

        # Look up reward
        return self.rewards_dict.get(doscar_formula, 0.0)
