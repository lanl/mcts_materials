#!/usr/bin/env python3
"""
Calculate rewards for compounds based on DOSCAR peak data.

For each compound, the reward is calculated as:
    reward = (1/10000.0) * sum((peak_height/peak_width) * exp(-0.5*(1/sigma)^2))

where sigma = 0.5 and the sum is over all peaks for that compound.

Usage:
    python calculate_doscar_rewards.py
"""

import pandas as pd
import numpy as np
from pathlib import Path


def calculate_compound_reward(peaks_df, sigma=0.5):
    """
    Calculate reward for a single compound based on its peaks.

    Args:
        peaks_df: DataFrame containing peaks for a single compound
        sigma: Gaussian width parameter (default: 0.5)

    Returns:
        Calculated reward value
    """
    # Calculate the exponential factor (constant for all peaks)
    exp_factor = np.exp(-0.5 * (1.0 / sigma) ** 2)

    # Calculate the sum of (peak_height / peak_width) * exp_factor
    peak_contributions = (peaks_df['PEAK_HEIGHT'] / peaks_df['PEAK_WIDTH']) * exp_factor
    total_sum = peak_contributions.sum()

    # Divide by 1000.0 to get final reward
    reward = total_sum / 1000.0

    return reward


def main():
    """Calculate rewards for all compounds in DOSCAR peaks data."""

    print("=" * 80)
    print("DOSCAR PEAKS REWARD CALCULATOR")
    print("=" * 80)

    # Step 1: Load DOSCAR peaks data
    print("\n1. Loading DOSCAR peaks data...")
    input_file = Path(__file__).parent.parent / "doscar_peaks_data_with_U.csv"

    if not input_file.exists():
        print(f"❌ Error: Input file not found: {input_file}")
        return 1

    try:
        df = pd.read_csv(input_file)
        print(f"   ✓ Loaded {len(df)} peak records")
        print(f"   ✓ Columns: {', '.join(df.columns)}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return 1

    # Step 2: Filter compounds - prefer core (non-valence) over valence
    print("\n2. Filtering compounds (prefer core over valence)...")

    # Separate core and valence compounds
    core_compounds = df[~df['COMPOUND_NAME'].str.endswith('_valence')]
    valence_compounds = df[df['COMPOUND_NAME'].str.endswith('_valence')]

    # Get base names from valence compounds (remove _valence suffix)
    valence_base_names = valence_compounds['COMPOUND_NAME'].str.replace('_valence', '').unique()
    core_names = core_compounds['COMPOUND_NAME'].unique()

    # Find valence compounds whose core version doesn't exist
    missing_core_bases = set(valence_base_names) - set(core_names)
    valence_to_include = valence_compounds[
        valence_compounds['COMPOUND_NAME'].str.replace('_valence', '').isin(missing_core_bases)
    ]

    # Combine core compounds with valence-only compounds
    filtered_df = pd.concat([core_compounds, valence_to_include])

    print(f"   ✓ Core compounds: {len(core_names)}")
    print(f"   ✓ Valence compounds (no core): {len(missing_core_bases)}")
    print(f"   ✓ Total compounds to process: {len(core_names) + len(missing_core_bases)}")

    # Step 3: Calculate rewards for each compound
    print("\n3. Calculating rewards for each compound...")
    sigma = 0.5
    print(f"   Using sigma = {sigma}")
    print(f"   Formula: (1/10000.0) * sum((peak_height/peak_width) * exp(-0.5*(1/sigma)^2))")

    # Group by compound name
    compound_groups = filtered_df.groupby('COMPOUND_NAME')
    n_compounds = len(compound_groups)

    # Calculate reward for each compound (unscaled)
    results = []
    unscaled_rewards = []
    for compound_name, group_df in compound_groups:
        # Calculate unscaled sum
        exp_factor = np.exp(-0.5 * (1.0 / sigma) ** 2)
        peak_contributions = (group_df['PEAK_HEIGHT'] / group_df['PEAK_WIDTH']) * exp_factor
        unscaled_sum = peak_contributions.sum()
        unscaled_rewards.append(unscaled_sum)

        # Calculate scaled reward
        reward_normalized = unscaled_sum / 10000.0
        results.append({
            'compound_name': compound_name,
            'reward_raw': unscaled_sum,
            'reward_normalized': reward_normalized
        })

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    print(f"   ✓ Calculated rewards for all {n_compounds} compounds")

    # Display unscaled statistics
    print("\n   Unscaled reward statistics (before dividing by 10000):")
    print(f"   Min (unscaled):  {min(unscaled_rewards):.2f}")
    print(f"   Max (unscaled):  {max(unscaled_rewards):.2f}")
    print(f"   Mean (unscaled): {np.mean(unscaled_rewards):.2f}")
    print(f"   Median (unscaled): {np.median(unscaled_rewards):.2f}")

    # Step 4: Display statistics
    print("\n4. Scaled Reward Statistics (÷10000):")
    print(f"   Mean reward:   {results_df['reward_normalized'].mean():.6f}")
    print(f"   Median reward: {results_df['reward_normalized'].median():.6f}")
    print(f"   Min reward:    {results_df['reward_normalized'].min():.6f}")
    print(f"   Max reward:    {results_df['reward_normalized'].max():.6f}")
    print(f"   Std dev:       {results_df['reward_normalized'].std():.6f}")

    # Step 5: Show top compounds
    print("\n5. Top 10 Compounds by Reward:")
    top_compounds = results_df.nlargest(10, 'reward_normalized')
    for i, (_, row) in enumerate(top_compounds.iterrows(), 1):
        print(f"   {i:2d}. {row['compound_name']:20s}  raw = {row['reward_raw']:10.2f}  normalized = {row['reward_normalized']:8.6f}")

    # Step 6: Save results
    print("\n6. Saving results...")
    output_file = Path(__file__).parent.parent / "doscar_rewards.csv"

    try:
        results_df.to_csv(output_file, index=False)
        print(f"   ✓ Results saved to: {output_file}")
        print(f"   ✓ Total compounds: {len(results_df)}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        return 1

    print("\n" + "=" * 80)
    print("✅ REWARD CALCULATION COMPLETE!")
    print("=" * 80)
    print(f"\nOutput file: {output_file}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
