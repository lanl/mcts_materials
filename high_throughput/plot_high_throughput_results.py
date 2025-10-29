#!/usr/bin/env python3
"""
Plot formation energy vs energy above hull for high-throughput results.

Reads from high_throughput_mace_results.csv and creates a scatter plot,
excluding rows where e_decomp = 0.0 (which indicates missing/failed decomposition calculations).

Usage:
    python plot_high_throughput_results.py
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def main():
    """Main plotting function."""
    print("=" * 80)
    print("HIGH-THROUGHPUT RESULTS VISUALIZATION")
    print("=" * 80)

    # Step 1: Load results
    print("\n1. Loading high-throughput results...")
    results_file = Path(__file__).parent.parent / "high_throughput_mace_results.full.csv"

    if not results_file.exists():
        print(f"ERROR: Results file not found: {results_file}")
        print("Please run high_throughput_mace_calculations.py first")
        return 1

    try:
        df = pd.read_csv(results_file)
        print(f"   ✓ Loaded {len(df)} total entries")
    except Exception as e:
        print(f"ERROR loading results file: {e}")
        return 1

    # Step 2: Filter data
    print("\n2. Filtering data...")
    print(f"   Total entries: {len(df)}")

    # Filter out entries where e_decomp = 0.0
    df_filtered = df[df['e_decomp'] != 0.0].copy()
    print(f"   Entries with valid e_decomp (≠ 0.0): {len(df_filtered)}")
    print(f"   Filtered out: {len(df) - len(df_filtered)} entries")

    if len(df_filtered) == 0:
        print("\nERROR: No valid data to plot after filtering")
        print("All entries have e_decomp = 0.0")
        return 1

    # Step 3: Data statistics
    print("\n3. Data statistics (filtered data)...")
    print(f"\n   Formation Energy (e_form):")
    print(f"     Min:  {df_filtered['e_form'].min():.4f} eV/atom")
    print(f"     Max:  {df_filtered['e_form'].max():.4f} eV/atom")
    print(f"     Mean: {df_filtered['e_form'].mean():.4f} eV/atom")
    print(f"     Std:  {df_filtered['e_form'].std():.4f} eV/atom")

    print(f"\n   Energy Above Hull (e_above_hull):")
    print(f"     Min:  {df_filtered['e_above_hull'].min():.4f} eV/atom")
    print(f"     Max:  {df_filtered['e_above_hull'].max():.4f} eV/atom")
    print(f"     Mean: {df_filtered['e_above_hull'].mean():.4f} eV/atom")
    print(f"     Std:  {df_filtered['e_above_hull'].std():.4f} eV/atom")

    # Count stable compounds
    stable_count = len(df_filtered[df_filtered['e_above_hull'] <= 0.0])
    metastable_count = len(df_filtered[
        (df_filtered['e_above_hull'] > 0.0) &
        (df_filtered['e_above_hull'] <= 0.1)
    ])

    print(f"\n   Thermodynamically stable (e_above_hull ≤ 0): {stable_count}")
    print(f"   Metastable (0 < e_above_hull ≤ 0.1): {metastable_count}")

    # Step 4: Create visualization
    print("\n4. Creating visualization...")

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color points by stability
    colors = []
    for e_hull in df_filtered['e_above_hull']:
        if e_hull <= 0.0:
            colors.append('green')  # Stable
        elif e_hull <= 0.1:
            colors.append('orange')  # Metastable
        else:
            colors.append('red')  # Unstable

    # Scatter plot
    scatter = ax.scatter(
        df_filtered['e_form'],
        df_filtered['e_above_hull'],
        c=colors,
        alpha=0.6,
        s=30,
        edgecolors='black',
        linewidth=0.5
    )

    # Add reference lines
    ax.axhline(y=0, color='darkgreen', linestyle='--', linewidth=2,
               alpha=0.7, label='Convex Hull (e_hull = 0)')
    ax.axhline(y=0.1, color='orange', linestyle='--', linewidth=1.5,
               alpha=0.5, label='Metastable threshold (e_hull = 0.1)')
    ax.axvline(x=0, color='blue', linestyle='--', linewidth=1.5,
               alpha=0.5, label='Formation favorability (e_form = 0)')

    # Shade the ideal region (both negative)
    ax.fill_between(
        [df_filtered['e_form'].min(), 0],
        -0.5,
        0,
        alpha=0.1,
        color='green',
        label='Ideal region (both negative)'
    )

    # Labels and title
    ax.set_xlabel('Formation Energy (eV/atom)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Energy Above Hull (eV/atom)', fontsize=14, fontweight='bold')
    ax.set_title(
        f'High-Throughput Materials Discovery Results\n'
        f'{len(df_filtered):,} compounds ({stable_count} stable, {metastable_count} metastable)',
        fontsize=16,
        fontweight='bold',
        pad=20
    )

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_elements = [
        Patch(facecolor='green', alpha=0.6, edgecolor='black', label=f'Stable (e_hull ≤ 0): {stable_count}'),
        Patch(facecolor='orange', alpha=0.6, edgecolor='black', label=f'Metastable (0 < e_hull ≤ 0.1): {metastable_count}'),
        Patch(facecolor='red', alpha=0.6, edgecolor='black',
              label=f'Unstable (e_hull > 0.1): {len(df_filtered) - stable_count - metastable_count}'),
        Line2D([0], [0], color='darkgreen', linestyle='--', linewidth=2, alpha=0.7, label='Convex Hull (e_hull = 0)'),
        Line2D([0], [0], color='orange', linestyle='--', linewidth=1.5, alpha=0.5, label='Metastable threshold (e_hull = 0.1)'),
        Line2D([0], [0], color='blue', linestyle='--', linewidth=1.5, alpha=0.5, label='Formation favorability (e_form = 0)'),
    ]

    ax.legend(
        handles=legend_elements,
        loc='best',
        fontsize=10,
        framealpha=0.9
    )

    # Grid
    ax.grid(alpha=0.3, linestyle=':')

    # Adjust layout
    plt.tight_layout()

    # Save figure
    output_file = Path(__file__).parent.parent / "high_throughput_results_plot.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved plot to: {output_file}")

    # Step 5: Print top stable compounds
    print("\n5. Top 10 most stable compounds (by e_above_hull)...")
    top_stable = df_filtered.nsmallest(10, 'e_above_hull')

    print(f"\n   {'Rank':<6} {'Compound':<20} {'E_form':<12} {'E_hull':<12} {'E_decomp':<12}")
    print("   " + "-" * 70)

    for i, (_, row) in enumerate(top_stable.iterrows(), 1):
        print(f"   {i:<6} {row['name']:<20} "
              f"{row['e_form']:<12.4f} {row['e_above_hull']:<12.4f} "
              f"{row['e_decomp']:<12.4f}")

    # Step 6: Print compounds with both negative values
    print("\n6. Compounds with BOTH negative formation energy AND negative energy above hull...")
    dual_negative = df_filtered[
        (df_filtered['e_form'] < 0) &
        (df_filtered['e_above_hull'] < 0)
    ].sort_values('e_above_hull')

    print(f"   Found {len(dual_negative)} compounds\n")

    if len(dual_negative) > 0:
        print(f"   {'Rank':<6} {'Compound':<20} {'E_form':<12} {'E_hull':<12}")
        print("   " + "-" * 60)

        for i, (_, row) in enumerate(dual_negative.head(20).iterrows(), 1):
            print(f"   {i:<6} {row['name']:<20} "
                  f"{row['e_form']:<12.4f} {row['e_above_hull']:<12.4f}")
    else:
        print("   No compounds found with both values negative")

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE!")
    print("=" * 80)
    print(f"\nPlot saved to: {output_file}")
    print(f"Data points plotted: {len(df_filtered):,}")
    print(f"Data points excluded (e_decomp = 0): {len(df) - len(df_filtered):,}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
