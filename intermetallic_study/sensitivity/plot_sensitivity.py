"""
Plot sensitivity analysis results for MCTS hyperparameter studies.

Generates 6 publication-quality figures (3"×3" each) showing:
- Number of unique compounds explored vs. best reward found
- One figure per hyperparameter study
- Summary statistics tables for sample efficiency analysis

© 2026. Triad National Security, LLC. All rights reserved.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_trajectory(result_dir: Path) -> Tuple[List[int], List[float]]:
    """
    Load trajectory from MCTS results.

    Returns:
        (unique_compounds, best_rewards) where each list element is cumulative count/best
    """
    # Load trajectory from convergence.csv
    convergence_path = result_dir / "convergence.csv"
    if not convergence_path.exists():
        return [], []

    df = pd.read_csv(convergence_path)

    # Extract unique_materials and best_reward columns
    unique_compounds = df['unique_materials'].tolist()
    best_rewards = df['best_reward'].tolist()

    return unique_compounds, best_rewards


def calculate_metrics(
    unique: List[int],
    rewards: List[float],
    design_space_size: int = 108
) -> Dict[str, float]:
    """
    Calculate sample efficiency metrics for MCTS run.

    Args:
        unique: List of unique compounds explored at each iteration
        rewards: List of best rewards at each iteration
        design_space_size: Total compounds in design space (108 for U-only)

    Returns:
        Dict with metrics:
        - final_reward: Best reward achieved
        - final_coverage: % of design space explored
        - compounds_to_90: Compounds needed to reach 90% of final reward
        - compounds_to_95: Compounds needed to reach 95% of final reward
        - best_reward_first_20: Best reward in first 20 compounds
        - best_reward_first_50: Best reward in first 50 compounds
    """
    if not unique or not rewards:
        return {
            "final_reward": 0.0,
            "final_coverage": 0.0,
            "compounds_to_90": None,
            "compounds_to_95": None,
            "best_reward_first_20": None,
            "best_reward_first_50": None,
        }

    final_reward = rewards[-1]
    final_coverage = (unique[-1] / design_space_size) * 100

    # Find compounds needed to reach 90% and 95% of final reward
    threshold_90 = final_reward * 0.90
    threshold_95 = final_reward * 0.95

    compounds_to_90 = None
    compounds_to_95 = None

    for u, r in zip(unique, rewards):
        if compounds_to_90 is None and r >= threshold_90:
            compounds_to_90 = u
        if compounds_to_95 is None and r >= threshold_95:
            compounds_to_95 = u

    # Best reward in first N compounds
    best_reward_first_20 = None
    best_reward_first_50 = None

    for u, r in zip(unique, rewards):
        if u <= 20:
            if best_reward_first_20 is None or r > best_reward_first_20:
                best_reward_first_20 = r
        if u <= 50:
            if best_reward_first_50 is None or r > best_reward_first_50:
                best_reward_first_50 = r
        if u > 50:
            break

    return {
        "final_reward": final_reward,
        "final_coverage": final_coverage,
        "compounds_to_90": compounds_to_90,
        "compounds_to_95": compounds_to_95,
        "best_reward_first_20": best_reward_first_20,
        "best_reward_first_50": best_reward_first_50,
    }


def plot_sensitivity_panel(
    ax: plt.Axes,
    data: Dict[str, Tuple[List[int], List[float]]],
    param_name: str,
    param_labels: List[str],
    title: str,
):
    """
    Plot one sensitivity panel with reference lines.

    Args:
        ax: Matplotlib axes
        data: Dict mapping config_name -> (unique_compounds, best_rewards)
        param_name: Parameter being varied (for legend)
        param_labels: Human-readable labels for each config
        title: Plot title
    """
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(data)))

    # Find best reward across all configs for reference lines
    max_reward = 0.0
    for unique, rewards in data.values():
        if rewards:
            max_reward = max(max_reward, max(rewards))

    # Plot trajectories
    for (config_name, (unique, rewards)), color, label in zip(
        sorted(data.items()), colors, param_labels
    ):
        if not unique or not rewards:
            continue
        ax.plot(unique, rewards, '-o', markersize=3, linewidth=1.5,
                color=color, label=label, alpha=0.8)

    # Add reference lines
    if max_reward > 0:
        # 90% and 95% thresholds
        ax.axhline(max_reward * 0.90, color='gray', linestyle='--',
                   linewidth=0.8, alpha=0.5, label='90% best')
        ax.axhline(max_reward * 0.95, color='gray', linestyle=':',
                   linewidth=0.8, alpha=0.5, label='95% best')

    # 50% coverage marker (54 compounds for 108-compound space)
    ax.axvline(54, color='red', linestyle='--', linewidth=0.8,
               alpha=0.4, label='50% coverage')

    ax.set_xlabel("Unique Compounds Explored", fontsize=9)
    ax.set_ylabel("Best Reward Found", fontsize=9)
    ax.set_xlim(0, 108)  # Full design space
    ax.legend(fontsize=6, loc='lower right', framealpha=0.9, ncol=1)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.2, linewidth=0.5)


def generate_summary_statistics(
    sensitivity_dir: Path,
    studies: List[Dict],
    output_dir: Path
):
    """
    Generate summary statistics tables for all studies.

    Args:
        sensitivity_dir: Path to sensitivity/ directory
        studies: List of study definitions
        output_dir: Where to save summary tables
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []

    for study in studies:
        study_dir = sensitivity_dir / study["name"] / "results"

        for config_name, label in zip(study["configs"], study["labels"]):
            result_dir = study_dir / config_name
            if not result_dir.exists():
                continue

            unique, rewards = load_trajectory(result_dir)
            if not unique or not rewards:
                continue

            metrics = calculate_metrics(unique, rewards)

            all_stats.append({
                "study": study["name"],
                "config": config_name,
                "label": label,
                **metrics
            })

    if not all_stats:
        print("No data found for summary statistics")
        return

    # Create DataFrame
    df = pd.DataFrame(all_stats)

    # Save full table as CSV
    csv_path = output_dir / "summary_statistics.csv"
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"Saved summary statistics: {csv_path}")

    # Generate per-study summary tables
    for study in studies:
        study_df = df[df['study'] == study['name']]
        if study_df.empty:
            continue

        print(f"\n{'='*80}")
        print(f"{study['title']}")
        print(f"{'='*80}")

        # Format table for display
        display_df = study_df[['label', 'final_reward', 'final_coverage',
                                'compounds_to_90', 'compounds_to_95',
                                'best_reward_first_20', 'best_reward_first_50']].copy()

        display_df.columns = ['Config', 'Final Reward', 'Coverage %',
                              'To 90%', 'To 95%', 'Best@20', 'Best@50']

        print(display_df.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
        print()

        # Interpretation
        best_config = study_df.loc[study_df['final_reward'].idxmax()]
        most_efficient = study_df.loc[study_df['compounds_to_95'].idxmin()] if study_df['compounds_to_95'].notna().any() else None

        print("Interpretation:")
        print(f"  Best final reward: {best_config['label']} ({best_config['final_reward']:.4f})")
        if most_efficient is not None:
            print(f"  Most sample-efficient (to 95%): {most_efficient['label']} ({most_efficient['compounds_to_95']:.0f} compounds)")

        # Check for over-exploration
        high_coverage = study_df[study_df['final_coverage'] > 75]
        if not high_coverage.empty:
            print(f"  WARNING: High coverage (>75%) detected for: {', '.join(high_coverage['label'].tolist())}")
            print(f"           These configs may be over-exploring (approaching exhaustive search)")


def generate_sensitivity_figures(sensitivity_dir: Path, output_dir: Path):
    """
    Generate all six sensitivity analysis figures and summary statistics.

    Args:
        sensitivity_dir: Path to intermetallic_study/sensitivity/
        output_dir: Where to save figures
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    studies = [
        {
            "name": "starting_material",
            "title": "Starting Material Sensitivity",
            "configs": ["cr_sn", "fe_sn", "cu_sn", "ni_ge", "w_pb"],
            "labels": ["Cr₆Sn₆U", "Fe₆Sn₆U", "Cu₆Sn₆U", "Ni₆Ge₆U", "W₆Pb₆U"],
        },
        {
            "name": "iterations",
            "title": "Iterations Sensitivity",
            "configs": ["iter_250", "iter_500", "iter_1000", "iter_2000", "iter_4000"],
            "labels": ["250", "500", "1000", "2000", "4000"],
        },
        {
            "name": "termination_limit",
            "title": "Termination Limit Sensitivity",
            "configs": ["limit_25", "limit_50", "limit_100", "limit_200", "limit_500"],
            "labels": ["25", "50", "100", "200", "500"],
        },
        {
            "name": "rollout_depth",
            "title": "Rollout Depth Sensitivity",
            "configs": ["depth_1", "depth_2", "depth_3", "depth_5"],
            "labels": ["1", "2", "3", "5"],
        },
        {
            "name": "n_rollout",
            "title": "N Rollout Sensitivity",
            "configs": ["rollout_0", "rollout_1", "rollout_2", "rollout_4", "rollout_8"],
            "labels": ["0", "1", "2", "4", "8"],
        },
        {
            "name": "move_step",
            "title": "Move Step Sensitivity",
            "configs": ["step_1", "step_2", "step_3", "step_5"],
            "labels": ["1", "2", "3", "5"],
        },
    ]

    # Generate figures
    for study in studies:
        print(f"Generating {study['name']} figure...")

        # Load data for all configs in this study
        data = {}
        study_dir = sensitivity_dir / study["name"] / "results"

        for config_name in study["configs"]:
            result_dir = study_dir / config_name
            if result_dir.exists():
                trajectory = load_trajectory(result_dir)
                data[config_name] = trajectory
            else:
                print(f"  Warning: {result_dir} not found, skipping")

        if not data:
            print(f"  No data found for {study['name']}, skipping")
            continue

        # Create 3"×3" figure
        fig, ax = plt.subplots(figsize=(3, 3), dpi=300)

        plot_sensitivity_panel(
            ax=ax,
            data=data,
            param_name=study["name"],
            param_labels=study["labels"],
            title="",
        )

        plt.tight_layout()

        # Save figure
        output_path = output_dir / f"{study['name']}_sensitivity.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")

        plt.close()

    # Generate summary statistics
    print("\n" + "="*80)
    print("GENERATING SUMMARY STATISTICS")
    print("="*80)
    generate_summary_statistics(sensitivity_dir, studies, output_dir)


def main():
    """Generate all sensitivity figures and summary statistics."""
    # Script is in sensitivity/ directory
    sensitivity_dir = Path(__file__).parent
    output_dir = sensitivity_dir / "figures"

    print("="*80)
    print("MCTS Sensitivity Analysis - Figures & Statistics Generator")
    print("="*80)
    print()

    generate_sensitivity_figures(sensitivity_dir, output_dir)

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nOutputs saved to: {output_dir}/")
    print("  - 6 sensitivity figures (PNG)")
    print("  - summary_statistics.csv")
    print()


if __name__ == "__main__":
    main()
