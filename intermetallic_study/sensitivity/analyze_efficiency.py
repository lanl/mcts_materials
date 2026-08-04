"""
Analyze MCTS sample efficiency across sensitivity studies.

This script generates detailed metrics to evaluate whether MCTS is effective
at guided search vs. just doing exhaustive high-throughput screening.

Key metrics:
- Sample efficiency: compounds needed to reach 90%/95% of best reward
- Coverage at convergence: % of design space explored
- Early performance: best reward in first 20/50 compounds
- Over-exploration detection: flags configs with >75% coverage

Usage:
    python analyze_efficiency.py

© 2026. Triad National Security, LLC. All rights reserved.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def load_trajectory(result_dir: Path) -> Tuple[List[int], List[float]]:
    """Load trajectory from convergence.csv."""
    convergence_path = result_dir / "convergence.csv"
    if not convergence_path.exists():
        return [], []

    df = pd.read_csv(convergence_path)
    unique_compounds = df['unique_materials'].tolist()
    best_rewards = df['best_reward'].tolist()

    return unique_compounds, best_rewards


def calculate_metrics(
    unique: List[int],
    rewards: List[float],
    design_space_size: int = 108
) -> Dict[str, float]:
    """
    Calculate sample efficiency metrics.

    Args:
        unique: Unique compounds explored at each iteration
        rewards: Best rewards at each iteration
        design_space_size: Total compounds in design space (108 for U-only)

    Returns:
        Dict with efficiency metrics
    """
    if not unique or not rewards:
        return {
            "final_reward": 0.0,
            "final_coverage": 0.0,
            "compounds_to_90": None,
            "compounds_to_95": None,
            "best_reward_first_20": None,
            "best_reward_first_50": None,
            "efficiency_90": None,  # final_reward / compounds_to_90
            "efficiency_95": None,  # final_reward / compounds_to_95
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

    # Efficiency metrics
    efficiency_90 = final_reward / compounds_to_90 if compounds_to_90 else None
    efficiency_95 = final_reward / compounds_to_95 if compounds_to_95 else None

    return {
        "final_reward": final_reward,
        "final_coverage": final_coverage,
        "compounds_to_90": compounds_to_90,
        "compounds_to_95": compounds_to_95,
        "best_reward_first_20": best_reward_first_20,
        "best_reward_first_50": best_reward_first_50,
        "efficiency_90": efficiency_90,
        "efficiency_95": efficiency_95,
    }


def analyze_all_studies(sensitivity_dir: Path):
    """Generate efficiency analysis for all studies."""

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
        print("No data found. Run experiments first.")
        return

    # Create DataFrame
    df = pd.DataFrame(all_stats)

    # Save full table as CSV
    output_dir = sensitivity_dir / "figures"
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "efficiency_metrics.csv"
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"Saved efficiency metrics: {csv_path}\n")

    # Print per-study analysis
    for study in studies:
        study_df = df[df['study'] == study['name']]
        if study_df.empty:
            continue

        print("="*90)
        print(f"{study['title']}")
        print("="*90)

        # Main metrics table
        display_df = study_df[['label', 'final_reward', 'final_coverage',
                                'compounds_to_90', 'compounds_to_95',
                                'best_reward_first_20', 'best_reward_first_50']].copy()

        display_df.columns = ['Config', 'Final Reward', 'Coverage %',
                              'To 90%', 'To 95%', 'Best@20', 'Best@50']

        print(display_df.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
        print()

        # Interpretation
        best_final = study_df.loc[study_df['final_reward'].idxmax()]
        print(f"Best final reward: {best_final['label']} ({best_final['final_reward']:.4f})")

        # Sample efficiency
        valid_90 = study_df[study_df['compounds_to_90'].notna()]
        if not valid_90.empty:
            most_efficient_90 = valid_90.loc[valid_90['compounds_to_90'].idxmin()]
            print(f"Most sample-efficient (90%): {most_efficient_90['label']} "
                  f"({int(most_efficient_90['compounds_to_90'])} compounds, "
                  f"{most_efficient_90['compounds_to_90']/108*100:.1f}% coverage)")

        valid_95 = study_df[study_df['compounds_to_95'].notna()]
        if not valid_95.empty:
            most_efficient_95 = valid_95.loc[valid_95['compounds_to_95'].idxmin()]
            print(f"Most sample-efficient (95%): {most_efficient_95['label']} "
                  f"({int(most_efficient_95['compounds_to_95'])} compounds, "
                  f"{most_efficient_95['compounds_to_95']/108*100:.1f}% coverage)")

        # Early performance (fast mode proxy)
        valid_20 = study_df[study_df['best_reward_first_20'].notna()]
        if not valid_20.empty:
            best_early = valid_20.loc[valid_20['best_reward_first_20'].idxmax()]
            print(f"Best early performance (first 20): {best_early['label']} "
                  f"(reward={best_early['best_reward_first_20']:.4f})")

        # Over-exploration warning
        high_coverage = study_df[study_df['final_coverage'] > 75]
        if not high_coverage.empty:
            print(f"\n  WARNING: High coverage (>75%) detected for: {', '.join(high_coverage['label'].tolist())}")
            print("   These configs may be over-exploring (approaching exhaustive search)")
            print("   MCTS should find good materials with <50% coverage in this 108-compound space")

        # Efficiency assessment
        print("\nSample Efficiency Assessment:")
        for _, row in study_df.iterrows():
            if pd.notna(row['compounds_to_95']):
                cov = row['compounds_to_95'] / 108 * 100
                if cov < 25:
                    rating = "EXCELLENT"
                elif cov < 50:
                    rating = "GOOD"
                elif cov < 75:
                    rating = "MARGINAL"
                else:
                    rating = "POOR (near-exhaustive)"
                print(f"  {row['label']:15s}: {rating:20s} ({row['compounds_to_95']:.0f} compounds to 95%)")

        print()

    # Cross-study summary
    print("="*90)
    print("CROSS-STUDY SUMMARY")
    print("="*90)

    # Best configs by different criteria
    print("\nBest configurations by criterion:")
    print(f"  Highest final reward: {df.loc[df['final_reward'].idxmax()]['study']} / {df.loc[df['final_reward'].idxmax()]['label']}")

    valid_eff = df[df['compounds_to_95'].notna()]
    if not valid_eff.empty:
        best_eff_row = valid_eff.loc[valid_eff['compounds_to_95'].idxmin()]
        print(f"  Most sample-efficient: {best_eff_row['study']} / {best_eff_row['label']} "
              f"({int(best_eff_row['compounds_to_95'])} compounds)")

    valid_early = df[df['best_reward_first_20'].notna()]
    if not valid_early.empty:
        best_early_row = valid_early.loc[valid_early['best_reward_first_20'].idxmax()]
        print(f"  Best fast mode (first 20): {best_early_row['study']} / {best_early_row['label']} "
              f"(reward={best_early_row['best_reward_first_20']:.4f})")

    print(f"\nMean coverage across all configs: {df['final_coverage'].mean():.1f}%")
    print(f"Configs with >75% coverage: {len(df[df['final_coverage'] > 75])} / {len(df)}")

    print()


def main():
    """Run efficiency analysis."""
    sensitivity_dir = Path(__file__).parent

    print("="*90)
    print("MCTS Sample Efficiency Analysis")
    print("="*90)
    print()
    print("Evaluating whether MCTS outperforms random/exhaustive search")
    print("Good MCTS: finds best materials with <50% design space coverage")
    print("Bad MCTS: requires >75% coverage (essentially high-throughput screening)")
    print()

    analyze_all_studies(sensitivity_dir)

    print("="*90)
    print("ANALYSIS COMPLETE")
    print("="*90)
    print()


if __name__ == "__main__":
    main()
