"""
Plot sensitivity analysis results for MCTS hyperparameter studies.

Generates 4 publication-quality figures (3"×3" each) showing:
- Number of unique compounds explored vs. best reward found
- One figure per hyperparameter: starting_material, termination_limit, rollout_depth, move_step

© 2026. Triad National Security, LLC. All rights reserved.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_trajectory(result_dir: Path) -> Tuple[List[int], List[float]]:
    """
    Load trajectory from MCTS results.

    Returns:
        (unique_compounds, best_rewards) where each list element is cumulative count/best
    """
    import pandas as pd

    # Load trajectory from convergence.csv
    convergence_path = result_dir / "convergence.csv"
    if not convergence_path.exists():
        return [], []

    df = pd.read_csv(convergence_path)

    # Extract unique_materials and best_reward columns
    unique_compounds = df['unique_materials'].tolist()
    best_rewards = df['best_reward'].tolist()

    return unique_compounds, best_rewards


def plot_sensitivity_panel(
    ax: plt.Axes,
    data: Dict[str, Tuple[List[int], List[float]]],
    param_name: str,
    param_labels: List[str],
    title: str,
):
    """
    Plot one sensitivity panel.

    Args:
        ax: Matplotlib axes
        data: Dict mapping config_name -> (unique_compounds, best_rewards)
        param_name: Parameter being varied (for legend)
        param_labels: Human-readable labels for each config
        title: Plot title
    """
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(data)))

    for (config_name, (unique, rewards)), color, label in zip(
        sorted(data.items()), colors, param_labels
    ):
        if not unique or not rewards:
            continue
        ax.plot(unique, rewards, '-o', markersize=3, linewidth=1.5,
                color=color, label=label, alpha=0.8)

    ax.set_xlabel("Unique Compounds Explored", fontsize=9)
    ax.set_ylabel("Best Reward Found", fontsize=9)
    ax.set_xlim(0, 100)
    ax.legend(fontsize=7, loc='lower right', framealpha=0.9)
    ax.tick_params(labelsize=8)


def generate_sensitivity_figures(sensitivity_dir: Path, output_dir: Path):
    """
    Generate all four sensitivity analysis figures.

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
            "name": "move_step",
            "title": "Move Step Sensitivity",
            "configs": ["step_1", "step_2", "step_3", "step_5"],
            "labels": ["1", "2", "3", "5"],
        },
    ]

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


def main():
    """Generate all sensitivity figures."""
    # Script is in sensitivity/ directory
    sensitivity_dir = Path(__file__).parent
    output_dir = sensitivity_dir / "figures"

    print("Generating sensitivity analysis figures...")
    generate_sensitivity_figures(sensitivity_dir, output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
