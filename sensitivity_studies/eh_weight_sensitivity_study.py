#!/usr/bin/env python3
"""
Energy Above Hull Weight Sensitivity Study

Tests different eh_weight values for the weighted rollout method to find
the optimal balance between formation energy and hull stability.

Usage:
    python eh_weight_sensitivity_study.py

NOTE: This study requires energy above hull calculations.
      You must provide your Materials Project API key below.
"""

# CONFIGURATION: Add your Materials Project API key here (REQUIRED for this study)
MP_API_KEY = None  # Get your key from: https://materialsproject.org/api

import sys
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Add the package to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcts_crystal import (
    MCTSTreeNode,
    MCTS,
    MaceEnergyCalculator,
    TreeVisualizer,
    ResultsAnalyzer
)
from ase.io import read


def run_single_mcts_replicate(eh_weight, replicate_id, n_iterations=1000):
    """
    Run a single MCTS replicate with the specified eh_weight.

    Args:
        eh_weight: Weight for energy above hull
        replicate_id: Replicate number (1-10)
        n_iterations: Number of MCTS iterations

    Returns:
        dict: Results dictionary with statistics
    """
    print(f"\n{'='*80}")
    print(f"Running replicate {replicate_id} with eh_weight={eh_weight}")
    print(f"{'='*80}")

    # Load starting structure
    structure_path = Path(__file__).parent.parent / "examples" / "mat_Pb6U1W6_sg191.cif"
    atoms = read(str(structure_path))

    # Set up energy calculator
    csv_file = Path(__file__).parent.parent / "examples" / "mace_calculations.csv"
    energy_calc = MaceEnergyCalculator(
        csv_file=str(csv_file),
        mp_api_key=MP_API_KEY
    )

    # Initialize MCTS
    root_node = MCTSTreeNode(atoms, f_block_mode='u_only', exploration_constant=0.1)
    mcts = MCTS(root_node)

    # Track iteration history manually
    iteration_history = []
    cumulative_best_e_form = float('inf')
    cumulative_best_e_above_hull = float('inf')
    cumulative_best_reward = -float('inf')
    cumulative_best_stable_count = 0  # Count of compounds with both negative

    # Run MCTS iteration by iteration to track progress
    for iter_num in range(n_iterations):
        if mcts.terminated:
            break

        # Selection
        select_chain = mcts.select_node(mode='epsilon')

        if mcts.terminated:
            break

        # Record statistics
        mcts.stat_node_visited()

        # Expansion and simulation
        reward, renew_t_to_terminate = mcts.expansion_simulation(
            rollout_depth=1,
            n_rollout=5,
            energy_calculator=energy_calc,
            rollout_method='weighted',
            eh_weight=eh_weight
        )

        # Back-propagation
        mcts.back_propagation(reward, select_chain, renew_t_to_terminate)

        # Update statistics
        mcts.stat_node_visited()

        # Update cumulative best values by checking ALL discovered compounds in stat_dict
        stable_count = 0
        best_reward_compound = None
        best_reward_e_form = None
        best_reward_e_hull = None

        for formula, stats in mcts.stat_dict.items():
            # stats format: [best_reward, visit_count, terminated, e_above_hull, e_form]
            compound_reward = stats[0]
            e_form = stats[4]
            e_above_hull = stats[3]

            cumulative_best_e_form = min(cumulative_best_e_form, e_form)
            cumulative_best_e_above_hull = min(cumulative_best_e_above_hull, e_above_hull)

            # Track the compound with best reward (what MCTS actually recommends)
            if compound_reward > cumulative_best_reward:
                cumulative_best_reward = compound_reward
                best_reward_compound = formula
                best_reward_e_form = e_form
                best_reward_e_hull = e_above_hull

            # Count stable compounds (both negative)
            if e_form < 0 and e_above_hull < 0:
                stable_count += 1

        cumulative_best_stable_count = max(cumulative_best_stable_count, stable_count)

        # Track history every 10 iterations
        if iter_num % 10 == 0 or iter_num == n_iterations - 1:
            iteration_history.append({
                'iteration': iter_num,
                'best_e_form': cumulative_best_e_form if cumulative_best_e_form != float('inf') else 0.0,
                'best_e_above_hull': cumulative_best_e_above_hull if cumulative_best_e_above_hull != float('inf') else 0.0,
                'best_reward': cumulative_best_reward if cumulative_best_reward != -float('inf') else 0.0,
                'best_reward_compound': best_reward_compound,
                'best_reward_e_form': best_reward_e_form if best_reward_e_form is not None else 0.0,
                'best_reward_e_hull': best_reward_e_hull if best_reward_e_hull is not None else 0.0,
                'stable_count': stable_count,
                'compounds_explored': len(mcts.stat_dict)
            })

    # Final result
    iterations_completed = iter_num + 1 if not mcts.terminated else iter_num

    # Collect statistics
    analyzer = ResultsAnalyzer(csv_file=str(csv_file))
    efficiency = analyzer.analyze_search_efficiency(mcts.stat_dict)

    # Get top compounds
    top_compounds = analyzer.get_top_compounds(mcts.stat_dict, n_top=10)

    # Count fully stable compounds
    stable_compounds = []
    for formula, stats in mcts.stat_dict.items():
        e_form = stats[4]
        e_above_hull = stats[3]
        if e_form < 0 and e_above_hull < 0:
            stable_compounds.append({
                'formula': formula,
                'e_form': e_form,
                'e_above_hull': e_above_hull
            })

    # Find the compound with highest reward (the recommended compound)
    recommended_compound = None
    recommended_reward = -float('inf')
    recommended_e_form = None
    recommended_e_hull = None

    for formula, stats in mcts.stat_dict.items():
        if stats[0] > recommended_reward:
            recommended_reward = stats[0]
            recommended_compound = formula
            recommended_e_form = stats[4]
            recommended_e_hull = stats[3]

    # Package results
    result_data = {
        'replicate_id': replicate_id,
        'eh_weight': eh_weight,
        'iterations_completed': iterations_completed,
        'compounds_explored': len(mcts.stat_dict),
        'best_compound': mcts.best_node.get_chemical_formula() if mcts.best_node else None,
        'best_formation_energy': mcts.best_node.e_form if mcts.best_node else None,
        'best_energy_above_hull': mcts.best_node.e_above_hull if mcts.best_node else None,
        'recommended_compound': recommended_compound,
        'recommended_reward': recommended_reward,
        'recommended_e_form': recommended_e_form,
        'recommended_e_hull': recommended_e_hull,
        'stable_compounds_count': len(stable_compounds),
        'compounds_near_hull_100meV': efficiency['compounds_near_hull_100meV'],
        'search_diversity': efficiency['search_diversity'],
        'top_10_compounds': top_compounds.to_dict('records'),
        'iteration_history': iteration_history,
        'stable_compounds': stable_compounds,
        'stat_dict': {k: v for k, v in mcts.stat_dict.items()}
    }

    print(f"✓ Replicate {replicate_id} completed")
    print(f"  Recommended compound (highest reward): {recommended_compound}")
    if recommended_compound:
        print(f"    Formation energy: {recommended_e_form:.4f} eV/atom")
        print(f"    Energy above hull: {recommended_e_hull:.4f} eV/atom")
        print(f"    Reward: {recommended_reward:.4f}")
    print(f"  Compounds explored: {len(mcts.stat_dict)}")
    print(f"  Fully stable compounds: {len(stable_compounds)}")

    return result_data


def run_sensitivity_study(eh_weights=[1.0, 3.0, 5.0, 7.0, 10.0, 15.0], n_replicates=5, n_iterations=1000):
    """
    Run sensitivity study across different eh_weight values.

    Args:
        eh_weights: List of eh_weight values to test
        n_replicates: Number of replicates per weight
        n_iterations: Number of MCTS iterations per replicate

    Returns:
        list: All results data
    """
    all_results = []

    for weight in eh_weights:
        print(f"\n{'#'*80}")
        print(f"# TESTING EH_WEIGHT: {weight}")
        print(f"{'#'*80}")

        for rep in range(1, n_replicates + 1):
            try:
                result = run_single_mcts_replicate(weight, rep, n_iterations)
                all_results.append(result)
            except Exception as e:
                print(f"ERROR in replicate {rep} for weight {weight}: {e}")
                continue

    return all_results


def create_sensitivity_visualizations(all_results, output_dir):
    """
    Create comprehensive sensitivity visualizations.

    Args:
        all_results: List of result dictionaries
        output_dir: Directory to save visualizations
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Convert to DataFrame for easier analysis
    summary_data = []
    for result in all_results:
        summary_data.append({
            'eh_weight': result['eh_weight'],
            'replicate_id': result['replicate_id'],
            'best_formation_energy': result['best_formation_energy'],
            'best_energy_above_hull': result['best_energy_above_hull'],
            'compounds_explored': result['compounds_explored'],
            'stable_compounds_count': result['stable_compounds_count'],
            'compounds_near_hull_100meV': result['compounds_near_hull_100meV'],
            'search_diversity': result['search_diversity']
        })

    df = pd.DataFrame(summary_data)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'

    # 1. Best Formation Energy vs EH Weight
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='eh_weight', y='best_formation_energy', ax=ax)
    sns.swarmplot(data=df, x='eh_weight', y='best_formation_energy',
                  color='black', alpha=0.5, ax=ax)
    ax.set_xlabel('Energy Above Hull Weight', fontsize=12)
    ax.set_ylabel('Best Formation Energy (eV/atom)', fontsize=12)
    ax.set_title('Best Formation Energy by EH Weight', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'formation_energy_vs_weight.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Best Energy Above Hull vs EH Weight
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='eh_weight', y='best_energy_above_hull', ax=ax)
    sns.swarmplot(data=df, x='eh_weight', y='best_energy_above_hull',
                  color='black', alpha=0.5, ax=ax)
    ax.set_xlabel('Energy Above Hull Weight', fontsize=12)
    ax.set_ylabel('Best Energy Above Hull (eV/atom)', fontsize=12)
    ax.set_title('Best Energy Above Hull by EH Weight', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Convex Hull')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path / 'energy_above_hull_vs_weight.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Stable Compounds Count vs EH Weight
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='eh_weight', y='stable_compounds_count', ax=ax)
    sns.swarmplot(data=df, x='eh_weight', y='stable_compounds_count',
                  color='black', alpha=0.5, ax=ax)
    ax.set_xlabel('Energy Above Hull Weight', fontsize=12)
    ax.set_ylabel('Fully Stable Compounds Found', fontsize=12)
    ax.set_title('Thermodynamically Stable Compounds (e_form < 0 AND e_hull < 0)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'stable_compounds_vs_weight.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Convergence Curves (3 panels: recommended compound properties)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Define colors for different weights
    weights = sorted(df['eh_weight'].unique())
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(weights)))
    color_map = dict(zip(weights, colors))

    for weight in weights:
        weight_results = [r for r in all_results if r['eh_weight'] == weight]

        # Formation energy of RECOMMENDED compound
        for result in weight_results:
            history = result['iteration_history']
            iterations = [h['iteration'] for h in history]
            fe_history = [h['best_reward_e_form'] for h in history]
            axes[0].plot(iterations, fe_history, alpha=0.3, color=color_map[weight])

        # Energy above hull of RECOMMENDED compound
        for result in weight_results:
            history = result['iteration_history']
            iterations = [h['iteration'] for h in history]
            eh_history = [h['best_reward_e_hull'] for h in history]
            axes[1].plot(iterations, eh_history, alpha=0.3, color=color_map[weight])

        # Stable compounds count convergence
        for result in weight_results:
            history = result['iteration_history']
            iterations = [h['iteration'] for h in history]
            stable_history = [h['stable_count'] for h in history]
            axes[2].plot(iterations, stable_history, alpha=0.3, color=color_map[weight])

    # Add mean curves
    for weight in weights:
        weight_results = [r for r in all_results if r['eh_weight'] == weight]

        if weight_results:
            ref_iterations = [h['iteration'] for h in weight_results[0]['iteration_history']]

            # Formation energy of recommended compound mean
            fe_rec_at_iterations = []
            for iter_point in ref_iterations:
                fe_values = []
                for result in weight_results:
                    history = result['iteration_history']
                    for h in history:
                        if h['iteration'] >= iter_point:
                            fe_values.append(h['best_reward_e_form'])
                            break
                if fe_values:
                    fe_rec_at_iterations.append(np.mean(fe_values))

            axes[0].plot(ref_iterations[:len(fe_rec_at_iterations)], fe_rec_at_iterations,
                        linewidth=3, label=f'α={weight}', color=color_map[weight])

            # Energy above hull of recommended compound mean
            eh_rec_at_iterations = []
            for iter_point in ref_iterations:
                eh_values = []
                for result in weight_results:
                    history = result['iteration_history']
                    for h in history:
                        if h['iteration'] >= iter_point:
                            eh_values.append(h['best_reward_e_hull'])
                            break
                if eh_values:
                    eh_rec_at_iterations.append(np.mean(eh_values))

            axes[1].plot(ref_iterations[:len(eh_rec_at_iterations)], eh_rec_at_iterations,
                        linewidth=3, label=f'α={weight}', color=color_map[weight])

            # Stable count mean
            stable_at_iterations = []
            for iter_point in ref_iterations:
                stable_values = []
                for result in weight_results:
                    history = result['iteration_history']
                    for h in history:
                        if h['iteration'] >= iter_point:
                            stable_values.append(h['stable_count'])
                            break
                if stable_values:
                    stable_at_iterations.append(np.mean(stable_values))

            axes[2].plot(ref_iterations[:len(stable_at_iterations)], stable_at_iterations,
                        linewidth=3, label=f'α={weight}', color=color_map[weight])

    axes[0].set_xlabel('Iteration', fontsize=12)
    axes[0].set_ylabel('Formation Energy (eV/atom)', fontsize=12)
    axes[0].set_title('Formation Energy of Recommended Compound\n(Trade-off: Lower weight → Better FE)', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10, title='EH Weight (α)')
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel('Iteration', fontsize=12)
    axes[1].set_ylabel('Energy Above Hull (eV/atom)', fontsize=12)
    axes[1].set_title('Energy Above Hull of Recommended Compound\n(Trade-off: Higher weight → Better EH)', fontsize=14, fontweight='bold')
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Convex Hull')
    axes[1].legend(fontsize=10, title='EH Weight (α)')
    axes[1].grid(alpha=0.3)

    axes[2].set_xlabel('Iteration', fontsize=12)
    axes[2].set_ylabel('Stable Compounds Found', fontsize=12)
    axes[2].set_title('Stable Compound Discovery\n(Both e_form < 0 AND e_hull < 0)', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=10, title='EH Weight (α)')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'convergence_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Pareto Front / Trade-off Visualization
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # For each weight, get top 10 compounds by reward from all replicates
    weight_top_compounds = {}
    weight_recommendations = {}  # Keep for heatmap

    for weight in weights:
        weight_results = [r for r in all_results if r['eh_weight'] == weight]
        compounds_data = []
        all_compounds_by_reward = []

        for result in weight_results:
            # Get recommended compound
            if result['recommended_compound']:
                compounds_data.append({
                    'compound': result['recommended_compound'],
                    'e_form': result['recommended_e_form'],
                    'e_hull': result['recommended_e_hull']
                })

            # Get top 10 compounds by reward from this replicate
            stat_dict = result['stat_dict']
            # Sort by reward (stat_dict[formula] = [best_reward, visit_count, terminated, e_above_hull, e_form])
            sorted_compounds = sorted(stat_dict.items(), key=lambda x: x[1][0], reverse=True)[:10]

            for formula, stats in sorted_compounds:
                all_compounds_by_reward.append({
                    'compound': formula,
                    'e_form': stats[4],
                    'e_hull': stats[3],
                    'reward': stats[0]
                })

        weight_recommendations[weight] = compounds_data
        weight_top_compounds[weight] = all_compounds_by_reward

    # Left panel: Scatter plot showing top compounds by weight
    for weight in weights:
        compounds = weight_top_compounds[weight]
        if compounds:
            e_forms = [c['e_form'] for c in compounds]
            e_hulls = [c['e_hull'] for c in compounds]
            # Vary size by reward rank (higher reward = larger point)
            sizes = [100 + 50 * (10 - i % 10) for i in range(len(compounds))]
            axes[0].scatter(e_forms, e_hulls, s=sizes, alpha=0.5,
                          color=color_map[weight], label=f'α={weight}',
                          edgecolors='black', linewidth=0.5)

    # Mark the ideal region
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Hull Stability')
    axes[0].axvline(x=0, color='blue', linestyle='--', alpha=0.5, linewidth=2, label='Formation Favorability')

    # Shade the "ideal" quadrant
    axes[0].fill_between([-1, 0], -0.1, 0, alpha=0.1, color='green', label='Ideal Region')

    axes[0].set_xlabel('Formation Energy (eV/atom)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Energy Above Hull (eV/atom)', fontsize=14, fontweight='bold')
    axes[0].set_title('Recommended Compounds by Weight\n(Each point = compound from one replicate)',
                     fontsize=15, fontweight='bold')
    axes[0].legend(fontsize=10, loc='best')
    axes[0].grid(alpha=0.3)
    axes[0].set_xlim(-0.8, 0.1)
    axes[0].set_ylim(-0.1, 0.3)

    # Right panel: Show unique compounds and their frequency
    # Aggregate all recommended compounds by weight
    compound_by_weight = {}
    for weight in weights:
        compounds = weight_recommendations[weight]
        compound_counts = {}
        for c in compounds:
            name = c['compound']
            if name not in compound_counts:
                compound_counts[name] = {
                    'count': 0,
                    'e_form': c['e_form'],
                    'e_hull': c['e_hull']
                }
            compound_counts[name]['count'] += 1
        compound_by_weight[weight] = compound_counts

    # Find all unique compounds across all weights
    all_compounds = set()
    for weight_compounds in compound_by_weight.values():
        all_compounds.update(weight_compounds.keys())

    # Create a heatmap showing which compounds are recommended at which weights
    compound_list = sorted(all_compounds)
    weight_list = sorted(weights)

    # Create matrix: rows = compounds, columns = weights
    matrix = np.zeros((len(compound_list), len(weight_list)))
    for i, compound in enumerate(compound_list):
        for j, weight in enumerate(weight_list):
            if compound in compound_by_weight[weight]:
                matrix[i, j] = compound_by_weight[weight][compound]['count']

    # Only show compounds that appear in at least one weight
    active_compounds = [i for i, row in enumerate(matrix) if row.sum() > 0]
    filtered_compounds = [compound_list[i] for i in active_compounds]
    filtered_matrix = matrix[active_compounds, :]

    # Sort by which weight they first appear prominently
    first_appearance = []
    for i, row in enumerate(filtered_matrix):
        if row.sum() > 0:
            # Find weighted average of weight index
            avg_weight_idx = np.average(range(len(row)), weights=row+0.1)
            first_appearance.append(avg_weight_idx)
        else:
            first_appearance.append(999)

    sort_order = np.argsort(first_appearance)
    filtered_compounds = [filtered_compounds[i] for i in sort_order]
    filtered_matrix = filtered_matrix[sort_order, :]

    # Plot heatmap
    im = axes[1].imshow(filtered_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')

    axes[1].set_xticks(range(len(weight_list)))
    axes[1].set_xticklabels([f'{w}' for w in weight_list], fontsize=11)
    axes[1].set_yticks(range(len(filtered_compounds)))
    axes[1].set_yticklabels(filtered_compounds, fontsize=9)

    axes[1].set_xlabel('EH Weight (α)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Recommended Compound', fontsize=14, fontweight='bold')
    axes[1].set_title('Compound Recommendations vs Weight\n(Color intensity = frequency across replicates)',
                     fontsize=15, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[1])
    cbar.set_label('# Replicates', fontsize=11)

    # Add annotations for high-frequency cells
    for i in range(len(filtered_compounds)):
        for j in range(len(weight_list)):
            if filtered_matrix[i, j] > 0:
                text = axes[1].text(j, i, int(filtered_matrix[i, j]),
                                   ha="center", va="center", color="black", fontsize=9)

    # Save without tight_layout (colorbar conflict)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.08)
    plt.savefig(output_path / 'weight_tradeoff_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 6. Summary statistics table
    summary_stats = df.groupby('eh_weight').agg({
        'best_formation_energy': ['mean', 'std', 'min'],
        'best_energy_above_hull': ['mean', 'std', 'min'],
        'stable_compounds_count': ['mean', 'std', 'max'],
        'compounds_explored': ['mean', 'std']
    }).round(4)

    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(summary_stats)

    # Save summary statistics
    summary_stats.to_csv(output_path / 'summary_statistics.csv')

    print(f"\n✓ All visualizations saved to: {output_path.absolute()}")


def main():
    """Main execution function."""
    print("="*80)
    print("EH WEIGHT SENSITIVITY STUDY")
    print("="*80)
    print("\nThis study will:")
    print("  • Test 6 different eh_weight values: 1.0, 3.0, 5.0, 7.0, 10.0, 15.0")
    print("  • Run 5 replicates for each weight")
    print("  • Generate convergence comparison plots")
    print("  • Find optimal weight for discovering stable compounds")
    print("\nTotal runs: 30 (6 weights × 5 replicates)")
    print("Estimated time: ~30-45 minutes (depending on cache hit rate)")

    output_dir = Path(__file__).parent.parent / "eh_weight_sensitivity_results"

    # Run sensitivity study
    all_results = run_sensitivity_study(
        eh_weights=[1.0, 3.0, 5.0, 7.0, 10.0, 15.0],
        n_replicates=5,
        n_iterations=1000
    )

    # Save raw results
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / 'raw_results.json', 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    print(f"\n✓ Raw results saved to: {output_dir / 'raw_results.json'}")

    # Create visualizations
    print("\nGenerating sensitivity visualizations...")
    create_sensitivity_visualizations(all_results, output_dir)

    print("\n" + "="*80)
    print("EH WEIGHT SENSITIVITY STUDY COMPLETE!")
    print("="*80)
    print(f"\n📁 Results directory: {output_dir.absolute()}")
    print("\n📊 Files created:")
    print("   • weight_tradeoff_analysis.png - KEY FIGURE showing compound transitions")
    print("   • convergence_comparison.png")
    print("   • formation_energy_vs_weight.png")
    print("   • energy_above_hull_vs_weight.png")
    print("   • stable_compounds_vs_weight.png")
    print("   • summary_statistics.csv")
    print("   • raw_results.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
