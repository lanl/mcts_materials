"""
Visualization tools for MCTS tree and results.
"""

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import math
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from matplotlib.patches import Rectangle, Patch
from collections import defaultdict
from .node import MCTSTreeNode
from .mcts import MCTS


class TreeVisualizer:
    """
    Visualization tools for MCTS tree structure and results.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer.
        
        Args:
            figsize: Figure size for plots
        """
        self.figsize = figsize
        
        # Define element groups
        self.transition_metals = {
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
            'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg'
        }
        self.group_IV = {'C', 'Si', 'Ge', 'Sn', 'Pb'}
    
    def formula_to_elements(self, formula: str) -> List[str]:
        """Extract element symbols from chemical formula."""
        return re.findall(r'[A-Z][a-z]?', formula)
        
    def plot_tree_expansion(self, mcts: MCTS, max_depth: int = 3, 
                          save_path: Optional[str] = None):
        """
        Plot the MCTS tree expansion.
        
        Args:
            mcts: MCTS instance
            max_depth: Maximum depth to visualize
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Build networkx graph from MCTS tree
        G = nx.DiGraph()
        pos = {}
        node_colors = []
        node_sizes = []
        node_labels = {}
        
        # Traverse tree and build graph
        self._build_graph_recursive(
            G, pos, node_colors, node_sizes, node_labels,
            mcts.root, 0, 0, max_depth, 0
        )
        
        # Draw the graph
        nx.draw(G, pos, ax=ax,
                node_color=node_colors,
                node_size=node_sizes,
                with_labels=False,
                arrows=True,
                edge_color='gray',
                arrowsize=20,
                alpha=0.8)
        
        # Add labels
        for node, (x, y) in pos.items():
            if node in node_labels:
                ax.text(x, y, node_labels[node], 
                       ha='center', va='center', 
                       fontsize=8, weight='bold')
        
        ax.set_title('MCTS Tree Expansion', fontsize=16, weight='bold')
        ax.axis('off')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='lightblue', markersize=10,
                      label='Unexplored'),
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='orange', markersize=10,
                      label='Explored'),
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='red', markersize=15,
                      label='Best Node')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Tree visualization saved to {save_path}")
            
        return fig
        
    def _build_graph_recursive(self, G, pos, node_colors, node_sizes, node_labels,
                             node: MCTSTreeNode, x: float, y: float, 
                             max_depth: int, current_depth: int):
        """
        Recursively build networkx graph from MCTS tree.
        """
        if current_depth > max_depth:
            return
            
        node_id = id(node)
        G.add_node(node_id)
        pos[node_id] = (x, y)
        
        # Color based on visit count and best node
        if node.t_of_visit == 0:
            node_colors.append('lightblue')
            node_sizes.append(300)
        elif hasattr(node, '_is_best') and node._is_best:
            node_colors.append('red')
            node_sizes.append(500)
        else:
            node_colors.append('orange')
            node_sizes.append(max(300, min(1000, node.t_of_visit * 50)))
            
        # Add label with chemical formula
        formula = node.get_chemical_formula()
        short_formula = formula[:6] + '...' if len(formula) > 8 else formula
        node_labels[node_id] = short_formula
        
        # Add children
        if node.children and current_depth < max_depth:
            n_children = len(node.children)
            child_spacing = 2.0 / max(1, n_children - 1) if n_children > 1 else 0
            start_x = x - 1.0
            
            for i, child in enumerate(node.children):
                child_x = start_x + i * child_spacing
                child_y = y - 1.5
                
                child_id = id(child)
                G.add_edge(node_id, child_id)
                
                self._build_graph_recursive(
                    G, pos, node_colors, node_sizes, node_labels,
                    child, child_x, child_y, max_depth, current_depth + 1
                )
                
    def plot_energy_distribution(self, stat_dict: Dict, top_n: int = 20,
                               save_path: Optional[str] = None, csv_file: Optional[str] = None):
        """
        Plot distribution of formation energies found.
        Uses compound names from MCTS tree but energy values from CSV file for accurate plotting/sorting.
        
        Args:
            stat_dict: Statistics dictionary from MCTS
            top_n: Number of top compounds to show
            save_path: Path to save the plot
            csv_file: Path to CSV file with actual formation energies
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Convert to DataFrame
        df = pd.DataFrame(stat_dict).T
        if df.empty:
            print("No data to plot")
            return fig
            
        df.columns = ['best_reward', 'visit_count', 'terminated', 'e_above_hull', 'e_form']
        df = df.reset_index()
        df.columns = ['formula'] + list(df.columns[1:])
        
        # Load formation energy values from CSV file if provided for more accurate data
        csv_energy_lookup = {}
        if csv_file and Path(csv_file).exists():
            try:
                csv_df = pd.read_csv(csv_file)
                if 'name' in csv_df.columns and 'e_form' in csv_df.columns:
                    csv_energy_lookup = dict(zip(csv_df['name'], csv_df['e_form']))
                    print(f"Loaded {len(csv_energy_lookup)} formation energy values from CSV")
            except Exception as e:
                print(f"Warning: Could not load CSV formation energy data: {e}")
        
        # Update formation energy values from CSV where available
        for idx, row in df.iterrows():
            formula = row['formula']
            if formula in csv_energy_lookup:
                df.at[idx, 'e_form'] = csv_energy_lookup[formula]
        
        df = df.sort_values('e_form')
        
        # Plot 1: Top compounds by formation energy
        top_df = df.head(top_n)
        
        bars1 = ax1.bar(range(len(top_df)), top_df['e_form'], 
                       color='skyblue', alpha=0.7)
        ax1.set_xlabel('Compounds (sorted by formation energy)')
        ax1.set_ylabel('Formation Energy (eV/atom)')
        ax1.set_title(f'Top {top_n} Compounds by Formation Energy')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add compound labels on x-axis (rotated)
        ax1.set_xticks(range(len(top_df)))
        ax1.set_xticklabels(top_df['formula'], rotation=45, ha='right', fontsize=8)
        
        # Plot 2: Visit count vs Formation energy scatter
        ax2.scatter(df['visit_count'], df['e_form'], alpha=0.6, c='orange')
        ax2.set_xlabel('Visit Count')
        ax2.set_ylabel('Formation Energy (eV/atom)')
        ax2.set_title('Formation Energy vs Visit Count')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Energy distribution plot saved to {save_path}")
            
        return fig
    
    def _load_formation_energies_lookup(self, csv_file: str) -> dict:
        """Load formation energies from CSV file into lookup dict."""
        import os
        lookup = {}
        try:
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                for _, row in df.iterrows():
                    formula = row['name']
                    e_form = float(row['e_form'])
                    lookup[formula] = e_form
        except Exception as e:
            print(f"Warning: Could not load formation energies from {csv_file}: {e}")
        return lookup
    
    def _normalize_formula_for_lookup(self, formula: str) -> str:
        """Normalize chemical formula for matching."""
        try:
            import re
            pattern = r'([A-Z][a-z]?)(\d*)'
            matches = re.findall(pattern, formula)
            element_counts = {}
            for element, count in matches:
                count = int(count) if count else 1
                element_counts[element] = element_counts.get(element, 0) + count
            sorted_elements = sorted(element_counts.keys())
            normalized_parts = []
            for element in sorted_elements:
                count = element_counts[element]
                if count == 1:
                    normalized_parts.append(element)
                else:
                    normalized_parts.append(f"{element}{count}")
            return ''.join(normalized_parts)
        except Exception:
            return formula
    
    def _get_formation_energy_from_lookup(self, formula: str, lookup: dict) -> Optional[float]:
        """Get formation energy from lookup dict with flexible matching."""
        # Try exact match first
        if formula in lookup:
            return lookup[formula]
        
        # Try normalized matching
        normalized_input = self._normalize_formula_for_lookup(formula)
        for cached_formula, e_form in lookup.items():
            if self._normalize_formula_for_lookup(cached_formula) == normalized_input:
                return e_form
        
        return None
    
    def plot_iteration_progress(self, mcts: MCTS, save_path: Optional[str] = None, csv_file: Optional[str] = None):
        """
        Plot iteration progress showing cumulative discovery of compounds by formation energy.
        
        Args:
            mcts: MCTS instance
            save_path: Path to save the plot
            csv_file: Path to CSV file with actual formation energies
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        if not mcts.stat_dict:
            # Create empty plots if no data
            ax1.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax1.transAxes)
            ax2.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax2.transAxes)
            plt.suptitle('MCTS Search Progress (No Data)', fontsize=14)
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Iteration progress plot saved to {save_path}")
            return fig
            
        # Convert stat_dict to DataFrame
        df = pd.DataFrame(mcts.stat_dict).T
        df.columns = ['best_reward', 'visit_count', 'terminated', 'e_above_hull', 'e_form']
        df = df.reset_index()
        df.columns = ['formula'] + list(df.columns[1:])
        
        # Simulate discovery order based on visit counts (higher visit count = discovered earlier)
        # This is an approximation since we don't have exact iteration timing
        total_compounds = len(df)
        
        # Sort by visit count (descending) to approximate discovery order
        df_sorted = df.sort_values('visit_count', ascending=False)
        
        # Create simulated iteration discovery
        discovery_iterations = []
        cumulative_best_rewards = []
        cumulative_compounds = []
        
        current_best_reward = -float('inf')
        
        for i, (_, row) in enumerate(df_sorted.iterrows()):
            iteration = i + 1
            reward = row['best_reward']
            
            # Update best reward if this is better
            if reward > current_best_reward:
                current_best_reward = reward
                
            discovery_iterations.append(iteration)
            cumulative_best_rewards.append(current_best_reward)
            cumulative_compounds.append(i + 1)
        
        # Plot 1: Best reward discovery over time
        color1 = 'tab:red'
        ax1.plot(discovery_iterations, cumulative_best_rewards, color=color1, linewidth=2, marker='o', markersize=4)
        ax1.set_xlabel('Compound Discovery Order')
        ax1.set_ylabel('Best Reward Found', color=color1)
        ax1.set_title('Best Reward Discovery Progress')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(alpha=0.3)
        
        # Add annotations for significant improvements
        for i in range(1, len(cumulative_best_rewards)):
            if cumulative_best_rewards[i] > cumulative_best_rewards[i-1]:
                formula = df_sorted.iloc[i]['formula']
                ax1.annotate(f'{formula[:8]}', 
                           (discovery_iterations[i], cumulative_best_rewards[i]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
        
        # Plot 2: Cumulative compounds discovered vs formation energy distribution
        # The e_form is now directly available in the stat_dict
        formation_energies = df_sorted['e_form'].values
        
        ax2.plot(discovery_iterations, formation_energies, color='tab:blue', linewidth=2, marker='s', markersize=4, alpha=0.7)
        ax2.set_xlabel('Compound Discovery Order')
        ax2.set_ylabel('Formation Energy (eV/atom)', color='tab:blue')
        ax2.set_title('Formation Energy Discovery Pattern')
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        ax2.grid(alpha=0.3)
        
        # Add horizontal line for stable compounds (E_form < 0)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Stability threshold')
        ax2.legend()
        
        plt.suptitle(f'MCTS Search Progress Analysis ({total_compounds} compounds discovered)', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Iteration progress plot saved to {save_path}")
            
        return fig
        
    def plot_energy_above_hull_distribution(self, stat_dict: Dict, top_n: int = 20,
                                           save_path: Optional[str] = None, csv_file: Optional[str] = None):
        """
        Plot distribution of energy above hull values found.
        Uses compound names from MCTS tree but energy values from CSV file for accurate plotting/sorting.
        
        Args:
            stat_dict: Statistics dictionary from MCTS
            top_n: Number of top compounds to show
            save_path: Path to save the plot
            csv_file: Path to CSV file with actual energy above hull values
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Convert to DataFrame
        df = pd.DataFrame(stat_dict).T
        if df.empty:
            print("No data to plot")
            return fig
            
        df.columns = ['best_reward', 'visit_count', 'terminated', 'e_above_hull', 'e_form']
        df = df.reset_index()
        df.columns = ['formula'] + list(df.columns[1:])
        
        # Load energy values from CSV file if provided for more accurate data
        csv_energy_lookup = {}
        if csv_file and Path(csv_file).exists():
            try:
                csv_df = pd.read_csv(csv_file)
                if 'name' in csv_df.columns and 'e_above_hull' in csv_df.columns:
                    csv_energy_lookup = dict(zip(csv_df['name'], csv_df['e_above_hull']))
                    print(f"Loaded {len(csv_energy_lookup)} energy above hull values from CSV")
            except Exception as e:
                print(f"Warning: Could not load CSV energy data: {e}")
        
        # Update energy above hull values from CSV where available
        for idx, row in df.iterrows():
            formula = row['formula']
            if formula in csv_energy_lookup:
                df.at[idx, 'e_above_hull'] = csv_energy_lookup[formula]
        
        # Sort by energy above hull (most stable first)
        df = df.sort_values('e_above_hull')
        
        # Plot 1: Top compounds by energy above hull (most stable)
        top_df = df.head(top_n)
        
        # Color bars based on stability (green for stable, red for unstable)
        colors = ['green' if x <= 0 else 'red' for x in top_df['e_above_hull']]
        
        bars1 = ax1.bar(range(len(top_df)), top_df['e_above_hull'], 
                       color=colors, alpha=0.7)
        ax1.set_xlabel('Compounds (sorted by energy above hull)')
        ax1.set_ylabel('Energy Above Hull (eV/atom)')
        ax1.set_title(f'Top {top_n} Most Stable Compounds by Energy Above Hull')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add stability threshold line
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5, 
                   label='Stability threshold (0 eV/atom)')
        ax1.legend()
        
        # Add compound labels on x-axis (rotated)
        ax1.set_xticks(range(len(top_df)))
        ax1.set_xticklabels(top_df['formula'], rotation=45, ha='right', fontsize=8)
        
        # Plot 2: Visit count vs Energy above hull scatter
        # Color points by stability
        colors_scatter = ['green' if x <= 0 else 'red' for x in df['e_above_hull']]
        ax2.scatter(df['visit_count'], df['e_above_hull'], alpha=0.6, c=colors_scatter)
        ax2.set_xlabel('Visit Count')
        ax2.set_ylabel('Energy Above Hull (eV/atom)')
        ax2.set_title('Energy Above Hull vs Visit Count')
        ax2.grid(alpha=0.3)
        
        # Add stability threshold line
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5, 
                   label='Stability threshold')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Energy above hull distribution plot saved to {save_path}")
            
        return fig
    
    def plot_energy_above_hull_progress(self, mcts: MCTS, save_path: Optional[str] = None, csv_file: Optional[str] = None):
        """
        Plot iteration progress showing best energy above hull and formation energy on dual y-axis,
        and best reward/iteration in a separate plot.
        
        Args:
            mcts: MCTS instance
            save_path: Path to save the plot
            csv_file: Path to CSV file with actual energy above hull values
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        if not mcts.stat_dict:
            # Create empty plots if no data
            ax1.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax1.transAxes)
            ax2.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax2.transAxes)
            plt.suptitle('MCTS Search Progress - Best Energies and Rewards (No Data)', fontsize=14)
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Energy above hull iteration progress plot saved to {save_path}")
            return fig
            
        # Convert stat_dict to DataFrame
        df = pd.DataFrame(mcts.stat_dict).T
        df.columns = ['best_reward', 'visit_count', 'terminated', 'e_above_hull', 'e_form']
        df = df.reset_index()
        df.columns = ['formula'] + list(df.columns[1:])
        
        # Sort by visit count (descending) to approximate discovery order
        df_sorted = df.sort_values('visit_count', ascending=False)
        
        # Create simulated iteration discovery with tracking of best energies and rewards
        discovery_iterations = []
        best_hull_energies = []
        best_formation_energies = []
        best_rewards = []
        
        current_best_hull_energy = float('inf')
        current_best_formation_energy = float('inf')  
        current_best_reward = float('-inf')
        
        for i, (_, row) in enumerate(df_sorted.iterrows()):
            iteration = i + 1
            hull_energy = row['e_above_hull']
            formation_energy = row['e_form']
            reward = row['best_reward']
            
            # Update best (lowest) energy above hull
            if hull_energy < current_best_hull_energy:
                current_best_hull_energy = hull_energy
                
            # Update best (lowest) formation energy
            if formation_energy < current_best_formation_energy:
                current_best_formation_energy = formation_energy
                
            # Update best (highest) reward
            if reward > current_best_reward:
                current_best_reward = reward
                
            discovery_iterations.append(iteration)
            best_hull_energies.append(current_best_hull_energy)
            best_formation_energies.append(current_best_formation_energy)
            best_rewards.append(current_best_reward)
        
        # Plot 1: Single plot with dual y-axis for best energy above hull and formation energy
        ax1_twin = ax1.twinx()
        
        # Best energy above hull (red) on primary y-axis
        line1 = ax1.plot(discovery_iterations, best_hull_energies, 'tab:red', 
                        linewidth=2, marker='o', markersize=3, label='Best Energy Above Hull')
        
        # Best formation energy (blue) on secondary y-axis
        line2 = ax1_twin.plot(discovery_iterations, best_formation_energies, 'tab:blue', 
                             linewidth=2, marker='s', markersize=3, label='Best Formation Energy')
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Energy Above Hull (eV/atom)', color='tab:red')
        ax1_twin.set_ylabel('Formation Energy (eV/atom)', color='tab:blue')
        ax1.set_title('Best Energies Over Iterations')
        ax1.tick_params(axis='y', labelcolor='tab:red')
        ax1_twin.tick_params(axis='y', labelcolor='tab:blue')
        
        # Add horizontal line for stability threshold (E_hull = 0) on primary axis
        ax1.axhline(y=0, color='tab:red', linestyle='--', alpha=0.5, label='Stability threshold')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')
        
        ax1.grid(alpha=0.3)
        
        # Plot 2: Best reward per iteration (separate plot)
        ax2.plot(discovery_iterations, best_rewards, 'tab:green', 
                linewidth=2, marker='D', markersize=3)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Best Reward')
        ax2.set_title('Best Reward Over Iterations')
        ax2.grid(alpha=0.3)
        
        total_compounds = len(df)
        plt.suptitle(f'MCTS Search Progress ({total_compounds} compounds discovered)', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Energy above hull iteration progress plot saved to {save_path}")
            
        return fig
        
    def plot_search_progress(self, mcts: MCTS, save_path: Optional[str] = None):
        """
        Plot search progress over time.
        
        Args:
            mcts: MCTS instance
            save_path: Path to save the plot
        """
        # This would require tracking rewards over iterations
        # For now, create a placeholder plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot([0, 100], [mcts.max_reward, mcts.max_reward], 'r-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Best Reward Found')
        ax.set_title('MCTS Search Progress')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Search progress plot saved to {save_path}")
            
        return fig
        
    def create_summary_plot(self, mcts: MCTS, save_path: Optional[str] = None):
        """
        Create a comprehensive summary plot.
        
        Args:
            mcts: MCTS instance  
            save_path: Path to save the plot
        """
        fig = plt.figure(figsize=(16, 10))
        
        # Tree visualization (top half)
        ax1 = plt.subplot(2, 2, (1, 2))
        
        # Mark best node
        if mcts.best_node:
            mcts.best_node._is_best = True
            
        # Build and plot tree (simplified version)
        self._plot_tree_simple(ax1, mcts.root, max_depth=3)
        ax1.set_title('MCTS Tree Structure', fontsize=14, weight='bold')
        
        # Energy distribution (bottom left)
        ax2 = plt.subplot(2, 2, 3)
        if mcts.stat_dict:
            df = pd.DataFrame(mcts.stat_dict).T
            df.columns = ['best_reward', 'visit_count', 'terminated', 'e_above_hull']
            df['e_form'] = -df['best_reward']
            df = df.sort_values('e_form').head(10)
            
            bars = ax2.bar(range(len(df)), df['e_form'], color='lightcoral', alpha=0.7)
            ax2.set_xlabel('Top 10 Compounds')
            ax2.set_ylabel('Formation Energy (eV/atom)')
            ax2.set_title('Lowest Formation Energies Found')
            ax2.grid(axis='y', alpha=0.3)
        
        # Statistics (bottom right)
        ax3 = plt.subplot(2, 2, 4)
        stats_text = f"""
        MCTS Run Statistics:
        
        Total Compounds Explored: {len(mcts.stat_dict)}
        Best Reward: {mcts.max_reward:.4f}
        Best Formation Energy: {-mcts.max_reward:.4f} eV/atom
        
        Best Compound: {mcts.best_node.get_chemical_formula() if mcts.best_node else 'None'}
        """
        
        ax3.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        ax3.set_title('Summary Statistics', fontsize=14, weight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Summary plot saved to {save_path}")
            
        return fig
        
    def _plot_tree_simple(self, ax, root: MCTSTreeNode, max_depth: int = 3):
        """
        Simple tree plotting for summary view.
        """
        ax.text(0.5, 0.9, f"Root: {root.get_chemical_formula()}", 
               ha='center', va='center', transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='lightgreen'))
        
        if root.children:
            n_children = min(len(root.children), 5)  # Limit for visibility
            for i, child in enumerate(root.children[:n_children]):
                x_pos = 0.1 + (0.8 * i / max(1, n_children - 1))
                ax.text(x_pos, 0.7, f"{child.get_chemical_formula()[:8]}", 
                       ha='center', va='center', transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7),
                       fontsize=8)
                
                # Draw connection line
                ax.plot([0.5, x_pos], [0.85, 0.75], 'k-', alpha=0.5, transform=ax.transAxes)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
    def plot_radial_tree_visualization(self, mcts: MCTS, output_dir: str,
                                     csv_file: Optional[str] = None,
                                     show_labels: bool = True):
        """
        Create radial tree visualization colored by formation energy.

        Args:
            mcts: MCTS instance
            output_dir: Output directory path
            csv_file: Path to MACE calculations CSV file
            show_labels: Whether to display node labels (default: True)
        """
        # Build tree data from MCTS
        tree_data = self._build_tree_data_from_mcts(mcts)
        
        if not tree_data:
            print("No tree data available for visualization")
            return
            
        # Build graph
        G = nx.DiGraph()
        
        for node_id, info in tree_data.items():
            formula = info["formula"]
            ucb = info.get("ucb")
            reward = info.get("best_reward", 0)
            
            if ucb is not None:
                label = f"{formula}\nUCB: {ucb:.2f}\nR: {reward:.2f}"
            else:
                label = f"{formula}\nR: {reward:.2f}"
                
            G.add_node(node_id, label=label, reward=reward)
            
            parent_id = info.get("parent_id")
            if parent_id is not None:
                G.add_edge(parent_id, node_id)
        
        # Find root node
        try:
            root_id = next(node_id for node_id, info in tree_data.items() 
                          if info.get("parent_id") is None)
        except StopIteration:
            print("No root node found in tree data")
            return
        
        # Compute radial layout
        pos = self._radial_layout_scaled(G, root_id, radius_step=2.5)
        
        # Load formation energies from CSV for coloring
        compound_energies = self._load_compound_energies(output_dir, csv_file)
        
        # Create labels with formation energy info (only if show_labels is True)
        labels = {}
        if show_labels:
            for node_id, info in tree_data.items():
                formula = info["formula"]
                ucb = info.get("ucb")
                reward = info.get("best_reward", 0)
                e_form = compound_energies.get(node_id, None)

                if e_form is not None:
                    labels[node_id] = f"{formula}\nE_form: {e_form:.3f}\nR: {reward:.2f}"
                else:
                    labels[node_id] = (f"{formula}\nUCB: {ucb:.2f}\nR: {reward:.2f}"
                                     if ucb is not None else f"{formula}\nR: {reward:.2f}")

        # Color nodes by formation energy
        node_colors = self._get_node_colors(G, root_id, compound_energies)

        # Create the visualization
        plt.figure(figsize=(18, 16))
        nx.draw(G, pos, with_labels=show_labels, labels=labels if show_labels else {},
                node_color=node_colors, edge_color='gray', node_size=2000,
                font_size=8, font_weight='bold', arrows=True,
                connectionstyle='arc3,rad=0.1')
        
        # Add legend
        legend_elements = [
            Patch(facecolor='red', label='Root Node'),
            Patch(facecolor='darkgreen', label='Very Stable (E_form < -0.2 eV/atom)'),
            Patch(facecolor='lightgreen', label='Stable (E_form < 0 eV/atom)'),
            Patch(facecolor='yellow', label='Metastable (0 ≤ E_form < 0.2 eV/atom)'),
            Patch(facecolor='lightcoral', label='Unstable (E_form ≥ 0.2 eV/atom)'),
            Patch(facecolor='lightgray', label='No Energy Data')
        ]
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
        
        plt.title("MCTS Tree Exploration Colored by Formation Energy\n"
                 "(Lower Formation Energy = More Thermodynamically Stable)", 
                  fontsize=14, pad=20)
        plt.axis('equal')
        plt.tight_layout()
        
        # Save the plot
        output_path = Path(output_dir) / 'radial_tree_visualization.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Radial tree visualization saved to {output_path}")
        
    def _build_tree_data_from_mcts(self, mcts: MCTS) -> Dict:
        """Build tree data dictionary from MCTS instance."""
        tree_data = {}
        
        def traverse_node(node: MCTSTreeNode, node_id: int = 0, parent_id: Optional[int] = None):
            """Recursively traverse MCTS tree and build data."""
            # Get node information
            formula = node.get_chemical_formula()
            ucb = node.get_ucb() if node.parent is not None else None
            
            tree_data[node_id] = {
                "formula": formula,
                "ucb": ucb,
                "best_reward": node.get_rewards(total=False),
                "total_reward": node.get_rewards(total=True),
                "visit_count": node.t_of_visit,
                "parent_id": parent_id
            }
            
            # Traverse children
            for i, child in enumerate(node.children):
                child_id = node_id * 100 + i + 1  # Simple ID scheme
                traverse_node(child, child_id, node_id)
        
        # Start traversal from root
        traverse_node(mcts.root, node_id=0)
        
        return tree_data
    
    def _polar_to_cartesian(self, r: float, theta: float) -> Tuple[float, float]:
        """Convert polar coordinates to cartesian."""
        return (r * math.cos(theta), r * math.sin(theta))
    
    def _count_descendants(self, G: nx.DiGraph, node) -> int:
        """Recursively count number of descendants for node (including itself)."""
        children = list(G.successors(node))
        return 1 + sum(self._count_descendants(G, child) for child in children)
    
    def _radial_layout_scaled(self, G: nx.DiGraph, root, radius_step: float = 2.5, 
                            angle_range: Tuple[float, float] = (0, 2 * math.pi), 
                            level: int = 0, pos: Optional[Dict] = None, 
                            total_descendants: Optional[int] = None) -> Dict:
        """Place nodes in radial layout, scaling angular span by subtree size."""
        if pos is None:
            pos = {}
        if total_descendants is None:
            total_descendants = self._count_descendants(G, root)
        
        # Place the root
        if level == 0:
            pos[root] = (0, 0)
        else:
            angle_mid = sum(angle_range) / 2
            pos[root] = self._polar_to_cartesian(level * radius_step, angle_mid)
        
        children = list(G.successors(root))
        if not children:
            return pos
        
        # Allocate angular space proportionally
        angle_start, angle_end = angle_range
        angle_total = angle_end - angle_start
        subtree_sizes = [self._count_descendants(G, child) for child in children]
        total_size = sum(subtree_sizes)
        
        angle_cursor = angle_start
        for i, child in enumerate(children):
            child_span = (angle_total * subtree_sizes[i] / total_size 
                         if total_size > 0 else angle_total / len(children))
            child_angle_range = (angle_cursor, angle_cursor + child_span)
            angle_cursor += child_span
            pos.update(self._radial_layout_scaled(
                G, child, radius_step, child_angle_range, level + 1, pos, total_descendants
            ))
        return pos
    
    def _load_compound_energies(self, output_dir: str, csv_file: Optional[str] = None) -> Dict:
        """Load compound formation energies from CSV file."""
        compound_energies = {}
        
        # Define possible CSV file locations
        if csv_file:
            csv_paths = [Path(csv_file)]
        else:
            csv_paths = [
                Path(output_dir).parent / "mace_calculations.csv",
                Path(output_dir) / "mace_calculations.csv",
                Path("examples/mace_calculations.csv"),
                Path("mace_calculations.csv")
            ]
        
        df = None
        for csv_path in csv_paths:
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    print(f"Loading formation energies from: {csv_path}")
                    break
                except Exception as e:
                    print(f"Failed to load {csv_path}: {e}")
                    continue
        
        if df is not None and 'name' in df.columns and 'e_form' in df.columns:
            # Create a mapping from formula to energy
            energy_map = dict(zip(df['name'], df['e_form']))
            
            # Map node IDs to energies (this is a simplified approach)
            # In practice, you'd want to match formulas properly
            compound_energies = energy_map
        else:
            print("Warning: Could not load formation energies from CSV")
            
        return compound_energies
    
    def plot_formation_energy_by_elements(self, stat_dict: Dict, csv_file: Optional[str] = None, 
                                        save_path: Optional[str] = None):
        """
        Plot formation energy vs transition metal grouped by Group IV element.
        Similar to the provided example code but using MCTS discovered compounds.
        
        Args:
            stat_dict: Statistics dictionary from MCTS
            csv_file: Path to CSV file with formation energies
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Load formation energies from CSV if provided
        energy_by_elements = {}
        if csv_file and Path(csv_file).exists():
            df = pd.read_csv(csv_file)
            for name, e in zip(df['name'], df['e_form']):
                elements = frozenset(self.formula_to_elements(name))
                energy_by_elements[elements] = e
        
        # Gather data for each (Group IV, TM) pair from MCTS tree
        data = defaultdict(dict)  # group_IV → {TM: min(E_form)}
        
        # Convert stat_dict to DataFrame for easier processing
        df_stats = pd.DataFrame(stat_dict).T
        df_stats.columns = ['best_reward', 'visit_count', 'terminated', 'e_above_hull', 'e_form']
        df_stats = df_stats.reset_index()
        df_stats.columns = ['formula'] + list(df_stats.columns[1:])
        
        for _, row in df_stats.iterrows():
            formula = row['formula']
            elements = self.formula_to_elements(formula)
            tm = next((el for el in elements if el in self.transition_metals), None)
            gv = next((el for el in elements if el in self.group_IV), None)
            
            if tm and gv:
                # Use formation energy from stat_dict, or from CSV if available
                e_form_elements = frozenset(elements)
                if e_form_elements in energy_by_elements:
                    e = energy_by_elements[e_form_elements]
                else:
                    e = row['e_form']  # Use from MCTS stat_dict
                
                if tm not in data[gv]:
                    data[gv][tm] = e
                else:
                    data[gv][tm] = min(data[gv][tm], e)  # Keep min value
        
        if not data:
            ax.text(0.5, 0.5, 'No compounds with Group IV + Transition Metal found', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Formation Energy vs. Transition Metal (No Data)')
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            return fig
        
        # Set common x-axis (alphabetically sorted TMs)
        all_tms = sorted({tm for group in data.values() for tm in group})
        
        ax.grid(True)
        
        for gv in sorted(data.keys()):
            y_values = [data[gv].get(tm, np.nan) for tm in all_tms]
            ax.plot(all_tms, y_values, marker='o', label=f'Group IV: {gv}')
        
        ax.set_xlabel("Transition Metal (alphabetical)")
        ax.set_ylabel("Minimum Formation Energy (eV/atom)")
        ax.set_title("Minimum Formation Energy vs. Transition Metal (Grouped by Group IV Element)")
        ax.tick_params(axis='x', rotation=90)
        ax.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Formation energy by elements plot saved to {save_path}")
            
        return fig
    
    def plot_energy_above_hull_by_elements(self, stat_dict: Dict, csv_file: Optional[str] = None, 
                                         save_path: Optional[str] = None):
        """
        Plot energy above hull vs transition metal grouped by Group IV element.
        
        Args:
            stat_dict: Statistics dictionary from MCTS
            csv_file: Path to CSV file with energy above hull values
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Load energy above hull values from CSV if provided
        hull_energy_by_elements = {}
        if csv_file and Path(csv_file).exists():
            df = pd.read_csv(csv_file)
            if 'e_above_hull' in df.columns:
                for name, e_hull in zip(df['name'], df['e_above_hull']):
                    elements = frozenset(self.formula_to_elements(name))
                    hull_energy_by_elements[elements] = e_hull
        
        # Gather data for each (Group IV, TM) pair from MCTS tree
        data = defaultdict(dict)  # group_IV → {TM: min(E_above_hull)}
        
        # Convert stat_dict to DataFrame for easier processing
        df_stats = pd.DataFrame(stat_dict).T
        df_stats.columns = ['best_reward', 'visit_count', 'terminated', 'e_above_hull', 'e_form']
        df_stats = df_stats.reset_index()
        df_stats.columns = ['formula'] + list(df_stats.columns[1:])
        
        for _, row in df_stats.iterrows():
            formula = row['formula']
            elements = self.formula_to_elements(formula)
            tm = next((el for el in elements if el in self.transition_metals), None)
            gv = next((el for el in elements if el in self.group_IV), None)
            
            if tm and gv:
                # Use energy above hull from stat_dict, or from CSV if available
                e_hull_elements = frozenset(elements)
                if e_hull_elements in hull_energy_by_elements:
                    e_hull = hull_energy_by_elements[e_hull_elements]
                else:
                    e_hull = row['e_above_hull']  # Use from MCTS stat_dict
                
                if tm not in data[gv]:
                    data[gv][tm] = e_hull
                else:
                    data[gv][tm] = min(data[gv][tm], e_hull)  # Keep min value (most stable)
        
        if not data:
            ax.text(0.5, 0.5, 'No compounds with Group IV + Transition Metal found', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Energy Above Hull vs. Transition Metal (No Data)')
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            return fig
        
        # Set common x-axis (alphabetically sorted TMs)
        all_tms = sorted({tm for group in data.values() for tm in group})
        
        ax.grid(True)
        
        for gv in sorted(data.keys()):
            y_values = [data[gv].get(tm, np.nan) for tm in all_tms]
            ax.plot(all_tms, y_values, marker='o', label=f'Group IV: {gv}')
        
        # Add stability threshold line
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, 
                  label='Stability threshold (0 eV/atom)')
        
        ax.set_xlabel("Transition Metal (alphabetical)")
        ax.set_ylabel("Minimum Energy Above Hull (eV/atom)")
        ax.set_title("Minimum Energy Above Hull vs. Transition Metal (Grouped by Group IV Element)")
        ax.tick_params(axis='x', rotation=90)
        ax.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Energy above hull by elements plot saved to {save_path}")
            
        return fig
    
    def _get_node_colors(self, G: nx.DiGraph, root_id: int, 
                        compound_energies: Dict) -> List[str]:
        """Determine node colors based on formation energy."""
        node_colors = []
        
        for node in G.nodes():
            if node == root_id:
                node_colors.append('red')
            else:
                # Get formula from node label and look up energy
                node_data = G.nodes[node]
                label = node_data.get('label', '')
                
                # Extract formula from label (first line)
                formula = label.split('\n')[0] if label else ''
                e_form = compound_energies.get(formula, None)
                
                if e_form is not None:
                    if e_form < -0.2:  # Very stable
                        node_colors.append('darkgreen')
                    elif e_form < 0:   # Stable
                        node_colors.append('lightgreen')
                    elif e_form < 0.2:  # Metastable
                        node_colors.append('yellow')
                    else:              # Unstable
                        node_colors.append('lightcoral')
                else:
                    node_colors.append('lightgray')  # No energy data
        
        return node_colors