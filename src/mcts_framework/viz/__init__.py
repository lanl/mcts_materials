"""
Visualization and analysis for completed MCTS runs.

analysis (metrics, ranking, text report) is pure-Python and always importable.
plots (convergence, property distribution, search tree) lazily import
matplotlib/networkx from the optional [viz] extra inside each function.

© 2026. Triad National Security, LLC. All rights reserved.
"""

from .analysis import compute_metrics, rank_materials, generate_report
from .plots import (
    plot_convergence,
    plot_property_distribution,
    plot_search_tree,
)

__all__ = [
    "compute_metrics",
    "rank_materials",
    "generate_report",
    "plot_convergence",
    "plot_property_distribution",
    "plot_search_tree",
]