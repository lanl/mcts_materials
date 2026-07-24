"""
Post-run analysis of MCTS studies (publication figures/tables).

These utilities operate on FINISHED run outputs (the compounds CSV a run writes)
plus the high-throughput MACE cache and DOSCAR peak data - they do not run a
search. The logic is reward/parameter-driven (gamma, beta, reward variant, top-N
size) and reads those parameters from the run's own Config, so a single
implementation reproduces every study variant with no per-study duplicated code
and no chemistry-specific assumptions.

Modules:
    design_space : load the MACE cache + rDOS, compute the composite score, and
                   rank the design space (the "true" ranking used to check MCTS
                   search coverage).
    tables       : LaTeX top-N candidate tables (N configurable).

© 2026. Triad National Security, LLC. All rights reserved.
"""

from .design_space import (
    full_formula_key,
    load_design_space,
    rank_design_space,
    score_by_method,
)
from .tables import write_top_n_table

__all__ = [
    "score_by_method",
    "full_formula_key",
    "load_design_space",
    "rank_design_space",
    "write_top_n_table",
]
