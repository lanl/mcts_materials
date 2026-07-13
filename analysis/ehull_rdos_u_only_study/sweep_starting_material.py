"""Starting-material sweep for this study's convergence_by_starting_material.png.

Four compositions spanning a ladder of move-graph edit distance (in the
substitution-move graph defined in mcts_crystal/node.py) to the true global-
best U-only compound (UZr6Pb6, see compute_global_u_only_ranks in
generate_figures.py): d=2, 4, 6, 8 out of a max possible d=9. This spread
shows convergence speed degrading as the starting point gets farther from
the optimum.

Reuses the shared replicate-running harness (run_sweep/save_sweep_results)
from sensitivity_studies/scripts/common.py - that module also backs the
other (generic MCTS-hyperparameter) sweeps there, so it stays put; only this
sweep is study-specific enough to live alongside generate_figures.py.
Output: sensitivity_studies/results/starting_material_sweep/convergence_data.csv
(path determined by common.py's own location, not this script's).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'sensitivity_studies' / 'scripts'))
from common import run_sweep, save_sweep_results, sweep_result_path

SWEEP_NAME = 'starting_material_sweep'

VALUES = {
    'V6Ge6U (d=2)': dict(transition_metal='V', group_iv='Ge'),
    'Ru6Ge6U (d=4)': dict(transition_metal='Ru', group_iv='Ge'),
    'Pd6Ge6U (d=6)': dict(transition_metal='Pd', group_iv='Ge'),
    'Cu6Ge6U (d=8)': dict(transition_metal='Cu', group_iv='Ge'),
}

if __name__ == '__main__':
    df = run_sweep(SWEEP_NAME, VALUES, checkpoint_path=sweep_result_path(SWEEP_NAME))
    save_sweep_results(df, SWEEP_NAME)
