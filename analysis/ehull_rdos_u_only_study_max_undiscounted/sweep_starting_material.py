"""Starting-material sweep for this study's convergence_by_starting_material.png.

Max-undiscounted study:
  - gamma = NORMALIZED_GAMMA (1/max raw r_DOS = 1/2516.1664410449775)
  - rollout_aggregation = 'max'
  - rollout_discount = 1.0 (no 0.9^depth decay on extra rollout samples)

The global-best U-only compound under NORMALIZED_GAMMA is UTi6Sn6, so the same
Cr/Fe/Ni/Pt6Sn6U d=2/4/6/8 ladder as the _normalized and _final studies is reused.

Output: sensitivity_studies/results/starting_material_sweep_max_undiscounted/convergence_data.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'sensitivity_studies' / 'scripts'))
from common import run_sweep, save_sweep_results, sweep_result_path

SWEEP_NAME = 'starting_material_sweep_max_undiscounted'

NORMALIZED_GAMMA = 1.0 / 2516.1664410449775

VALUES = {
    'Cr6Sn6U (d=2)': dict(transition_metal='Cr', group_iv='Sn', gamma=NORMALIZED_GAMMA, rollout_aggregation='max', rollout_discount=1.0),
    'Fe6Sn6U (d=4)': dict(transition_metal='Fe', group_iv='Sn', gamma=NORMALIZED_GAMMA, rollout_aggregation='max', rollout_discount=1.0),
    'Ni6Sn6U (d=6)': dict(transition_metal='Ni', group_iv='Sn', gamma=NORMALIZED_GAMMA, rollout_aggregation='max', rollout_discount=1.0),
    'Pt6Sn6U (d=8)': dict(transition_metal='Pt', group_iv='Sn', gamma=NORMALIZED_GAMMA, rollout_aggregation='max', rollout_discount=1.0),
}

if __name__ == '__main__':
    df = run_sweep(SWEEP_NAME, VALUES, checkpoint_path=sweep_result_path(SWEEP_NAME))
    save_sweep_results(df, SWEEP_NAME)
