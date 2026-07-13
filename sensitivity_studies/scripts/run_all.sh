#!/bin/bash
# Run all sensitivity sweeps + generate all figures.
# Each sweep is ~25 replicate MCTS runs (5 values x 5 seeds, or 3x3x5 for the
# rollout-params sweep) at 500 iterations each; all U-only ehull_rdos
# compounds are already cached in high_throughput_mace_results.full.csv, so
# this completes in well under 20 minutes total (no live MACE/MP calls).

set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
RESULTS="../results"

echo "=== c (exploration_constant) sweep ==="
python sweep_c.py
python plot_sweep.py "${RESULTS}/c_sweep/convergence_data.csv" \
    "" "${RESULTS}/c_sweep/convergence_vs_c.png" --label-prefix="C = "

echo ""
echo "=== starting material sweep ==="
echo "(moved to analysis/ehull_rdos_u_only_study/sweep_starting_material.py - run that"
echo " directly, then analysis/ehull_rdos_u_only_study/generate_figures.py to replot)"

echo ""
echo "=== selection_mode sweep ==="
python sweep_selection_mode.py
python plot_sweep.py "${RESULTS}/selection_mode_sweep/convergence_data.csv" \
    "" "${RESULTS}/selection_mode_sweep/convergence_vs_selection_mode.png"

echo ""
echo "=== rollout/termination params sweep ==="
python sweep_rollout_params.py
python plot_sweep.py "${RESULTS}/rollout_params_sweep/convergence_data_n_rollout.csv" \
    "n_rollout" "${RESULTS}/rollout_params_sweep/convergence_vs_n_rollout.png"
python plot_sweep.py "${RESULTS}/rollout_params_sweep/convergence_data_rollout_depth.csv" \
    "" "${RESULTS}/rollout_params_sweep/convergence_vs_rollout_depth.png" --label-prefix="Depth = "
python plot_sweep.py "${RESULTS}/rollout_params_sweep/convergence_data_termination_limit.csv" \
    "" "${RESULTS}/rollout_params_sweep/convergence_vs_termination_limit.png" --label-prefix="Limit = "
# termination_limit's effect is on search length, not the reward curve (the
# optimum is found well before termination ever triggers) - see this instead:
python plot_termination_iterations.py "${RESULTS}/rollout_params_sweep/convergence_data_termination_limit.csv" \
    "termination_limit: search length" "${RESULTS}/rollout_params_sweep/iterations_vs_termination_limit.png"

echo ""
echo "=== ALL SENSITIVITY SWEEPS COMPLETE ==="
