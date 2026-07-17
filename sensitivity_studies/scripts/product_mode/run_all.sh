#!/bin/bash
# Product-mode sensitivity sweeps: c, rollout_depth, termination_limit.
# All runs use n_rollout=1, rollout_method=ehull_rdos_product, gamma=1.0,
# move_step=1, starting from Pd6Ge6U (d=6 from UTi6Sn6).
# Results go to sensitivity_studies/results/product_mode/.

set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd ../../../ && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
RESULTS="../../results/product_mode"
SCRIPTS_DIR="$(dirname "${BASH_SOURCE[0]}")/../../scripts"

echo "=== c (exploration_constant) sweep ==="
"${PYTHON}" sweep_c.py
python "${SCRIPTS_DIR}/plot_sweep.py" "${RESULTS}/c_sweep/convergence_data.csv" \
    "" "${RESULTS}/c_sweep/convergence_vs_c.png" --label-prefix="C = "

echo ""
echo "=== rollout_depth + termination_limit sweep ==="
"${PYTHON}" sweep_rollout_params.py
python "${SCRIPTS_DIR}/plot_sweep.py" "${RESULTS}/rollout_params_sweep/convergence_data_rollout_depth.csv" \
    "" "${RESULTS}/rollout_params_sweep/convergence_vs_rollout_depth.png" --label-prefix="Depth = "
python "${SCRIPTS_DIR}/plot_sweep.py" "${RESULTS}/rollout_params_sweep/convergence_data_termination_limit.csv" \
    "" "${RESULTS}/rollout_params_sweep/convergence_vs_termination_limit.png" --label-prefix="Limit = "
python "${SCRIPTS_DIR}/plot_termination_iterations.py" \
    "${RESULTS}/rollout_params_sweep/convergence_data_termination_limit.csv" \
    "termination_limit: search length" \
    "${RESULTS}/rollout_params_sweep/iterations_vs_termination_limit.png"

echo ""
echo "=== ALL PRODUCT-MODE SENSITIVITY SWEEPS COMPLETE ==="
