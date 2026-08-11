#!/bin/bash
# Run all u_only grid search configs in parallel
# Grid: rollout_depth ∈ {1,2,3} × n_rollout ∈ {1,3,5,7}

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Find repo root (mcts-materials directory)
REPO_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"
cd "$REPO_ROOT"

echo "========================================"
echo "U-Only Grid Search (12 configs)"
echo "========================================"
echo ""
echo "Parameters:"
echo "  rollout_depth: {1, 2, 3}"
echo "  n_rollout: {1, 3, 5, 7}"
echo "  move_step: 1"
echo "  iterations: 1000"
echo "  termination_limit: 100"
echo ""
echo "Running all configs in PARALLEL..."
echo ""

# Launch all configs in parallel
for config in "$SCRIPT_DIR/configs"/*.yaml; do
    config_name=$(basename "$config" .yaml)
    echo "Launching $config_name..."
    mcts-run run --config "$config" &
done

echo ""
echo "All configs launched. Waiting for completion..."
wait

echo ""
echo "========================================"
echo "U-Only Grid Search Complete!"
echo "========================================"
echo "Results saved in: results/"
