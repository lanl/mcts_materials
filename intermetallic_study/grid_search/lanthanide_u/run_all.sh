#!/bin/bash
# Run all lanthanide_u grid search configs in parallel
# Grid: rollout_depth ∈ {1,2,3} × n_rollout ∈ {1,3,5,7}
#
# NOTE: Requires MP_API_KEY to be set in environment
# Export it before running: export MP_API_KEY=your_key_here

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Find repo root (mcts-materials directory)
REPO_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"
cd "$REPO_ROOT"

# Check for MP_API_KEY
if [ -z "$MP_API_KEY" ]; then
    echo "ERROR: MP_API_KEY environment variable not set"
    echo ""
    echo "Lanthanide configs use ehull_rdos_product rollout method which requires"
    echo "an MP API key. Export it before running:"
    echo ""
    echo "  export MP_API_KEY=your_key_here"
    echo "  ./run_all.sh"
    echo ""
    echo "Or source the .env file from the repo root:"
    echo ""
    echo "  source .env"
    echo "  ./run_all.sh"
    exit 1
fi

echo "========================================"
echo "Lanthanide+U Grid Search (12 configs)"
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
echo "Lanthanide+U Grid Search Complete!"
echo "========================================"
echo "Results saved in: results/"
