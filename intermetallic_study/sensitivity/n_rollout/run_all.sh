#!/bin/bash
# Run all n_rollout sensitivity configs

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Running n_rollout sensitivity study..."
echo "========================================"

for config in configs/*.yaml; do
    config_name=$(basename "$config" .yaml)
    echo ""
    echo "Running $config_name..."
    mcts-run run --config "$config"
done

echo ""
echo "All n_rollout sensitivity runs complete!"
