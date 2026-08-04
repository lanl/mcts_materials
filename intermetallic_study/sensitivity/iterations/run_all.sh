#!/bin/bash
# Run all iterations sensitivity configs

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Running iterations sensitivity study..."
echo "========================================"

for config in configs/*.yaml; do
    config_name=$(basename "$config" .yaml)
    echo ""
    echo "Running $config_name..."
    mcts-run run --config "$config"
done

echo ""
echo "All iterations sensitivity runs complete!"
