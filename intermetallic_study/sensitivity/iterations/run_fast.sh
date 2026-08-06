#!/bin/bash
# Run all FAST MODE configs for this sensitivity study

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

STUDY_NAME=$(basename "$SCRIPT_DIR")

echo "Running ${STUDY_NAME} sensitivity study (FAST MODE)..."
echo "========================================"

for config in configs/*_fast.yaml; do
    config_name=$(basename "$config" .yaml)
    echo ""
    echo "Running $config_name..."
    mcts-run run --config "$config"
done

echo ""
echo "All ${STUDY_NAME} FAST MODE runs complete!"
