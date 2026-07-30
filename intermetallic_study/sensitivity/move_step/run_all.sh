#!/bin/bash
# Run all move_step sensitivity configs

set -e

CONFIGS=(
  "step_1"
  "step_2"
  "step_3"
  "step_5"
)

echo "=== Move Step Sensitivity Study ==="
echo "Running ${#CONFIGS[@]} configurations..."

for config in "${CONFIGS[@]}"; do
  echo ""
  echo ">>> Running: $config"
  mcts-run run --config "configs/${config}.yaml"
done

echo ""
echo "=== All runs complete! ==="
