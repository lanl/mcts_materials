#!/bin/bash
# Run all rollout_depth sensitivity configs

set -e

CONFIGS=(
  "depth_1"
  "depth_2"
  "depth_3"
  "depth_5"
)

echo "=== Rollout Depth Sensitivity Study ==="
echo "Running ${#CONFIGS[@]} configurations..."

for config in "${CONFIGS[@]}"; do
  echo ""
  echo ">>> Running: $config"
  mcts-run run --config "configs/${config}.yaml"
done

echo ""
echo "=== All runs complete! ==="
