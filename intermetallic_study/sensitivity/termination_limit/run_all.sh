#!/bin/bash
# Run all termination_limit sensitivity configs

set -e

CONFIGS=(
  "limit_25"
  "limit_50"
  "limit_100"
  "limit_200"
  "limit_500"
)

echo "=== Termination Limit Sensitivity Study ==="
echo "Running ${#CONFIGS[@]} configurations..."

for config in "${CONFIGS[@]}"; do
  echo ""
  echo ">>> Running: $config"
  mcts-run run --config "configs/${config}.yaml"
done

echo ""
echo "=== All runs complete! ==="
