#!/bin/bash
# Run all starting material sensitivity configs

set -e

CONFIGS=(
  "cr_sn"
  "fe_sn"
  "cu_sn"
  "ni_ge"
  "w_pb"
)

echo "=== Starting Material Sensitivity Study ==="
echo "Running ${#CONFIGS[@]} configurations..."

for config in "${CONFIGS[@]}"; do
  echo ""
  echo ">>> Running: $config"
  mcts-run run --config "configs/${config}.yaml"
done

echo ""
echo "=== All runs complete! ==="
