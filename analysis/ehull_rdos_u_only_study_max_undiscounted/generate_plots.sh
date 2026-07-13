#!/bin/bash
# Generate all figures for the max-undiscounted ehull_rdos U-only study
# (beta=1.0, gamma=1/2516.1664410449775, rollout_aggregation=max, rollout_discount=1.0 -
# see run_study.sh and generate_figures.py's NORMALIZED_GAMMA)

set -e

echo "=================================================="
echo "Generating figures for the max-undiscounted ehull_rdos U-only study"
echo "gamma=0.00039742998860786596 (fixed), rollout_aggregation=max, rollout_discount=1.0"
echo "=================================================="

echo ""

echo "Step: Generating all figures (PNG) using Python plotting pipeline"
python generate_figures.py

echo ""
echo "=================================================="
echo "ALL FIGURES GENERATED"
echo "=================================================="
