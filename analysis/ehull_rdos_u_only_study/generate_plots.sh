#!/bin/bash
# Generate all figures for the ehull_rdos U-only study (beta/gamma from config.json,
# default 1.0/0.0001 - see config.example.json)
# Run this from inside analysis/ehull_rdos_u_only_study/ after run_study.sh
# has produced all_compounds.csv, convergence_history.csv, and mcts_object.pkl here.

set -e

echo "=================================================="
echo "Generating figures for the ehull_rdos U-only study"
echo "Weights: beta/gamma from config.json (default beta=1.0, gamma=0.0001)"
echo "=================================================="

echo ""

echo "Step: Generating all figures (PNG) using Python plotting pipeline"
python generate_figures.py

echo ""
echo "=================================================="
echo "ALL FIGURES GENERATED"
echo "=================================================="
