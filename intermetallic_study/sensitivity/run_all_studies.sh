#!/bin/bash
# Master script to run all sensitivity studies
# Usage:
#   ./run_all_studies.sh            # Run all studies sequentially
#   ./run_all_studies.sh --parallel # Run all studies in parallel (requires more CPU cores)

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# List of all sensitivity studies
STUDIES=(
    "starting_material"
    "iterations"
    "termination_limit"
    "rollout_depth"
    "n_rollout"
    "move_step"
)

echo "========================================"
echo "MCTS Sensitivity Studies - Master Runner"
echo "========================================"
echo ""
echo "Studies to run: ${STUDIES[@]}"
echo ""

# Check if running in parallel mode
if [ "$1" == "--parallel" ]; then
    echo "Running all studies in PARALLEL..."
    echo ""

    for study in "${STUDIES[@]}"; do
        echo "Launching $study study in background..."
        (cd "$study" && bash run_all.sh > "../logs/${study}.log" 2>&1) &
    done

    echo ""
    echo "All studies launched. Waiting for completion..."
    wait
    echo ""
    echo "All parallel runs complete!"

else
    echo "Running all studies SEQUENTIALLY..."
    echo ""

    for study in "${STUDIES[@]}"; do
        echo ""
        echo "========================================"
        echo "Running $study study..."
        echo "========================================"
        cd "$study"
        bash run_all.sh
        cd ..
    done

    echo ""
    echo "All sequential runs complete!"
fi

echo ""
echo "========================================"
echo "Generating sensitivity figures..."
echo "========================================"
python plot_sensitivity.py

echo ""
echo "========================================"
echo "ALL SENSITIVITY STUDIES COMPLETE!"
echo "========================================"
echo ""
echo "Results saved in:"
for study in "${STUDIES[@]}"; do
    echo "  - $study/results/"
done
echo ""
echo "Figures saved in: figures/"
echo ""
echo "To view results, check the convergence.csv and summary.json files in each result directory."
