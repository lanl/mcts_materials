#!/bin/bash
# Master script to run all sensitivity studies in FAST MODE only
# Usage:
#   ./run_all_fast.sh            # Run all fast mode studies sequentially
#   ./run_all_fast.sh --parallel # Run all fast mode studies in parallel

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
echo "MCTS Sensitivity Studies - FAST MODE Runner"
echo "========================================"
echo ""
echo "Studies to run: ${STUDIES[@]}"
echo ""

# Check if running in parallel mode
if [ "$1" == "--parallel" ]; then
    echo "Running all FAST MODE studies in PARALLEL..."
    echo ""

    for study in "${STUDIES[@]}"; do
        echo "Launching $study (fast mode) study in background..."
        (cd "$study" && bash run_fast.sh > "../logs/${study}_fast.log" 2>&1) &
    done

    echo ""
    echo "All studies launched. Waiting for completion..."
    wait
    echo ""
    echo "All parallel runs complete!"

else
    echo "Running all FAST MODE studies SEQUENTIALLY..."
    echo ""

    for study in "${STUDIES[@]}"; do
        echo ""
        echo "========================================"
        echo "Running $study (fast mode) study..."
        echo "========================================"
        cd "$study"
        bash run_fast.sh
        cd ..
    done

    echo ""
    echo "All sequential runs complete!"
fi

echo ""
echo "========================================"
echo "ALL FAST MODE SENSITIVITY STUDIES COMPLETE!"
echo "========================================"
echo ""
echo "Results saved in:"
for study in "${STUDIES[@]}"; do
    echo "  - $study/results_fast/"
done
echo ""
echo "To compare with thorough mode, use analyze_efficiency.py"
