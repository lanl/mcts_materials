#!/bin/bash
# Run both u_only and lanthanide_u grid searches
# Usage:
#   ./run_both.sh              # Run sequentially (u_only then lanthanide_u)
#   ./run_both.sh --parallel   # Run both in parallel

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Find repo root (mcts-materials directory)
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_ROOT"

echo "========================================"
echo "Grid Search: Both Design Spaces"
echo "========================================"
echo ""

if [ "$1" == "--parallel" ]; then
    echo "Running u_only and lanthanide_u grids in PARALLEL..."
    echo ""

    # Check for MP_API_KEY for lanthanide runs
    if [ -z "$MP_API_KEY" ]; then
        echo "WARNING: MP_API_KEY not set. Lanthanide configs will fail."
        echo "Export it first: export MP_API_KEY=your_key_here"
        echo ""
        exit 1
    fi

    # Launch both in background
    (bash "$SCRIPT_DIR/u_only/run_all.sh" > "$SCRIPT_DIR/u_only_grid.log" 2>&1) &
    U_ONLY_PID=$!

    (bash "$SCRIPT_DIR/lanthanide_u/run_all.sh" > "$SCRIPT_DIR/lanthanide_u_grid.log" 2>&1) &
    LANTHA_PID=$!

    echo "Launched both grids:"
    echo "  - u_only (PID $U_ONLY_PID)"
    echo "  - lanthanide_u (PID $LANTHA_PID)"
    echo ""
    echo "Waiting for completion..."

    status=0
    wait "$U_ONLY_PID" || status=$?
    wait "$LANTHA_PID" || status=$?
    exit "$status"
    echo ""
    echo "Both grids complete!"

else
    echo "Running grids SEQUENTIALLY..."
    echo ""

    echo "Step 1/2: Running u_only grid..."
    bash "$SCRIPT_DIR/u_only/run_all.sh"

    echo ""
    echo "Step 2/2: Running lanthanide_u grid..."

    # Check for MP_API_KEY
    if [ -z "$MP_API_KEY" ]; then
        echo "ERROR: MP_API_KEY not set for lanthanide_u runs"
        echo "Export it first: export MP_API_KEY=your_key_here"
        exit 1
    fi

    bash "$SCRIPT_DIR/lanthanide_u/run_all.sh"

    echo ""
    echo "Both grids complete!"
fi

echo ""
echo "========================================"
echo "Results:"
echo "  - u_only: u_only/results/"
echo "  - lanthanide_u: lanthanide_u/results/"
echo "========================================"
