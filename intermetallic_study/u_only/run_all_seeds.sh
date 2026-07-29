#!/bin/bash
# Run the U-only product-mode study (single seed)
#
# Usage:
#   bash run_all_seeds.sh

set -e

STUDY_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$STUDY_DIR/../.." && pwd)"

echo "=========================================="
echo "U-Only Product-Mode Study"
echo "=========================================="
echo "Study dir: $STUDY_DIR"
echo "Repo root: $REPO_ROOT"
echo ""

# Check for required data files
if [ ! -f "$REPO_ROOT/high_throughput_mace_results.full.csv" ]; then
    echo "ERROR: Missing high_throughput_mace_results.full.csv in repo root"
    exit 1
fi

if [ ! -f "$REPO_ROOT/doscar_peaks_data_with_U.csv" ]; then
    echo "ERROR: Missing doscar_peaks_data_with_U.csv in repo root"
    exit 1
fi

# Check for MP API key (optional but recommended)
if [ -z "$MP_API_KEY" ]; then
    # Try to load from .env file
    ENV_FILE="$REPO_ROOT/.env"
    if [ -f "$ENV_FILE" ]; then
        export $(grep -v '^#' "$ENV_FILE" | xargs)
        if [ -n "$MP_API_KEY" ]; then
            echo "✓ Loaded MP_API_KEY from .env file"
        fi
    fi
fi

if [ -z "$MP_API_KEY" ]; then
    echo "WARNING: MP_API_KEY not set. Using cached energies only."
    echo "  To use Materials Project, either:"
    echo "    1. Create .env file: echo 'MP_API_KEY=your-key' > .env"
    echo "    2. Or export directly: export MP_API_KEY='your-key'"
    echo ""
fi

cd "$REPO_ROOT"

# Check if mcts-run is available
if ! command -v mcts-run &> /dev/null; then
    echo "ERROR: mcts-run not found. Install with: pip install -e ."
    exit 1
fi

echo "Running seed 0..."
echo ""

# Run seed 0
START_TIME=$(date +%s)

seed=0
echo "=========================================="
echo "Running seed $seed..."
echo "=========================================="
mcts-run run --config "intermetallic_study/u_only/configs/seed_${seed}.yaml" 2>&1 | \
    tee "intermetallic_study/u_only/results/seed_${seed}.log"
echo ""

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "=========================================="
echo "Seed 0 complete!"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "=========================================="
echo ""
echo "Results saved to: intermetallic_study/u_only/results/seed_0/"
echo ""
echo "=========================================="
echo "Generating figures and tables..."
echo "=========================================="
python3 intermetallic_study/u_only/generate_figures.py --top-n 15
if [ $? -eq 0 ]; then
    echo "✓ Figures and tables generated successfully"
else
    echo "✗ Error generating figures and tables"
fi

echo ""
echo "Next steps:"
echo "  1. View results: cat intermetallic_study/u_only/results/seed_0/summary.json"
echo "  2. View figures: open intermetallic_study/u_only/figures/"
echo "  3. View tables: cat intermetallic_study/u_only/top15_recommendations.txt"
echo ""
