#!/bin/bash
# Pre-flight check before running sensitivity studies
# Validates configs, dependencies, and data files

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "========================================"
echo "Sensitivity Studies - Pre-flight Check"
echo "========================================"
echo ""

# Check Python dependencies
echo "Checking Python dependencies..."
python3 -c "import pandas, matplotlib, numpy" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "  ✓ Required packages installed (pandas, matplotlib, numpy)"
else
    echo "  ✗ Missing Python packages. Install with:"
    echo "    pip install pandas matplotlib numpy"
    exit 1
fi

# Check mcts-run command
echo ""
echo "Checking mcts-run installation..."
if command -v mcts-run &> /dev/null; then
    echo "  ✓ mcts-run command available"
else
    echo "  ✗ mcts-run not found. Install with:"
    echo "    cd /Users/blaubach/codex/mcts-materials && pip install -e '.[intermetallic,viz]'"
    exit 1
fi

# Check data files
echo ""
echo "Checking required data files..."
REPO_ROOT="/Users/blaubach/codex/mcts-materials"

if [ -f "$REPO_ROOT/high_throughput_mace_results.full.csv" ]; then
    echo "  ✓ high_throughput_mace_results.full.csv found"
else
    echo "  ✗ high_throughput_mace_results.full.csv missing in repo root"
    exit 1
fi

if [ -f "$REPO_ROOT/doscar_peaks_data_with_U.csv" ]; then
    echo "  ✓ doscar_peaks_data_with_U.csv found"
else
    echo "  ✗ doscar_peaks_data_with_U.csv missing in repo root"
    exit 1
fi

# Check MP API key
echo ""
echo "Checking Materials Project API key..."
if [ -z "$MP_API_KEY" ]; then
    echo "  ✗ MP_API_KEY not set. Set with:"
    echo "    export MP_API_KEY='your-key-here'"
    echo "    or source ../set_api_key.sh"
    exit 1
else
    echo "  ✓ MP_API_KEY is set"
fi

# Validate sample configs
echo ""
echo "Validating sample configurations..."

SAMPLE_CONFIGS=(
    "starting_material/configs/cr_sn.yaml"
    "iterations/configs/iter_1000.yaml"
    "termination_limit/configs/limit_100.yaml"
    "rollout_depth/configs/depth_3.yaml"
    "n_rollout/configs/rollout_2.yaml"
    "move_step/configs/step_1.yaml"
)

FAILED=0
for config in "${SAMPLE_CONFIGS[@]}"; do
    if [ -f "$config" ]; then
        mcts-run validate --config "$config" > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo "  ✓ $config"
        else
            echo "  ✗ $config (validation failed)"
            FAILED=1
        fi
    else
        echo "  ✗ $config (file not found)"
        FAILED=1
    fi
done

if [ $FAILED -eq 1 ]; then
    echo ""
    echo "Some configs failed validation. Please fix before running."
    exit 1
fi

# Count total configs
echo ""
echo "Counting configurations..."
TOTAL_CONFIGS=$(find . -name "*.yaml" -path "*/configs/*" | wc -l | tr -d ' ')
echo "  Total configs: $TOTAL_CONFIGS (expected 28)"

if [ "$TOTAL_CONFIGS" -ne 28 ]; then
    echo "  ⚠️  Warning: Expected 28 configs, found $TOTAL_CONFIGS"
fi

# Check disk space (rough estimate: 28 runs × 100MB = ~3GB)
echo ""
echo "Checking disk space..."
AVAILABLE_GB=$(df -g . | tail -1 | awk '{print $4}')
if [ "$AVAILABLE_GB" -gt 5 ]; then
    echo "  ✓ Sufficient disk space (~${AVAILABLE_GB}GB available)"
else
    echo "  ⚠️  Warning: Low disk space (${AVAILABLE_GB}GB available, ~3GB needed)"
fi

echo ""
echo "========================================"
echo "Pre-flight check PASSED!"
echo "========================================"
echo ""
echo "Ready to run sensitivity studies:"
echo "  Sequential: ./run_all_studies.sh"
echo "  Parallel:   ./run_all_studies.sh --parallel"
echo ""
echo "Estimated runtime:"
echo "  Sequential: ~28 hours"
echo "  Parallel:   ~4-5 hours (with 6+ cores)"
echo ""
