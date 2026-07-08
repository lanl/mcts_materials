#!/bin/bash
# Lanthanide+U study: Yb start, move_step=3.
# Starting material: Cr6Sn6Yb (rank 697/1702, 59th percentile).
# Same reward/rollout params as the main study but with ±3 move set,
# matching the discovery efficiency study's extended-mode hyperparameters.

set -e

STUDY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${STUDY_DIR}/../../" && pwd)"
OUTPUT_DIR="${STUDY_DIR}/Yb_start_step3"
F_BLOCK_MODE="lanthanides_u"
NORMALIZED_GAMMA="0.00039742998860786596"

mkdir -p "${OUTPUT_DIR}"

echo "=================================================="
echo "Lanthanide+U study — Yb start, move_step=3"
echo "Start: Cr6Sn6Yb (rank 697/1702, 59th percentile)"
echo "gamma=${NORMALIZED_GAMMA}, rollout=max, discount=1.0"
echo "Output: ${OUTPUT_DIR}"
echo "=================================================="

cd "${REPO_ROOT}"
"${REPO_ROOT}/.venv/bin/python" run_mcts.py \
    --structure examples/mat_Pb6U1W6_sg191.cif \
    --transition-metal Cr \
    --group-iv Sn \
    --f-block-element Yb \
    --f-block-mode ${F_BLOCK_MODE} \
    --move-step 3 \
    --termination-limit 25 \
    --rollout-method ehull_rdos \
    --seed 42 \
    --rollout-depth 2 \
    --n-rollout 2 \
    --gamma "${NORMALIZED_GAMMA}" \
    --rollout-aggregation max \
    --rollout-discount 1.0 \
    --output "${OUTPUT_DIR}"

echo ""
echo "Run complete. Regenerating figures..."
cd "${STUDY_DIR}"
"${REPO_ROOT}/.venv/bin/python" generate_figures.py
