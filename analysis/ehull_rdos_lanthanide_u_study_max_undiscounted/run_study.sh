#!/bin/bash
# Lanthanide+U max-undiscounted study: same reward/rollout params as the
# U-only max_undiscounted study, but f-block expanded to all lanthanides + U
# (1,728-compound design space).
#   gamma = 1/(max raw r_DOS across the 108 U-only compounds) = 1/2516.1664410449775
#         = 0.00039742998860786596  (same normalization as u_only study)
#   rollout_aggregation = max
#   rollout_discount    = 1.0  (no depth decay)
#   f-block-mode        = lanthanides_u  (lanthanides La-Lu + U, ±1 f-block moves)

set -e

STUDY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${STUDY_DIR}/../../" && pwd)"
OUTPUT_DIR="${STUDY_DIR}"
F_BLOCK_MODE="lanthanides_u"

NORMALIZED_GAMMA="0.00039742998860786596"

echo "=================================================="
echo "EHULL + RDOS STUDY (Lanthanide+U, max undiscounted)"
echo "Reward = beta*ehull_reward(E_hull) + gamma*r_DOS, gamma=${NORMALIZED_GAMMA}"
echo "where ehull_reward = -tanh(120 * (E_hull - 0.05))"
echo "Rollout aggregation: max (no depth discount, rollout_discount=1.0)"
echo "F-block mode: ${F_BLOCK_MODE} (1,728-compound design space)"
echo "=================================================="

MP_API_KEY_ARG=()
if [ -n "${MP_API_KEY}" ]; then
    MP_API_KEY_ARG=(--mp-api-key "${MP_API_KEY}")
elif [ ! -f "${REPO_ROOT}/config.json" ]; then
    echo "Note: no config.json found and MP_API_KEY env var not set."
    echo "Copy config.example.json to config.json and add your Materials Project API key,"
    echo "or export MP_API_KEY=your_key before running this script."
fi

mkdir -p "${OUTPUT_DIR}"

cd "${REPO_ROOT}"
"${REPO_ROOT}/.venv/bin/python" run_mcts.py \
    --structure examples/mat_Pb6U1W6_sg191.cif \
    --f-block-mode ${F_BLOCK_MODE} \
    --termination-limit 25 \
    --rollout-method ehull_rdos \
    --seed 42 \
    "${MP_API_KEY_ARG[@]}" \
    --rollout-depth 2 \
    --n-rollout 2 \
    --gamma "${NORMALIZED_GAMMA}" \
    --rollout-aggregation max \
    --rollout-discount 1.0 \
    --output "${OUTPUT_DIR}"

echo ""
echo "=================================================="
echo "MCTS RUN COMPLETE - Generating table..."
echo "=================================================="
cd "${STUDY_DIR}"
"${REPO_ROOT}/.venv/bin/python" generate_figures.py
