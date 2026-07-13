#!/bin/bash
# Reproduce the U-only ehull_rdos study: sharp tanh E_hull + DOSCAR reward
# Reward = beta*ehull_reward(E_hull) + gamma*r_DOS, where ehull_reward = -tanh(120*(E_hull-0.05))
# beta/gamma default to 1.0/0.0001 (see config.json/config.example.json - gamma is a single
# value shared by the MCTS run and all analysis/plotting scripts)
# F-block mode: u_only (108 composition design space)
#
# Starting material: Cr6Sn6U (transition_metal/group_iv from config.json,
# substituted onto the Pb6U1W6 sg191 structure below) - chosen to be centrally
# located between the four target compounds, see note below.
#
# Requires (see README.md "Data Availability" for schema/how to obtain):
#   - high_throughput_mace_results.full.csv (repo root)
#   - doscar_peaks_data_with_U.csv (repo root) - rDOS is always computed in real
#     time from this file, there is no precomputed rewards cache
#   - a Materials Project API key, supplied via config.json (preferred, gitignored)
#     or the MP_API_KEY environment variable below
#
# See config.example.json in this directory for the exact effective settings
# this study used (transition_metal/group_iv/gamma/selection_mode/exploration_constant/
# iterations - mp_api_key is a placeholder, never a real key). It documents
# the repo-root config.json this study was run against; it is not itself read
# by any script here.
#
# selection_mode, exploration_constant, iterations, transition_metal, group_iv
# are NOT hardcoded as CLI flags here - they come from config.json so there is
# one source of truth for these hyperparameters across run_mcts.py and the
# analysis/plotting scripts. See config.json/config.example.json and
# README.md "Child Selection Methods".
#
# config.json currently sets transition_metal=Cr, group_iv=Sn (starting
# material Cr6Sn6U) - chosen to minimize the worst-case graph distance to the
# four experimentally-synthesized target compounds (U-Sn-V, U-Sn-Nb, U-Ge-Cr,
# U-Ge-Co): distances from Cr6Sn6U are 1, 2, 1, 3 respectively (max=3), vs
# 3, 2, 2, 5 from the original Pb6U1W6 start (max=5).

set -e

STUDY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${STUDY_DIR}/../../" && pwd)"
OUTPUT_DIR="${STUDY_DIR}"
F_BLOCK_MODE="u_only"

echo "=================================================="
echo "EHULL + RDOS STUDY (U-only)"
echo "Reward = beta*ehull_reward(E_hull) + gamma*r_DOS"
echo "where ehull_reward = -tanh(120 * (E_hull - 0.05))"
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
python run_mcts.py \
    --structure examples/mat_Pb6U1W6_sg191.cif \
    --f-block-mode ${F_BLOCK_MODE} \
    --termination-limit 25 \
    --rollout-method ehull_rdos \
    --seed 42 \
    "${MP_API_KEY_ARG[@]}" \
    --rollout-depth 3 \
    --n-rollout 2 \
    --output "${OUTPUT_DIR}"

echo ""
echo "=================================================="
echo "MCTS RUN COMPLETE - Generating figures..."
echo "=================================================="
cd "${STUDY_DIR}"
bash generate_plots.sh
