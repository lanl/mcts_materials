#!/bin/bash
# Extended-mode discovery efficiency study — sequential execution.
#
# Changes vs. the original run_study.sh / run_tm_variant.sh:
#   --f-block-mode lanthanides_u_extended   (±3 lanthanide jumps instead of ±1)
#   5 independent seeds (0–4) per starting material
#
# Output: extended_mode/{LAN}_{TM}_s{SEED}/
# Runs ONE job at a time; skips directories that already have convergence_history.csv.

set -e

STUDY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${STUDY_DIR}/../../" && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
CIF="${REPO_ROOT}/examples/mat_Pb6U1W6_sg191.cif"
NORMALIZED_GAMMA="0.00039742998860786596"
ITERATIONS=500

mkdir -p "${STUDY_DIR}/extended_mode"

run_one() {
    local LAN="$1"
    local TM="$2"
    local SEED="$3"
    local OUTDIR="${STUDY_DIR}/extended_mode/${LAN}_${TM}_s${SEED}"
    # Skip if already completed
    if [ -f "${OUTDIR}/convergence_history.csv" ]; then
        echo "[$(date +%H:%M:%S)] Skipping ${LAN}-${TM} seed=${SEED} (already done)"
        return 0
    fi
    mkdir -p "${OUTDIR}"
    echo "[$(date +%H:%M:%S)] Starting ${LAN}-${TM} seed=${SEED}..."
    "${PYTHON}" "${REPO_ROOT}/run_mcts.py" \
        --structure "${CIF}" \
        --f-block-element "${LAN}" \
        --transition-metal "${TM}" \
        --f-block-mode lanthanides_u \
        --move-step 3 \
        --rollout-method ehull_rdos \
        --gamma "${NORMALIZED_GAMMA}" \
        --rollout-aggregation max \
        --rollout-discount 1.0 \
        --termination-limit 25 \
        --rollout-depth 2 \
        --n-rollout 2 \
        --iterations "${ITERATIONS}" \
        --seed "${SEED}" \
        --output "${OUTDIR}" \
        > "${OUTDIR}/run.log" 2>&1
    echo "[$(date +%H:%M:%S)] Done ${LAN}-${TM} seed=${SEED}"
}

N_PARALLEL=4   # max concurrent MCTS processes — tune to taste
LANS="Eu Sm Gd Tb Nd Pr Ce Er Yb"
SEEDS="0 1 2 3 4"

# Collect all (LAN, TM, SEED) triples into an array
jobs=()
for TM in Cu Fe Cr; do
    for LAN in ${LANS}; do
        for SEED in ${SEEDS}; do
            jobs+=("$LAN $TM $SEED")
        done
    done
done
for SEED in ${SEEDS}; do
    jobs+=("Yb Mn $SEED")
    jobs+=("Yb Ti $SEED")
done

total=${#jobs[@]}
echo "Total jobs: ${total} (N_PARALLEL=${N_PARALLEL}, ITERATIONS=${ITERATIONS})"

running=0
for job in "${jobs[@]}"; do
    read -r LAN TM SEED <<< "$job"
    run_one "$LAN" "$TM" "$SEED" &
    running=$((running + 1))
    if [ "$running" -ge "$N_PARALLEL" ]; then
        wait -n 2>/dev/null || wait   # wait for any one job to finish
        running=$((running - 1))
    fi
done
wait   # drain remaining jobs

echo "All extended-mode runs complete."
