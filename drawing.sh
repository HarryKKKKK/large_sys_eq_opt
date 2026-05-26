#!/bin/bash

set -euo pipefail

WORKDIR="$(pwd)"
cd "$WORKDIR"

SCRIPT="visualization/plot_snapshot_series.py"
OUTDIR="figs/validation"

mkdir -p "$OUTDIR"

SOLVERS=("hll" "hllc" "exact" "force")

echo "===== PLOT VALIDATION FIGURES ====="
echo "Workdir: $WORKDIR"
echo "Output directory: $OUTDIR"
echo ""

# ============================================================
# 1. Shock-bubble: density contour
# ============================================================

# CASE="shock_bubble"
# FIELD="rho"
# T_END="0.0011741"

# echo "===== Shock-bubble density plots ====="

# for SOLVER in "${SOLVERS[@]}"; do
#     INPUT="outputs/gpu_${CASE}_${SOLVER}_n1_snapshot_*.csv"
#     OUTPUT="${OUTDIR}/gpu_${CASE}_${SOLVER}_${FIELD}_series.png"

#     if compgen -G "$INPUT" > /dev/null; then
#         echo "Plotting: case=${CASE}, solver=${SOLVER}, field=${FIELD}"

#         python "$SCRIPT" \
#             --input "$INPUT" \
#             --field "$FIELD" \
#             --t-end "$T_END" \
#             --levels 45 \
#             --output "$OUTPUT"

#         echo "Saved: $OUTPUT"
#     else
#         echo "Skipping ${CASE} ${SOLVER}: no files matched ${INPUT}"
#     fi

#     echo ""
# done

# ============================================================
# 2. Blast-wave: pressure filled contour
# ============================================================

CASE="blast_wave"
FIELD="rho"
T_END="0.2"

echo "===== Blast-wave pressure plots ====="

for SOLVER in "${SOLVERS[@]}"; do
    INPUT="outputs/gpu_${CASE}_${SOLVER}_n1_snapshot_*.csv"
    OUTPUT="${OUTDIR}/gpu_${CASE}_${SOLVER}_${FIELD}_series.png"

    if compgen -G "$INPUT" > /dev/null; then
        echo "Plotting: case=${CASE}, solver=${SOLVER}, field=${FIELD}"

        python "visualization/blast_wave.py" \
            --input "$INPUT" \
            --field "$FIELD" \
            --t-end "$T_END" \
            --levels 45 \
            --output "$OUTPUT"

        echo "Saved: $OUTPUT"
    else
        echo "Skipping ${CASE} ${SOLVER}: no files matched ${INPUT}"
    fi

    echo ""
done

echo "===== DONE ====="
echo "Figures saved in: $OUTDIR"