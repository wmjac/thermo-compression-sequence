#!/usr/bin/env bash
#SBATCH --job-name=mixture-P0-slice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=slurm-%x-%j.out

set -eo pipefail
module purge
module load anaconda3/2026.7
conda activate "${HOME}/.local/conda/envs/torch-gpu"
set -u

# Copy to _best and edit. These are TWO-DIMENSIONAL SLICES with d-2 fixed
# coordinates, not a reconstruction of the full d-dimensional P=0 surface.
TOPDIR="/home/wjacobs/git/thermo-compression-sequence"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:?Submit from your _best directory}"
RUN_DIR="${SUBMIT_DIR}/analysis01"
FEATURES_DIR="${SUBMIT_DIR}/post_training02/features"
OUTPUT_DIR="${SUBMIT_DIR}/p_zero_slice01"
DIMENSIONS=(2 3 4 5 6)
AXES=(1 2)
VOXEL_SIZE=0.01                     # Edit according to current feature-coordinate ranges.
HULL_POLICY="full"                 # full: both hull tests; projected: legacy projected domain.
HULL_TOL_MULT=1.0                  # epsilon = this * sqrt(mean_sequence(feature_rmsd^2)).
LP_NUM_POINTS="all"                # Optional integer sample; all can be costly for 500k sequences.
LP_SEED=0
# Explicitly no offsets below. Example:
# OFFSET_ARGS=(--offset 3 0.01 --offset 4 -0.02)
# Offset coordinates must be fixed and present in every selected dimension.
OFFSET_ARGS=()
DEVICE="cuda"
THREADS=1
PYTHON="python3"

export OMP_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${TOPDIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${SUBMIT_DIR}"
"${PYTHON}" -c 'import sys, torch; print(f"Python: {sys.executable} | Torch: {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}")'
echo "git commit                         : $(git -C "${TOPDIR}" rev-parse HEAD)"
echo "scope                              : 2D affine slices only"
echo "canonical run                      : ${RUN_DIR}"
echo "feature export                     : ${FEATURES_DIR}"
echo "output                             : ${OUTPUT_DIR}"

"${PYTHON}" "${TOPDIR}/analysis/calc_pressure_zeros_v1.py" \
    --run-dir "${RUN_DIR}" \
    --features-dir "${FEATURES_DIR}" \
    --dimensions "${DIMENSIONS[@]}" \
    --axes "${AXES[@]}" \
    "${OFFSET_ARGS[@]}" \
    --output-dir "${OUTPUT_DIR}" \
    --voxel-size "${VOXEL_SIZE}" \
    --hull-policy "${HULL_POLICY}" \
    --hull-tol-mult "${HULL_TOL_MULT}" \
    --lp-num-points "${LP_NUM_POINTS}" \
    --lp-seed "${LP_SEED}" \
    --lp-feasibility-tol 0.000000001 \
    --lp-time-limit 60.0 \
    --max-grid-points 250000 \
    --c-min 0.00000001 \
    --c-tot-max 0.25 \
    --n-brackets 64 \
    --min-scan-refinements 1 \
    --max-scan-refinements 3 \
    --root-xtol 0.000000000001 \
    --root-rtol 0.000000000001 \
    --pressure-atol 0.0000000001 \
    --pressure-rtol 0.00000001 \
    --slope-threshold 0.000001 \
    --root-maxiter 100 \
    --match-atol 0.0001 \
    --match-rtol 0.25 \
    --evaluation-dtype float64 \
    --batch-size 2048 \
    --device "${DEVICE}" \
    --threads "${THREADS}" \
    --progress-every 20 \
    --overwrite no

# The Python logs already contain the exact invocation and all resolved settings.
cp -- "$0" "${OUTPUT_DIR}/submitted_pressure_zero_slice.slurm"

