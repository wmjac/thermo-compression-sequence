#!/usr/bin/env bash
#SBATCH --job-name=mixture-free-energy-slice
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

# Copy to your _best directory and edit before submission.
# Copy the geometry/hull settings from the matching P=0 job for the same mask.
TOPDIR="/home/wjacobs/git/thermo-compression-sequence"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:?Submit from your _best directory}"
RUN_DIR="${SUBMIT_DIR}/analysis01"
FEATURES_DIR="${SUBMIT_DIR}/post_training02/features"
OUTPUT_DIR="${SUBMIT_DIR}/free_energy_slice01"
DIMENSIONS=(3)
AXES=(1 2)
C_TOT=0.1                       # Fixed TOTAL CHAIN concentration in nm^-3.
VOXEL_SIZE=0.01
HULL_POLICY="full"
HULL_TOL_MULT=1.0
LP_NUM_POINTS="all"              # Optional integer subsample, e.g. 10000.
LP_SEED=0
# Offsets ADD to the sequence-pool mean in fixed coordinates.
# Example: OFFSET_ARGS=(--offset 3 0.01)
# A fixed-coordinate offset must be valid in every selected dimension.
OFFSET_ARGS=()
EVALUATION_DTYPE="float64"
BATCH_SIZE=2048
DEVICE="cuda"
THREADS=1
PROGRESS_EVERY=20                 # Evaluation batches; LP progress also logs by time.
ARROW_STRIDE=5
ARROW_SCALE="auto"                # Or a positive global quiver scale; larger -> shorter.
SAVE_MEMBER_VALUES="yes"
OVERWRITE="no"
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
echo "canonical run                      : ${RUN_DIR}"
echo "feature export                     : ${FEATURES_DIR}"
echo "fixed c_tot (nm^-3)                 : ${C_TOT}"
echo "output                             : ${OUTPUT_DIR}"

"${PYTHON}" "${TOPDIR}/analysis/calc_free_energy_slice_v1.py" \
    --run-dir "${RUN_DIR}" \
    --features-dir "${FEATURES_DIR}" \
    --dimensions "${DIMENSIONS[@]}" \
    --axes "${AXES[@]}" \
    "${OFFSET_ARGS[@]}" \
    --c-tot "${C_TOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --voxel-size "${VOXEL_SIZE}" \
    --hull-policy "${HULL_POLICY}" \
    --hull-tol-mult "${HULL_TOL_MULT}" \
    --lp-num-points "${LP_NUM_POINTS}" \
    --lp-seed "${LP_SEED}" \
    --lp-feasibility-tol 0.000000001 \
    --lp-time-limit 60.0 \
    --max-grid-points 250000 \
    --evaluation-dtype "${EVALUATION_DTYPE}" \
    --batch-size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --threads "${THREADS}" \
    --progress-every "${PROGRESS_EVERY}" \
    --arrow-stride "${ARROW_STRIDE}" \
    --arrow-scale "${ARROW_SCALE}" \
    --save-member-values "${SAVE_MEMBER_VALUES}" \
    --overwrite "${OVERWRITE}"

cp -- "$0" "${OUTPUT_DIR}/submitted_free_energy_slice_v1.slurm"
