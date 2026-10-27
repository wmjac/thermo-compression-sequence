#!/usr/bin/env bash
#SBATCH --job-name=mixture-post-training
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

# Copy to an _best directory, edit these settings, then sbatch from that directory.
# Existing trained/canonicalized models and canonical measures are reused.
TOPDIR="/home/wjacobs/git/thermo-compression-sequence"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:?Submit from your _best directory}"
RUN_DIR="${SUBMIT_DIR}/analysis01"
OUTPUT_ROOT="${SUBMIT_DIR}/post_training01"
SEQUENCE_LIST="${SUBMIT_DIR}/af20_sequences.txt"
TEST_JSONS=("/EDIT/path/to/independent_test.json")  # () explicitly means no test set.
DIMENSIONS=(1 2 3 4 5 6)
REFERENCE_DIMENSION=6                   # Explicit scientific choice; inspect spectra/stability.
MEMBERSHIP_SYMMETRY="reversal"         # Use exact for literal sequence membership.
PYTHON="python3"
DEVICE="cuda"
THREADS=1

export OMP_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${TOPDIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${SUBMIT_DIR}"
[[ -f "${SEQUENCE_LIST}" ]] || { echo "Missing sequence list: ${SEQUENCE_LIST}" >&2; exit 2; }
for path in "${TEST_JSONS[@]}"; do
    [[ -f "${path}" ]] || { echo "Missing test JSON: ${path}" >&2; exit 2; }
done
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "Choose a fresh OUTPUT_ROOT: ${OUTPUT_ROOT}" >&2; exit 2; }
mkdir -p "${OUTPUT_ROOT}"
cp -- "$0" "${OUTPUT_ROOT}/submitted_post_training.slurm"
"${PYTHON}" -c 'import sys, torch; print(f"Python: {sys.executable} | Torch: {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}")'
echo "git commit                         : $(git -C "${TOPDIR}" rev-parse HEAD)"
echo "canonical run                      : ${RUN_DIR}"
echo "reference dimension                : ${REFERENCE_DIMENSION}"

"${PYTHON}" "${TOPDIR}/analysis/export_sequence_features_v1.py" \
    --run-dir "${RUN_DIR}" \
    --dimensions "${DIMENSIONS[@]}" \
    --sequence-list "${SEQUENCE_LIST}" \
    --test-json "${TEST_JSONS[@]}" \
    --membership-symmetry "${MEMBERSHIP_SYMMETRY}" \
    --output-dir "${OUTPUT_ROOT}/features" \
    --batch-size 2048 \
    --device "${DEVICE}" \
    --threads "${THREADS}" \
    --save-member-features no \
    --overwrite no

# rho_chem and rho_mix are read from canonicalization, never replaced by CLI priors.
"${PYTHON}" "${TOPDIR}/analysis/compare_dimensions_v1.py" \
    --run-dir "${RUN_DIR}" \
    --dimensions "${DIMENSIONS[@]}" \
    --reference-dimension "${REFERENCE_DIMENSION}" \
    --output-dir "${OUTPUT_ROOT}/dimension_comparison" \
    --rank-rtol 0.0000000001 \
    --rank-atol 0.0 \
    --degeneracy-rtol 0.01 \
    --random-controls 100 \
    --seed 0 \
    --response-analysis yes \
    --save-responses yes \
    --batch-size 2048 \
    --device "${DEVICE}" \
    --threads "${THREADS}" \
    --overwrite no
