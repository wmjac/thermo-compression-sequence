#!/usr/bin/env bash
#SBATCH --job-name=mixture-canonical
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
conda activate "${HOME}/.local/conda/envs/torch"

set -u

# Submit with sbatch from the unpacked project root; SLURM spools this script elsewhere.
TOPDIR="${SLURM_SUBMIT_DIR:?Submit this script from the mixture-models project root}"
cd "${TOPDIR}"
[[ -f "${TOPDIR}/pyproject.toml" ]] || { echo "Not a mixture-models project root: ${TOPDIR}" >&2; exit 2; }

# The Della MIG allocation provides one CPU core.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${TOPDIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON="python3"
"${PYTHON}" -c 'import sys, torch; print(f"Python: {sys.executable} | Torch: {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}")'
if git -C "${TOPDIR}" rev-parse --verify HEAD >/dev/null 2>&1; then
    echo "git commit                         : $(git -C "${TOPDIR}" rev-parse HEAD)"
else
    echo "git commit                         : unversioned tarball; source hashes will be recorded"
fi
echo

# Edit these explicit input paths before submitting. Add further batches to TRAIN_JSONS.
RUN_DIR="${TOPDIR}/runs/run01"
TRAIN_JSONS=("${TOPDIR}/data/train/batch000001.json")
TEST_JSONS=("${TOPDIR}/data/test/batch000001.json")

# A=0.3651 is the repository README example, not an automatically fitted value.
"${PYTHON}" -m mixture_models canonicalize \
    --run-dir "${RUN_DIR}" \
    --dimensions 2 4 6 8 10 \
    --sequence-pool training \
    --sequence-weighting uniform \
    --concentration-prior uniform \
    --composition-prior gem \
    --ctot-min 0.0 \
    --ctot-max 0.25 \
    --dirichlet-total 0.3651 \
    --num-samples 50000 \
    --batch-size 2000 \
    --encoding-batch-size 2048 \
    --eps-mass 0.000001 \
    --max-components 50 \
    --eps-effective 0.01 \
    --metric-aggregation batch-mean \
    --metric-eigenvalue-tolerance 0.000000000001 \
    --align-max-iter 200 \
    --align-tolerance 0.00000001 \
    --diagnostic-points 1000 \
    --seed 0 \
    --device cuda \
    --threads 1 \
    --overwrite no

