#!/usr/bin/env bash
#SBATCH --job-name=mixture-train
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

"${PYTHON}" -m mixture_models train \
    --run-dir "${RUN_DIR}" \
    --dimensions 2 4 6 8 10 \
    --training-json "${TRAIN_JSONS[@]}" \
    --model fixed_length_transformer_mlp \
    --num-folds 5 \
    --num-models 5 \
    --seed 0 \
    --batch-sizes 2048 \
    --learning-rates 0.001 0.005 0.01 \
    --restarts 5 \
    --max-epochs 20000 \
    --warmup-epochs 100 \
    --grad-clip 1.0 \
    --patience 500 \
    --min-delta 0.0001 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --adam-eps 0.00000001 \
    --weight-decay 0.0 \
    --amsgrad no \
    --dtype float32 \
    --validation-aggregation batch-mean \
    --device cuda \
    --threads 1 \
    --num-workers 0 \
    --pin-memory auto \
    --persistent-workers yes \
    --prefetch-factor 4 \
    --deterministic no \
    --checkpoint-every 1 \
    --plot-every 200 \
    --resume no

