#!/usr/bin/env bash
#SBATCH --job-name=mixture-analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=slurm-%x-%j.out

# Standalone post-training workflow; compatible with the original sequential code.
# Requires the plotting update that adds logarithmic loss/RMSE PNG outputs.
# Put best/d_NNN symlinks in the submission directory. Keep their source runs.
# Submit from that directory: sbatch analysis.slurm
# Each submission uses a NEW ANALYSIS_LABEL; existing output is never replaced.
# Validation uses only each selected model's original held-out fold; test uses
# all selected models. Reported pressure losses are sample means. The original
# checkpoint-selection scores (batch means) are also retained, at full precision.
set -eo pipefail
module purge
module load anaconda3/2026.7
conda activate "${HOME}/.local/conda/envs/torch-gpu"
set -u

# EDIT THESE SETTINGS. All model CLI options are explicit below.
TOPDIR="/home/wjacobs/git/thermo-compression-sequence"
COLLECTION_DIR="${SLURM_SUBMIT_DIR:?Submit from the _best directory}"
ANALYSIS_LABEL="analysis01"
DIMENSIONS=(1 2 3 4 5 6)
EXPECTED_NUM_FOLDS=5
EXPECTED_NUM_MODELS=5
RUN_TEST="yes"                       # Set no if no independent test data exist yet.
TEST_JSONS=("/EDIT/path/to/independent_test.json")
PYTHON="python3"
DEVICE="cuda"
THREADS=1

# Canonical rho_mix: uniform distinct training-sequence pool; c_tot uniform on
# [0, 0.25] nm^-3; truncated GEM with total concentration A=0.3651.
# Edit the explicit canonicalize options below to change this distribution.
ANALYSIS_DIR="${COLLECTION_DIR}/${ANALYSIS_LABEL}"
[[ "${ANALYSIS_LABEL}" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Invalid ANALYSIS_LABEL" >&2; exit 2; }
[[ -d "${TOPDIR}/src/mixture_models" ]] || { echo "Invalid TOPDIR: ${TOPDIR}" >&2; exit 2; }
[[ "${RUN_TEST}" == yes || "${RUN_TEST}" == no ]] || { echo "RUN_TEST must be yes or no" >&2; exit 2; }
if [[ "${RUN_TEST}" == yes ]]; then
    (( ${#TEST_JSONS[@]} > 0 )) || { echo "Set TEST_JSONS or RUN_TEST=no" >&2; exit 2; }
    for path in "${TEST_JSONS[@]}"; do
        [[ -f "${path}" ]] || { echo "Missing independent test JSON: ${path}" >&2; exit 2; }
    done
fi
cd "${COLLECTION_DIR}"
export OMP_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${TOPDIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON}" -c 'import sys, torch; print(f"Python: {sys.executable} | Torch: {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}")'
echo "git commit                         : $(git -C "${TOPDIR}" rev-parse HEAD)"

# Prepare per-dimension run views. Original data, weights and histories are read
# through links; metadata is copied. Canonical/evaluation outputs are local.
"${PYTHON}" - "${COLLECTION_DIR}" "${ANALYSIS_DIR}" "${TOPDIR}" \
    "${EXPECTED_NUM_FOLDS}" "${EXPECTED_NUM_MODELS}" "${DIMENSIONS[@]}" <<'PY_PREPARE'
import hashlib
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from mixture_models.data import load_run_records, sha256_file
from mixture_models.provenance import read_json, timestamp, write_json
from mixture_models.selection import load_selection

collection, output, code = (Path(p).resolve() for p in sys.argv[1:4])
expected_folds, expected_models = map(int, sys.argv[4:6])
dimensions = list(map(int, sys.argv[6:]))
if not dimensions or min(dimensions) <= 0 or len(dimensions) != len(set(dimensions)):
    raise ValueError("DIMENSIONS must contain distinct positive integers.")
if expected_folds < 2 or expected_models <= 0 or expected_models % expected_folds:
    raise ValueError("Expected models must be a positive multiple of expected folds.")
if output.exists():
    raise ValueError(f"Choose a new ANALYSIS_LABEL; output already exists: {output}")
entries, reference_uids = [], None
for dimension in dimensions:
    tag = f"d_{dimension:03d}"
    link = collection / "best" / tag
    if not link.is_symlink():
        raise ValueError(f"Expected a symlink to an original run's best/{tag}: {link}")
    best = link.resolve(strict=True)
    if best.name != tag or best.parent.name != "best":
        raise ValueError(f"Link must resolve to <source_run>/best/{tag}: {best}")
    source = best.parent.parent
    _, selection = load_selection(source, dimension)  # Checks every checkpoint SHA256.
    if selection["dimension"] != dimension or selection["num_folds"] != expected_folds or selection["num_models"] != expected_models:
        raise ValueError(f"Unexpected selection dimension/folds/model count: {best}")
    counts = Counter(item["fold"] for item in selection["models"])
    if counts != Counter({fold: expected_models // expected_folds for fold in range(expected_folds)}):
        raise ValueError(f"Incomplete or unbalanced fold representation: {best}")
    records, inputs = load_run_records(source)  # Checks archived training JSON hashes.
    folds = read_json(source / "folds.json")
    uids = [record.uid for record in records]
    if folds["uid"] != uids or folds["num_folds"] != expected_folds:
        raise ValueError(f"Fold assignments do not match archived training data: {source}")
    if len(folds["validation_fold"]) != len(uids) or set(folds["validation_fold"]) != set(range(expected_folds)):
        raise ValueError(f"Invalid or empty validation folds: {source}")
    if reference_uids is None:
        reference_uids = sorted(uids)
    elif reference_uids != sorted(uids):
        raise ValueError("Dimensions use different training pools; put them in separate collections.")
    for item in selection["models"]:
        for filename in ("history.npz", "config.json", "metrics.json"):
            if not (source / item["trial_path"] / filename).is_file():
                raise FileNotFoundError(source / item["trial_path"] / filename)
    for filename in ("config.json", "inputs.json", "folds.json"):
        if not (source / filename).is_file():
            raise FileNotFoundError(source / filename)
    fold_identity = repr(sorted(zip(uids, folds["validation_fold"]))).encode()
    entries.append({"dimension": dimension, "source_run": str(source),
        "best_directory": str(best), "selection_sha256": sha256_file(best / "selection.json"),
        "num_models": selection["num_models"], "num_folds": selection["num_folds"],
        "num_training_systems": len(records), "training_inputs": inputs,
        "source_metadata_sha256": {name: sha256_file(source / name)
            for name in ("config.json", "inputs.json", "folds.json")},
        "fold_assignment_sha256": hashlib.sha256(fold_identity).hexdigest(),
        "selected_models": [{key: item[key] for key in ("checkpoint", "checkpoint_sha256", "fold",
            "trial_path", "best_epoch", "best_validation_loss")} for item in selection["models"]]})

output.mkdir(parents=True)
(output / "canonical").mkdir()
(output / "evaluation").mkdir()
for entry in entries:
    dimension = entry["dimension"]
    tag = f"d_{dimension:03d}"
    source = Path(entry["source_run"])
    view = output / "run_views" / tag
    (view / "best").mkdir(parents=True)
    (view / "best" / tag).symlink_to(entry["best_directory"], target_is_directory=True)
    for name in ("inputs", "trials"):
        (view / name).symlink_to(source / name, target_is_directory=True)
    for name in ("config.json", "inputs.json", "folds.json"):
        shutil.copy2(source / name, view / name)
    for name in ("canonical", "evaluation"):
        (view / name).symlink_to(output / name, target_is_directory=True)
    write_json(view / "analysis_source.json", entry)
    print(f"d={dimension}: {entry['num_models']} models from {source}", flush=True)

same_folds = len({entry["fold_assignment_sha256"] for entry in entries}) == 1
if not same_folds:
    print("Fold assignments differ across dimensions; each dimension will use its original folds.", flush=True)
commit = subprocess.check_output(["git", "-C", str(code), "rev-parse", "HEAD"], text=True).strip()
write_json(output / "collection.json", {"created_at": timestamp(), "status": "prepared",
    "collection_directory": str(collection), "code_directory": str(code), "git_commit": commit,
    "same_training_pool": True, "same_fold_assignments": same_folds, "dimensions": entries,
    "notes": "Read-only source links; original fold assignments; no training or model reselection."})
PY_PREPARE

# Save the exact Slurm-spooled script, including local edits, as run documentation.
cp -- "$0" "${ANALYSIS_DIR}/submitted_analysis.slurm"

# Validate independent test data and overlap before expensive canonicalization.
if [[ "${RUN_TEST}" == yes ]]; then
    FIRST_VIEW="${ANALYSIS_DIR}/run_views/$(printf 'd_%03d' "${DIMENSIONS[0]}")"
    "${PYTHON}" - "${FIRST_VIEW}" "${TEST_JSONS[@]}" <<'PY_TEST_INPUTS'
import sys
from mixture_models.data import load_json_files, load_run_records
from mixture_models.evaluation import reject_training_overlap
training, _ = load_run_records(sys.argv[1])
test, _ = load_json_files(sys.argv[2:])
reject_training_overlap(test, training)
print(f"Independent test data: {len(test)} systems", flush=True)
PY_TEST_INPUTS
else
    echo "RUN_TEST=no: only validation and canonical analyses will be produced."
fi

for dimension in "${DIMENSIONS[@]}"; do
    VIEW="${ANALYSIS_DIR}/run_views/$(printf 'd_%03d' "${dimension}")"
    "${PYTHON}" -m mixture_models canonicalize \
        --run-dir "${VIEW}" \
        --dimensions "${dimension}" \
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
        --device "${DEVICE}" \
        --threads "${THREADS}" \
        --overwrite no

    "${PYTHON}" -m mixture_models evaluate \
        --run-dir "${VIEW}" \
        --dimensions "${dimension}" \
        --split validation \
        --data-json \
        --name validation \
        --basis canonical \
        --batch-size 2048 \
        --save-hessians no \
        --allow-training-overlap no \
        --device "${DEVICE}" \
        --threads "${THREADS}" \
        --overwrite no

    if [[ "${RUN_TEST}" == yes ]]; then
        "${PYTHON}" -m mixture_models evaluate \
            --run-dir "${VIEW}" \
            --dimensions "${dimension}" \
            --split test \
            --data-json "${TEST_JSONS[@]}" \
            --name test \
            --basis canonical \
            --batch-size 2048 \
            --save-hessians no \
            --allow-training-overlap no \
            --device "${DEVICE}" \
            --threads "${THREADS}" \
            --overwrite no
    fi
done

# Aggregate across dimensions (the existing CLI writes summaries per invocation).
"${PYTHON}" - "${ANALYSIS_DIR}" "${RUN_TEST}" <<'PY_SUMMARIZE'
import csv
import sys
from pathlib import Path
from mixture_models.data import sha256_file
from mixture_models.plotting import plot_dimension_comparison, plot_dimension_losses
from mixture_models.provenance import read_json, timestamp, write_json

output = Path(sys.argv[1])
collection = read_json(output / "collection.json")
entries = sorted(collection["dimensions"], key=lambda entry: entry["dimension"])
splits = ["validation", "test"] if sys.argv[2] == "yes" else ["validation"]
rows, members = [], []
summaries = {}
for split in splits:
    summary = []
    for entry in entries:
        dimension = entry["dimension"]
        metrics = read_json(output / "evaluation" / f"d_{dimension:03d}" / split / "metrics.json")
        config = read_json(output / "evaluation" / f"d_{dimension:03d}" / split / "config.json")
        if config["selection_sha256"] != entry["selection_sha256"]:
            raise ValueError(f"Selected models changed during analysis for d={dimension}.")
        best = Path(entry["best_directory"])
        if sha256_file(best / "selection.json") != entry["selection_sha256"]:
            raise ValueError(f"Source selection changed during analysis for d={dimension}.")
        for model in entry["selected_models"]:
            if sha256_file(best / model["checkpoint"]) != model["checkpoint_sha256"]:
                raise ValueError(f"Source checkpoint changed during analysis for d={dimension}.")
        summary.append(metrics)
        rows.append({"dimension": dimension, "split": split, "num_models": entry["num_models"],
            "num_folds": entry["num_folds"], "source_run": entry["source_run"], **metrics["ensemble"]})
        for model in metrics["individual_models"]:
            selected = next(item for item in entry["selected_models"] if item["checkpoint"] == model["checkpoint"])
            members.append({"dimension": dimension, "split": split, "source_run": entry["source_run"],
                "trial_path": selected["trial_path"], "best_epoch": selected["best_epoch"], **model})
    write_json(output / "evaluation" / f"summary_{split}.json", summary)
    plot_dimension_losses(summary, output / "evaluation" / f"loss_by_dimension_{split}.png")
    summaries[split.title()] = summary
    for item in summary:
        print(f"d={item['dimension']} {split}: {item['ensemble']['pressure_loss']:.17g}")
plot_dimension_comparison(summaries, output / "evaluation" / "loss_by_dimension.png")
for filename, data in (("loss_summary.csv", rows), ("individual_model_losses.csv", members)):
    with (output / "evaluation" / filename).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
collection.update(status="complete", completed_at=timestamp(), evaluated_splits=splits)
write_json(output / "collection.json", collection)
print(f"Completed analysis: {output}", flush=True)
PY_SUMMARIZE
