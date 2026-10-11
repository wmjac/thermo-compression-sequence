# Mixture models

A standalone workflow for pressure-based training of thermodynamic mixture models,
determination of an ensemble canonical basis, and validation/test evaluation.

The implemented model is `fixed_length_transformer_mlp`: a 20-residue Transformer
encoder with mean pooling and sequence-reversal averaging, followed by an MLP
free-energy decoder. The architecture comes from `wmjac/idpmixture`; the training
protocol comes from Bethany's supplied fixed-length free-energy training scripts.
See [docs/training_procedure.md](docs/training_procedure.md) for the exact mapping,
preserved choices, and explicit refactor changes.

## Install and check

Use your existing PyTorch environment. From the unpacked project root:

```bash
python3 -m pip install -e '.[test]'
python3 -m pytest -q
python3 examples/run_smoke.py --output smoke-output
```

The smoke script creates explicitly synthetic data, pools two JSON batches, trains
two dimensions with five folds and ten retained models per dimension, checks
resumption, canonicalizes the models, and evaluates validation and independent
test records. It also checks that changing basis preserves pressure predictions.
It is an execution check, not a production training run or a convergence study.

The SLURM scripts use the source tree through `PYTHONPATH`, so editable installation
is optional there if all dependencies are already installed. Python >=3.10 and
PyTorch >=2.3 are required. The bundled validation was run with CPU PyTorch 2.8.0;
Della GPU execution must be checked in your environment.

## Input JSONs and units

Each training and test JSON must have this top-level block, using the actual units
of its stored values:

```json
"units": {
  "pressure": "atm",
  "potential_energy": "kcal/mol",
  "temperature": "K",
  "concentration": "nm^-3",
  "nominal_temperature": 300.0
}
```

The units shown above are an example. Conversion always uses each file's declared
units and nominal temperature. Concentrations become nm^-3; pressures become
kT nm^-3; potential energies become kT; observed temperatures become K. Means,
standard deviations, and SEMs are converted consistently. Unknown or missing
units cause an explicit error. See [docs/units.md](docs/units.md).

The existing `systems` structure is retained. The required physical inputs are
`input_data_header.mixture_properties.chains`, `chain concentration`, and
`observables.pressure.mean`. The list `chains` contains distinct `[sequence, count]`
entries. Counts determine each component's fraction of the total chain
concentration. The number of mixture components is unrestricted. Optional
monomer concentration is checked against lengths and counts. The general loader
accepts variable sequence lengths; the selected model currently requires 20.

The pressure loss is

\[
L=\left\langle\left(\frac{P-\widehat P}{c_{\mathrm{tot}}k_BT}\right)^2\right\rangle
 =\left\langle\left(\frac{\beta P-\beta\widehat P}{c_{\mathrm{tot}}}\right)^2\right\rangle.
\]

It uses total chain concentration and total pressure, including the ideal chain
term. SEMs are preserved for analysis; they do not weight training loss.
The current model takes sequences and concentrations as inputs; nominal
temperature defines the conversion, rather than an additional model input.

## Training

```bash
mixture-models train \
  --run-dir runs/run01 \
  --training-json data/train/batch000001.json data/train/batch000002.json \
  --model fixed_length_transformer_mlp \
  --dimensions 2:10:2 \
  --num-folds 5 --num-models 5 --seed 0
```

Dimension ranges use inclusive endpoints: `2:10:2` means 2, 4, 6, 8, 10.
All training JSONs are pooled before constructing one randomly shuffled fold
assignment. Systems, rather than individual observations or components, are the
split units. The same assignments are used across dimensions, hyperparameters,
and restarts. Independent test data do not enter the training pool or selection.

Scientific defaults are supplied by the selected model module:

| Choice | Default |
|---|---|
| Batch-size grid | 2048 |
| Learning-rate grid | 0.001, 0.005, 0.01 |
| Restarts per grid point per fold | 5 |
| Optimizer | Adam: betas 0.9/0.999, epsilon 1e-8, weight decay 0, AMSGrad off |
| Maximum epochs | 20000 |
| Linear warmup | 100 epochs, stepped after each optimizer update |
| Gradient norm clipping | 1.0 |
| Patience | 500 epochs |
| Required validation improvement | 1e-4 |
| Checkpoint-validation aggregation | Mean of minibatch mean losses |
| Training dtype | float32 |
| Folds / retained models | 5 / 5 |

Thus the default grid trains **75 candidate models per dimension** and retains
five. `--num-models` is the retained ensemble size, not the number of candidate
training runs. It must be a positive multiple of `--num-folds`. Selection takes
the best `num_models / num_folds` distinct candidates from each fold. The grid
must contain enough candidates for that quota.

All resolved settings are saved in `config.json`; all candidate histories and
checkpoints remain available. Metadata, converted input archives, selection
scores, and canonical arrays retain full stored precision. Keeping the validated
float32 training dtype is deliberate; the refactor does not silently switch
optimization to float64. `--dtype float64` is available as an explicit change.

Use `mixture-models train --help` to inspect every option. CLI overrides change
the resolved model protocol and are recorded; the supplied production SLURM
scripts explicitly state every CLI option.

## Continuation and selection

Repeat the original training command with `--resume yes` to continue an
interrupted run. Data hashes, source code, and resolved configuration must match.
The run stores input snapshots, so those files can be supplied if the originals
have moved. Continuation is from the last saved epoch boundary, including
optimizer, scheduler, patience, histories, and Python/NumPy/PyTorch RNG states.
The default saves every epoch; `--checkpoint-every` can change that interval.
Completed trials are skipped. SIGINT/SIGTERM requests, when received by Python,
finish the current epoch and save continuation state. An interruption exits with
status 75, which prevents dependent `afterok` jobs from starting prematurely.

Changing the training protocol requires a new run. The source snapshot can be
used to recover the exact implementation of an older run:

```bash
PYTHONPATH=/absolute/path/to/run01/source_snapshot python3 -m mixture_models train ... --resume yes
```

To select a different ensemble size from completed candidates without retraining:

```bash
mixture-models select --run-dir runs/run01 --dimensions 2:10:2 --num-models 10
```

Selection writes exactly k checkpoints under `best/d_XXX/`. Repeating an identical
selection preserves its manifest. Changing selection requires a new canonical
basis; evaluation checks the linkage rather than silently using stale transforms.

## Canonical basis

```bash
mixture-models canonicalize \
  --run-dir runs/run01 --dimensions 2:10:2 \
  --sequence-pool training --sequence-weighting uniform \
  --concentration-prior uniform --ctot-min 0.0 --ctot-max 0.25 \
  --composition-prior gem --dirichlet-total 0.3651 \
  --num-samples 50000 --batch-size 2000 --seed 0
```

`--sequence-pool training` uses the lexicographically sorted distinct sequences in
the training pool. Alternatively supply a one-sequence-per-line text file, a
one-dimensional string `.npy`, a JSON list/`{"sequences": [...]}`, or a dataset
JSON. External pools preserve file order except dataset JSONs, whose distinct
sequences are sorted. The saved `features.npz` records the actual pool and order.

This choice defines the chemical sequence measure and is scientifically material.
Using the entire HP sequence universe instead of the observed training sequences
requires explicitly supplying that external pool.

The prior uses uniform total concentration in the specified nm^-3 interval and
truncated GEM stick-breaking compositions, following the repository's large-n
approximation to symmetric Dirichlet(A/n). A=0.3651 is the repository README's
example for approximately 2.5 effective components at threshold 0.01; it is not
automatically calibrated or assumed exact for another pool. Supply the desired
full-precision A explicitly. Observed component-cardinality statistics are saved.

For each selected model the workflow estimates M = E[g g^T], computes its
principal symmetric square root, applies per-model uncentered PCA with positive
mean signs, performs generalized orthogonal Procrustes alignment, then applies
uncentered PCA to the aligned ensemble mean. It uses every selected member.
The coordinate convention is `z_canonical = z_native @ A.T`.

Sampling draws, gradients' outer-product batch means, convergence diagnostics,
all constituent transformations, inverse transformations, native/aligned features,
and ensemble mean features are archived. Singular or ill-conditioned metrics
cause an explicit error after saving the estimates; no regularization or dimension
reduction is inserted. Procrustes convergence status is recorded.

## Evaluation

```bash
mixture-models evaluate \
  --run-dir runs/run01 --dimensions 2:10:2 \
  --split validation --name validation --basis canonical

mixture-models evaluate \
  --run-dir runs/run01 --dimensions 2:10:2 \
  --split test --name test --basis canonical \
  --data-json data/test/batch000001.json data/test/batch000002.json
```

Validation uses the saved training snapshots and folds. Each system is evaluated
with only the selected members trained with its fold held out. With five folds
and five retained models, each system therefore has one out-of-fold prediction;
with ten models it has two. Test predictions use the entire ensemble. Both
individual eligible-model losses and ensemble losses are reported.

Reported evaluation loss averages over systems. This is separately labelled from
the legacy checkpoint criterion, which averages minibatch means. Ensemble pressure
is the mean of individual-model pressures. Ensemble spread is their population
standard deviation, rather than a SEM. Individual features are never averaged
before being fed to a decoder.

Outputs include full-precision predictions, eligibility masks, free-energy
densities, gradients, pressure parity plots, residuals, selected-model loss
curves, and loss versus dimension. `--save-hessians yes` also archives Hessians.
`--basis native` permits evaluation before canonicalization. New evaluation labels
preserve earlier outputs; `--overwrite yes` explicitly replaces an existing label.

The Python API supports sequence embeddings and queries in canonical coordinates:

```python
from mixture_models.evaluation import Ensemble

ensemble = Ensemble("runs/run01", dimension=10, device="cpu", basis="canonical")
features = ensemble.encode_sequences(sequences)      # (k, n, d)
terms = ensemble.thermodynamics_from_latent(x, hessian=True)
# x: (k, batch, d), or a shared (batch, d) point
# terms: individual-model free_energy_excess, pressure_excess, gradient, hessian
```

Queries use each member's inverse transformation and its own decoder. Gradients
and Hessians are transformed consistently with the new coordinates.

## Della SLURM

Edit the explicit data paths, dimensions, and desired prior parameters in the
scripts, then submit **from this unpacked project root**:

```bash
sbatch slurm/workflow.slurm
```

The workflow trains, canonicalizes, evaluates validation, and evaluates test data
in that order. Separate `train.slurm`, `canonicalize.slurm`, and `evaluate.slurm`
are also provided. They use one task, one CPU core, the MIG partition, one GPU,
72 hours, `anaconda3/2026.7`, and `${HOME}/.local/conda/envs/torch`. Every option
for every invoked command is explicit. DataLoader workers default to zero for
the one-core allocation. `TOPDIR` comes from `SLURM_SUBMIT_DIR`, avoiding Slurm's
spooled-script location.

## Run layout

```text
runs/run01/
  config.json, provenance.json, inputs.json, folds.json, dataset.npz
  inputs/                         original JSON snapshots
  source_snapshot/mixture_models/  exact source used for training
  invocations/                    each training/resume invocation
  trials/d_XXX/fold_YY/<trial>/    config, metrics, best.pt, last.pt, history.npz, loss.png
  best/d_XXX/                     selection.json and exactly k selected checkpoints
  canonical/d_XXX/                config, metrics, transforms, features, samples, diagnostics, plots
  evaluation/d_XXX/<label>/        config, metrics, predictions, converted data, plots
  evaluation/                     per-label loss-by-dimension summaries and plots
```

Original source files used as references are bundled under `reference_sources/`
for auditing and numerical comparisons. Their hashes and repository provenance
are in `SOURCE_PROVENANCE.json`. Runtime provenance records source hashes, Git
commit when available, package/backend settings, seeds, hardware, and environment.
See [VALIDATION.md](VALIDATION.md) for the checks performed before delivery.
