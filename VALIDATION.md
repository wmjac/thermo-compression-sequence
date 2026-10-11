# Delivery validation

Validated on CPU with Python 3.12 and PyTorch 2.8.0+cpu. Production GPU/SLURM
execution and long-run scientific convergence were not tested in this workspace.

## Scientific and reproducibility tests

The bundled pytest suite contains 20 passing checks, covering:

- Declared-unit conversions, including atm/Pa, molar concentrations, scaled
  number-density units, molar energies, temperature offsets, and SEM scaling.
- Explicit errors for missing, unknown, or dimensionally incorrect units.
- General ingestion of sequences of different lengths, with the current
  model enforcing its own length-20 restriction.
- Pressure, normalized loss, and parameter-gradient equivalence to Bethany's
  supplied training model in float32 and float64.
- Equivalence of free-energy density, pressure, gradients, and Hessians to the
  authoritative repository decoder, including the zero-density limit.
- Sequence-reversal symmetry.
- Mixture-prior sampling equality to the original sampler with matching RNG
  state, including small-pool truncation.
- Canonical transformation consistency, thermodynamic distance preservation,
  chemical-potential and Hessian invariance, arbitrary ensemble size, and
  agreement with the original PCA/Procrustes functions.
- Exact model/history/RNG agreement between uninterrupted training and an
  epoch-boundary interruption followed by resumption.
- Presence of every command's public CLI options in its production SLURM script.

The float32 first-Adam-step comparison identified differences of about 1e-5
only in the attention **key-bias** entries. Exact attention probabilities are
invariant to these entries because they add the same constant to every softmax
logit. Floating-point gradient noise is amplified by Adam's epsilon in those
unobservable directions. The test checks the remaining parameter entries and
post-update predictions/loss. On the diagnostic fixture the maximum normalized
pressure difference after the update was about 1.3e-7 and the float32 loss
difference was zero. No parameter freezing, gradient modification, or optimizer
change was introduced to conceal this numerical effect.

These are algebraic and small-scale numerical comparisons; they do not claim
bitwise reproduction of historical stochastic optimization trajectories.

## End-to-end smoke workflow

`examples/run_smoke.py` passed with the delivered source. It creates synthetic
JSON data with explicit units, pools two 15-system training batches, and uses a
disjoint 10-system test batch. It trains dimensions 2 and 3 with five folds,
two candidates per fold, and three epochs per candidate. For both dimensions it:

1. Retains exactly ten checkpoints, two from each fold.
2. Repeats complete-run resumption and verifies an unchanged selection manifest.
3. Estimates the metric from 200 prior mixtures and determines the canonical basis.
4. Evaluates validation and test data, including Hessians and diagnostic plots.
5. Confirms two eligible held-out members for each validation system.
6. Confirms native and canonical pressure predictions agree within float32 tolerance.

A separate pilot used 30 training systems split across two files and ten disjoint
test systems from the uploaded simulation dataset, with the agreed units block
added to scratch copies. Training, selection of ten members, canonicalization,
and both evaluation modes completed. Native/canonical pressure comparisons for
that pilot had zero maximum absolute difference in the tested batches. Original
uploaded data files were not modified.

All four SLURM scripts passed Bash syntax checks. Source, tests, and examples
passed Python compilation checks. The release archive excludes generated run
outputs and dependencies; `CHECKSUMS.sha256` inventories the delivered files.
