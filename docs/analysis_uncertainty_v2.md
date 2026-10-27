# Test-loss and coordinate uncertainty, v2

Unpack this additive overlay in the repository root. It requires the earlier
post-training v1 overlay and the original src/mixture_models package. All
delivered filenames are new. No training protocol, package source, selection,
checkpoint, units, or canonical prior is changed. Existing training-resume
package source hashes are preserved. No installation is required.

## Coordinate standard deviations

Use analysis/export_sequence_features_v2.py in place of
analysis/export_sequence_features_v1.py, with the same CLI options.
The TSV filename remains features_d_NNN.tsv. Existing columns and their order
remain unchanged; the new final columns are z_1_std ... z_d_std.

For a sequence s, with k selected models in their saved aligned coordinates:

    z_j_mean(s) = mean_m z_mj(s)
    z_j_std(s)  = sqrt(mean_m (z_mj(s) - z_j_mean(s))**2)
    feature_rmsd(s)**2 = sum_j z_j_std(s)**2

These are population SDs across models (ddof=0), independently for every
sequence and coordinate. They do not measure variation across sequences, and
are not standard errors of the feature mean. A single selected model gives zero
population spread. There is no division by sqrt(k) or by the number of axes.
Existing RMSD uses precisely the v1 summation order. All exported values use
float64 aggregation and .17g text; inference retains the checkpoint dtype.
Optional member-feature NPY output and dataset-membership definitions are unchanged.
Metadata records the new definitions, actual model count, and v2 script hashes.
The descriptor-augmentation script continues to work with this expanded TSV.

Example, run from the repository root with the existing torch environment:

~~~bash
python3 analysis/export_sequence_features_v2.py \
  --run-dir /path/to/af20_batch000001_best/analysis01 \
  --dimensions 1 2 3 4 5 6 \
  --sequence-list /path/to/af20_sequences.txt \
  --test-json /path/to/test.json \
  --membership-symmetry reversal \
  --output-dir /path/to/af20_batch000001_best/post_training02/features \
  --batch-size 2048 --device cuda --threads 1 \
  --save-member-features no --overwrite no
~~~

## Test loss across selected models

The original evaluation already archives metrics.json individual-model losses
and predictions.npz pressure spread per test system. It does not provide an
aggregate SD of model test losses. analysis/evaluate_ensemble_v2.py adds that
summary and error-bar plots. It accepts normal runs and the combined
analysis01/run_views/d_NNN layout.

Two different central values are kept explicit:

    L_m       = mean_i [ (P_i - P_mi) / c_tot,i ]**2
    L_member  = mean_m L_m
    SD_member = sqrt(mean_m (L_m - L_member)**2)
    L_ensemble = mean_i [ (P_i - mean_m P_mi) / c_tot,i ]**2

Pressures are already in kT nm^-3 and concentrations in nm^-3; these losses
are dimensionless. Per-model losses are sample means over the same test systems.
Generally L_member differs from L_ensemble. The saved check is:

    L_member = L_ensemble + mean_i Var_m(P_mi / c_tot,i)

Error bars are centered on L_member, not L_ensemble. The plots also show
L_ensemble separately, preserving the original predictive-loss interpretation.
On RMSE plots the member curve is mean_m sqrt(L_m), with SD_m sqrt(L_m);
it is neither sqrt(mean_m L_m) nor a square root of an SD.

SDs have ddof=0 for consistency with existing ensemble spreads and feature
RMSD. They are descriptive model disagreement, not confidence intervals or
SEM estimates: cross-validation members share training data. The actual selected
model count is used (normally five). Validation remains the original out-of-fold
evaluation; losses from different held-out folds are not assigned these
same-test-set ensemble error bars.

### Existing test evaluation: no model inference required

~~~bash
python3 analysis/evaluate_ensemble_v2.py \
  --run-dir /path/to/af20_batch000001_best/analysis01 \
  --dimensions 1 2 3 4 5 6 \
  --output-dir /path/to/af20_batch000001_best/post_training02/test_evaluation \
  --reuse-predictions yes --data-json \
  --name test --basis canonical --batch-size 2048 \
  --save-hessians no --allow-training-overlap no \
  --device cpu --threads 1 --overwrite no
~~~

This reuses evaluation/d_NNN/test/{metrics.json,config.json,predictions.npz}.
Original evaluation files are read-only. Checkpoint identities, saved selection
and canonical-transform hashes, member order, complete test eligibility, and
agreement of saved losses with cached predictions are verified. No test JSON
paths should be supplied in reuse mode: the saved dataset is authoritative.
The batch-size, save-hessians, and allow-training-overlap options affect fresh
evaluation only; reuse mode performs no new overlap decision or Hessian calculation.

### Fresh test evaluation

Use the same command with --reuse-predictions no and provide paths after
--data-json. The original mixture_models.evaluation.run_evaluation first
creates the standard test artifacts; this script then summarizes them.
--name chooses that evaluation label. Use a fresh label if an evaluation
already exists. --overwrite yes explicitly permits replacement of the old
evaluation in fresh mode as well as this script's output directory.
The original test-data unit conversion and overlap checks remain in force.
Canonicalization and training are not repeated.

### New output directory

- metrics_d_NNN.json: original metric content plus test_model_spread.
- summary_test.json: the same detailed metrics across requested dimensions.
- loss_summary.tsv: dimension/model/system counts; ensemble-prediction loss
  and RMSE; member mean/SD of loss and RMSE; normalized prediction variance.
  The fields are ensemble_pressure_loss, ensemble_normalized_rmse,
  member_pressure_loss_mean, member_pressure_loss_std,
  member_normalized_rmse_mean, member_normalized_rmse_std, and
  mean_normalized_prediction_variance.
- individual_model_losses.tsv: per-member losses and original metadata,
  including checkpoint, fold, and checkpoint validation score.
- loss_by_dimension_test.png: linear loss, member mean +/- SD, and the
  separate ensemble-prediction curve.
- loss_by_dimension_test_log.png: the same on a log loss ordinate.
- rmse_by_dimension_test_log.png: mean member RMSE +/- its SD and the
  ensemble-prediction RMSE on a log ordinate.
- metadata.json: definitions, CLI/provenance/source identities, output hashes,
  and completion status. Only a completed run has status: complete.

Log plots omit zero points and nonpositive lower SD endpoints, with an
annotation. They do not replace numerical values by a positive floor.
All underlying values remain available at full text precision in the TSVs.
These new figures live in the new output directory; old plots/CSV/JSON files
are not silently rewritten in reuse mode.

## Slurm

slurm/post_training_analysis_v2.slurm is the updated explicit-options template.
It adds test uncertainties, calls the v2 feature exporter, and retains the v1
dimension-comparison analysis. Copy it into the _best directory as your local
analysis.slurm (or another name), edit its settings, and submit as before.
Use OUTPUT_ROOT=.../post_training02 (the default) to retain old outputs.

With REUSE_TEST_PREDICTIONS=yes, RUN_DIR must already contain the desired
test evaluation under TEST_EVALUATION_NAME. For new test data set
REUSE_TEST_PREDICTIONS=no and choose a fresh TEST_EVALUATION_NAME.
Set RUN_TEST_UNCERTAINTY=no if no independent test set exists.
The template still requires the sequence list and configured JSON paths for
feature membership. All CLI values are explicit; existing rho_chem/rho_mix
remain archived canonicalization choices. GPU/CPU requests match the prior
MIG template and can be edited for an existing A100 allocation.

Checks:

~~~bash
python3 -m unittest discover -s tests -p test_analysis_uncertainty_v2.py -v
~~~
