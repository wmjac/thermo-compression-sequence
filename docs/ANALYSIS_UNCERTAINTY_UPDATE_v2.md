# Analysis uncertainty update, v2

Unpack in /home/wjacobs/git/thermo-compression-sequence (the repository root):

    tar -xzf /path/to/thermo-compression-sequence_analysis_uncertainty_v2.tar.gz

This is an additive update on top of the post-training v1 scripts. Existing
files, package/training code, model weights, and canonical bases are preserved.
All uncertainty calculations use the actual selected ensemble count.

Changes:

1. analysis/export_sequence_features_v2.py uses the v1 CLI and appends
   z_1_std ... z_d_std after the existing feature_rmsd column. Each is the
   population standard deviation across models in the saved aligned canonical
   basis. Existing z means and RMSD retain their definitions and precision.

2. analysis/evaluate_ensemble_v2.py adds mean and population SD across model
   test losses, plus linear/log loss and log RMSE error-bar plots. The loss of
   the ensemble-averaged prediction is reported separately. Its default reuses
   saved test predictions without model inference or original-output changes.
   Fresh test evaluation is also supported via the original evaluation routine.

3. slurm/post_training_analysis_v2.slurm runs both updates and the existing
   dimension comparisons, with all CLI options explicit. Copy this template
   into the _best directory, edit its settings, and submit. A fresh
   post_training02 output is the default. Existing canonicalization is reused.

To rerun just the coordinate export, change the filename in your current
Slurm command from export_sequence_features_v1.py to export_sequence_features_v2.py
and choose a fresh feature output directory. All other options stay the same.

To add error bars to an already completed test evaluation, use
evaluate_ensemble_v2.py with --reuse-predictions yes; this does not require a GPU.
The template's RUN_DIR should point to the existing analysis01 directory.
See docs/analysis_uncertainty_v2.md for commands, formulas, output fields,
reuse/fresh options, and how to select the evaluation label.

SDs use ddof=0, consistent with existing model spread and vector RMSD.
They describe disagreement among models, not standard errors or confidence
intervals. Validation fold handling is unchanged.

Dependencies: the existing project/torch environment and post-training v1
analysis/_post_training_common_v1.py. No pip install is required.
