# Verification of the analysis uncertainty update

Passed before packaging:

- Five numerical unittest checks, including known five-model losses with
  unequal concentrations; the distinction between mean member loss and loss
  of mean prediction; mean member RMSE versus sqrt(mean loss); single-member
  behavior; invalid data; and the feature coordinate/RMSD identity.
- Real synthetic-data checkpoints at dimensions 1, 2, and 3, with two selected
  members per dimension. This also verifies that the implementation uses the
  actual ensemble size instead of hard-coding five.
- Fresh independent-test evaluation through the original evaluation routine,
  followed by reuse through combined run_views. The loss-summary TSV,
  individual-model TSV, and detailed JSON were byte-identical between paths.
- Per-model loss SDs and RMSE SDs checked directly against cached predictions.
- v1 versus v2 feature exports with uneven sequence batches: every existing
  TSV field and optional per-model NPY value was identical; the new coordinate
  SD columns exactly round-tripped float64 population SDs of aligned features.
- Rejection of an evaluation labelled validation when test uncertainty is
  requested, preserving the distinction between held-out-fold evaluation and
  the same-test-set model comparison.
- Existing training/canonical files remained unchanged. Cached analysis did
  not alter original evaluation files. The package source hash did not change.
- Linear/log loss and log RMSE error-bar plots generated and visually inspected;
  all-zero plots and nonpositive lower SD endpoint handling checked.
- New script hashes, completion markers, and output hashes checked.
- Python syntax, Slurm shell syntax, and explicit coverage of CLI options in
  the updated template checked.

Integration used the earlier vectorized source plus plotting v3, the existing
post-training v1 helpers, and this additive overlay. Environment: Python 3.12,
PyTorch 2.8.0+cpu, NumPy 2.3.5, Matplotlib 3.10.8. Real checkpoint integration
uses small synthetic-data fixtures, not production AF20/EK20 results.
GPU execution and live Slurm submission were not exercised here.

Run the included numerical tests from the repository root:

    python3 -m unittest discover -s tests -p test_analysis_uncertainty_v2.py -v
