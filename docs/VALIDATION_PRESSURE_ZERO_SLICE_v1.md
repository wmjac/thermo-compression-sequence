# Verification of the two-dimensional P = 0 slice implementation

Passed before packaging:

- Ten numerical/geometry/plot tests: known five-model pressure curves,
  per-model and ensemble roots, rejection of negative slopes, vacuum exclusion,
  upper-boundary roots, multiple positive-slope roots, bracket refinement,
  discontinuous/nonfinite objective failure, invalid tolerances, ambiguous
  root matching, fixed-coordinate geometry, full-dimensional hull exclusion,
  deterministic extrema handling, and degenerate projections.
- Plot-domain checks inspect actual Matplotlib scatter and mesh inputs:
  neither dots nor colored cells include hull-excluded grid centers.
- Pinned legacy regression at idpmixture commit
  c425574a9fb61b24f6a8f2db748e18ef50cdc28b: projected lattice coordinates,
  projected mask, fixed plane, LP flags, and positive-slope roots agree on
  matched inputs. Legacy functions were retrieved by their recorded blob SHAs.
- Float64 canonical/native pressure invariance checked independently.
  Autodifferentiated directional slopes agree with finite differences of
  float64 pressure evaluations.
- Full pipeline with actual synthetic-data architecture checkpoints at d=2,3,
  using two selected models, exported canonical features, and the full hull.
  The refined fixture included a root-bearing slice and a root-free slice,
  projected exclusions, and a point excluded by the full-dimensional test.
- Exact fixed-coordinate TSV values, actual root counts, empty-root headers,
  residuals/slopes, evaluation dtype, logs with exact calls, completion status,
  and output hashes checked.
- Forced Brent nonconvergence: root/per-dimension metadata reports failed,
  readable logs preserve the cause and maxiter=1 call, and no misleading
  no-root heatmap is produced.
- Source training, evaluation, canonical, and checkpoint artifacts remain
  unchanged; package source hashes remain unchanged.
- Solution, no-root, and unavailable-model-spread maps rendered and visually
  checked. Legends sit outside the data area so no grid centers are hidden.
- Python syntax, Slurm shell syntax, and explicit template coverage of all
  33 CLI options checked (the repeated offset option is explicitly empty by
  default and documented with examples).

The integration fixtures are small synthetic-data runs, not production
AF20/EK20 measurements. Execution used Python 3.12, PyTorch 2.8.0+cpu,
NumPy 2.3.5, SciPy 1.17.0, and Matplotlib 3.10.8.
GPU execution and live Slurm submission were not exercised in this environment.

Run the included tests from the project root:

    python3 -m unittest discover -s tests -p test_pressure_zero_slice_v1.py -v
