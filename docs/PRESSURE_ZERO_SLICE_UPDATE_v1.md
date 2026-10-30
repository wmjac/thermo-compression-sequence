# Two-dimensional P = 0 slice calculation

Unpack this additive overlay in the thermo-compression-sequence repository root.
It requires the earlier post-training v1 scripts and existing project environment.
No pip install, training changes, new checkpoints, or new canonicalization.

1. Copy slurm/pressure_zero_slice_v1.slurm to the _best directory.
2. Edit RUN_DIR, FEATURES_DIR, OUTPUT_DIR, DIMENSIONS, AXES, VOXEL_SIZE, and
   optional offsets/hull settings. All solver options are explicit.
3. Submit from that directory using sbatch.

With the template defaults:

    _best/p_zero_slice01/solver.log
    _best/p_zero_slice01/d_003/lattice.tsv
    _best/p_zero_slice01/d_003/slice_definition.json
    _best/p_zero_slice01/d_003/solver.log
    _best/p_zero_slice01/d_003/c_tot_largest_root.png

This is a 2D affine slice: d-2 coordinates are fixed and explicitly recorded in
columns such as z_3_fixed. Human-readable logs record the exact invocation,
fixed values, numerical settings, and detailed solver calls.

The primary heatmap colors retained grid centers by the largest accepted
positive-slope P=0 concentration; no-root centers are black dots. Excluded
centers have no colored cells or marks. All root branches remain in tables.

Defaults address legacy numerical stability: normalized pressure objective,
float64 decoder evaluation, autodifferentiated slopes, independent residual
checks, and bracket-grid refinement. Numerical failures are logged explicitly
instead of becoming no-root points.

Default hull policy is full: projected plus d-dimensional LP membership.
Use --hull-policy projected to reproduce the old projected-only root domain.
For large sequence clouds, optional LP sampling is explicit, seeded, and recorded.

The solver itself has no 2D assumptions and can support future higher-dimensional
domain constructors. See docs/pressure_zero_slice_v1.md for all definitions,
CLI usage, units, outputs, and deliberate changes relative to idpmixture.
