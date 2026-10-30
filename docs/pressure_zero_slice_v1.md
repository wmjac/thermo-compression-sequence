# P = 0 on a two-dimensional canonical slice

This is a two-dimensional affine slice in learned dimension d. Two canonical
coordinates vary; d-2 coordinates are fixed. It does not reconstruct the full
d-dimensional P=0 surface or establish phase coexistence. The concentration
solver is independent of the 2D geometry/plotting code for future extensions.

## Setup and invocation

Requires analysis/_post_training_common_v1.py from the earlier post-training
overlay, the existing src/mixture_models package, and your usual torch, SciPy,
NumPy, Matplotlib environment. Either v1 or v2 feature exports are accepted.
No installation or new training/canonicalization is needed.

Use slurm/pressure_zero_slice_v1.slurm from your _best directory. Defaults:
RUN_DIR=analysis01, FEATURES_DIR=post_training02/features,
OUTPUT_DIR=p_zero_slice01, dimensions 2 through 6, axes 1 and 2.
Grid spacing is explicit and must suit your canonical-coordinate ranges.

~~~bash
python3 analysis/calc_pressure_zeros_v1.py \
  --run-dir /path/to/af20_batch000001_best/analysis01 \
  --features-dir /path/to/af20_batch000001_best/post_training02/features \
  --dimensions 3 --axes 1 2 \
  --output-dir /path/to/af20_batch000001_best/p_zero_slice01 \
  --voxel-size 0.01 --hull-policy full --hull-tol-mult 1.0 \
  --lp-num-points all --lp-seed 0 --lp-feasibility-tol 1e-9 \
  --lp-time-limit 60 --max-grid-points 250000 \
  --c-min 1e-8 --c-tot-max 0.25 --n-brackets 64 \
  --min-scan-refinements 1 --max-scan-refinements 3 \
  --root-xtol 1e-12 --root-rtol 1e-12 \
  --pressure-atol 1e-10 --pressure-rtol 1e-8 \
  --slope-threshold 1e-6 --root-maxiter 100 \
  --match-atol 1e-4 --match-rtol 0.25 --evaluation-dtype float64 \
  --batch-size 2048 --device cuda --threads 1 \
  --progress-every 20 --overwrite no
~~~

Axis indices are 1-based. Require d>=2; dimension 1 has no 2D slice.
--offset 3 0.01 adds 0.01 to the mean of fixed coordinate 3; repeat for
other fixed dimensions. Every offset dimension must be fixed and present in
all requested d values. The unmodified means, offsets, and resulting fixed
values are all saved.

## Geometry and hull policy

The geometry uses ensemble-mean sequence features from the chosen TSV, with
the same saved canonical basis. Export completion, TSV hash, selected-model
identity/count, and canonical-configuration identity are verified.

The sequence population is exactly the exported TSV rows. Their arithmetic
mean fixes the other coordinates, following the old script. This is not
silently replaced by a rho_chem-weighted mean or a smaller sequence pool.
Archived rho_chem and rho_mix remain the choices defining the canonical basis.
The lattice is origin-aligned: coordinates (k*h,l*h), h=voxel-size. Integer
bounding-box limits use floor/ceil, followed by the projected hull test.

The legacy global RMSD tolerance is recovered from the exported vector RMSDs:

    global_RMSD = sqrt(mean_sequence(feature_rmsd(sequence)^2))
    epsilon = hull_tol_mult * global_RMSD

There is no division by d or sqrt(k). This geometric convention is not a
confidence region. The LP solves:

    weights >= 0
    sum(weights) = 1
    abs(Z_lp.T @ weights - query) <= epsilon, componentwise

Default --hull-policy full retains centers passing both projected and full LP
tests. This is an explicit change from the legacy script, which annotated full
membership and solved all projected-hull points. --hull-policy projected
restores that legacy domain; full LP flags are still saved.

All points are used by default. Exact duplicates are removed without changing
the chosen hull. Integer --lp-num-points enables seeded subsampling, retaining
all coordinate bounding-box extrema; too few requested points is an error.
Actual original row indices are saved in lp_indices.npy.

All-points LPs can dominate runtime with a 500k-sequence cloud. No automatic
sampling occurs. A sampled hull is a subset of the full cloud: outside the
sampled hull does not establish exclusion from the full sequence hull. This
limitation is recorded prominently whenever sampling is selected.

LP numerical tolerance is separate from geometric epsilon. Time-limit and
numerical failures abort explicitly instead of being labelled outside.
Degenerate projections fail clearly; no artificial jitter is added.

## Numerical pressure calculation

Each member evaluates its own decoder and saved inverse transform at the
same canonical latent point x=c*zbar, preserving the legacy shared-ray scan:

    beta P_m(c) = c + beta P_ex,m(c*zbar)
    ensemble_pressure(c) = mean_m beta P_m(c)

These are latent-space comparisons. They are not predictions for a fixed
physical sequence mixture whose coordinate would differ between encoders.
Concentrations are nm^-3; physical pressures are kT nm^-3; solver pressures
are beta P in nm^-3. No legacy factor-of-ten or atm conversion is added.

The scalar root objective is Q(c)=beta P(c)/c for c>c-min, with the same
positive roots and the trivial vacuum zero removed. Member curves use Q_m.
Float64 decoder evaluation is the default, including in-memory conversion of
loaded parameters. Saved checkpoints/features remain unchanged.
--evaluation-dtype checkpoint permits an explicit arithmetic-precision comparison.
Original and evaluation dtypes are recorded.

Brent uses separate concentration xtol/rtol. Every candidate must satisfy:

    abs(beta P) <= pressure_atol + pressure_rtol*c

Only exact floating-point grid zeros are endpoint candidates. Arbitrary
near-zero grid pressures do not become roots without a bracket.
Slope uses the transformed autodifferentiated free-energy Hessian:

    d(beta P_m)/dc = 1 + c*zbar.T @ H_ex,m(c*zbar) @ zbar

The ensemble slope averages member derivatives. Accepted roots require slope
strictly above slope-threshold. This replaces the legacy tiny finite difference
applied to float32 decoder evaluations.

The concentration scan starts with n-brackets intervals and doubles resolution.
After at least min-scan-refinements, all member and ensemble accepted root sets
must agree with the preceding scan within 10*(root_xtol+root_rtol*abs(c)).
Otherwise refinement continues up to max-scan-refinements. Finite-grid
agreement is a convergence diagnostic, not proof that arbitrarily narrow root
pairs are absent. Use a resolution study when needed.

Nonconvergence, failed pressure residuals, nonfinite values, and unresolved LPs
abort with failed metadata and a readable log. These failures never become
black no-root points. No-root means no accepted positive-slope root was found
within the configured window. The concentration window is never automatically
expanded. Positive slope is mechanical stability along the fixed ray only;
composition stability and coexistence require additional calculations.

## Ensemble-root uncertainty

Roots of ensemble-mean pressure and means of member roots are distinct outputs.
Every member is solved independently along the common canonical ray.
All accepted roots are saved, even if they cannot be matched.

Matching is unique-nearest, local to each grid center:

    abs(c_member-c_ensemble) <= match_atol + match_rtol*abs(c_ensemble)

Equal-distance matches or multiple member roots assigned to one ensemble root
are ambiguous and excluded from that summary. This is not global branch
continuation. Raw roots and matching outcomes remain available.

Matched concentration SD uses the population denominator (ddof=0); it is blank
with fewer than two matches. The mean, support count, and complete/partial/
insufficient status are explicit. These are descriptive model disagreements,
not SEMs/confidence intervals. Pressure SD at the ensemble root uses every
selected member and is also saved.

## Exact outputs

The --output-dir root has metadata.json and solver.log. Each d_NNN/ contains:

| File | Contents |
| --- | --- |
| solver.log | Human-readable exact invocation, fixed values, hull policy, dtype, root method/settings, every candidate call/result, refinement progress and failures. UTC timestamps. |
| slice_definition.json | Varied axes, exact fixed values, means, offsets, explicit 2D scope. |
| lattice.tsv | Every bounding-box center, including exclusions, with full coordinates/statuses. |
| roots.tsv | Accepted ensemble-pressure roots from the final scan. |
| member_roots.tsv | Accepted per-model roots from the final scan. |
| root_candidates.tsv | Candidates at all refinements, including rejected slopes. |
| root_summary.tsv | Ensemble root, matched-member mean/SD, support, pressure SD. |
| root_matching.tsv | Matched, ambiguous, and out-of-tolerance member-root outcomes. |
| lp_indices.npy | Original row indices used by the LP. |
| grid.npz | Full coordinates, indices, hull equations/statuses, retention mask. |
| c_tot_largest_root.png | Primary solution heatmap, largest accepted root at each retained center. |
| c_tot_model_std.png | Model concentration SD near that largest root. |
| c_tot_root_rank_NN.png | Additional maps if multiple roots occur, by descending local concentration. |
| metadata.json | Definitions, sources, settings, units, counts, immutable output hashes, status. |

For d=4, axes 1 and 2, lattice columns include:

    point_id grid_k grid_l z_1_varied z_2_varied z_3_fixed z_4_fixed ...

Exact fixed values repeat in every row, at full precision. Other columns:
inside_projected_hull, full_hull_status, retained, n_solutions, solver_status,
final_brackets, refinements. n_solutions is the actual count, correcting the
old 0/1 flag. It is blank for excluded points. Only retained, successfully
scanned centers can have zero solutions.

Local root indices are 1-based ascending concentration. Member indices are
zero-based; member=-1 means the ensemble-mean curve. The primary map explicitly
selects the largest local root; all branches remain in tables.

Plots show the two varied coordinates of zbar. Solutions color the heatmap;
retained centers with no accepted ensemble root have black dots. Excluded
centers have neither colored cells nor markers. The SD map uses grey cells
where a root exists but concentration SD is unavailable. Ranked maps can have
blank retained cells with too few roots for that rank. Captions state fixed
coordinate values; TSV/JSON/log retain the exact values.

Numeric tables use .17g for float64 round trips. Logs are excluded from
immutable hashes because completion/failure messages can be appended.
Completion/failure is recorded at both levels. Existing outputs require
--overwrite yes and must be a previous post-training output or empty directory;
input directories and ancestors are protected.

## Baseline and verification

Baseline: wmjac/idpmixture commit c425574a9fb61b24f6a8f2db748e18ef50cdc28b,
calc_P_zeros.py, calc_fe_slice.py and eval_model.py. Deliberate changes are
recorded in the root metadata: actual ensemble size, current unit/model API,
default full-hull filtering, normalized objective, double decoder evaluation,
autodifferentiated slopes, residual/refinement checks, model spread, correct
root counts, full precision, explicit slice/fixed-coordinate output and
masked heatmaps. Training code and canonicalization are unchanged.

~~~bash
python3 -m unittest discover -s tests -p test_pressure_zero_slice_v1.py -v
~~~
