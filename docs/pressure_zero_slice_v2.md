# P=0 slice reporting with explicit upper-bound caps, v2

This update addresses an incomplete high-concentration branch: a pressure
curve can have small ascending zero crossings but still be negative at the
largest scanned concentration. Reporting the largest root found within the
window can then produce a misleading low concentration in the primary map.

## Install and run

Unpack in your top-level `thermo-compression-sequence` directory:

```bash
tar -xzf thermo-compression-sequence_pressure_zero_slice_v2.tar.gz
```

This is an additive overlay. It requires the existing P=0 slice v1 and
post-training v1 helpers already used in your workflow. All v1 files, model
code, trained checkpoints, canonicalization, and training protocols are kept.
No package installation is required.

Copy `slurm/pressure_zero_slice_v2.slurm` into your `_best` directory, edit it
like the previous slice template, and submit it there. It invokes
`analysis/calc_pressure_zeros_v2.py` and defaults to a fresh `p_zero_slice02`
output directory. Every CLI setting is explicit in the Slurm file, including
`--upper-bound-policy cap`. The original solver still operates on a **2D slice**
with all other coordinates fixed as recorded in the output.

## Reporting rule

For each retained lattice point, let

\[
\epsilon_P = \texttt{pressure\_atol}
             +\texttt{pressure\_rtol}\,c_{\mathrm{tot,max}}.
\]

Under `--upper-bound-policy cap`, if

\[
\beta P(c_{\mathrm{tot,max}})<-\epsilon_P,
\]

the **reported concentration is exactly `c_tot_max`**, with
`reported_status=capped_at_c_tot_max` and `upper_bound_censored=1`. This applies
even when smaller positive-slope roots exist. The final negative-pressure
interval is still open at the upper boundary. A finite higher root has not
been computed or mathematically established by this test.

Otherwise the reported concentration is the largest accepted in-range root,
or empty if none exists. A genuine root at the upper boundary is a root, not
a cap. Small negative endpoint pressures within the tolerance are not capped.
Positive pressure at the upper boundary without a root continues to be reported
as no positive-slope root; it is not assigned `c_tot_max`.

`--upper-bound-policy roots-only` restores v1 reporting values while retaining
the new endpoint-pressure diagnostics. All calculations use kT and nm^-3.

The upper-bound pressure comes from the existing root scan: this adds no model
evaluation. The v1 scan grid, Brent calls, numerical tolerances, slope tests,
candidate roots, and matching procedure are reused. In particular, finite-grid
root detection has the same limitations as v1. Caps do not repair a missed
crossing when the endpoint pressure is positive.

## Outputs

- **`c_tot_reported.png` is the new primary map.** It shows actual reported
  roots and caps. Open triangles identify points capped at `c_tot_max`.
  Black dots indicate retained points with neither a root nor a cap.
  Hull-excluded points have no marks or colors. Fixed coordinates are labeled.
- `lattice.tsv` adds `reported_c_tot`, `reported_status`,
  `upper_bound_censored`, `upper_bound_pressure_negative`,
  `beta_pressure_at_c_tot_max`, `upper_bound_beta_pressure_tolerance`, and
  `largest_root_in_range`. The existing `n_solutions` and `solver_status`
  continue to describe actual accepted roots. A capped point can have zero
  actual roots or several smaller roots.
- `curve_summary.tsv` contains these reporting values and root counts for
  the mean curve (`member=-1`) and every individual model (zero based).
  The same cap rule is applied independently to each curve.
- `roots.tsv`, `member_roots.tsv`, `root_candidates.tsv`, `root_summary.tsv`,
  and `root_matching.tsv` retain their original meanings. Caps are never
  inserted into root lists or averaged into root-concentration uncertainty.
- `c_tot_largest_root.png` and the root-rank maps remain available as raw
  in-range-root diagnostics. Use **`c_tot_reported.png`** for the capped result.
- Both levels of `metadata.json` and `solver.log` document the policy, exact
  command, threshold, source hashes, counts, and output provenance. Per-dimension
  counts include `points_capped_at_c_tot_max`.

Additional crossings are retained without smoothing or an arbitrary cutoff.
This change addresses upper-window reporting; it does not automatically
classify fluctuations, select phases by free energy, or solve coexistence.

## Uploaded AF20 sweeps motivating the change

All five supplied sweeps used d=3, the same five selected models, and the same
canonicalization. Their table hashes match their archived metadata. They use
each member's own sequence encoding for the specified physical mixture;
the lattice calculation continues to use a shared canonical ray.

Approximate roots below are interpolated from the sweep tables, not newly
evaluated checkpoint roots. Pressure is in kT nm^-3; concentrations are nm^-3.

| Sweep | Composition | Mean P(0.25) | Ascending mean crossings | v2 reported value |
| --- | --- | ---: | --- | --- |
| 1 | A20 | 2.719609 | None | No root |
| 2 | F20 | 1.110067 | 0.008789, 0.246116 | 0.246116 |
| 3 | Equimolar F20/A20 | 3.357427 | 0.042912 | 0.042912 |
| 4 | A10F10 | 1.856238 | 0.054428, 0.117744 | 0.117744 |
| 5 | AFFFFFFFFAFFFFFFFFFA | -1.123888 | 0.008190 | **0.25, capped** |

Every individual model in sweep 5 also has negative pressure at 0.25. In
sweep 2, two individual models have negative endpoint pressure even though
the ensemble mean has an ascending root just below 0.25. The per-curve table
makes that boundary information available without mixing caps into root SDs.

## Checks

Run the new analytical tests after unpacking:

```bash
python3 -m unittest discover -s tests -p test_pressure_zero_slice_v2.py
```

The tests cover multiple negative-pressure intervals, a higher root beyond the
window, no in-range ascending root, true boundary roots, positive-pressure
no-root curves, tolerance handling, configurable `c_tot_max`, unchanged raw
roots and evaluation counts, and plot masking/markers.

Before delivery, all six analytical tests passed. Reporting on all five
uploaded AF20 sweeps reproduced the outcomes in the table above. A CPU
integration using saved synthetic-data checkpoints (d=2 and d=3, two members
each) produced 17 caps in d=2 and none in d=3. Its five raw-root/matching tables
were byte-identical to v1, and model-evaluation counts were unchanged.
Output masks, cap markers, logs, metadata, hashes, and fixed coordinates were
checked. Every file in the original v1 archive manifest was unchanged.
GPU inference and live Slurm submission were not exercised.
