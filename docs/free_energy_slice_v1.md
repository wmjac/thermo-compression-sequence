# Excess free-energy and gradient slices, v1

This additive analysis evaluates the selected canonicalized decoders at one
CLI-specified total chain concentration, using the same affine feature grid
and hull routines as the current P=0 calculation. It does not modify model,
training, canonicalization, or P=0 source files.

## Installation and submission

Unpack in the top-level `thermo-compression-sequence` directory:

```bash
tar -xzf thermo-compression-sequence_free_energy_slice_v1.tar.gz
```

The existing post-training/P=0 helpers must already be present:
`analysis/_post_training_common_v1.py`, `_pressure_zero_geometry_v1.py`, and
`_pressure_zero_io_v1.py`. These are shared directly; the overlay does not
replace or duplicate them. Use the existing Python environment; no `pip
install` is required.

Copy `slurm/free_energy_slice_v1.slurm` into your `_best` directory, edit its
settings, and submit it from there. The template matches the existing Della
MIG environment and gives explicit values for every CLI option. Its example
concentration is `C_TOT=0.1`, in **nm^-3**. The Python option **`--c-tot` is
required**, with no implicit default. Choose a separate output directory for
each concentration or slice.

To reproduce a P=0 job's domain exactly, copy its dimensions, axes, offsets,
voxel size, hull policy, tolerance multiplier, LP point count, and LP seed.
The same feature export and canonical run must be used. The usual
`analysis01/run_views/d_NNN` combined-run layout is supported.

A direct example, from your `_best` directory:

```bash
python3 /home/wjacobs/git/thermo-compression-sequence/analysis/calc_free_energy_slice_v1.py \
    --run-dir "$PWD/analysis01" \
    --features-dir "$PWD/post_training02/features" \
    --dimensions 3 \
    --axes 1 2 \
    --c-tot 0.1 \
    --output-dir "$PWD/free_energy_slice01" \
    --voxel-size 0.1 \
    --hull-policy full \
    --hull-tol-mult 1 \
    --lp-num-points 10000 \
    --lp-seed 0 \
    --device cpu
```

Add offsets with the existing convention, e.g. `--offset 3 0.2` adds 0.2
to the exported sequence-pool mean of coordinate 3. It does not set the
coordinate to 0.2. Offsets may only select unvaried coordinates and must be
valid in every requested dimension. Exact fixed values are repeated in the
TSV and saved in `slice_definition.json`; plot captions use display rounding.

## Calculation and units

At each shared canonical feature point, every selected model evaluates

\[
\boldsymbol{x}=c_{\mathrm{tot}}\bar{\boldsymbol z},\qquad
f_m^{\mathrm{ex}}=\Psi_m(\boldsymbol{x}).
\]

The existing thermodynamic API supplies the native-to-canonical gradient
transformation and the architecture's validated concentration scaling. No
legacy simulation-unit rescaling is added. The two reported derivatives are

\[
\boldsymbol g_{x,m}=\nabla_x\Psi_m,\qquad
\boldsymbol g_{\bar z,m}=c_{\mathrm{tot}}\boldsymbol g_{x,m}.
\]

The plots use the two varied components of the **positive gradient with
respect to the displayed feature coordinates**, at fixed concentration and
fixed other coordinates. The arrows are gradients, not a dynamical velocity
field. This evaluates a fixed-concentration slice, rather than following a
P=0 surface with concentration varying across the grid.

All internal energy quantities use kT and concentrations use nm^-3:

| Quantity | Units/convention |
| --- | --- |
| `c_tot` | Total chain number density, nm^-3 |
| `f_ex` | Excess free-energy **density**, kT nm^-3 |
| `grad_x_a` | Derivative with respect to canonical latent concentration; kT per unit canonical feature |
| `grad_zbar_a` | Derivative with respect to the displayed feature coordinate; kT nm^-3 per unit canonical feature |

The ideal free energy is not added. A feature-space point can represent many
chemical compositions, whose ideal mixing entropies need not be the same.
No pressure roots, phase coexistence, smoothing, or convexity constraints are
introduced by this script. The archived rho_chem/rho_mix and canonical
transforms are reused without recomputation. The grid population mean is
the same uniform arithmetic mean over exported feature rows as in P=0.

All models evaluate the **same** canonical point, matching P=0. Physical
sequence mixtures evaluated with each member's own sequence encoding remain
the role of `composition_sweep_v1.py`.

## Output files

The output root has `metadata.json` and `analysis.log`. Each `d_NNN/` contains:

| File | Contents |
| --- | --- |
| **`lattice.tsv`** | Every bounding-box grid point, coordinates, masks, concentration, ensemble mean and population SD of the free energy and the two in-plane gradient components. |
| `member_fields.tsv` | Individual-model values at evaluated points, with `point_id`, zero-based `model_index`, checkpoint/fold/member identity. Controlled by `--save-member-values yes/no` (default yes). Join coordinates from `lattice.tsv` by `point_id`. |
| `grid.npz`, `lp_indices.npy` | The same geometry arrays and selected hull-cloud rows used by P=0. |
| `slice_definition.json` | Exact varied/fixed coordinates, offsets, population mean, shared-point convention, and fixed concentration. |
| `metadata.json`, `analysis.log` | Invocation, resolved settings, model and feature hashes, canonical config, units, precision, LP call, gradient definitions, plot scaling, counts, and completion status. |
| `free_energy_excess.png` | Mean excess free-energy-density heatmap. |
| **`free_energy_excess_gradient.png`** | The same heatmap with the in-plane ensemble-mean gradient arrows. |
| `free_energy_excess_std.png` | Population SD of excess free-energy density across selected models. |

For axes 1 and 2, the principal TSV value columns are:

```text
f_ex_mean  f_ex_std
grad_x_1_mean  grad_x_1_std  grad_x_2_mean  grad_x_2_std
grad_zbar_1_mean  grad_zbar_1_std  grad_zbar_2_mean  grad_zbar_2_std
```

Coordinate names match P=0: `z_1_varied`, `z_2_varied`, and, for example,
`z_3_fixed`. Gradient suffixes use the actual 1-based selected axis numbers,
even when their order is reversed. `retained=1` and `evaluated=1` identify
evaluated points in a complete run. Excluded rows have blank numerical field
values. Full-hull flags are still reported under the projected policy.
With LP subsampling, an outside flag refers to the selected hull cloud and
does not prove exclusion from the full sequence-pool hull.

Means and population SDs are computed independently for each scalar or
gradient component (`ddof=0`), not SEMs. One selected model gives zero spread.
Stored values and coordinate transformations use float64 and TSV output uses
17 significant digits. Decoder evaluation defaults to float64, matching P=0;
`--evaluation-dtype checkpoint` preserves the checkpoint dtype. Saved models
are not changed. Zero concentration is allowed as a vacuum-limit diagnostic.

## Arrow controls

`--arrow-stride 5` selects every fifth grid index along each axis for display;
all retained grid points are still evaluated and saved. Reduce it to 1 for
every nonzero vector. Arrows originate only at evaluated points; excluded
points remain blank. A single scale preserves relative vector magnitudes.

`--arrow-scale auto` chooses a global Matplotlib quiver scale. Its resolved
numeric value is recorded in `d_NNN/metadata.json` under `plotting`.
Alternatively supply a positive number; larger scale means shorter arrows.
No vector-by-vector normalization is applied. Axes have equal geometric
aspect ratio, and a reference arrow supplies a magnitude key.

## Checks

```bash
python3 -m unittest discover -s tests -p test_free_energy_slice_v1.py
```

The calculation references `wmjac/idpmixture/calc_fe_slice.py`, Git blob
`3931bd10306b81c5150145499a5de2df0eb16f54`, inspected through GitHub during
implementation. Geometry and filtering use the current P=0 helpers, rather
than copying legacy geometry or unit handling.

Before delivery, four analytical tests passed, covering a five-member nonlinear
field, finite-difference chain-rule checks, component spreads, zero/empty
evaluations, reversed axes, nonfinite failure handling, and plot masking.
Saved-checkpoint integration at d=2 and d=3 (two selected members each) verified
exact equality of the grid/hull/LP arrays with P=0, canonical gradients against
independent native-decoder finite differences, TSV member means/SDs, fixed
coordinates, zero-concentration and empty slices, both dtype modes, protected
outputs, and logs/hashes. Model inputs and source were unchanged. Example plots
were inspected visually. CPU checks were run; live Slurm/GPU execution was not
exercised.
