# Post-training analyses, v1

These scripts analyze existing pressure-trained, selected and canonicalized
models. They do not train, select models, modify canonical coordinates, change
units, or substitute a new mixture distribution. Their source is outside
`src/mixture_models`, so installing the overlay does not change the package's
strict training-resume source manifest. Script hashes and the repository commit
are included in analysis metadata.

## Inputs and directory layouts

`--run-dir` accepts either a conventional run directory containing `best/`,
`canonical/`, `inputs.json`, `inputs/` and `folds.json`, or the combined
`_best/analysis01` directory created by analysis_v1.slurm/analysis_v2.slurm. The
latter is recognized by `run_views/d_NNN/`; each dimension retains its own
archived training data and provenance. Keep the source runs and links intact.

Selected checkpoint hashes, canonical selection identity, completion status,
and all archived canonical artifact hashes are verified before analysis.
No existing analysis output is replaced unless `--overwrite yes` is explicit.
Overwrite applies to a previous output from these scripts or an empty directory;
it does not clear arbitrary directories containing other data.
Incomplete outputs have `status: in_progress`; only success writes `complete`.

## Enumerating sequences

```bash
python3 analysis/enumerate_sequences_v1.py \
  --amino-acids AF --length 20 --reversal-symmetry yes \
  --output af20_sequences.txt --count-only no --overwrite no
```

The alphabet is a string of distinct uppercase standard amino-acid letters.
The sequence length belongs to this enumeration task; it is not hard-coded to
20. Exported sequences are subsequently validated by the model architecture.
Output is sorted, headerless ASCII, one sequence per line, directly accepted
by the feature-export script. A `.metadata.json` sidecar records the alphabet,
length, reversal choice, representative convention, count and file hash.

With `--reversal-symmetry no`, all q^L oriented sequences are written. With
`yes`, the lexicographically smaller of each sequence/reverse pair is written,
and each palindrome is written once. The count is
(q^L + q^ceil(L/2))/2. For AF20 or EK20 the counts are 1,048,576 without reversal
identification and 524,800 with it. `--count-only yes` prints the count without
creating a file. Enumeration streams and does not retain a giant deduplication
set. Very large spaces are not silently truncated.

## Exporting features

With the usual conda environment active, run from the repository root:

```bash
python3 analysis/export_sequence_features_v1.py \
  --run-dir /path/to/af20_batch000001_best/analysis01 \
  --dimensions 1 2 3 4 5 6 \
  --sequence-list /path/to/af20_sequences.txt \
  --test-json /path/to/independent_test.json \
  --membership-symmetry reversal \
  --output-dir /path/to/af20_batch000001_best/post_training01/features \
  --batch-size 2048 --device cuda --threads 1 \
  --save-member-features no --overwrite no
```

`--test-json` accepts multiple files. Explicitly supply the flag without paths
if there is no test set; metadata records that no test set was defined. Files
are loaded using the existing JSON reader and mandatory units block.
Sequence input can also be a one-dimensional string NPY, a JSON list, or a JSON
object with a `sequences` list. Duplicate sequences are rejected, not silently
removed. Output order is input order.

Each `features_d_NNN.tsv` has columns:

```
sequence  in_training  in_test  z_1 ... z_d  feature_rmsd
```

The actual delimiter is a tab. Membership columns are 0/1. Training membership
means occurrence in any mixture in the archived pooled training data, including
systems assigned to validation folds; it does not mean occurrence in every
individual member's fitting fold. Test membership means occurrence in the
supplied test data. Mixture-level overlap is not inferred from these flags.
With `exact`, membership requires the literal sequence. With `reversal`, a
sequence and its reverse share a membership key. This is helpful when the
enumerator has retained just one representative. It changes flags only; it
does not transform or deduplicate the input list or change model evaluation.

For selected models m=1,...,k, let z_m(s) be the feature after its SAVED canonical
transform. The TSV feature is mean_m z_m(s). Its scalar uncertainty is

```
sqrt(mean_m(sum_j((z_mj(s) - mean_m(z_mj(s)))**2)))
```

There is no division by d or sqrt(k), and this is not a standard error. All
means and RMSDs are accumulated in float64; `.17g` TSV values round-trip these
floats. Encoder inference uses the saved checkpoint dtype, with no claim of
extra precision beyond that dtype. Only one dimension and a sequence batch
are resident at once. Optional `--save-member-features yes` writes disk-backed
float64 NPY arrays of shape (k, number_of_sequences, d), in selected-model order.
TSVs and metadata always suffice to recover the mean, RMSD and membership.

## Dimension comparisons

```bash
python3 analysis/compare_dimensions_v1.py \
  --run-dir /path/to/af20_batch000001_best/analysis01 \
  --dimensions 1 2 3 4 5 6 --reference-dimension 6 \
  --output-dir /path/to/af20_batch000001_best/post_training01/dimension_comparison \
  --rank-rtol 1e-10 --rank-atol 0.0 --degeneracy-rtol 0.01 \
  --random-controls 100 --seed 0 \
  --response-analysis yes --save-responses yes \
  --batch-size 2048 --device cuda --threads 1 --overwrite no
```

The reference dimension D must be supplied and included in the dimensions;
all compared dimensions must be <=D. Inspect the reference spectrum, member
spread, subspace stability and predictive results to choose a reference without
unnecessary noisy directions. The script does not silently select a physical
noise cutoff. `rank-rtol` and `rank-atol` identify numerical null directions:
retain lambda > max(rank_atol, rank_rtol*lambda_max). All eigenvalues and masks
are saved, including discarded numerical directions. Rank-deficient comparisons
report their effective ranks; missing principal angles are not replaced by zero.
Negative PSD-Gram eigenvalues within the specified numerical eigentolerance
are clipped to zero; larger negative values raise an error. No positive spectral
floor is added. Subtractive squared-residual roundoff is also checked before
clipping a tiny negative value to zero; `tm_loss_raw` retains the original value.

### Measures and reference EYM modes

The analysis uses the canonicalization sequence pool and its archived
rho_chem, NOT the external feature-export list. If rho_chem is absent (as in
the original canonicalization format), it is uniform over that pool. Explicit
archived uniform or per-sequence weights are supported; unsupported or
inconsistent definitions fail instead of being replaced. Pools can have a
different ordering across dimensions: identities are checked and features are
reordered. Weights and archived rho_mix definitions must agree across dimensions.
The resolved chemical pool and weights are saved in `chemical_measure.npz`.

For each selected reference model, let Z contain canonical sequence features
and W=diag(rho_chem). Its saved metric kernel is Gamma=Z Z^T. Nonzero EYM
eigenvalues and modes are obtained from C=Z^T W Z, a D-by-D matrix. The
sqrt(W)-weighted chemical eigenvectors are Q=sqrt(W) Z U Lambda^(-1/2).
No centering, translation or fitted scale factor is applied. A dense
number_of_sequences-by-number_of_sequences kernel is never formed.

EYM analyses are performed separately for each selected reference member.
The final shared canonical coordinate order is not assumed to be the EYM order
of each member. Averaging aligned coordinates is used only in the explicitly
labelled descriptive mean-geometry output; it is never treated as a physical
ensemble encoder/decoder or a D-dimensional ensemble TM kernel.

### Geometry, mixing and controls

Principal angles compare weighted FEATURE-COLUMN SUBSPACES on the common
sequence population. They are not angles between individual sequence vectors
in unequal-dimensional ambient coordinate spaces. Orthonormal bases are
constructed without centering. Angles are acos of overlap singular values.
Cross comparisons target the leading d reference EYM modes, with full-reference
containment also reported. A small angle means a retained direction agrees;
overlap fractions and effective ranks must be considered alongside angles.
Principal angles detect changes in the subspace, not changes of basis within
an identical subspace. The latter have zero principal angles and are described
by the signed loading matrices and mixing statistics instead.

Outputs include:

- `reference_spectrum.tsv`, `reference_spectrum_summary.tsv`,
  `reference_eigensystems.npz`: per-member spectra/eigenvectors and member spread.
- `reference_truncation.tsv`: EYM spectral tails and sqrt(tail) for every d.
- `cross_dimension_geometry.tsv`: all lower/reference member pairs, angles,
  overlap, full-reference containment, padded Procrustes RMSD and residual to
  the leading d reference coordinates.
- `within_dimension_geometry.tsv`: the same diagnostics for every unordered
  pair of distinct selected members at each dimension, including D.
- `reference_leading_subspace_controls.tsv`: compare the leading d EYM chemical
  subspaces between reference ensemble members at every d. This detects uncertain
  leading directions even when the full D-dimensional reference span is stable.
- `principal_angles.tsv`: individual chemical and (when enabled) response-space angles.
- `ensemble_mean_geometry.tsv`: descriptive aligned-mean comparisons and the
  old-style normalization sqrt(within_d_rmsd^2+within_D_rmsd^2). This ratio is a
  diagnostic scale, not a statistical z score or an uncertainty-of-the-mean.
- `mode_overlaps_d_NNN.npz`: signed overlap matrices in each member's own EYM
  basis and projections of the saved canonical axes onto reference modes.
- `mode_mixing.tsv`: leading-mode capture, tail participation, unexplained
  fraction, entropy and effective number of reference modes per lower axis.
  Effective mode count is 1/sum(p_alpha^2) after normalizing squared overlaps
  within the reference span; it does not hide the separately reported unexplained fraction.
- `random_subspace_controls.tsv`: Haar random d-subspaces inside the identified
  reference span, with spectra matched to its leading d eigenvalues for the
  Procrustes control. The chemical ambient population and reference model stay
  fixed; these are not random sequences or new thermodynamic priors.
- `comparison_summary.tsv`: empirical pair/control summaries.

Procrustes permits orthogonal rotations and reflections, but no translations or
scaling. The full padded residual includes the unavoidable discarded reference
tail; therefore the leading-d residual and reference-truncation control are also
reported. Near-degenerate eigenvalues permit rotations with little physical
meaning. `degeneracy-rtol` groups nearby reference eigenvalues and the mixing
table includes effective BLOCK participation, so rotations within such a block
can be distinguished from mixing separated modes. Inspect eigensystems for
degeneracy of lower-model modes as well. Mode labels within degenerate spaces
should not be interpreted as uniquely identifiable physical directions.

Selected models and their pairwise comparisons share training data. Pairwise
standard deviations and quantiles are descriptive ensemble controls; they are
not independent-sample confidence intervals. No significance p-values are assigned.

### Common-mixture response analysis and actual TM loss

Enabled by default. The exact archived mixture draws from the reference
canonicalization are reused for every model, with the SAME canonical sequence
pool, concentrations and compositions. No rho parameters are exposed as
analysis overrides. All mixtures are already in nm^-3; decoder responses are
dimensionless excess chemical potentials through the existing model API.

The original canonical metric aggregation is preserved. For `batch-mean`,
each archived sample in batch b receives weight 1/(number_of_batches*batch_size_b).
For `sample-mean`, all samples receive 1/number_of_samples. Analysis batching
controls device memory only; it never changes those archived quadrature weights.
`shared_mixture_samples.npz` saves draws, weights and canonical batch sizes.

Write mu_ref(c,s)=z_ref(s).g_ref(c) and mu_low(c,s)=z_low(s).g_low(c), where
gradients and features refer to the same saved canonical coordinates. The TM
loss is the rho_chem/rho_mix weighted mean of (mu_ref-mu_low)^2. Chemical and
response cross-Gram matrices evaluate it without a sequence-by-mixture matrix.
The function-space reference EYM modes and response overlap are obtained by
factoring these same small matrices. This is an analysis of existing models
trained with PRESSURE LOSS; it does not optimize any model on TM loss.

- `tm_losses.tsv`: actual reference-relative TM loss/RMSE for all lower/reference
  pairs, its EYM tail bound, best function-subspace projection residual, and
  within-dimension TM disagreements as controls.
- `response_mode_retention.tsv`: fraction of each reference chemical-potential
  function mode represented by the lower-model RESPONSE span on these mixtures.
  This complements the chemical-feature subspace comparison.
- `reference_leading_function_controls.tsv`: stability of the leading d EYM
  function subspaces between reference members under the same mixture quadrature.
- `reference_function_coefficients.npz`: coefficients mapping each reference
  member's saved canonical decoder gradients to its orthonormal EYM functions.
- `common_quadrature_reference_spectrum.tsv`: reference spectra computed from
  these re-evaluated responses. These are the spectra used for the TM bounds.
- `responses_d_NNN.npy`: optional float64 canonical decoder gradients, shape
  (k, number_of_archived_mixtures, d), written when `--save-responses yes`.

The saved-metric spectrum and re-evaluated reference spectrum may differ by
floating-point evaluation error; both are retained and explicitly labelled.
Lower-dimensional canonicalization runs may have used different Monte Carlo
seeds; actual TM losses nevertheless use one identical reference quadrature.
Existing canonicalization convergence diagnostics remain available in source
artifacts. Fresh Monte Carlo replicates are not generated by this analysis.

The EYM tail is a lower bound for reference-relative TM loss, not pressure
loss. The reference itself is a learned model, not exact simulation
thermodynamics. Thermodynamic integrability can favor learned directions that
differ from the leading reference modes; disagreement is not automatically an
optimization failure. With `--response-analysis no`, geometry and controls are
still computed, but actual TM losses/response-space overlaps are omitted.

PNG outputs include reference spectra, subspace comparisons, per-d mode-overlap
heat maps and actual TM-loss comparisons with reference EYM tails and reference
ensemble disagreement. All numeric outputs retain full floating-point precision.
TM-loss-versus-dimension plots include linear loss, log loss, and log RMSE PNGs.
Their curves average pairwise squared losses before taking a square root for
RMSE; individual pair values are also shown. Zero values on logarithmic plots
are omitted and annotated without applying an artificial positive floor.

## Verification

```bash
python3 -m unittest discover -s tests -p test_post_training_analysis_v1.py -v
```

Tests check reversal-orbit counts and representatives, explicit/uniform chemical
measures, known rotated/missing subspaces, absence of centering, numerical rank,
degenerate-block mixing, exact archived-mixture reconstruction, and low-rank TM
loss/eigenvalues against a directly formed small dense operator.

The release also undergoes a CPU integration using synthetic data and actual
architecture checkpoints at d=1,2,3. This verifies TSV round-trip precision,
reversal-aware membership, combined run views, shared-mixture quadrature with
an unequal final canonical batch, TM bounds and read-only source artifacts.
These tiny synthetic training settings are verification fixtures, not proposed
production training parameters. GPU performance is not established by CPU tests.
