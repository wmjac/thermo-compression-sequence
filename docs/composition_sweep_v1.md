# Fixed-composition concentration sweep, v1

Unpack the archive in the top-level `thermo-compression-sequence` directory:

```bash
tar -xzf thermo-compression-sequence_composition_sweep_v1.tar.gz
```

The archive adds `analysis/composition_sweep_v1.py`, this document, and a small
numerical test. It replaces no existing project files and changes no training,
checkpoint, canonicalization, or P=0 solver code. No `pip install` is needed.
Use the same Python environment as the training/evaluation workflow. The script
locates `src/` relative to itself, so it also works from a different working
directory. No other analysis overlay is required.

## Example: equimolar E20/K20

From your EK `_best` directory, using its existing `analysis01` canonical run:

```bash
python3 /home/wjacobs/git/thermo-compression-sequence/analysis/composition_sweep_v1.py \
    --run-dir "$PWD/analysis01" \
    --dimension 2 \
    --component EEEEEEEEEEEEEEEEEEEE 0.5 \
    --component KKKKKKKKKKKKKKKKKKKK 0.5 \
    --c-min 1e-8 \
    --c-max 0.25 \
    --num-points 2001 \
    --spacing log \
    --output "$PWD/e20_k20_d002.tsv" \
    --basis canonical \
    --evaluation-dtype float64 \
    --batch-size 1024 \
    --device cpu \
    --threads 1
```

This is a small inference task; it can run inside your existing allocation.
Use `--device cuda` to use an allocated GPU. Repeat `--component SEQUENCE X`
for any number of distinct sequences. For a single sequence, supply fraction
`1`. Fractions are **mole fractions of chains**, not residue fractions or
absolute concentrations. They must sum to one within absolute tolerance
`1e-12`; they are never renormalized. Zero fractions are allowed. Exact duplicate
sequences are rejected so that their fractions can be combined explicitly.
The architecture validates all sequence lengths and residue symbols.

`--run-dir` accepts an ordinary saved run, an `analysis01` directory containing
`run_views/d_NNN`, or an individual dimension's run view. `--basis canonical`
requires that selection's completed canonicalization. `--basis native` uses
the selected checkpoints directly and does not require canonicalization.
Physical predictions are invariant to the basis up to numerical precision.

Both grid endpoints are included. `--spacing linear --c-min 0` includes the
vacuum limit, with ideal free energy zero there. Logarithmic spacing requires
strictly positive `--c-min`. The concentration window is supplied directly
in **nm^-3**; there is no simulation-unit conversion in this script.

## Quantities and conventions

Every model evaluates the same specified physical mixture, using its own
sequence features and its own decoder. Features are not averaged across
models before decoding. Specifically,

\[
c_i=x_i c_{\mathrm{tot}},\qquad
\boldsymbol{x}^{(m)}=\sum_i c_i\boldsymbol{z}^{(m)}_i.
\]

The existing thermodynamic API supplies the excess free-energy density and
pressure, including all existing decoder concentration scaling:

\[
P^{\mathrm{ex}}=\boldsymbol{x}\cdot\nabla f^{\mathrm{ex}}-f^{\mathrm{ex}}.
\]

In kT energy units, the added ideal contributions are

\[
f^{\mathrm{id}}=\sum_i c_i[\ln(c_i/c_{\mathrm{ref}})-1],\quad
P^{\mathrm{id}}=c_{\mathrm{tot}},\quad c_{\mathrm{ref}}=1\,\mathrm{nm}^{-3}.
\]

The zero-concentration limit uses `0 ln(0) = 0`. All four quantities are
**densities**, in **kT nm^-3**. Here `f` is Helmholtz free-energy density,
not free energy per chain. The ideal reference fixes a linear free-energy
convention; the excess contribution retains the checkpoint's convention.
Pressure training alone does not determine arbitrary species-dependent linear
terms in free energy. Keep the same conventions when comparing compositions.
The ideal reference above is separate from the architecture's existing decoder
reference concentration of 0.1 nm^-3, which this script does not change.

Sequence encodings use checkpoint precision, matching ordinary inference and
sequence-feature exports. The decoder is then evaluated in float64 by default,
matching the P=0 diagnostic default. `--evaluation-dtype checkpoint` retains
the original decoder precision. Both choices and actual dtypes are recorded;
saved checkpoint files are never changed. Aggregation and text output use
float64 and 17 significant digits.

## Output files

For `--output e20_k20_d002.tsv`:

| File | Contents |
| --- | --- |
| `e20_k20_d002.tsv` | One row per concentration: `c_tot_nm^-3`, `num_models`, `f_ideal`, `P_ideal`, then `f_ex_mean`, `f_ex_std`, `P_ex_mean`, `P_ex_std`, `f_mean`, `f_std`, `P_mean`, `P_std`. |
| `e20_k20_d002.members.tsv` | Every selected model's four predictions at every concentration, with checkpoint, fold, and member identity. `model_index` is zero based. |
| `e20_k20_d002.metadata.json` | Exact command, arguments, composition, units, ideal reference, features for each model, model identities and hashes, canonical config if used, source/Git provenance, output hashes, and completion status. |

Standard deviations describe the population spread over the selected models
(`ddof=0`), not SEMs or confidence intervals. With one model the spread is zero.
All selected models contribute equally, independent of their fold: this is
inference on a specified composition, not held-out validation scoring.

Existing output files are protected unless `--overwrite` is supplied. That
option replaces only these three files. A complete metadata file is written
last; a failed run may leave incomplete tables without completion metadata.
The sweep evaluates the chosen window without root finding, clipping negative
pressures, or solving phase coexistence. It does not establish whether the
chosen sequences or concentrations lie within the training distribution.

For numerical checks after unpacking:

```bash
python3 -m unittest discover -s tests -p test_composition_sweep_v1.py
```

Validation before delivery: four analytical/validation tests passed. Integration
on saved synthetic-data checkpoints (d=2, two selected models) verified native
and canonical agreement, independent direct decoder predictions, non-equimolar
weighting, both grids, zero concentration, output means/SDs and hashes, both
decoder precision options, and protection of saved inputs and existing outputs.
These checks ran on CPU; GPU inference was not exercised.
