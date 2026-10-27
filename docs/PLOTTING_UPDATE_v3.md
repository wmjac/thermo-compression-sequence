# Logarithmic loss and RMSE plots

Unpack this overlay in the thermo-compression-sequence repository root:

```bash
tar -xzf /path/to/thermo-compression-sequence_plotting_v3.tar.gz
```

This replaces only `src/mixture_models/plotting.py` and adds
`slurm/analysis_v2.slurm` and `examples/replot_dimension_losses_v1.py`.
It works with both the original and vectorized training versions.
No model, training, checkpoint-selection, canonicalization, unit-conversion,
evaluation, or numerical-metric calculations are changed.

Future `evaluate` calls automatically write the existing linear plot and two
additional PNGs. For each evaluation name (for example, validation):

- `loss_by_dimension_validation.png`: loss on a linear y-axis.
- `loss_by_dimension_validation_log.png`: loss on a logarithmic y-axis.
- `rmse_by_dimension_validation_log.png`: sqrt(loss) on a logarithmic y-axis.

Dimension d stays linear. RMSE is the dimensionless normalized pressure RMSE:
sqrt(mean(((P - P_hat)/(c_tot kT))**2)). The square root is taken after computing
the saved ensemble pressure loss. It is not an average of model-wise RMSEs.
Zero values are omitted and annotated in logarithmic plots without adding a
floor or changing archived values.

For future combined-run analyses, copy `slurm/analysis_v2.slurm` to each `_best`
directory and edit its settings as for v1 (test JSONs, paths, ensemble size and
rho_mix). It adds the same per-evaluation PNGs and combined validation/test plots:
`loss_by_dimension.png`, `loss_by_dimension_log.png`, and
`rmse_by_dimension_log.png`. Original analysis_v1.slurm files are not replaced.

Existing analyses need only replotting. From either `_best` directory, run:

```bash
python3 /home/wjacobs/git/thermo-compression-sequence/examples/replot_dimension_losses_v1.py analysis01/evaluation
```

Use the existing PyTorch conda environment (NumPy and Matplotlib are needed;
the replot helper does not import PyTorch or use a GPU). No installation is
required. Optionally select evaluations with `--names validation test`.
The helper reads per-dimension metrics, falling back to summary JSONs. It
updates PNGs only, including the existing linear plots; metrics, input data,
checkpoints and run completion status are not modified. Incomplete analyses
will plot only dimensions with saved metrics; replot after completing the run.

As with any source edit, strict training-resume source checks see this change.
Use the original checkout/source snapshot when resuming an original training run.
