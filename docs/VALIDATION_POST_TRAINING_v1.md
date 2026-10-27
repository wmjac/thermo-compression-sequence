# Post-training v1 verification

All listed checks passed before packaging.

- 11 unittest checks: reversal orbits/counts; default/explicit chemical measures;
  common-measure identity reordering and changed-prior rejection; output-directory
  protection; known rotations and missing directions; uncentered geometry;
  weighted numerical rank; degenerate-block mixing; low-rank TM loss and EYM
  eigenvalues against a dense operator; exact mixture construction; unequal-batch
  canonical quadrature weights.
- Full AF20 reversal enumeration: 524,800 unique, sorted representatives of
  length 20; exactly 1,024 palindromes. Count without reversal: 1,048,576.
- Synthetic-data checkpoint integration: original model, d=1,2,3, two selected
  models per dimension, two fitting/validation folds, one training epoch. These
  are deliberately tiny verification fixtures, not production protocol defaults.
- Export: ordinary run input, uneven encoding batches, exact float64 TSV
  round-trip against optional member NPYs; population vector RMSD; input ordering;
  exact and reversal-aware training/test membership; explicit absence of test data.
- Comparison: combined run_views input; original canonical priors; 48 archived
  mixtures with canonical batch sizes 10,10,10,10,8 and analysis batches of 13.
  Original batch-mean quadrature weights preserved exactly. All TM losses obeyed
  their reference EYM bounds and projection residuals within numerical tolerance.
- Leading chemical and function subspace controls verified at d=1,2,3.
  Reference EYM function orthonormality errors below 1e-14 on the fixture.
- Geometry-only path with random controls disabled; feature export/checkpoint
  loading under the vectorized-source overlay; explicit empty test dataset.
- Linear/log TM loss and log TM RMSE PNGs produced; scientific plots inspected.
- Input training/canonicalization file hashes unchanged after integration.
- Python syntax, Slurm shell syntax, and explicit template coverage of all
  export/comparison CLI options checked.

Numerical integration environment: Python 3.12, PyTorch 2.8.0+cpu,
NumPy 2.3.5, SciPy 1.17.0, Matplotlib 3.10.8. GPU inference/performance and a live
Della Slurm submission were not exercised in this environment.

Run the packaged scientific tests with:

```bash
python3 -m unittest discover -s tests -p test_post_training_analysis_v1.py -v
```
