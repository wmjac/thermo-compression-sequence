# Post-training sequence and EYM analyses

Unpack this overlay in the thermo-compression-sequence repository root. It adds
scripts under `analysis/`, a Slurm template, documentation and scientific tests.
No existing project files, trained models, canonical bases, or training source
hashes are changed. Both the original and vectorized training versions are
supported. No pip installation is required.

1. Generate a sequence list (run from the repository root):

```bash
python3 analysis/enumerate_sequences_v1.py \
    --amino-acids AF --length 20 --reversal-symmetry yes \
    --output /path/to/af20_batch000001_best/af20_sequences.txt \
    --count-only no --overwrite no
```

Replace AF with EK and edit the output path for EK20. There are 524,800
reversal classes, or 1,048,576 oriented sequences with `--reversal-symmetry no`.

2. Copy `slurm/post_training_analysis_v1.slurm` into each `_best` directory.
Edit the canonical `RUN_DIR` (normally `analysis01`), sequence-list/test paths,
dimensions, reference dimension and output directory. Copy your HPC resource
directives if you want to use a different GPU. Submit from the `_best` directory.

The template runs the two analysis scripts independently and gives explicit
values for every CLI option. Measures are inherited from canonicalization.
Do not point RUN_DIR at `_best/best`: use the directory containing `canonical/`
and, for combined runs, `run_views/`.

See `docs/post_training_analysis_v1.md` for independent commands, file schemas,
mathematical definitions, controls, interpretation and verification.
