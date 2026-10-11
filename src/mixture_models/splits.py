import numpy as np


def make_folds(records, num_folds, seed):
    if num_folds < 2 or num_folds > len(records):
        raise ValueError("num_folds must be between 2 and the number of pooled systems.")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(records))
    validation = np.array_split(permutation, num_folds)
    assignments = np.empty(len(records), dtype=np.int64)
    for fold, indices in enumerate(validation):
        assignments[indices] = fold
    # Preserve source order within each fold's datasets, matching the legacy loader.
    return {"seed": seed, "num_folds": num_folds,
            "split_unit": "simulation system", "uid": [r.uid for r in records],
            "validation_fold": assignments.tolist(), "permutation": permutation.tolist()}
